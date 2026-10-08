#!/usr/bin/env python3
"""Run paired untrained-vs-traffic-trained Traffic-Qwen V1 seeds on Jinan."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JEVLIGHT = Path(os.environ.get("JEVLIGHT_ROOT", str(ROOT / ".deps/JevLight")))
SPLITS = ("train.jsonl", "validation.jsonl", "test.jsonl")
METRICS = (
    "reward", "avg_queue", "avg_waiting_time_seconds", "avg_travel_time_seconds",
    "model_decisions", "request_fallbacks", "guardrail_corrections", "runtime_seconds",
    "phase_switches", "current_phase_retention", "maximum_non_service_time_seconds",
    "latency_p50_ms", "latency_p95_ms", "GPU_peak_memory_GiB",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\nCWD: " + str(cwd) + "\n\n")
        log.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reproducibility_v1.json")
    parser.add_argument("--jevlight-root", type=Path,
                        default=Path(os.environ.get("JEVLIGHT_ROOT", str(DEFAULT_JEVLIGHT))))
    parser.add_argument("--only-seeds", type=int, nargs="*", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = args.only_seeds or cfg["seeds"]
    horizon = args.horizon or cfg["horizon_seconds"]
    jevlight = args.jevlight_root.resolve()
    if not (jevlight / ".git").exists() or not (jevlight / "utils/oneline.py").is_file():
        raise FileNotFoundError(f"JevLight checkout not found: {jevlight}")
    data_src = jevlight / "data/traffic_qwen/v1_dual_choice"
    data_dst = ROOT / "data/traffic_qwen/v1_dual_choice"
    data_dst.mkdir(parents=True, exist_ok=True)
    expected = {}
    manifest_path = ROOT / "data/manifests/source_splits.sha256"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in SPLITS:
            expected[parts[1]] = parts[0]
    split_hashes = {}
    for name in SPLITS:
        src = data_src / name
        dst = data_dst / name
        if not src.is_file():
            raise FileNotFoundError(src)
        actual = digest(src)
        if name in expected and actual != expected[name]:
            raise ValueError(f"Source split checksum mismatch for {name}: {actual} != {expected[name]}")
        if dst.exists():
            if digest(dst) != actual:
                raise FileExistsError(f"Refusing to overwrite different local data: {dst}")
        else:
            shutil.copy2(src, dst)
        split_hashes[name] = {"sha256": actual, "bytes": src.stat().st_size,
                              "source": f"JevLight/data/traffic_qwen/v1_dual_choice/{name}", "working_copy": f"data/traffic_qwen/v1_dual_choice/{name}"}

    traffic = jevlight / "data/Jinan/3_4" / cfg["traffic_file"]
    roadnet = jevlight / "data/Jinan/3_4/roadnet_3_4.json"
    for path in (traffic, roadnet):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_root = ROOT / "results/reproducibility"
    runs_root = output_root / "runs"
    train_root = output_root / "training"
    logs_root = output_root / "logs"
    for path in (runs_root, train_root, logs_root):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(jevlight))),
        "WANDB_MODE": "offline", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    base_train = json.loads((ROOT / cfg["training_config"]).read_text(encoding="utf-8"))
    base_train.update({
        "model_name": cfg["base_model"], "model_revision": cfg["base_model_revision"],
        "local_files_only": True,
        "train_data": cfg["train_split"], "validation_data": cfg["validation_split"],
        "test_data": cfg["test_split"],
    })

    manifest = {
        "experiment": cfg["experiment"], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": seeds, "horizon_seconds": horizon, "dataset": cfg["dataset"],
        "traffic_file": "Jinan/3_4/anon_3_4_jinan_real.json", "traffic_file_sha256": digest(traffic),
        "roadnet_file": "Jinan/3_4/roadnet_3_4.json", "roadnet_file_sha256": digest(roadnet),
        "jevlight_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=jevlight, text=True).strip(),

        "dataset_splits": split_hashes, "base_model": cfg["base_model"],
        "base_model_revision": cfg["base_model_revision"], "training": base_train,
        "guardrail": cfg["guardrail"], "fallback": cfg["fallback"],
        "python": sys.version, "started_monotonic": time.monotonic(),
    }
    (output_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for seed in seeds:
        print(f"=== Seed {seed}: training ===", flush=True)
        train_cfg = dict(base_train)
        train_cfg["seed"] = seed
        train_cfg["output_dir"] = f"checkpoints/reproducibility/trained-seed-{seed}"
        train_cfg["smoke_output_dir"] = f"checkpoints/reproducibility/smoke-seed-{seed}"
        train_cfg["artifact_dir"] = f"results/reproducibility/training/seed-{seed}"
        cfg_path = ROOT / f"configs/.generated_reproducibility_seed_{seed}.json"
        if cfg_path.exists():
            raise FileExistsError(f"Refusing to overwrite config: {cfg_path}")
        cfg_path.write_text(json.dumps(train_cfg, indent=2) + "\n", encoding="utf-8")
        run([sys.executable, str(ROOT / "scripts/train_traffic_qwen.py"), "--config", str(cfg_path)],
            cwd=ROOT, env=env, log_path=logs_root / f"train-seed-{seed}.log")
        training_metrics = json.loads((ROOT / train_cfg["artifact_dir"] / "metrics.json").read_text(encoding="utf-8"))
        if training_metrics.get("status") != "passed":
            raise RuntimeError(f"Training did not pass for seed {seed}; see {logs_root / f'train-seed-{seed}.log'}")

        for arm, model_path in (
            ("untrained", cfg["base_model"]),
            ("traffic_trained", str(ROOT / train_cfg["output_dir"])),
        ):
            output = runs_root / f"seed-{seed}" / arm
            command = [
                sys.executable, str(ROOT / "scripts/evaluate_cityflow.py"),
                "--dataset", cfg["dataset"], "--traffic_file", cfg["traffic_file"],
                "--run_counts", str(horizon), "--model_path", model_path,
                "--model_revision", cfg["base_model_revision"],
                "--device", "cuda", "--seed", str(seed), "--proj_name", f"Repro-{arm}-seed-{seed}",
                "--output_dir", str(output), "--no_guardrail", "--no_fallback", "--skip_benchmark_update",
            ]
            print(f"=== Seed {seed}: evaluating {arm} ({horizon}s) ===", flush=True)
            run(command, cwd=jevlight, env=env, log_path=logs_root / f"eval-seed-{seed}-{arm}.log")
            metrics_path = output / "traffic_qwen_metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing evaluation metrics after run: {metrics_path}")

    rows = []
    for seed in seeds:
        pair = {}
        for arm in ("untrained", "traffic_trained"):
            path = runs_root / f"seed-{seed}" / arm / "traffic_qwen_metrics.json"
            metric = json.loads(path.read_text(encoding="utf-8"))
            pair[arm] = metric
            rows.append({"seed": seed, "model": arm, **{key: metric.get(key) for key in METRICS},
                         "phase_distribution": json.dumps(metric.get("phase_distribution", {})),
                         "duration_distribution_seconds": json.dumps(metric.get("duration_distribution_seconds", {})),
                         "engine_seed": json.loads(next((runs_root / f"seed-{seed}" / arm / "environment_manifests").glob("environment_*.json")).read_text()).get("cityflow_engine_seed")})
        pair["improvements_percent"] = {
            key: ((pair["untrained"].get(key) - pair["traffic_trained"].get(key)) /
                  pair["untrained"].get(key) * 100)
            for key in ("avg_queue", "avg_waiting_time_seconds", "avg_travel_time_seconds")
            if pair["untrained"].get(key) not in (None, 0) and pair["traffic_trained"].get(key) is not None
        }

    with (output_root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arms = {arm: [json.loads((runs_root / f"seed-{seed}" / arm / "traffic_qwen_metrics.json").read_text()) for seed in seeds]
            for arm in ("untrained", "traffic_trained")}
    summary = {"experiment": cfg["experiment"], "seeds": seeds, "horizon_seconds": horizon,
               "metrics_by_arm": {}, "paired_improvements_percent": {}}
    for arm, measurements in arms.items():
        summary["metrics_by_arm"][arm] = {}
        for key in METRICS:
            values = [float(x[key]) for x in measurements if x.get(key) is not None]
            if values:
                summary["metrics_by_arm"][arm][key] = {"values": values, "mean": mean(values),
                    "sample_std": stdev(values) if len(values) > 1 else None}
    for key in ("avg_queue", "avg_waiting_time_seconds", "avg_travel_time_seconds"):
        diffs = [((float(base[key]) - float(trained[key])) / float(base[key]) * 100)
                 for base, trained in zip(arms["untrained"], arms["traffic_trained"])
                 if base.get(key) not in (None, 0) and trained.get(key) is not None]
        summary["paired_improvements_percent"][key] = {"values": diffs, "mean": mean(diffs),
            "sample_std": stdev(diffs) if len(diffs) > 1 else None}
    summary["interpretation_limit"] = (
        "Descriptive comparison over three seeds on one Jinan traffic episode; not an independent multi-scenario test."
    )
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["summary"] = str(output_root / "summary.json")
    (output_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
