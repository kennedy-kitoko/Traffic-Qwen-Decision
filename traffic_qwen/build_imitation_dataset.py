"""Build decision-only V1 imitation rows from the untouched guardrail trace."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from traffic_qwen.action_space import ACTION_IDS, decode, encode
from traffic_qwen.dataset_schema import sample

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = ROOT / "artifacts/baseline_guardrail/state_action.json"
OUT = ROOT / "data/traffic_qwen/v1"
DT = 5
EMBARGO_SECONDS = 60


def _phase(value):
    if not isinstance(value, str):
        raise ValueError(f"Missing phase action: {value!r}")
    match = re.fullmatch(r"Phase[-_ ]([1-4])", value, re.I)
    if not match:
        raise ValueError(f"Unexpected phase encoding: {value!r}")
    return int(match.group(1)) - 1


def _state(record, iid, t, current_phase_proxy, elapsed_proxy, recent_phases, recent_durations):
    moves = {}
    for name, values in record["state"].items():
        moves[name] = {
            "cells": values.get("cells"),
            "queue_vehicles": values.get("queue_len"),
            "avg_wait_seconds": values.get("avg_wait_time"),
        }
    incoming = {name: {"cells": val.get("cells"), "queue_proxy": val.get("queue_len")}
                for name, val in record.get("state_incoming", {}).items()}
    return {
        "intersection_id": iid,
        "simulation_time_seconds": t,
        "current_phase_command_proxy": current_phase_proxy,
        "seconds_since_phase_command_proxy": elapsed_proxy,
        "recent_phase_commands_proxy": list(recent_phases),
        "recent_applied_durations_seconds": list(recent_durations),
        "movement_lanes": moves,
        "incoming_approach_proxy": incoming,
        "approaching_speed": record.get("approaching_speed"),
        "available_fields": ["movement cells", "movement queue length", "movement average wait",
                             "incoming cells/queue proxy", "approaching speed",
                             "recent decisions from the trace"],
        "unavailable_fields": ["exact approaching vehicle count", "downstream occupancy",
                               "recent departures", "per-vehicle max wait", "neighbor states",
                               "CityFlow cur_phase at observation", "elapsed green from CityFlow",
                               "phase service age by movement"],
    }


def build(trace_path=DEFAULT_TRACE, out_dir=OUT):
    raw = json.loads(Path(trace_path).read_text())
    if not isinstance(raw, list) or len(raw) != 12 or any(not isinstance(x, list) for x in raw):
        raise ValueError("Expected the verified trace layout: 12 intersection lists")
    lengths = {len(x) for x in raw}
    if len(lengths) != 1:
        raise ValueError(f"Intersections have unequal snapshot counts: {lengths}")
    n = lengths.pop()
    ids = [f"intersection_{i // 3 + 1}_{i % 3 + 1}" for i in range(12)]
    records = []
    applied_phase = [0] * 12
    phase_age = [0] * 12
    phase_history = [[] for _ in range(12)]
    duration_history = [[] for _ in range(12)]
    for tick in range(n):
        for i, iid in enumerate(ids):
            r = raw[i][tick]
            if not isinstance(r, dict) or "state" not in r or "action" not in r:
                raise ValueError(f"Invalid record at intersection {iid}, tick {tick}")
            recorded_phase = _phase(r["action"])
            if r.get("decision_source") == "laya":
                if r.get("duration") not in (15, 20, 25, 30, 35, 40):
                    raise ValueError(f"Missing/invalid decision duration at {iid}, tick {tick}: {r.get('duration')}")
                guard = r.get("guardrail") or {}
                # The recorded state is sampled before this decision is applied;
                # action is the phase output at this tick, so state current_phase
                # must come from the previous snapshot.
                state = _state(r, iid, tick * DT, applied_phase[i], phase_age[i],
                               phase_history[i][-4:], duration_history[i][-4:])
                gold = encode(recorded_phase, int(r["duration"]))
                records.append({
                    **sample(state, gold, {
                        "episode_id": "jinan_guardrail_single_run",
                        "intersection_id": iid,
                        "simulation_time_seconds": tick * DT,
                        "provenance": "applied JevLight action after guardrail; imitation label, not optimal",
                        "source_model_phase_index": guard.get("model_phase", recorded_phase),
                        "applied_phase_index": recorded_phase,
                        "source_confidence": r.get("decision_confidence"),
                        "source_decision_probabilities": r.get("decision_probabilities"),
                        "guardrail_intervened_phase": bool(guard),
                        "guardrail_phase_details": guard or None,
                        "reward_immediate": None,
                        "future_consequences": None,
                        "duration_label_quality": "actual applied duration field; no counterfactual validation",
                    }),
                    "_tick": tick,
                })
            if recorded_phase != applied_phase[i]:
                applied_phase[i], phase_age[i] = recorded_phase, 0
                phase_history[i].append(recorded_phase)
            if r.get("decision_source") == "laya" and r.get("duration") is not None:
                duration_history[i].append(int(r["duration"]))
            phase_age[i] += DT
    if not records:
        raise ValueError("No Laya decision rows found")

    # Chronological split with a 60 s exclusion buffer around boundaries.
    boundary1, boundary2 = int(n * .70), int(n * .85)
    parts = {"train": [], "validation": [], "test": []}
    excluded = 0
    for row in records:
        tick = row.pop("_tick")
        if abs(tick - boundary1) * DT <= EMBARGO_SECONDS or abs(tick - boundary2) * DT <= EMBARGO_SECONDS:
            excluded += 1
            continue
        split = "train" if tick < boundary1 else ("validation" if tick < boundary2 else "test")
        parts[split].append(row)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in parts.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    all_rows = [r for rows in parts.values() for r in rows]
    distributions = collections.Counter(r["gold"]["traffic_action"]["label"] for r in all_rows)
    if set(distributions) - set(ACTION_IDS):
        raise AssertionError("Unexpected action label")
    def feature_signature(row):
        state = row["state"]
        features = {k: v for k, v in state.items()
                    if k not in ("intersection_id", "simulation_time_seconds", "seconds_since_phase_command_proxy")}
        return hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()
    signatures = [feature_signature(r) for r in all_rows]
    split_signatures = {name: {feature_signature(r) for r in rows} for name, rows in parts.items()}
    cross_split_exact_overlap = {
        "train_validation": len(split_signatures["train"] & split_signatures["validation"]),
        "train_test": len(split_signatures["train"] & split_signatures["test"]),
        "validation_test": len(split_signatures["validation"] & split_signatures["test"]),
    }
    report = {
        "source_trace": str(trace_path), "source_immutable": True,
        "layout": {"intersections": 12, "snapshots_each": n, "snapshot_interval_seconds_assumed": DT},
        "decision_rows_before_split": len(records), "usable_rows": len(all_rows),
        "split_counts": {k: len(v) for k, v in parts.items()}, "embargo_excluded_rows": excluded,
        "split_method": "chronological contiguous time blocks 70/15/15 with +/-60s embargo at boundaries; one episode only, not a true unseen-episode test",
        "action_distribution": dict(sorted(distributions.items())),
        "phase_distribution": dict(sorted(collections.Counter(decode(a).phase for a in distributions.elements()).items())),
        "duration_distribution": dict(sorted(collections.Counter(decode(a).duration_seconds for a in distributions.elements()).items())),
        "guardrail_phase_interventions": sum(bool(r["metadata"]["guardrail_intervened_phase"]) for r in all_rows),
        "guardrail_total_corrections_from_benchmark_artifact": 944,
        "guardrail_count_note": "The trace identifies 880 phase interventions. The benchmark summary reports 944 total guardrail corrections; raw proposed duration was not separately logged, so the remaining duration-only corrections cannot be reconstructed from this trace.",
        "per_intersection": dict(sorted(collections.Counter(r["metadata"]["intersection_id"] for r in all_rows).items())),
        "duplicate_feature_state_count_ignoring_time_and_intersection": len(signatures) - len(set(signatures)),
        "exact_feature_overlap_across_splits": cross_split_exact_overlap,
        "missing_fields": ["CityFlow cur_phase at observation", "exact elapsed green", "downstream occupancy", "recent departures", "exact approaching counts", "per-vehicle max wait", "elapsed phase service age by movement", "neighbor states", "immediate reward", "counterfactual outcomes"],
        "label_limitations": ["labels imitate JevLight after its guardrail; not optimal labels", "single Jinan episode", "duration labels not independently validated", "guardrail phase was recoverable from trace intervention records"],
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    print(json.dumps(build(args.trace, args.out), indent=2))
