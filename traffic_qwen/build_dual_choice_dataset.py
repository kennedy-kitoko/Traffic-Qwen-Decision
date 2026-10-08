"""Convert the existing, split-preserved Traffic-Qwen V1 labels to two Clef choices."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PHASES = [f"phase_{i}" for i in range(4)]
DURATIONS = ["15", "20", "25", "30", "35", "40"]
PHASE_DESCRIPTIONS = [
    "JevLight phase 1: eastbound and westbound through traffic (WT_ET).",
    "JevLight phase 2: northbound and southbound through traffic (NT_ST).",
    "JevLight phase 3: eastbound and westbound left turns (WL_EL).",
    "JevLight phase 4: northbound and southbound left turns (NL_SL).",
]


def _label(row: dict) -> tuple[str, str]:
    gold = row.get("gold", {})
    value = gold.get("traffic_action", {})
    value = value.get("label", value.get("choice", "")) if isinstance(value, dict) else value
    match = re.fullmatch(r"phase_([0-3])_(15|20|25|30|35|40)", str(value))
    if not match:
        raise ValueError(f"Unsupported V1 action label: {value!r}")
    return f"phase_{match.group(1)}", match.group(2)


def convert_row(row: dict) -> dict:
    phase, duration = _label(row)
    questions = {
        "phase": {
            "type": "choice",
            "instructions": "Choose the signal phase that should be activated to reduce congestion and waiting while preventing starvation.",
            "criteria": {key: PHASE_DESCRIPTIONS[i] for i, key in enumerate(PHASES)},
        },
        "duration": {
            "type": "choice",
            "instructions": "Choose the green duration appropriate for the selected phase and observed traffic conditions.",
            "criteria": {key: f"Green time of {key} seconds." for key in DURATIONS},
        },
    }
    return {
        "state": row["state"],
        "questions": questions,
        "gold": {
            "phase": {"label": phase, "choice": phase,
                      "probabilities": {key: float(key == phase) for key in PHASES}},
            "duration": {"label": duration, "choice": duration,
                         "probabilities": {key: float(key == duration) for key in DURATIONS}},
        },
        "metadata": {**row.get("metadata", {}), "source_label": row.get("gold", {}).get("traffic_action", {}).get("label"),
                     "label_provenance": "existing guarded JevLight V1 applied action; imitation only"},
    }


def _read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_report(splits: dict[str, list[dict]]) -> dict:
    report = {"split_counts": {}, "splits": {}, "cross_split_state_overlap": {}, "cross_split_episode_overlap": {}}
    state_keys, episode_keys = {}, {}
    for name, rows in splits.items():
        phase_counts, duration_counts, combo_counts, intersections, times = (Counter() for _ in range(5))
        hashes, state_ids, episodes, missing = [], set(), set(), Counter()
        for row in rows:
            phase, duration = _label(row)
            phase_counts[phase] += 1
            duration_counts[duration] += 1
            combo_counts[f"{phase}_{duration}"] += 1
            md, state = row.get("metadata", {}), row.get("state", {})
            intersection = md.get("intersection_id", state.get("intersection_id"))
            t = md.get("simulation_time_seconds", state.get("simulation_time_seconds"))
            episode = md.get("episode_id")
            intersections[str(intersection)] += 1
            times[str(t)] += 1
            identity = (episode, intersection, t)
            state_ids.add(json.dumps(identity, sort_keys=True))
            if episode is not None:
                episodes.add(str(episode))
            canonical = json.dumps(row["state"], sort_keys=True, separators=(",", ":"))
            hashes.append(hashlib.sha256(canonical.encode()).hexdigest())
            for field in ("state", "gold", "questions"):
                if not row.get(field):
                    missing[field] += 1
        report["split_counts"][name] = len(rows)
        report["splits"][name] = {
            "rows": len(rows), "phase_counts": dict(phase_counts), "duration_counts_seconds": dict(duration_counts),
            "phase_duration_counts": dict(combo_counts), "duplicate_state_rows": len(hashes) - len(set(hashes)),
            "missing_required_fields": dict(missing), "intersection_decisions": dict(intersections),
            "time_min_seconds": min((int(x) for x in times if x != "None"), default=None),
            "time_max_seconds": max((int(x) for x in times if x != "None"), default=None),
            "episode_ids": sorted(episodes), "state_keys": state_ids,
        }
        state_keys[name] = state_ids
        episode_keys[name] = episodes
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            key = f"{left}__{right}"
            report["cross_split_state_overlap"][key] = len(state_keys[left] & state_keys[right])
            report["cross_split_episode_overlap"][key] = sorted(episode_keys[left] & episode_keys[right])
    for split in report["splits"].values():
        split.pop("state_keys")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/traffic_qwen/v1")
    parser.add_argument("--output", default="data/traffic_qwen/v1_dual_choice")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in ("train", "validation", "test"):
        source_rows = _read(source / f"{name}.jsonl")
        converted = [convert_row(row) for row in source_rows]
        (output / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in converted), encoding="utf-8")
        splits[name] = source_rows
    report = make_report(splits)
    report["dataset_note"] = "Preserved existing V1 splits. Labels imitate guarded JevLight actions; not optimal or counterfactual labels."
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2))


if __name__ == "__main__":
    main()
