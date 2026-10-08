import json
import tempfile
import unittest
from pathlib import Path

from traffic_qwen.action_space import ACTION_IDS, decode, encode
from traffic_qwen.dataset_schema import sample
from traffic_qwen.build_imitation_dataset import build


class ActionSpaceTests(unittest.TestCase):
    def test_exactly_24_unique_roundtrips(self):
        self.assertEqual(len(ACTION_IDS), 24)
        self.assertEqual(len(set(ACTION_IDS)), 24)
        for action in ACTION_IDS:
            decoded = decode(action)
            self.assertEqual(encode(decoded.phase, decoded.duration_seconds), action)

    def test_rejects_invalid_action(self):
        with self.assertRaises(ValueError):
            decode("phase_4_15")


class DatasetSchemaTests(unittest.TestCase):
    def test_row_schema_and_choice(self):
        row = sample({"intersection_id": "i"}, ACTION_IDS[0], {"episode_id": "e"})
        self.assertEqual(set(("state", "questions", "gold", "metadata")), set(row))
        q = row["questions"]["traffic_action"]
        self.assertEqual(q["type"], "choice")
        self.assertEqual(set(q["criteria"]), set(ACTION_IDS))
        self.assertEqual(row["gold"]["traffic_action"]["label"], ACTION_IDS[0])

    def test_real_trace_build_has_isolated_chronological_splits(self):
        trace = Path("artifacts/baseline_guardrail/state_action.json")
        if not trace.exists():
            self.skipTest("JevLight baseline trace is not present")
        with tempfile.TemporaryDirectory() as td:
            report = build(trace, Path(td))
            self.assertEqual(report["decision_rows_before_split"], 1143)
            self.assertEqual(sum(report["split_counts"].values()), report["usable_rows"])
            self.assertGreater(report["embargo_excluded_rows"], 0)
            self.assertEqual(report["exact_feature_overlap_across_splits"]["train_test"], 0)
            for split in ("train", "validation", "test"):
                rows = [json.loads(line) for line in (Path(td) / f"{split}.jsonl").read_text().splitlines()]
                for item in rows:
                    self.assertIn(item["gold"]["traffic_action"]["label"], ACTION_IDS)
                    self.assertIsNone(item["metadata"]["reward_immediate"])


if __name__ == "__main__":
    unittest.main()
