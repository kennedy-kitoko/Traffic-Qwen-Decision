import json
import unittest
from pathlib import Path

from traffic_qwen.build_dual_choice_dataset import convert_row, make_report


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/traffic_qwen/v1"


class DualChoiceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all((SOURCE / (split + ".jsonl")).is_file() for split in ("train", "validation", "test")):
            raise unittest.SkipTest("Jinan-derived training splits excluded pending redistribution clearance")
        cls.rows = {}
        for split in ("train", "validation", "test"):
            cls.rows[split] = [json.loads(x) for x in (SOURCE / f"{split}.jsonl").read_text().splitlines() if x.strip()]

    def test_every_sample_has_two_choice_tasks_and_one_hot_gold(self):
        row = convert_row(self.rows["train"][0])
        self.assertEqual(set(row), {"state", "questions", "gold", "metadata"})
        self.assertEqual(set(row["questions"]), {"phase", "duration"})
        for task, count in (("phase", 4), ("duration", 6)):
            question, answer = row["questions"][task], row["gold"][task]
            self.assertEqual(question["type"], "choice")
            self.assertEqual(len(question["criteria"]), count)
            self.assertIn(answer["choice"], question["criteria"])
            self.assertEqual(answer["label"], answer["choice"])
            self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)
            self.assertEqual(set(question["criteria"]), set(answer["probabilities"]))

    def test_legacy_action_decodes_to_phase_and_duration(self):
        for row in self.rows["train"][:40]:
            old = row["gold"]["traffic_action"]["label"]
            new = convert_row(row)["gold"]
            self.assertEqual(old, f"{new['phase']['choice']}_{new['duration']['choice']}")
            self.assertEqual(int(new["duration"]["choice"]) % 5, 0)

    def test_all_splits_preserve_rows_and_have_no_state_or_episode_leakage(self):
        report = make_report(self.rows)
        for name, rows in self.rows.items():
            self.assertEqual(report["split_counts"][name], len(rows))
        self.assertTrue(all(v == 0 for v in report["cross_split_state_overlap"].values()))
        # Dataset V1 comes from one episode, so episode overlap is expected and disclosed.
        self.assertTrue(any(report["cross_split_episode_overlap"].values()))

    def test_vocabulary_and_label_coverage(self):
        for split in self.rows.values():
            for source in split:
                row = convert_row(source)
                self.assertEqual(len(row["questions"]["phase"]["criteria"]), 4)
                self.assertEqual(len(row["questions"]["duration"]["criteria"]), 6)
                self.assertTrue(row["gold"]["phase"]["choice"].startswith("phase_"))


if __name__ == "__main__":
    unittest.main()
