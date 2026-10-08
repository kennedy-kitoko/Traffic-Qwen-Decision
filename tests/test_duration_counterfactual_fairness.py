import json
import unittest
from collections import defaultdict
from pathlib import Path


class DurationCounterfactualFairnessTests(unittest.TestCase):
    def test_real_cityflow_branches_hold_requested_green_then_resume(self):
        path = Path('/home/kennedy/JevLight/artifacts/traffic_qwen/v2_1_results.json')
        if not path.exists():
            self.skipTest('run the bounded V2.1 pilot first')
        data = json.loads(path.read_text())
        groups = defaultdict(list)
        for row in data['results']:
            groups[(row['state_id'], row['phase'])].append(row)
        verified = 0
        for rows in groups.values():
            if len({r['duration'] for r in rows}) != 6:
                continue
            by_duration = {r['duration']: r for r in rows}
            for duration in (15, 20, 25, 30, 35, 40):
                row = by_duration[duration]
                self.assertEqual(row['horizon_seconds'], 120)
                self.assertEqual(row['candidate_green_seconds'], duration)
                self.assertGreaterEqual(row['maxpressure_takeover_seconds'], duration)
                if row['phase'] != row['starting_phase']:
                    self.assertGreaterEqual(row['candidate_transition_yellow_seconds'], 5)
                verified += 1
            break
        self.assertEqual(verified, 6)


if __name__ == '__main__':
    unittest.main()
