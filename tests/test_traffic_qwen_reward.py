import unittest

from traffic_qwen.reward import combine_components, discounted_return, normalize_component, spillback_lane_samples


class RewardTests(unittest.TestCase):
    def test_component_units_normalize_and_clip_independently(self):
        reward = combine_components({
            "queue_reduction": 25,
            "waiting_reduction": 5000,
            "departed_vehicles": 10,
            "switching_penalty": 2,
            "starvation_penalty": 10000,
            "spillback_penalty": 10,
        })
        self.assertAlmostEqual(reward["total"], 0.30 + 0.25 + 0.20 - 0.05 - 0.10 - 0.10)
        self.assertEqual(reward["queue_reduction"]["normalized"], 1)
        self.assertEqual(reward["starvation_penalty"]["normalized"], -1)

    def test_spillback_penalty_changes_reward(self):
        clear = combine_components({"queue_reduction": 0})["total"]
        blocked = combine_components({"queue_reduction": 0, "spillback_penalty": 10})["total"]
        self.assertLess(blocked, clear)

    def test_downstream_occupancy_triggers_spillback_samples(self):
        lanes = ["exit_a", "exit_b"]
        lengths = {"exit_a": 100.0, "exit_b": 60.0}
        clear, _ = spillback_lane_samples({"exit_a": 1, "exit_b": 1}, lengths, lanes)
        blocked, max_ratio = spillback_lane_samples({"exit_a": 12, "exit_b": 1}, lengths, lanes)
        self.assertEqual(clear, 0)
        self.assertEqual(blocked, 1)
        self.assertGreaterEqual(max_ratio, 0.8)

    def test_switch_penalty_is_bounded_and_discounted(self):
        self.assertEqual(normalize_component("switching_penalty", 100), -1)
        self.assertGreater(discounted_return([{"queue_reduction": 1}, {"queue_reduction": 1}], .5),
                           discounted_return([{"queue_reduction": 1}], .5))


if __name__ == "__main__":
    unittest.main()
