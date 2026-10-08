import unittest

from traffic_qwen.continuation_policy import MaxPressureContinuation


class MaxPressureContinuationTests(unittest.TestCase):
    def test_matches_jevlight_four_phase_pressure_map(self):
        # phase 1 = 1+4, phase 2 = 7+10, phase 3 = 0+3, phase 4 = 6+9
        state = {"cur_phase": [0], "traffic_movement_pressure_queue": [0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]}
        self.assertEqual(MaxPressureContinuation.choose(state), 0)

    def test_first_phase_wins_ties_like_numpy_argmax(self):
        state = {"cur_phase": [0], "traffic_movement_pressure_queue": [0] * 12}
        self.assertEqual(MaxPressureContinuation.choose(state), 0)

    def test_yellow_keeps_last_requested_phase(self):
        state = {"cur_phase": [-1], "traffic_movement_pressure_queue": [0] * 12}
        self.assertEqual(MaxPressureContinuation.choose(state, previous_action=3), 3)


if __name__ == '__main__':
    unittest.main()
