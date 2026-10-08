import unittest

from models.traffic_qwen_agent import TrafficQwenAgent


class TrafficQwenAgentProtocolTests(unittest.TestCase):
    def test_phase_and_duration_questions_match_training_vocabulary(self):
        questions = TrafficQwenAgent._questions()
        self.assertEqual(set(questions), {"phase", "duration"})
        self.assertEqual(list(questions["phase"]["criteria"]), list(TrafficQwenAgent.PHASE_OPTIONS))
        self.assertEqual(list(questions["duration"]["criteria"]), [str(x) for x in TrafficQwenAgent.DURATIONS])
        self.assertTrue(all(q["type"] == "choice" for q in questions.values()))

    def test_probability_validation_accepts_normalized_outputs(self):
        TrafficQwenAgent._validate_probabilities({"phase_0": .25, "phase_1": .25, "phase_2": .25, "phase_3": .25}, TrafficQwenAgent.PHASE_OPTIONS)
        TrafficQwenAgent._validate_probabilities({str(x): 1/6 for x in TrafficQwenAgent.DURATIONS}, tuple(str(x) for x in TrafficQwenAgent.DURATIONS))

    def test_probability_validation_fails_closed_on_bad_outputs(self):
        with self.assertRaises(ValueError):
            TrafficQwenAgent._validate_probabilities({"phase_0": 1.2}, TrafficQwenAgent.PHASE_OPTIONS)
        with self.assertRaises(ValueError):
            TrafficQwenAgent._validate_probabilities({str(x): .1 for x in TrafficQwenAgent.DURATIONS}, tuple(str(x) for x in TrafficQwenAgent.DURATIONS))


if __name__ == "__main__":
    unittest.main()
