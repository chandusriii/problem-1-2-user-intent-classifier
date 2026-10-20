import unittest
from src.intent_classifier import classify_query, engineer_features

class IntentClassifierTests(unittest.TestCase):
    def test_frame_scope(self):
        result = classify_query("What color is the car at 00:15?")
        self.assertEqual(result["scope"], "FRAME")
        self.assertEqual(result["level"], "L1_DIRECT")

    def test_clip_scope(self):
        result = classify_query("When does the presenter explain neural networks?")
        self.assertEqual(result["scope"], "CLIP")

    def test_whole_video_scope(self):
        result = classify_query("Summarize the entire video.")
        self.assertEqual(result["scope"], "WHOLE_VIDEO")

    def test_global_reasoning(self):
        result = classify_query("How does behavior change throughout the video?")
        self.assertEqual(result["level"], "L3_GRAPH")

    def test_features(self):
        features = engineer_features("How many times does the player score?")
        self.assertTrue(features["count_cue"])
        self.assertGreater(features["token_count"], 0)

    def test_empty_query(self):
        with self.assertRaises(ValueError):
            classify_query("  ")

if __name__ == "__main__":
    unittest.main()
