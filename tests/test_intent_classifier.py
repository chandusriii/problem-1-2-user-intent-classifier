import unittest
from src.intent_classifier import classify_query, engineer_features, normalize_query

class IntentClassifierTests(unittest.TestCase):
    def test_frame_scope(self):
        result = classify_query("What color is the car at 00:15?")
        self.assertEqual(result["scope"], "FRAME")
        self.assertEqual(result["level"], "L1_DIRECT")

    def test_clip_scope(self):
        result = classify_query("When does the presenter explain neural networks?")
        self.assertEqual(result["scope"], "CLIP")

    def test_timestamped_transcript_is_clip_not_frame(self):
        self.assertEqual(classify_query("What does the speaker say at 02:30?")["scope"], "CLIP")

    def test_whole_video_scope(self):
        self.assertEqual(classify_query("Summarize the entire video.")["scope"], "WHOLE_VIDEO")

    def test_global_reasoning(self):
        self.assertEqual(classify_query("How does behavior change throughout the video?")["level"], "L3_GRAPH")

    def test_empty_query(self):
        with self.assertRaises(ValueError):
            classify_query("")

    def test_whitespace_query(self):
        with self.assertRaises(ValueError):
            classify_query("   ")

    def test_compound_visual_then_temporal_question(self):
        result = classify_query("What color is the car after the presenter stops speaking?")
        self.assertEqual(result["scope"], "CLIP")

    def test_unseen_visual_paraphrase_at_timestamp(self):
        self.assertEqual(classify_query("Identify the hue of the vehicle at 00:15.")["scope"], "FRAME")

    def test_event_at_start(self):
        self.assertEqual(classify_query("What happens at the beginning of this recording?")["scope"], "CLIP")

    def test_video_level_count(self):
        self.assertEqual(classify_query("How many bridges can be seen throughout the video?")["scope"], "WHOLE_VIDEO")

    def test_feature_extraction_has_numeric_dimensions(self):
        features = engineer_features("How many times does the player score?")
        self.assertTrue(features["count_cue"])
        self.assertIsInstance(features["token_count"], int)
        self.assertIsInstance(features["char_count"], int)

if __name__ == "__main__":
    unittest.main()
