import os
import tempfile
import unittest

from src.intent_classifier import VideoRAGIntentClassifier, normalize_query
from src.train import EXPANDED_CORPUS

class TestUpgradedClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(tempfile.gettempdir(), "video_rag_intent_classifier_test.pkl")
        cls.classifier = VideoRAGIntentClassifier().fit(
            [x[0] for x in EXPANDED_CORPUS],
            [x[1] for x in EXPANDED_CORPUS],
            [x[2] for x in EXPANDED_CORPUS],
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)

    def test_frame_route_matching(self):
        result = self.classifier.predict_route("What color are the shoes worn at 00:12?")
        self.assertIn("FRAME", result["assigned_scopes"])
        self.assertIn(result["reasoning_tier"], {"L1_DIRECT", "L2_RETRIEVAL", "L3_GRAPH"})

    def test_whole_video_route_matching(self):
        result = self.classifier.predict_route("Summarize the full plot of this video.")
        self.assertIn("WHOLE_VIDEO", result["assigned_scopes"])

    def test_empty_query_raises_validation_error(self):
        with self.assertRaises(ValueError):
            self.classifier.predict_route("")

    def test_single_query_feature_alignment(self):
        from src.intent_classifier import extract_and_align_features

        matrix = extract_and_align_features(
            "What color is the car at 00:15?",
            self.classifier.vectorizer,
        )
        self.assertEqual(matrix.ndim, 2)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(
            matrix.shape[1],
            len(self.classifier.vectorizer.get_feature_names_out()) + 4,
        )

    def test_feature_matrix_shape(self):
        import numpy as np
        from src.intent_classifier import combine_features
        matrix = combine_features(self.classifier.vectorizer, ["What color is a car at 00:15?"])
        self.assertEqual(matrix.ndim, 2)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], len(self.classifier.vectorizer.get_feature_names_out()) + 4)

    def test_untrained_model_uses_rules_fallback(self):
        result = VideoRAGIntentClassifier().predict_route("Summarize this video.")
        self.assertEqual(result["classifier"], "rules_fallback_untrained")
        self.assertIn(result["assigned_scopes"][0], {"FRAME", "CLIP", "WHOLE_VIDEO"})

    def test_model_save_and_load_round_trip(self):
        self.classifier.save(self.model_path)
        loaded = VideoRAGIntentClassifier.load(self.model_path)
        result = loaded.predict_route("Summarize the full plot of this video.")
        self.assertIn("WHOLE_VIDEO", result["assigned_scopes"])

    def test_normalize_query_rejects_empty(self):
        with self.assertRaises(ValueError):
            normalize_query("  ")

if __name__ == "__main__":
    unittest.main()
