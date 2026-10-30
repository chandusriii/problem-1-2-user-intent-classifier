import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark import evaluate, load_labelled_rows
from src.intent_classifier import (
    MODEL_FORMAT_VERSION, VideoRAGIntentClassifier, combine_features,
    extract_and_align_features, extract_dense_signals, normalize_query,
)
from src.train import EXPANDED_CORPUS


class TestIntentClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = VideoRAGIntentClassifier().fit(
            [row[0] for row in EXPANDED_CORPUS],
            [row[1] for row in EXPANDED_CORPUS],
            [row[2] for row in EXPANDED_CORPUS],
        )

    def test_frame_route(self):
        result = self.classifier.predict_route("What color are the shoes at 00:12?")
        self.assertIn("FRAME", result["assigned_scopes"])

    def test_clip_route(self):
        result = self.classifier.predict_route("What does the speaker say at 02:30?")
        self.assertIn("CLIP", result["assigned_scopes"])

    def test_whole_video_route(self):
        result = self.classifier.predict_route("Summarize the full plot of this video.")
        self.assertIn("WHOLE_VIDEO", result["assigned_scopes"])

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.classifier.predict_route(" ")

    def test_non_string_query_is_rejected(self):
        with self.assertRaises(TypeError):
            normalize_query(None)

    def test_dense_features_are_bounded_and_stable_shape(self):
        features = extract_dense_signals("What happens after 00:15?")
        self.assertEqual(features.shape, (1, 2))
        self.assertEqual(features.tolist(), [[1.0, 1.0]])

    def test_dense_features_do_not_scale_with_query_length(self):
        short = extract_dense_signals("What happens after 00:15?")
        long = extract_dense_signals("What happens after 00:15? " + "very " * 200)
        self.assertEqual(short.tolist(), long.tolist())

    def test_single_query_features_remain_sparse(self):
        features = extract_and_align_features(
            "What color is the car at 00:15?", self.classifier.vectorizer
        )
        self.assertEqual(features.shape[0], 1)
        self.assertTrue(hasattr(features, "tocsr"))

    def test_batch_feature_alignment(self):
        features = combine_features(
            self.classifier.vectorizer,
            ["What color is the car?", "Summarize the full video."],
        )
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[1], len(self.classifier.vectorizer.get_feature_names_out()) + 2)

    def test_invalid_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            VideoRAGIntentClassifier().fit(
                ["query one", "query two"],
                ["UNKNOWN", "FRAME"],
                ["L1_DIRECT", "L2_RETRIEVAL"],
            )

    def test_single_label_head_is_rejected(self):
        with self.assertRaises(ValueError):
            VideoRAGIntentClassifier().fit(
                ["query one", "query two"],
                ["FRAME", "FRAME"],
                ["L1_DIRECT", "L2_RETRIEVAL"],
            )

    def test_untrained_fallback_whole_video(self):
        result = VideoRAGIntentClassifier().predict_route("Summarize this entire video.")
        self.assertEqual(result["assigned_scopes"], ["WHOLE_VIDEO"])
        self.assertEqual(result["classifier"], "rules_fallback_untrained")
        self.assertEqual(result["confidence_distribution"], {})

    def test_untrained_fallback_clip(self):
        result = VideoRAGIntentClassifier().predict_route("What does the speaker say?")
        self.assertEqual(result["assigned_scopes"], ["CLIP"])

    def test_untrained_fallback_frame(self):
        result = VideoRAGIntentClassifier().predict_route("What color is visible at 00:12?")
        self.assertEqual(result["assigned_scopes"], ["FRAME"])

    def test_invalid_multi_route_margin_is_rejected(self):
        with self.assertRaises(ValueError):
            self.classifier.predict_route("Explain this video", multi_route_margin=1.5)

    def test_model_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.pkl"
            self.classifier.save(path)
            loaded = VideoRAGIntentClassifier.load(path)
            self.assertEqual(loaded.format_version, MODEL_FORMAT_VERSION)
            self.assertIn(
                "WHOLE_VIDEO",
                loaded.predict_route("Summarize the full video.")["assigned_scopes"],
            )

    def test_legacy_artifact_is_rejected(self):
        import pickle
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pkl"
            legacy = VideoRAGIntentClassifier()
            legacy.is_trained = True
            del legacy.format_version
            with path.open("wb") as handle:
                pickle.dump(legacy, handle)
            with self.assertRaisesRegex(ValueError, "Unsupported or legacy"):
                VideoRAGIntentClassifier.load(path)

    def test_untrained_model_cannot_be_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                VideoRAGIntentClassifier().save(Path(directory) / "model.pkl")


class TestBenchmark(unittest.TestCase):
    def test_label_loader_requires_all_fields_and_valid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(json.dumps({"query": "one", "scope": "FRAME"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing keys"):
                load_labelled_rows(path)
            path.write_text(json.dumps({"query": "one", "scope": "NOPE", "tier": "L1_DIRECT"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid scope"):
                load_labelled_rows(path)

    def test_metrics_do_not_count_second_candidate_as_top1(self):
        class StubModel:
            def predict_route(self, query):
                return {
                    "assigned_scopes": ["FRAME", "CLIP"],
                    "reasoning_tier": "L2_RETRIEVAL",
                }
        report = evaluate(StubModel(), [
            {"query": "A", "scope": "CLIP", "tier": "L2_RETRIEVAL"},
            {"query": "B", "scope": "FRAME", "tier": "L2_RETRIEVAL"},
        ])
        self.assertEqual(report["scope_top1"]["accuracy"], 0.5)
        self.assertEqual(report["scope_candidate_recall"], 1.0)
        self.assertEqual(report["multi_route_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
