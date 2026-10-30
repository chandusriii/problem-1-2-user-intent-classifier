"""Unit tests for the ActivityNet-QA preprocessing and model-comparison utilities."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_activitynet import map_labels
from scripts.compare_models import load, build_models


class ActivityNetExperimentTests(unittest.TestCase):
    def test_proxy_labels_are_valid_and_not_empty(self):
        samples = [
            ("What color is the car at 00:15?", "FRAME", "L1_DIRECT"),
            ("What does the speaker say after the introduction?", "CLIP", "L2_RETRIEVAL"),
            ("How does the person's behavior change from beginning to end?", "WHOLE_VIDEO", "L3_GRAPH"),
        ]
        for query, expected_scope, expected_tier in samples:
            with self.subTest(query=query):
                self.assertEqual(map_labels(query), (expected_scope, expected_tier))

    def test_comparison_loader_rejects_missing_group_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(json.dumps({"query":"question","scope_proxy":"FRAME","tier_proxy":"L1_DIRECT"})+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "video_id"):
                load(path)

    def test_comparison_loader_reads_jsonl_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.jsonl"
            rows = [
                {"query":"What is visible?","scope_proxy":"FRAME","tier_proxy":"L1_DIRECT","video_id":"vid-a"},
                {"query":"What happens next?","scope_proxy":"CLIP","tier_proxy":"L2_RETRIEVAL","video_id":"vid-b"},
            ]
            path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
            self.assertEqual(len(load(path)), 2)

    def test_five_model_configurations_are_exposed(self):
        self.assertEqual(len(build_models()), 5)
        self.assertIn("word_tfidf_linear_svm", build_models())


if __name__ == "__main__":
    unittest.main()
