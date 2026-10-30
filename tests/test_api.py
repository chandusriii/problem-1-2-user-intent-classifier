import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.app import app
from src.intent_classifier import VideoRAGIntentClassifier
from src.train import EXPANDED_CORPUS


def build_trained_model():
    return VideoRAGIntentClassifier().fit(
        [row[0] for row in EXPANDED_CORPUS],
        [row[1] for row in EXPANDED_CORPUS],
        [row[2] for row in EXPANDED_CORPUS],
    )


class ApiTests(unittest.TestCase):
    def setUp(self):
        patcher = patch(
            "src.app.VideoRAGIntentClassifier.load",
            return_value=build_trained_model(),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_frame_route(self):
        response = self.client.post(
            "/route", json={"query": "What color is the car at 00:15?"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("FRAME", response.json()["assigned_scopes"])

    def test_clip_route(self):
        response = self.client.post(
            "/route", json={"query": "What does the speaker say at 02:30?"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("CLIP", response.json()["assigned_scopes"])

    def test_whole_video_route(self):
        response = self.client.post(
            "/route", json={"query": "Summarize the entire video."}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("WHOLE_VIDEO", response.json()["assigned_scopes"])

    def test_classify_alias_matches_route(self):
        payload = {"query": "What color is the car at 00:15?"}
        route = self.client.post("/route", json=payload)
        classify = self.client.post("/classify", json=payload)
        self.assertEqual(route.json(), classify.json())

    def test_blank_query_is_rejected(self):
        response = self.client.post("/route", json={"query": "   "})
        self.assertEqual(response.status_code, 422)

    def test_query_length_is_limited(self):
        response = self.client.post("/route", json={"query": "x" * 5001})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
