import unittest
from fastapi.testclient import TestClient
from src.app import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
        self.assertEqual(response.json()["engine_state"], "online")

    def test_frame_route(self):
        response = self.client.post("/route", json={"query": "What color is the car at 00:15?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assigned_scopes"][0], "FRAME")
        self.assertIn("classifier", response.json())

    def test_timestamped_speech_route(self):
        response = self.client.post("/route", json={"query": "What does the speaker say at 02:30?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assigned_scopes"][0], "CLIP")

    def test_classify_alias_matches_route(self):
        query = {"query": "What color is the car at 00:15?"}
        route_response = self.client.post("/route", json=query)
        classify_response = self.client.post("/classify", json=query)
        self.assertEqual(route_response.status_code, 200)
        self.assertEqual(classify_response.status_code, 200)
        self.assertEqual(
            classify_response.json()["assigned_scopes"],
            route_response.json()["assigned_scopes"],
        )
        self.assertEqual(
            classify_response.json()["reasoning_tier"],
            route_response.json()["reasoning_tier"],
        )

    def test_whole_video_route(self):
        response = self.client.post("/route", json={"query": "Summarize the entire video."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assigned_scopes"][0], "WHOLE_VIDEO")

    def test_blank_query(self):
        response = self.client.post("/route", json={"query": "   "})
        self.assertEqual(response.status_code, 422)

    def test_query_length_limit(self):
        response = self.client.post("/route", json={"query": "x" * 5001})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
