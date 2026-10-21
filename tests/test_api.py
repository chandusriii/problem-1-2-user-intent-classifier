import unittest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

class ApiTests(unittest.TestCase):
    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_frame_route(self):
        response = client.post("/route", json={"query":"What color is the car at 00:15?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assigned_scopes"][0], "FRAME")

    def test_timestamped_speech_route(self):
        response = client.post("/route", json={"query":"What does the speaker say at 02:30?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assigned_scopes"][0], "CLIP")

    def test_whole_video_route(self):
        response = client.post("/route", json={"query":"Summarize the entire video."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["assigned_scopes"][0], "WHOLE_VIDEO")

    def test_blank_query(self):
        response = client.post("/route", json={"query":"   "})
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()
