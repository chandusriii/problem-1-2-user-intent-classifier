"""Train and persist the TF-IDF multi-head baseline on a small development corpus."""
from __future__ import annotations

from src.intent_classifier import VideoRAGIntentClassifier

# Development examples only. A real quality claim requires a larger, reviewed,
# independently labelled corpus; these examples are not Video-MME ground truth.
EXPANDED_CORPUS = [
    ("What color is the car parked near the curb at 00:15?", "FRAME", "L1_DIRECT"),
    ("Identify the hue of the vehicle shown in this image.", "FRAME", "L1_DIRECT"),
    ("What is written on the sign at 01:10?", "FRAME", "L2_RETRIEVAL"),
    ("How many people are visible in this specific frame?", "FRAME", "L1_DIRECT"),
    ("Read the text printed on the presenter's shirt at 02:40.", "FRAME", "L2_RETRIEVAL"),
    ("Which object is visible at this moment?", "FRAME", "L1_DIRECT"),
    ("Describe the layout of the slide shown at 00:42.", "FRAME", "L1_DIRECT"),
    ("What logo appears in the upper corner of this image?", "FRAME", "L1_DIRECT"),
    ("What does the speaker say at 02:30?", "CLIP", "L2_RETRIEVAL"),
    ("What happens immediately after the transition?", "CLIP", "L2_RETRIEVAL"),
    ("When does the presenter explain neural networks?", "CLIP", "L2_RETRIEVAL"),
    ("How many times does the player shoot before scoring?", "CLIP", "L3_GRAPH"),
    ("Describe the sequence when the person enters the room.", "CLIP", "L2_RETRIEVAL"),
    ("What action is happening between 00:20 and 00:30?", "CLIP", "L2_RETRIEVAL"),
    ("Find the moment where the red car enters the scene.", "CLIP", "L2_RETRIEVAL"),
    ("What changes after the alarm sounds?", "CLIP", "L2_RETRIEVAL"),
    ("How does the person react to the announcement?", "CLIP", "L2_RETRIEVAL"),
    ("What is the speaker discussing during this segment?", "CLIP", "L2_RETRIEVAL"),
    ("Summarize the entire video recording concisely.", "WHOLE_VIDEO", "L2_RETRIEVAL"),
    ("What is the overall main theme of the full movie?", "WHOLE_VIDEO", "L3_GRAPH"),
    ("Provide a timeline of events across the total runtime.", "WHOLE_VIDEO", "L3_GRAPH"),
    ("What is this video about?", "WHOLE_VIDEO", "L2_RETRIEVAL"),
    ("List all main events throughout the video.", "WHOLE_VIDEO", "L2_RETRIEVAL"),
    ("How does the subject's behaviour evolve from beginning to end?", "WHOLE_VIDEO", "L3_GRAPH"),
]

def run_training_pipeline(model_path: str = "models/pipeline.pkl") -> VideoRAGIntentClassifier:
    queries = [row[0] for row in EXPANDED_CORPUS]
    scopes = [row[1] for row in EXPANDED_CORPUS]
    tiers = [row[2] for row in EXPANDED_CORPUS]
    model = VideoRAGIntentClassifier().fit(queries, scopes, tiers)
    model.save(model_path)
    print(f"Saved local development model to {model_path} ({len(queries)} examples).")
    print("Note: the bundled corpus is for engineering smoke tests, not Video-MME accuracy claims.")
    return model

if __name__ == "__main__":
    run_training_pipeline()
