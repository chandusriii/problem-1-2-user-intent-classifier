# Intent Classifier Design

## Objective
Predict one retrieval scope—FRAME, CLIP, or WHOLE_VIDEO—and independently predict a reasoning tier: L1_DIRECT, L2_RETRIEVAL, or L3_GRAPH.

## Baseline features
The query-only baseline normalizes whitespace and uses readable lexical features for timestamps, visible properties, event language, speech/transcripts, on-screen text, counting, causal cues, comparisons, and broad-video context. Global-summary requests take priority. Speech content is clip-based even if the question includes a timestamp. Static visual details at an instant are frame-based; temporal actions and sequences are clip-based.

## API
`src/app.py` exposes `GET /health` and `POST /route`. It validates query content and returns `assigned_scope`, `reasoning_tier`, and `retrieval_strategy`. The API delegates to the same classifier as the CLI and requires no saved model assets.

## Tests and limitations
Run `python -m unittest discover -s tests -v`. Unit tests are smoke checks rather than a full held-out benchmark. The provided VideoMME questions do not inherently include retrieval-scope labels; official accuracy, macro-F1, and per-class metrics require reviewed, human-labelled FRAME/CLIP/WHOLE_VIDEO targets. Simple lexical rules can miss paraphrases or compound intents; consider multiple routes or a broader-retrieval fallback when confidence is low.
