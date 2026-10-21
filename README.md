# Problem 1.2 — User Intent Classifier

A lightweight, CPU-only baseline for routing video-related questions.

## Independent outputs

**Retrieval scope**
- `FRAME`: evidence from a still frame or specific timestamp.
- `CLIP`: a temporal segment, event, sequence, or spoken passage.
- `WHOLE_VIDEO`: a summary, theme, or event inventory requiring broad coverage.

**Reasoning tier**
- `L1_DIRECT`: direct or targeted context.
- `L2_RETRIEVAL`: retrieval of relevant clips, captions, transcript, or OCR.
- `L3_GRAPH`: cross-section reasoning across video events.

The default inference uses interpretable rules and does not load a trained model binary. Retrieval scope and reasoning tier are separate outputs.

## Run locally (Python 3.11 recommended)

```bash
python -m pip install -r requirements.txt
python src/main.py --query "What color is the car at 00:15?"
python src/main.py --query "What does the speaker say at 02:30?"
python src/main.py --query "Summarize the entire video."
python -m unittest discover -s tests -v
```

## API

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. Endpoints are `GET /health` and `POST /route`.

## Validation boundary

The included checks exercise representative inputs and are smoke tests, not official VideoMME accuracy. The supplied VideoMME data does not directly provide the target retrieval-scope labels; official scope metrics require consistent annotation of the questions as FRAME, CLIP, or WHOLE_VIDEO. Rules may misroute unseen paraphrases and compound queries, so ambiguous production inputs should use broader or multiple retrieval routes.
