# User Intent Classifier: Design and Deployment

## Objective
Predict one of the assignment's primary retrieval scopes—FRAME, CLIP, or WHOLE_VIDEO—and recommend a routing strategy. Reasoning difficulty is an additional, independent output inspired by AdaVideoRAG.

## Feature engineering
Classification runs on query text alone: no video decoding or large model inference is required. We normalize whitespace and derive low-cost features including token count, character count, timestamp patterns, frame/shot cues, clip/scene cues, whole-video terms, temporal relations, counting language, causal language, comparison language, speech/transcript cues, OCR cues, and global-context indicators. These features are intentionally interpretable and cheap to compute.

The initial implementation is a deterministic rules baseline for easy inspection and CPU inference. With a larger labelled set, it can be compared against word/character TF-IDF plus Logistic Regression. Model probabilities, if used, need calibration before they are treated as confidence.

## Scope taxonomy
- **FRAME:** evidence can be obtained from one timestamp or image/frame.
- **CLIP:** a local segment, spoken passage, or event with limited temporal context.
- **WHOLE_VIDEO:** summaries, themes, event inventories, or questions whose evidence may be distributed throughout the video.

## Reasoning tier and route
- **L1_DIRECT:** direct or targeted frame/context path.
- **L2_RETRIEVAL:** retrieve relevant captions, transcript/ASR, OCR, and clips.
- **L3_GRAPH:** cross-section retrieval for global relationships and multi-hop evidence.

The three levels follow AdaVideoRAG's principle of routing straightforward questions to no retrieval, simple reasoning to naive multimodal retrieval, and hard reasoning to graph retrieval. This is an inspired lightweight baseline, not a reproduction of the paper's Qwen2.5-7B classifier.

## VideoMME sample and evaluation
The assignment requests the 10% VideoMME sample. The included JSONL is only an executable smoke-evaluation set with manually supplied scope labels; it is not the official VideoMME result. VideoMME's supplied question-answer labels must not be assumed to encode retrieval scope. To report a meaningful benchmark, download the assigned sample, apply and document a consistent scope-annotation guide, manually review labels, run the classifier on those labelled questions, and report accuracy, macro-F1, per-class precision/recall, mean and p95 inference latency. The official dataset must not be described as evaluated until that procedure is actually run.

## Deployment
The current module has no model-weight downloads and runs on CPU. It can be called directly from the upstream router or wrapped in a stateless FastAPI endpoint. In production, add request validation, versioned labels, low-confidence/ambiguous-query fallback, structured latency metrics, and error monitoring. Scale horizontally behind a queue or load balancer; retrieval work is invoked only after classification.

## Limitations
The current classifier is a transparent rule baseline rather than a trained semantic model. Rules can fail on paraphrases and compound queries. It does not promise perfect routing or reproduce AdaVideoRAG's reported results. A learned model and stronger conclusions require representative, held-out, scope-labelled data.
