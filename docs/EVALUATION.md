# Evaluation Protocol

The benchmark command requires one JSON object per JSONL line with exactly these required fields:

```json
{"query":"What is visible at 00:13?","scope":"FRAME","tier":"L1_DIRECT"}
{"query":"Summarize the complete recording.","scope":"WHOLE_VIDEO","tier":"L2_RETRIEVAL"}
```

Allowed scope labels: `FRAME`, `CLIP`, `WHOLE_VIDEO`.
Allowed reasoning labels: `L1_DIRECT`, `L2_RETRIEVAL`, `L3_GRAPH`.

## Metrics

- **Scope top-1 accuracy / macro F1:** uses only the highest-ranked predicted scope.
- **Scope candidate recall:** fraction where the reviewed scope appears anywhere among the model's returned candidate scopes. This is not top-1 accuracy.
- **Multi-route coverage:** proportion of queries for which the model returns more than one scope.
- **Reasoning-tier accuracy / macro F1:** evaluates the independently predicted reasoning tier.
- **Latency:** measured around the complete per-query prediction call; values depend on hardware and runtime.

The benchmark reports class-level precision, recall, F1 and support for both prediction heads. Its p95 latency uses the nearest-rank definition.

## Data integrity

- Never evaluate on the same examples used to fit the model.
- Keep paraphrases and near-duplicates within one split.
- Record dataset provenance, licensing, annotation guidance, annotator review and split methodology.
- Ensure every row has a valid scope and tier label.
- The bundled `data/intent_eval.jsonl` is illustrative only. It must not be treated as a held-out benchmark or evidence of generalization.
- The bundled development corpus is not official Video-MME ground truth.

## Reproducibility

Record Python and package versions, hardware, model artifact version, exact evaluation command and the data revision. Treat timing measurements as environment-specific, not universal performance claims.
