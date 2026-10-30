# ML Experiment Log

## E0 — ActivityNet-QA weak-supervision diagnostic

**Status:** completed in GitHub Actions. Report artifact: `activitynet-qa-proxy-benchmark`.

- Sample: 750 ActivityNet-QA query texts, 750 source video IDs, 590 unique normalized query strings.
- Grouped holdout: 616 train / 134 test rows; connected components prevent a shared video ID or exact normalized query from crossing the split.
- Proxy distributions: scope FRAME=490, CLIP=260, WHOLE_VIDEO=0; tier L1_DIRECT=479, L2_RETRIEVAL=229, L3_GRAPH=42.
- Five model families evaluated with independent scope and tier heads.

### Observed proxy-label results

| Model | Scope accuracy | Scope macro F1 | Tier accuracy | Tier macro F1 | Mean inference ms/query |
|---|---:|---:|---:|---:|---:|
| Word TF-IDF + Logistic Regression | 0.9627 | 0.6406 | 0.9627 | 0.9702 | 0.0313 |
| Word TF-IDF + Linear SVM | 0.9627 | 0.6406 | 0.9627 | 0.9702 | 0.0325 |
| Word+character TF-IDF + Logistic Regression | 0.9701 | 0.6457 | 0.9701 | 0.9760 | 0.1286 |
| Word TF-IDF + Random Forest | 0.9478 | 0.6296 | 0.9478 | 0.9423 | 0.6133 |
| Word TF-IDF + Extra Trees | 0.9627 | 0.6402 | 0.9552 | 0.9636 | 0.5320 |

**Interpretation:** all five models reproduce the proxy labels well, but the scope macro F1 is much lower than accuracy because WHOLE_VIDEO is absent and class support is uneven. This is not a meaningful scope benchmark. The results are only a pipeline diagnostic and cannot be described as real-world routing accuracy. The word+character model has the highest proxy metrics but is not proven to be the best model for actual routing.

### Required next experiment

Build a balanced, independently human-annotated evaluation set with both annotators blinded to the proxy labels. The current sample has no WHOLE_VIDEO proxy examples, so it cannot test that class at all. The heuristic labels must not be used as ground truth or as a training target for a quality claim.
