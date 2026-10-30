# ML Experiment Log

## E0 — ActivityNet-QA weak-supervision diagnostic

**Status:** automated model fitting and artifact export are wired into CI.

- Input: 750 ActivityNet-QA questions across 750 source video IDs.
- Targets: lexical heuristic proxies, not source-provided or human labels.
- Holdout: group split prevents shared video IDs and exact normalized query strings from crossing partitions.
- Models: Logistic Regression, Linear SVM, word+character Logistic Regression, Random Forest and Extra Trees.
- Metrics: accuracy, macro/weighted F1, per-class metrics, confusion matrices, fit time and inference time.
- Environment metadata is stored with the report.
- CI artifact: `activitynet-qa-proxy-benchmark`.

**Interpretation rule:** proxy scores are a pipeline diagnostic only. The next quality milestone is independent human annotation, inter-annotator agreement, adjudication and a frozen test set.
