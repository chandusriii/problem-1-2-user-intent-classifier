# Evaluation Protocol

The benchmark command expects reviewed JSONL records containing `query`, `scope` and `tier`.

Example:

```json
{"query":"What appears at 00:13?","scope":"FRAME","tier":"L1_DIRECT"}
{"query":"Summarize the complete recording","scope":"WHOLE_VIDEO","tier":"L2_RETRIEVAL"}
```

## Protocol requirements

- Do not evaluate on examples used to fit the model.
- Keep near-duplicate queries within one data split.
- Document how labels were defined and reviewed.
- Report support counts, accuracy, macro F1 and per-class precision/recall/F1 for both heads.
- Record package versions, hardware and the exact inference command.
- Treat timing values as measurements of the named local environment, not universal performance claims.

The bundled development corpus is a smoke-test dataset only. A report from it does not demonstrate generalization.
