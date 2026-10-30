# Human-labelled routing set

Populate `activitynet_qa_annotation_template.jsonl` independently using `docs/ANNOTATION_GUIDE.md`. Two annotators should label each row without seeing proxy labels or predictions. Keep the two raw annotation files separate, then adjudicate disagreements into a third file. Use pseudonymous annotator IDs. The template is intentionally blank and contains no fabricated labels.

## Inter-annotator agreement

After two independent annotation files are prepared, run:

```bash
python -m scripts.annotation_quality --annotator-a data/annotation/annotator_a.jsonl --annotator-b data/annotation/annotator_b.jsonl --output results/annotation_agreement.json
```

The report includes Cohen's kappa and raw agreement for both label dimensions, plus confusion matrices. Only matching source IDs are compared; unmatched counts are reported.
