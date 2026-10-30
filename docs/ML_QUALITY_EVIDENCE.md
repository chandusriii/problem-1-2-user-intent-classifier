# ML Quality Evidence: ActivityNet-QA Query Routing Experiments

## Dataset and attribution

The source dataset is **ActivityNet-QA**, published by the MILVLG authors. The upstream repository documents 58,000 video-question-answer pairs from 5,800 videos, with 32,000 training questions, 18,000 validation questions and 8,000 test questions. This experiment uses only question text and IDs; it does not use answer strings or video files.

- Source repository: https://github.com/MILVLG/activitynet-qa
- Paper: Yu et al., *ActivityNet-QA: A Dataset for Understanding Complex Web Videos via Question Answering*, AAAI 2019.
- Upstream licence note: the repository states MIT licence but restricts use to non-commercial purposes. Check upstream terms before reuse and retain attribution.

## Essential limitation: proxy labels

ActivityNet-QA does **not** provide the labels `FRAME`, `CLIP`, `WHOLE_VIDEO`, `L1_DIRECT`, `L2_RETRIEVAL`, or `L3_GRAPH`. `prepare_activitynet.py` derives provisional labels from question text. Those labels are heuristic proxies, not human ground truth. They are useful to test dataset ingestion, grouped evaluation, class imbalance, baselines and comparison code, but proxy-label scores must not be presented as real-world routing accuracy or official ActivityNet-QA results.

Before claiming model quality, replace proxy labels with human-adjudicated labels. Start with a stratified sample, write annotation guidance, use two independent annotators, resolve disagreements, and report Cohen's kappa (or another appropriate agreement statistic). Then run the same evaluation on reviewed labels.

## Leakage control and evaluation

`compare_models.py` uses `GroupShuffleSplit` grouped by `video_id`; questions from the same video cannot appear in both train and test. The seed is 42. Exact query duplicates across different videos may still occur; treat this as a remaining source of lexical overlap and consider a duplicate-family group key in the next revision.

The comparison includes word TF-IDF + Logistic Regression, word TF-IDF + Linear SVM, word+character TF-IDF + Logistic Regression, Random Forest and Extra Trees. Metrics include accuracy, macro F1, weighted F1, per-class precision/recall/F1/support, confusion matrices, and CPU train/inference time.

## Checked-in sample

A small 750-question sample is included at `data/external/activitynet_qa_questions_sample.jsonl` (500 questions from the official train question list and 250 from validation). The derived proxy-labelled copy is `data/processed/activitynet_qa_sample_proxy.jsonl`; its label distribution is summarized in `data/processed/activitynet_qa_sample_summary.json`. The sample intentionally contains question text and source identifiers, not answers or video media. It is a smoke-test sample, not a substitute for full source splits.

## Local run

Download the upstream `dataset/train_q.json` and `dataset/val_q.json` files from the source repository, then run:

```bash
python scripts/prepare_activitynet.py --train data/external/train_q.json --validation data/external/val_q.json
python scripts/compare_models.py --input data/processed/activitynet_qa_train_proxy.jsonl --output results/activitynet_model_comparison.json
```

Keep generated results out of Git until inspected. Do not mix validation data into the held-out test set or tune repeatedly on the test partition.
