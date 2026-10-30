# ML Quality Evidence: ActivityNet-QA Exploratory Benchmark

## Purpose and provenance

This experiment uses question text from [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa), introduced by Yu et al. (AAAI 2019). It uses query text and identifiers only, not answers or video files. The checked-in sample has 750 questions: 500 sampled from the upstream training question list and 250 from validation.

The upstream repository states an MIT licence alongside a non-commercial-use restriction. Retain attribution, review the upstream terms and do not use this data commercially without resolving that restriction.

## Critical target limitation

ActivityNet-QA does not provide FRAME / CLIP / WHOLE_VIDEO retrieval-scope labels or L1_DIRECT / L2_RETRIEVAL / L3_GRAPH reasoning-tier labels. The current targets are heuristic proxies inferred from query wording. They are not source annotations, human judgements, or ground truth.

This is therefore a weak-supervision diagnostic. Scores show how well text models reproduce a lexical heuristic; they do **not** measure real-world routing correctness, video understanding, or official ActivityNet-QA performance. High proxy scores may simply mean the model learned the heuristic's own keywords.

## Comparison protocol

`scripts/compare_models.py` compares:
1. Word unigram/bigram TF-IDF + Logistic Regression.
2. Word unigram/bigram TF-IDF + Linear SVM.
3. Combined word and character n-gram TF-IDF + Logistic Regression.
4. Word TF-IDF + feature selection + Random Forest.
5. Word TF-IDF + feature selection + Extra Trees.

Separate output heads predict scope and tier. The holdout uses GroupShuffleSplit (seed 42) over connected components: records sharing a video ID or exact normalized query remain in one partition. Metrics include accuracy, macro/weighted F1, per-class scores, confusion matrices, training time and mean inference time.

## Run locally

Download the upstream `dataset/train_q.json` and `dataset/val_q.json` inputs. Do not add complete upstream splits to Git.

```bash
python scripts/prepare_activitynet.py --train data/external/train_q.json --validation data/external/val_q.json
python scripts/compare_models.py --input data/processed/activitynet_qa_train_proxy.jsonl --output results/activitynet_model_comparison_proxy.json
```

CI executes the comparison on the checked-in 750-question sample and uploads the JSON report plus per-example predictions as the `activitynet-qa-proxy-benchmark` artifact. The report carries the proxy-label warning and records Python, NumPy and scikit-learn versions.

## What is required for a defensible quality claim

1. Finalize annotation guidance with clear, positive, negative and ambiguous examples for each class.
2. Have two annotators independently label an unbiased sample without seeing proxy labels or model predictions.
3. Measure Cohen's kappa per task, adjudicate disagreements, and preserve annotation provenance.
4. Freeze grouped train/validation/test splits before model selection. Group video IDs and exact/near duplicate question families.
5. Compare these baselines with the current router and a pretrained sentence-embedding model on the reviewed labels.
6. Report per-class metrics, confidence intervals, error analysis, latency, memory and class coverage.

Until such a reviewed dataset exists, do not present proxy scores as validated model quality.
