"""Grouped comparison of text models against explicitly weak routing proxies."""
from __future__ import annotations
import argparse, json, platform, time
from collections import Counter
from pathlib import Path
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SEED = 42
SCOPES = ["FRAME", "CLIP", "WHOLE_VIDEO"]
TIERS = ["L1_DIRECT", "L2_RETRIEVAL", "L3_GRAPH"]

def load(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("query", "scope_proxy", "tier_proxy", "video_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"line {line_no}: invalid {key}")
        rows.append(row)
    if len(rows) < 30:
        raise ValueError(f"Need at least 30 rows, got {len(rows)}")
    return rows

def vectorizer(max_features=12000):
    return TfidfVectorizer(ngram_range=(1, 2), max_features=max_features,
                           min_df=2, sublinear_tf=True, strip_accents="unicode")

def build_models():
    hybrid = FeatureUnion([
        ("word", vectorizer()),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                max_features=10000, min_df=2, sublinear_tf=True)),
    ])
    return {
        "word_tfidf_logistic_regression": Pipeline([
            ("features", vectorizer()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced",
                                              C=2.0, random_state=SEED)),
        ]),
        "word_tfidf_linear_svm": Pipeline([
            ("features", vectorizer()),
            ("classifier", LinearSVC(class_weight="balanced", C=1.0, random_state=SEED)),
        ]),
        "word_char_tfidf_logistic_regression": Pipeline([
            ("features", hybrid),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced",
                                              C=2.0, random_state=SEED)),
        ]),
        "word_tfidf_random_forest": Pipeline([
            ("features", vectorizer(5000)),
            ("select", SelectKBest(chi2, k=2000)),
            ("classifier", RandomForestClassifier(n_estimators=250, max_features="sqrt",
                min_samples_leaf=2, class_weight="balanced_subsample",
                n_jobs=-1, random_state=SEED)),
        ]),
        "word_tfidf_extra_trees": Pipeline([
            ("features", vectorizer(5000)),
            ("select", SelectKBest(chi2, k=2000)),
            ("classifier", ExtraTreesClassifier(n_estimators=250, max_features="sqrt",
                min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=SEED)),
        ]),
    }

def score(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=labels,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }

def connected_groups(rows):
    """Keep same-video and exact-normalized-query records in one fold."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    seen = {}
    for i, row in enumerate(rows):
        keys = (("video", row["video_id"]),
                ("query", " ".join(row["query"].lower().split())))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    return np.asarray([find(i) for i in range(len(rows))], dtype=object)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/activitynet_model_comparison_proxy.json"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.1 <= args.test_size <= 0.4:
        parser.error("--test-size must be between 0.1 and 0.4")
    rows = load(args.input)
    texts = np.asarray([r["query"] for r in rows], dtype=object)
    scope = np.asarray([r["scope_proxy"] for r in rows], dtype=object)
    tier = np.asarray([r["tier_proxy"] for r in rows], dtype=object)
    groups = connected_groups(rows)
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=SEED
    ).split(texts, scope, groups))
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise RuntimeError("Exact-query/video group leakage detected")
    if len(set(scope[train_idx])) < 2 or len(set(tier[train_idx])) < 2:
        raise ValueError("Training split has insufficient target diversity")
    report = {
        "experiment": "activitynet_qa_weak_supervision_diagnostic",
        "dataset": {
            "source": "ActivityNet-QA",
            "repository": "https://github.com/MILVLG/activitynet-qa",
            "rows": len(rows),
            "unique_videos": len({r["video_id"] for r in rows}),
            "unique_normalized_queries": len({" ".join(r["query"].lower().split()) for r in rows}),
            "train_rows": len(train_idx), "test_rows": len(test_idx),
            "train_groups": len(set(groups[train_idx])),
            "test_groups": len(set(groups[test_idx])),
            "split": "GroupShuffleSplit over connected video_id and exact-query groups",
            "seed": SEED,
            "scope_proxy_counts": dict(Counter(scope)),
            "tier_proxy_counts": dict(Counter(tier)),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "label_warning": (
            "Targets are lexical heuristic proxies, not ActivityNet-QA annotations "
            "or human ground truth. Metrics measure reproduction of the heuristic, "
            "not real-world routing accuracy."
        ),
        "models": {},
    }
    x_train, x_test = texts[train_idx].tolist(), texts[test_idx].tolist()
    predictions = []
    for name, template in build_models().items():
        scope_model = template
        # Fit an independently cloned pipeline for the second output head.
        from sklearn.base import clone
        tier_model = clone(template)
        start = time.perf_counter()
        scope_model.fit(x_train, scope[train_idx])
        tier_model.fit(x_train, tier[train_idx])
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scope_pred = scope_model.predict(x_test)
        tier_pred = tier_model.predict(x_test)
        inference_seconds = time.perf_counter() - start
        report["models"][name] = {
            "scope": score(scope[test_idx], scope_pred, SCOPES),
            "reasoning_tier": score(tier[test_idx], tier_pred, TIERS),
            "train_seconds_both_heads": train_seconds,
            "test_inference_seconds_both_heads": inference_seconds,
            "mean_inference_ms_per_query": inference_seconds * 1000 / len(test_idx),
        }
        for position, idx in enumerate(test_idx):
            predictions.append({
                "model": name, "source_id": rows[idx].get("source_id"),
                "video_id": rows[idx]["video_id"], "query": rows[idx]["query"],
                "scope_proxy": scope[idx], "scope_prediction": str(scope_pred[position]),
                "tier_proxy": tier[idx], "tier_prediction": str(tier_pred[position]),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred_path = args.output.with_name(args.output.stem + "_predictions.jsonl")
    pred_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                                 for x in predictions), encoding="utf-8")
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(args.output), "predictions": str(pred_path),
        "rows": len(rows), "train_rows": len(train_idx), "test_rows": len(test_idx),
        "warning": report["label_warning"],
        "models": {k: {
            "scope_macro_f1": v["scope"]["macro_f1"],
            "tier_macro_f1": v["reasoning_tier"]["macro_f1"],
            "mean_inference_ms_per_query": v["mean_inference_ms_per_query"],
        } for k, v in report["models"].items()},
    }, indent=2))

if __name__ == "__main__":
    main()
