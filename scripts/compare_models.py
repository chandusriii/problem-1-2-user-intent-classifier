"""Compare query-text classifiers with video-grouped train/test separation.
All routing labels are heuristic proxies, not benchmark ground truth.
"""
from __future__ import annotations
import argparse, json, time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

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
    if not rows:
        raise ValueError("No dataset rows")
    return rows

def make_pipeline(vectorizer, classifier):
    return Pipeline([("features", vectorizer), ("classifier", classifier)])

def build_models():
    def word():
        return TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True, strip_accents="unicode")
    def forest_vec():
        return TfidfVectorizer(ngram_range=(1, 2), max_features=3500, sublinear_tf=True, strip_accents="unicode")
    def hybrid():
        return FeatureUnion([("word", word()), ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000, min_df=2, sublinear_tf=True))])
    return {
        "word_tfidf_logistic_regression": (make_pipeline(word(), LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0, random_state=SEED)), make_pipeline(word(), LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0, random_state=SEED))),
        "word_tfidf_linear_svm": (make_pipeline(word(), LinearSVC(class_weight="balanced", C=1.0, random_state=SEED)), make_pipeline(word(), LinearSVC(class_weight="balanced", C=1.0, random_state=SEED))),
        "word_char_tfidf_logistic_regression": (make_pipeline(hybrid(), LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0, random_state=SEED)), make_pipeline(hybrid(), LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0, random_state=SEED))),
        "word_tfidf_random_forest": (make_pipeline(forest_vec(), RandomForestClassifier(n_estimators=250, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)), make_pipeline(forest_vec(), RandomForestClassifier(n_estimators=250, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))),
        "word_tfidf_extra_trees": (make_pipeline(forest_vec(), ExtraTreesClassifier(n_estimators=250, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=SEED)), make_pipeline(forest_vec(), ExtraTreesClassifier(n_estimators=250, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=SEED)))
    }

def metrics(y_true, y_pred, labels):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(), "labels": labels}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/activitynet_model_comparison.json"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    rows = load(args.input)
    texts = np.asarray([r["query"] for r in rows], dtype=object)
    scopes = np.asarray([r["scope_proxy"] for r in rows], dtype=object)
    tiers = np.asarray([r["tier_proxy"] for r in rows], dtype=object)
    groups = np.asarray([r["video_id"] for r in rows], dtype=object)
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=SEED).split(texts, scopes, groups))
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise RuntimeError("Video ID overlap detected across train and test")
    x_train, x_test = texts[train_idx].tolist(), texts[test_idx].tolist()
    report = {"dataset": {"source": "ActivityNet-QA", "repository": "https://github.com/MILVLG/activitynet-qa",
        "rows": len(rows), "unique_videos": len(set(groups)), "train_rows": len(train_idx), "test_rows": len(test_idx),
        "train_videos": len(set(groups[train_idx])), "test_videos": len(set(groups[test_idx])),
        "split": "GroupShuffleSplit by video_id; no overlap", "seed": SEED,
        "scope_proxy_counts": dict(Counter(scopes)), "tier_proxy_counts": dict(Counter(tiers)),
        "label_warning": "ActivityNet-QA does not provide routing labels. These heuristic proxy targets are not human ground truth."}, "models": {}}
    prediction_rows = []
    for name, (scope_model, tier_model) in build_models().items():
        start = time.perf_counter()
        scope_model.fit(x_train, scopes[train_idx]); tier_model.fit(x_train, tiers[train_idx])
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scope_pred = scope_model.predict(x_test); tier_pred = tier_model.predict(x_test)
        inference_seconds = time.perf_counter() - start
        report["models"][name] = {"scope": metrics(scopes[test_idx], scope_pred, SCOPES),
            "reasoning_tier": metrics(tiers[test_idx], tier_pred, TIERS), "train_seconds": train_seconds,
            "test_inference_seconds": inference_seconds, "mean_inference_ms_per_query": inference_seconds * 1000 / len(test_idx)}
        for pos, idx in enumerate(test_idx):
            prediction_rows.append({"model": name, "source_id": rows[idx].get("source_id"), "video_id": rows[idx]["video_id"],
                "query": rows[idx]["query"], "scope_proxy": scopes[idx], "scope_prediction": str(scope_pred[pos]),
                "tier_proxy": tiers[idx], "tier_prediction": str(tier_pred[pos])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_name(args.output.stem + "_predictions.jsonl")
    prediction_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in prediction_rows), encoding="utf-8")
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(args.output), "predictions": str(prediction_path),
        "rows": len(rows), "test_rows": len(test_idx),
        "models": {k: {"scope_macro_f1": v["scope"]["macro_f1"], "tier_macro_f1": v["reasoning_tier"]["macro_f1"]} for k, v in report["models"].items()}}, indent=2))
if __name__ == "__main__":
    main()
