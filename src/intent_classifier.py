"""Multi-head CPU classifier for video-query retrieval scope and reasoning tier."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SCOPES = ("FRAME", "CLIP", "WHOLE_VIDEO")
TIERS = ("L1_DIRECT", "L2_RETRIEVAL", "L3_GRAPH")

_TIMESTAMP = re.compile(r"(?<!\d)(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?(?!\d)")

def normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    normalized = re.sub(r"\s+", " ", query).strip()
    if not normalized:
        raise ValueError("query cannot be empty")
    return normalized

def extract_dense_signals(text: str) -> np.ndarray:
    """Return four numeric features in a stable 2-D row shape (1, 4)."""
    q = normalize_query(text)
    lower = q.lower()
    has_timestamp = bool(_TIMESTAMP.search(q))
    temporal = bool(re.search(r"\b(after|before|then|sequence|during|while|when|at the beginning|at the end)\b", lower))
    return np.asarray([[
        float(has_timestamp),
        float(temporal),
        float(len(q)),
        float(len(q.split())),
    ]], dtype=np.float64)


def extract_and_align_features(query_text: str, vectorizer: TfidfVectorizer) -> np.ndarray:
    """Combine one query's TF-IDF vector and dense structural signals safely."""
    query = normalize_query(query_text)
    sparse_features = vectorizer.transform([query]).toarray()
    dense_features = extract_dense_signals(query)
    if sparse_features.ndim != 2 or dense_features.ndim != 2:
        raise ValueError("TF-IDF and dense features must both be 2-D matrices")
    if sparse_features.shape[0] != dense_features.shape[0]:
        raise ValueError("TF-IDF and dense features must have matching row counts")
    return np.hstack((sparse_features, dense_features))


def combine_features(vectorizer: TfidfVectorizer, queries: list[str]) -> np.ndarray:
    """Horizontal TF-IDF + dense stacking with shape checks."""
    if not queries:
        raise ValueError("queries must not be empty")
    sparse_features = vectorizer.transform(queries).toarray()
    dense_features = np.vstack([extract_dense_signals(q) for q in queries])
    if sparse_features.ndim != 2 or dense_features.ndim != 2 or sparse_features.shape[0] != dense_features.shape[0]:
        raise ValueError("TF-IDF and dense feature matrices must have matching row counts")
    return np.hstack((sparse_features, dense_features))

class VideoRAGIntentClassifier:
    """Two independent Logistic Regression heads over shared text/structure features."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            max_features=1500,
            sublinear_tf=True,
        )
        self.scope_model = LogisticRegression(C=2.0, max_iter=300, class_weight="balanced")
        self.tier_model = LogisticRegression(C=1.5, max_iter=300, class_weight="balanced")
        self.is_trained = False

    def fit(self, queries: list[str], scopes: list[str], tiers: list[str]) -> "VideoRAGIntentClassifier":
        if not queries or len(queries) != len(scopes) or len(queries) != len(tiers):
            raise ValueError("queries, scopes, and tiers must be non-empty and have equal lengths")
        clean_queries = [normalize_query(q) for q in queries]
        invalid_scopes = set(scopes) - set(SCOPES)
        invalid_tiers = set(tiers) - set(TIERS)
        if invalid_scopes or invalid_tiers:
            raise ValueError(f"invalid labels: scopes={sorted(invalid_scopes)}, tiers={sorted(invalid_tiers)}")
        if len(set(scopes)) < 2 or len(set(tiers)) < 2:
            raise ValueError("training requires at least two distinct scope labels and two tier labels")
        sparse_features = self.vectorizer.fit_transform(clean_queries).toarray()
        dense_features = np.vstack([extract_dense_signals(q) for q in clean_queries])
        combined = np.hstack((sparse_features, dense_features))
        self.scope_model.fit(combined, np.asarray(scopes))
        self.tier_model.fit(combined, np.asarray(tiers))
        self.is_trained = True
        return self

    def predict_route(self, query: str, multi_route_margin: float = 0.15) -> dict[str, Any]:
        clean_query = normalize_query(query)
        if not 0.0 <= multi_route_margin <= 1.0:
            raise ValueError("multi_route_margin must be between 0 and 1")
        if not self.is_trained:
            # Do not import a nonexistent rules head from this ML module.
            # Use a conservative fallback without fabricating model probabilities.
            lower = clean_query.lower()
            if re.search(r"\\b(summarize|summary|overall|entire video|whole video|main theme|main topic)\\b", lower):
                fallback_scope, fallback_tier = "WHOLE_VIDEO", "L2_RETRIEVAL"
            elif re.search(r"\\b(say|said|speaker|speech|transcript|mention|talking|explain)\\b", lower):
                fallback_scope, fallback_tier = "CLIP", "L2_RETRIEVAL"
            elif _TIMESTAMP.search(clean_query) or re.search(r"\\b(color|colour|hue|frame|image|visible|wearing|sign)\\b", lower):
                fallback_scope, fallback_tier = "FRAME", "L1_DIRECT"
            else:
                fallback_scope, fallback_tier = "CLIP", "L2_RETRIEVAL"
            return {
                "query": clean_query,
                "assigned_scopes": [fallback_scope],
                "reasoning_tier": fallback_tier,
                "confidence_distribution": {},
                "classifier": "rules_fallback_untrained",
            }
        features = extract_and_align_features(clean_query, self.vectorizer)
        probabilities = self.scope_model.predict_proba(features)[0]
        learned_classes = list(self.scope_model.classes_)
        ordered = sorted(
            ((label, float(probabilities[i])) for i, label in enumerate(learned_classes)),
            key=lambda item: item[1],
            reverse=True,
        )
        top_scope, top_probability = ordered[0]
        final_scopes = [top_scope]
        if len(ordered) > 1 and top_probability - ordered[1][1] < multi_route_margin:
            final_scopes.append(ordered[1][0])
        tier = str(self.tier_model.predict(features)[0])
        return {
            "query": clean_query,
            "assigned_scopes": final_scopes,
            "reasoning_tier": tier,
            "confidence_distribution": {label: float(prob) for label, prob in ordered},
            "classifier": "tfidf_logistic_regression_multi_head",
        }

    def save(self, file_path: str = "models/pipeline.pkl") -> None:
        from pathlib import Path
        import pickle
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, file_path: str = "models/pipeline.pkl") -> "VideoRAGIntentClassifier":
        import pickle
        from pathlib import Path
        path = Path(file_path)
        with path.open("rb") as handle:
            instance = pickle.load(handle)
        if not isinstance(instance, cls):
            raise TypeError(f"saved object at {path} is not a {cls.__name__}")
        return instance
