"""CPU-oriented multi-head classifier for video-query routing."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SCOPES = ("FRAME", "CLIP", "WHOLE_VIDEO")
TIERS = ("L1_DIRECT", "L2_RETRIEVAL", "L3_GRAPH")

_TIMESTAMP = re.compile(r"(?<!\d)(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?(?!\d)")
_TEMPORAL_CUE = re.compile(r"\b(after|before|then|sequence|during|while|when|beginning|end)\b", re.I)
_WHOLE_VIDEO_CUE = re.compile(r"\b(summarize|summary|overall|entire video|whole video|main theme|main topic|full video)\b", re.I)
_CLIP_CUE = re.compile(r"\b(say|said|speaker|speech|transcript|mention|talking|explain|happens|action|sequence)\b", re.I)
_FRAME_CUE = re.compile(r"\b(color|colour|hue|frame|image|visible|wearing|sign|logo|object)\b", re.I)


def normalize_query(query: str) -> str:
    """Validate and normalize one user query."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    normalized = re.sub(r"\s+", " ", query).strip()
    if not normalized:
        raise ValueError("query cannot be empty")
    return normalized


def extract_dense_signals(text: str) -> np.ndarray:
    """Return timestamp, temporal, character-count and word-count features."""
    query = normalize_query(text)
    return np.asarray([[
        float(bool(_TIMESTAMP.search(query))),
        float(bool(_TEMPORAL_CUE.search(query))),
        float(len(query)),
        float(len(query.split())),
    ]], dtype=np.float64)


def _combine_sparse_and_dense(sparse_features, dense_features: np.ndarray) -> csr_matrix:
    """Combine sparse TF-IDF and dense signals without densifying the vocabulary."""
    if sparse_features.ndim != 2 or dense_features.ndim != 2:
        raise ValueError("feature inputs must be two-dimensional")
    if sparse_features.shape[0] != dense_features.shape[0]:
        raise ValueError("feature inputs must have matching row counts")
    return hstack((sparse_features, csr_matrix(dense_features)), format="csr")


def extract_and_align_features(query_text: str, vectorizer: TfidfVectorizer) -> csr_matrix:
    """Transform one query using the fitted vectorizer and dense signals."""
    query = normalize_query(query_text)
    return _combine_sparse_and_dense(
        vectorizer.transform([query]), extract_dense_signals(query)
    )


def combine_features(vectorizer: TfidfVectorizer, queries: list[str]) -> csr_matrix:
    """Transform a batch of queries into the model's sparse feature matrix."""
    if not queries:
        raise ValueError("queries must not be empty")
    clean_queries = [normalize_query(query) for query in queries]
    dense = np.vstack([extract_dense_signals(query) for query in clean_queries])
    return _combine_sparse_and_dense(vectorizer.transform(clean_queries), dense)


class VideoRAGIntentClassifier:
    """Two independent Logistic Regression heads over shared query features."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), analyzer="word", max_features=1500, sublinear_tf=True
        )
        self.scope_model = LogisticRegression(
            C=2.0, max_iter=300, class_weight="balanced", random_state=42
        )
        self.tier_model = LogisticRegression(
            C=1.5, max_iter=300, class_weight="balanced", random_state=42
        )
        self.is_trained = False

    def fit(self, queries: list[str], scopes: list[str], tiers: list[str]) -> "VideoRAGIntentClassifier":
        if not queries or len(queries) != len(scopes) or len(queries) != len(tiers):
            raise ValueError("queries, scopes, and tiers must be non-empty and have equal lengths")
        clean_queries = [normalize_query(query) for query in queries]
        invalid_scopes = set(scopes) - set(SCOPES)
        invalid_tiers = set(tiers) - set(TIERS)
        if invalid_scopes or invalid_tiers:
            raise ValueError(
                f"invalid labels: scopes={sorted(invalid_scopes)}, tiers={sorted(invalid_tiers)}"
            )
        if len(set(scopes)) < 2 or len(set(tiers)) < 2:
            raise ValueError("training requires at least two distinct scope and tier labels")

        self.vectorizer.fit(clean_queries)
        dense = np.vstack([extract_dense_signals(query) for query in clean_queries])
        features = _combine_sparse_and_dense(self.vectorizer.transform(clean_queries), dense)
        self.scope_model.fit(features, np.asarray(scopes))
        self.tier_model.fit(features, np.asarray(tiers))
        self.is_trained = True
        return self

    @staticmethod
    def _rules_fallback(query: str) -> tuple[str, str]:
        """Conservative heuristic fallback; it does not emit fake probabilities."""
        if _WHOLE_VIDEO_CUE.search(query):
            return "WHOLE_VIDEO", "L2_RETRIEVAL"
        if _CLIP_CUE.search(query):
            return "CLIP", "L2_RETRIEVAL"
        if _TIMESTAMP.search(query) or _FRAME_CUE.search(query):
            return "FRAME", "L1_DIRECT"
        return "CLIP", "L2_RETRIEVAL"

    def predict_route(self, query: str, multi_route_margin: float = 0.15) -> dict[str, Any]:
        clean_query = normalize_query(query)
        if not 0.0 <= multi_route_margin <= 1.0:
            raise ValueError("multi_route_margin must be between 0 and 1")
        if not self.is_trained:
            scope, tier = self._rules_fallback(clean_query)
            return {
                "query": clean_query,
                "assigned_scopes": [scope],
                "reasoning_tier": tier,
                "confidence_distribution": {},
                "classifier": "rules_fallback_untrained",
            }

        features = extract_and_align_features(clean_query, self.vectorizer)
        probabilities = self.scope_model.predict_proba(features)[0]
        ordered = sorted(
            ((str(label), float(probabilities[i])) for i, label in enumerate(self.scope_model.classes_)),
            key=lambda item: item[1],
            reverse=True,
        )
        scopes = [ordered[0][0]]
        if len(ordered) > 1 and ordered[0][1] - ordered[1][1] < multi_route_margin:
            scopes.append(ordered[1][0])
        tier = str(self.tier_model.predict(features)[0])
        return {
            "query": clean_query,
            "assigned_scopes": scopes,
            "reasoning_tier": tier,
            "confidence_distribution": {label: probability for label, probability in ordered},
            "classifier": "tfidf_logistic_regression_multi_head",
        }

    def save(self, file_path: str | Path = "models/pipeline.pkl") -> None:
        """Persist a fitted model locally. Only load trusted pickle artifacts."""
        if not self.is_trained:
            raise RuntimeError("cannot save an untrained classifier")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: str | Path = "models/pipeline.pkl") -> "VideoRAGIntentClassifier":
        """Load a trusted local model artifact."""
        path = Path(file_path)
        with path.open("rb") as handle:
            instance = pickle.load(handle)
        if not isinstance(instance, cls):
            raise TypeError(f"saved object at {path} is not a {cls.__name__}")
        return instance
