"""Lightweight, interpretable classifier for video-query retrieval scope."""
from __future__ import annotations
import re
from typing import Any

_TIMESTAMP = re.compile(r"(?<!\d)(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?(?!\d)")

def normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        raise ValueError("query cannot be empty")
    return query

def engineer_features(query: str) -> dict[str, Any]:
    """Extract numeric and boolean cues without video decoding or model downloads."""
    q = normalize_query(query)
    lower = q.lower()
    return {
        "token_count": len(re.findall(r"\b[\w']+\b", lower)),
        "char_count": len(q),
        "timestamp_count": len(_TIMESTAMP.findall(q)),
        "frame_cue": bool(re.search(r"\b(frame|image|picture|at this moment|in this shot)\b", lower)),
        "clip_cue": bool(re.search(r"\b(clip|scene|segment|moment|during|before|after|when|while)\b", lower)),
        "whole_video_cue": bool(re.search(r"\b(entire video|whole video|throughout the video|overall video|video as a whole|summari[sz]e|overview|all events|main events)\b", lower)),
        "temporal_relation": bool(re.search(r"\b(before|after|then|earlier|later|until|while|following)\b", lower)),
        "count_cue": bool(re.search(r"\b(how many|how often|count|number of times|every time|all occurrences)\b", lower)),
        "causal_cue": bool(re.search(r"\b(why|cause|caused|reason for|result in|lead to)\b", lower)),
        "comparison_cue": bool(re.search(r"\b(compare|difference|different|similar|versus|vs\.?|change from .* to|relationship between)\b", lower)),
        "global_cue": bool(re.search(r"\b(throughout|from beginning to end|from start to finish|across the video|overall)\b", lower)),
        "speech_cue": bool(re.search(r"\b(say|said|says|speaker|speech|transcript|mention|mentions|spoken|talking)\b", lower)),
        "ocr_cue": bool(re.search(r"\b(read|text on|written on|sign|subtitle|on-screen text)\b", lower)),
    }

def classify_query(query: str) -> dict[str, Any]:
    """Predict requested scope and reasoning tier and map them to a route."""
    q = normalize_query(query)
    f = engineer_features(q)
    lower = q.lower()

    # Required primary classes are FRAME, CLIP, and WHOLE_VIDEO.
    if f["whole_video_cue"] or f["global_cue"]:
        scope = "WHOLE_VIDEO"
    elif f["timestamp_count"] or f["frame_cue"] or (f["ocr_cue"] and re.search(r"\\bat\\s+\\d", lower)):
        scope = "FRAME"
    elif f["clip_cue"] or f["speech_cue"] or f["ocr_cue"]:
        scope = "CLIP"
    else:
        scope = "CLIP"

    # Keep reasoning level separate from retrieval scope.
    if (f["comparison_cue"] or f["causal_cue"]) and f["global_cue"]:
        level = "L3_GRAPH"
    elif f["count_cue"] or f["temporal_relation"] or f["causal_cue"] or f["comparison_cue"] or scope == "WHOLE_VIDEO":
        level = "L2_RETRIEVAL"
    else:
        level = "L1_DIRECT"

    route = {
        "L1_DIRECT": "DIRECT_OR_TARGETED_CONTEXT",
        "L2_RETRIEVAL": "RETRIEVE_RELEVANT_CLIPS",
        "L3_GRAPH": "CROSS_SECTION_GRAPH_RETRIEVAL",
    }[level]
    return {
        "query": q,
        "scope": scope,
        "level": level,
        "retrieval_strategy": route,
        "features": f,
        "classifier": "interpretable_rules_baseline",
    }

def classify_batch(queries: list[str]) -> list[dict[str, Any]]:
    """Batch queries while keeping classification stateless."""
    return [classify_query(query) for query in queries]
