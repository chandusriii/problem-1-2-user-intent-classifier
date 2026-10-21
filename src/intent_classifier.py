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
    """Extract stable lexical and structural cues from query text."""
    q = normalize_query(query)
    lower = q.lower()
    return {
        "token_count": len(re.findall(r"\b[\w']+\b", lower)),
        "char_count": len(q),
        "timestamp_count": len(_TIMESTAMP.findall(q)),
        "frame_cue": bool(re.search(r"\b(frame|image|picture|at this moment|in this shot|what color|what colour|hue|visible|wearing|which object)\b", lower)),
        "clip_cue": bool(re.search(r"\b(clip|scene|segment|moment|during|before|after|when|while|happens?|doing|action|sequence|enters?|leaves?|stops?|starts?|explains?)\b", lower)),
        "whole_video_cue": bool(re.search(r"\b(entire video|whole video|throughout the video|overall video|video as a whole|summari[sz]e|overview|all events|main events|main message|main topic|what is this video about)\b", lower)),
        "temporal_relation": bool(re.search(r"\b(before|after|then|earlier|later|until|while|following|at the beginning|at the start|at the end|in the beginning|first half|second half|middle of the video)\b", lower)),
        "count_cue": bool(re.search(r"\b(how many|how often|count|number of times|every time|all occurrences)\b", lower)),
        "causal_cue": bool(re.search(r"\b(why|cause|caused|reason for|result in|lead to)\b", lower)),
        "comparison_cue": bool(re.search(r"\b(compare|difference|different|similar|versus|vs\.?|change from .* to|relationship between)\b", lower)),
        "global_cue": bool(re.search(r"\b(throughout|from beginning to end|from start to finish|across the video|overall|over the course of the video)\b", lower)),
        "speech_cue": bool(re.search(r"\b(say|said|says|speaker|speech|transcript|mention|mentions|spoken|talking|explain|explains|explained|speaking|stops speaking)\b", lower)),
        "ocr_cue": bool(re.search(r"\b(read|text on|written on|sign|subtitle|on-screen text|written|displayed)\b", lower)),
    }

def classify_query(query: str) -> dict[str, Any]:
    """Predict retrieval scope and reasoning tier, and map them to a route."""
    q = normalize_query(query)
    f = engineer_features(q)
    lower = q.lower()

    # Broad summaries and questions about coverage across the video need global evidence.
    if f["whole_video_cue"] or f["global_cue"]:
        scope = "WHOLE_VIDEO"
    # Questions about words or speech need a transcript/temporal segment, even with a timestamp.
    elif f["speech_cue"]:
        scope = "CLIP"
    # A specific static visual fact at a timestamp is frame-scoped.
    elif f["timestamp_count"] or f["frame_cue"] or (
        f["ocr_cue"] and re.search(r"\bat\s+\d", lower)
    ):
        scope = "FRAME"
    # Temporal relationships and actions require an interval/sequence, not a single frame.
    elif f["clip_cue"] or f["temporal_relation"] or f["ocr_cue"]:
        scope = "CLIP"
    elif f["count_cue"]:
        scope = "WHOLE_VIDEO"
    else:
        scope = "CLIP"

    global_change = bool(
        re.search(r"\b(change|changes|changed|evolve|evolves|evolved|develop|develops|developed|progress|progresses|shift|shifts|transforms?)\b", lower)
        and (f["global_cue"] or f["whole_video_cue"])
    )
    if global_change or (f["comparison_cue"] and f["global_cue"]) or (f["causal_cue"] and f["global_cue"]):
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
