# Human Annotation Guide — Video Query Routing

Version: 1.0  
Unit: one user query, considered without access to the source video.

## Purpose

Assign two labels to each query:

1. **Retrieval scope:** the narrowest video span that can reasonably answer the query.
2. **Reasoning tier:** the amount of reasoning needed after retrieving the relevant evidence.

Annotate the information need expressed by the question. Do not guess whether an answer is true.

## Retrieval scope

### FRAME
One visual moment, still frame, or very short instant is sufficient.

Examples: "What color is the car at 00:15?", "What text is written on the sign?", "How many people are visible in this shot?"

A timestamp alone does not imply FRAME. Ask whether one visual observation is enough.

### CLIP
A bounded temporal segment, utterance, action, or sequence is needed.

Examples: "What does the speaker say after the introduction?", "What happens between the door opening and the person leaving?", "How does the player react immediately after scoring?"

### WHOLE_VIDEO
A global summary, theme, full timeline, or information distributed across the recording is needed.

Examples: "What is the overall topic of the video?", "Summarize the main events from beginning to end.", "How does the person's role change throughout the recording?"

## Reasoning tier

### L1_DIRECT
A direct observation or simple fact answers the query once its evidence is located.

Examples: "What color is the car?", "Is the person wearing a helmet?", "How many people are visible?"

### L2_RETRIEVAL
The system must locate a particular moment or segment and retrieve evidence, but the answer itself is direct.

Examples: "When does the speaker mention the launch?", "What does the presenter say after the chart appears?", "Find the moment where the dog enters the room."

### L3_GRAPH
The answer requires joining multiple observations, tracking temporal change, comparing events, or reasoning about relationships across time.

Examples: "How does the person's behavior change from the beginning to the end?", "Compare the two attempts and explain what changed.", "What sequence of events causes the final outcome?"

## Annotation workflow

1. Read the query without looking at proxy labels or model predictions.
2. Select one scope label and one tier label independently.
3. Record confidence as `high`, `medium`, or `low`.
4. Add a concise one-sentence rationale.
5. Set `needs_adjudication=true` if ambiguous or underspecified.
6. Do not use source answers, heuristic labels, or model predictions.

## Edge cases

- A timestamp is not sufficient by itself to imply FRAME. A timestamped action or spoken segment can be CLIP.
- "What is happening?" is usually CLIP unless it clearly asks what is visible in a single still image.
- A question about an entire video usually has WHOLE_VIDEO scope. Use L3_GRAPH only when integrating multiple events or tracking change; a direct aggregate may remain L2_RETRIEVAL.
- Judge the minimum evidence and reasoning required. Do not infer complexity from domain knowledge.
- Flag genuine ambiguity rather than silently inventing context.

## Agreement and adjudication

Use two independent annotators for each sampled query. Compute Cohen's kappa separately for scope and tier before adjudication. Review disagreements through a documented consensus process or a third adjudicator. Preserve raw labels separately from adjudicated labels. Report class counts, disagreement rate, kappa, adjudication rules, and excluded rows.

## Required JSONL record

One UTF-8 JSON object per line:

```json
{"query":"What color is the car at 00:15?","scope":"FRAME","tier":"L1_DIRECT","confidence":"high","rationale":"A single visual moment is sufficient for a direct appearance attribute.","annotator_id":"ann_01","needs_adjudication":false}
```

Do not include source answers, video transcripts, private metadata, or model-generated predictions in the annotation file.
