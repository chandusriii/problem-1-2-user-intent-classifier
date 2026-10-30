"""Prepare ActivityNet-QA question text for a query-only routing experiment."""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path

TIMESTAMP = re.compile(r"(?<!\\d)(?:\\d{1,2}:)?\\d{1,2}:\\d{2}(?:\\.\\d+)?(?!\\d)")
TEMPORAL = re.compile(r"\\b(after|before|then|sequence|during|while|when|beginning|end|first|next|later|until|earlier|subsequently)\\b", re.I)
WHOLE = re.compile(r"\\b(summarize|summary|overall|entire|whole video|full video|throughout|main topic|main theme|all events|across the video|in this video)\\b", re.I)
CLIP = re.compile(r"\\b(say|said|speaker|speech|transcript|mention|talking|explain|happen|action|sequence|when|how many times|respond|reaction|before|after)\\b", re.I)
FRAME = re.compile(r"\\b(color|colour|hue|frame|image|visible|wearing|sign|logo|object|shirt|person|people|what is shown|what do you see|number of)\\b", re.I)

def map_labels(q: str):
    s = " ".join(q.lower().split())
    if WHOLE.search(s) or any(x in s for x in ("video about","entire recording","whole recording","full recording","throughout the recording","from start to end","from beginning to end")):
        scope = "WHOLE_VIDEO"
    elif TEMPORAL.search(s) or CLIP.search(s) or ("between " in s and TIMESTAMP.search(s)):
        scope = "CLIP"
    elif TIMESTAMP.search(s) or FRAME.search(s):
        scope = "FRAME"
    else:
        scope = "CLIP"
    temporal_count = len(TEMPORAL.findall(s))
    if scope == "WHOLE_VIDEO" or temporal_count >= 2 or any(x in s for x in ("compare","relationship","why does","how does","evolve","timeline","cause and effect","multiple events")):
        tier = "L3_GRAPH"
    elif scope == "FRAME" and not TEMPORAL.search(s):
        tier = "L1_DIRECT"
    else:
        tier = "L2_RETRIEVAL"
    return scope, tier

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for split,path in (("train",args.train),("validation",args.validation)):
        entries=json.loads(path.read_text(encoding="utf-8"))
        rows=[]
        for item in entries:
            q=item.get("question")
            if not isinstance(q,str) or not q.strip(): continue
            scope,tier=map_labels(q)
            rows.append({"query":q.strip(),"scope_proxy":scope,"tier_proxy":tier,"source":"ActivityNet-QA","source_id":item.get("question_id"),"video_id":item.get("video_name"),"label_method":"heuristic_v1_requires_human_review"})
        target=args.out/f"activitynet_qa_{split}_proxy.jsonl"
        target.write_text("".join(json.dumps(r,ensure_ascii=False)+"\\n" for r in rows),encoding="utf-8")
        summary={"rows":len(rows),"scope_counts":dict(Counter(r["scope_proxy"] for r in rows)),"tier_counts":dict(Counter(r["tier_proxy"] for r in rows)),"unique_video_ids":len({r["video_id"] for r in rows}),"unique_questions":len({r["query"].lower() for r in rows})}
        (args.out/f"activitynet_qa_{split}_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(split,json.dumps(summary))
if __name__=="__main__": main()
