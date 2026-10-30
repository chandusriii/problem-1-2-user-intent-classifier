"""Compute inter-annotator agreement for human-reviewed routing labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix
SCOPES={"FRAME","CLIP","WHOLE_VIDEO"}
TIERS={"L1_DIRECT","L2_RETRIEVAL","L3_GRAPH"}

def load_annotations(path:Path):
    rows=[];seen=set()
    for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line)
        for field in ("source_id","scope","tier"):
            if not isinstance(row.get(field),str) or not row[field].strip():
                raise ValueError(f"line {line_no}: missing/invalid {field}")
        if row["scope"] not in SCOPES or row["tier"] not in TIERS:
            raise ValueError(f"line {line_no}: invalid scope/tier label")
        if row["source_id"] in seen: raise ValueError(f"line {line_no}: duplicate source_id")
        seen.add(row["source_id"]);rows.append(row)
    if not rows: raise ValueError("no annotations found")
    return rows

def summarize_agreement(first,second):
    a={x["source_id"]:x for x in first};b={x["source_id"]:x for x in second}
    common=sorted(set(a)&set(b))
    if not common: raise ValueError("annotator files share no source_id values")
    unmatched_first=sorted(set(a)-set(b));unmatched_second=sorted(set(b)-set(a))
    report={"n_common":len(common),"only_first":len(unmatched_first),"only_second":len(unmatched_second),"scope":{},"tier":{}}
    for field,labels in (("scope",sorted(SCOPES)),("tier",sorted(TIERS))):
        x=[a[k][field] for k in common];y=[b[k][field] for k in common]
        report[field]={"raw_agreement":sum(i==j for i,j in zip(x,y))/len(common),
          "cohen_kappa":float(cohen_kappa_score(x,y,labels=labels)),
          "confusion_matrix":confusion_matrix(x,y,labels=labels).tolist(),"labels":labels}
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--annotator-a",type=Path,required=True)
    parser.add_argument("--annotator-b",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("results/annotation_agreement.json"))
    args=parser.parse_args()
    report=summarize_agreement(load_annotations(args.annotator_a),load_annotations(args.annotator_b))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
