"""Compare query-text classifiers with leakage-safe video-grouped test partitioning."""
from __future__ import annotations
import argparse,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

SEED=42
SCOPES=["FRAME","CLIP","WHOLE_VIDEO"]
TIERS=["L1_DIRECT","L2_RETRIEVAL","L3_GRAPH"]

def load(path):
    rows=[]
    for n,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        r=json.loads(line)
        for k in ("query","scope_proxy","tier_proxy","video_id"):
            if not isinstance(r.get(k),str) or not r[k].strip(): raise ValueError(f"line {n}: invalid {k}")
        rows.append(r)
    if not rows: raise ValueError("No dataset rows")
    return rows

def text_pipeline(vec,model):
    return Pipeline([("features",vec),("classifier",model)])

def build_models():
    def word():
        return TfidfVectorizer(ngram_range=(1,2),max_features=12000,sublinear_tf=True,strip_accents="unicode")
    def forest_vec():
        return TfidfVectorizer(ngram_range=(1,2),max_features=3500,sublinear_tf=True,strip_accents="unicode")
    def hybrid():
        return FeatureUnion([("word",word()),("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),max_features=8000,min_df=2,sublinear_tf=True))])
    return {
      "word_tfidf_logistic_regression":(text_pipeline(word(),LogisticRegression(max_iter=2000,class_weight="balanced",C=2.0,random_state=SEED)),text_pipeline(word(),LogisticRegression(max_iter=2000,class_weight="balanced",C=2.0,random_state=SEED))),
      "word_tfidf_linear_svm":(text_pipeline(word(),LinearSVC(class_weight="balanced",C=1.0,random_state=SEED)),text_pipeline(word(),LinearSVC(class_weight="balanced",C=1.0,random_state=SEED))),
      "word_char_tfidf_logistic_regression":(text_pipeline(hybrid(),LogisticRegression(max_iter=2000,class_weight="balanced",C=2.0,random_state=SEED)),text_pipeline(hybrid(),LogisticRegression(max_iter=2000,class_weight="balanced",C=2.0,random_state=SEED))),
      "word_tfidf_random_forest":(text_pipeline(forest_vec(),RandomForestClassifier(n_estimators=250,max_features="sqrt",min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=SEED)),text_pipeline(forest_vec(),RandomForestClassifier(n_estimators=250,max_features="sqrt",min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=SEED))),
      "word_tfidf_extra_trees":(text_pipeline(forest_vec(),ExtraTreesClassifier(n_estimators=250,max_features="sqrt",min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=SEED)),text_pipeline(forest_vec(),ExtraTreesClassifier(n_estimators=250,max_features="sqrt",min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=SEED)))
    }

def evaluate(y,p,labels):
    return {"accuracy":float(accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,labels=labels,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y,p,labels=labels,average="weighted",zero_division=0)),"per_class":classification_report(y,p,labels=labels,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(y,p,labels=labels).tolist(),"labels":labels}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=Path("results/activitynet_model_comparison.json"))
    ap.add_argument("--test-size",type=float,default=.2)
    args=ap.parse_args()
    rows=load(args.input)
    X=np.asarray([r["query"] for r in rows],dtype=object)
    ys=np.asarray([r["scope_proxy"] for r in rows],dtype=object)
    yt=np.asarray([r["tier_proxy"] for r in rows],dtype=object)
    groups=np.asarray([r["video_id"] for r in rows],dtype=object)
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=args.test_size,random_state=SEED).split(X,ys,groups))
    assert not (set(groups[tr]) & set(groups[te])), "Video ID leakage detected"
    xtr,xte=X[tr].tolist(),X[te].tolist()
    summary={"dataset":{"source":"ActivityNet-QA","repository":"https://github.com/MILVLG/activitynet-qa","rows":len(rows),"unique_videos":len(set(groups)),"train_rows":len(tr),"test_rows":len(te),"train_videos":len(set(groups[tr])),"test_videos":len(set(groups[te])),"split":"GroupShuffleSplit by video_id; zero video overlap","seed":SEED,"scope_proxy_counts":dict(Counter(ys)),"tier_proxy_counts":dict(Counter(yt)),"label_warning":"The source dataset does not provide routing labels. All targets here are heuristic proxies inferred from query wording; results are not gold-label accuracy."},"models":{}}
    prediction_rows=[]
    for name,(sm,tm) in build_models().items():
        t=time.perf_counter();sm.fit(xtr,ys[tr]);tm.fit(xtr,yt[tr]);train_seconds=time.perf_counter()-t
        t=time.perf_counter();sp=sm.predict(xte);tp=tm.predict(xte);infer_seconds=time.perf_counter()-t
        summary["models"][name]={"scope":evaluate(ys[te],sp,SCOPES),"reasoning_tier":evaluate(yt[te],tp,TIERS),"train_seconds":train_seconds,"test_inference_seconds":infer_seconds,"mean_inference_ms_per_query":infer_seconds*1000/len(te)}
        for j,idx in enumerate(te):
            item={"source_id":rows[idx].get("source_id"),"video_id":rows[idx]["video_id"],"query":rows[idx]["query"],"scope_proxy":ys[idx],"tier_proxy":yt[idx]}
            item[name+"_scope_pred"]=str(sp[j]);item[name+"_tier_pred"]=str(tp[j])
            prediction_rows.append(item)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pred=args.output.with_name(args.output.stem+"_predictions.jsonl")
    pred.write_text("".join(json.dumps(r,ensure_ascii=False)+"\\n" for r in prediction_rows),encoding="utf-8")
    args.output.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"summary":str(args.output),"predictions":str(pred),"rows":len(rows),"test_rows":len(te),"models":{k:{"scope_macro_f1":v["scope"]["macro_f1"],"tier_macro_f1":v["reasoning_tier"]["macro_f1"]} for k,v in summary["models"].items()}},indent=2))
if __name__=="__main__":main()
