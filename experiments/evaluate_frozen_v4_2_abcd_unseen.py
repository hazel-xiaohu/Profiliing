#!/usr/bin/env python3
"""Evaluate the exact frozen Partition v4.2 ABCD models without refitting."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, types
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN = Path("/output/Partition/script/train_export_direct_log_xgboost_abcd.py")
FROZEN = Path("/output/Partition/artifacts/frozen_direct_log_xgboost_abcd_h100_20260812")

def load_training_contract():
    if "xgboost" not in sys.modules:
        stub=types.ModuleType("xgboost");stub.XGBRegressor=object;sys.modules["xgboost"]=stub
    spec=importlib.util.spec_from_file_location("frozen_abcd_contract",TRAIN)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def predict_frozen_json(path, x):
    """Vectorized inference matching the generated C evaluator (float splits)."""
    j=json.loads(path.read_text());learner=j["learner"]
    raw=learner["learner_model_param"]["base_score"].strip("[]")
    pred=np.full(len(x),float(raw),dtype=np.float64);x=np.asarray(x,dtype=np.float32)
    trees=learner["gradient_booster"]["model"]["trees"]
    rows=np.arange(len(x))
    for tree in trees:
        left=np.asarray(tree["left_children"],dtype=np.int32);right=np.asarray(tree["right_children"],dtype=np.int32)
        feature=np.asarray(tree["split_indices"],dtype=np.int32);threshold=np.asarray(tree["split_conditions"],dtype=np.float32)
        missing=np.asarray(tree["default_left"],dtype=np.int8);node=np.zeros(len(x),dtype=np.int32);active=np.ones(len(x),dtype=bool)
        while active.any():
            rr=rows[active];nn=node[active];leaf=left[nn]<0
            if leaf.any():pred[rr[leaf]]+=threshold[nn[leaf]].astype(np.float64)
            if (~leaf).any():
                r2=rr[~leaf];n2=nn[~leaf];v=x[r2,feature[n2]];go=np.where(np.isnan(v),missing[n2]!=0,v<threshold[n2]);node[r2]=np.where(go,left[n2],right[n2])
            active[rr[leaf]]=False
    return pred

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input-dir",required=True);ap.add_argument("--output",required=True);a=ap.parse_args()
    src,out=Path(a.input_dir),Path(a.output)
    if out.exists() and any(out.iterdir()):raise SystemExit(f"output must be empty: {out}")
    out.mkdir(parents=True,exist_ok=True);c=load_training_contract();frames=[];summary=[];per_matrix=[];per_block=[]
    manifest={"schema":"frozen_v4_2_abcd_unseen_v1","no_refit":True,"target":"log1p(kernel_time_ms)","prediction":"expm1(raw_prediction)","training_contract":str(TRAIN),"frozen_manifest":str(FROZEN/"freeze_manifest_strict_abcd.json"),"targets":{}}
    for target in c.TARGETS:
        req=set(["task_id","matrix_id","block_size","label_time_ms","hardware_id"]+c.FEATURES[target])
        for mean,cv in c.DERIVED[target].values():req|={mean,cv}
        d=pd.read_csv(src/f"{target}.model_ready_v4.csv",usecols=lambda x:x in req,low_memory=False)
        for new,(mean,cv) in c.DERIVED[target].items():
            d[new]=pd.to_numeric(d[mean],errors="coerce").fillna(0)*pd.to_numeric(d[cv],errors="coerce").fillna(0)
        y=pd.to_numeric(d.label_time_ms,errors="coerce").to_numpy(float);ok=np.isfinite(y)&(y>0);d=d.loc[ok].reset_index(drop=True);y=y[ok]
        model_path=FROZEN/f"{target}.direct_log_xgboost_ABCD.json"
        pred=np.maximum(np.expm1(predict_frozen_json(model_path,c.values(d,c.FEATURES[target]))),0)
        f=pd.DataFrame({"target":target,"task_id":d.task_id.astype(str),"matrix_id":d.matrix_id.astype(str),"block_size":pd.to_numeric(d.block_size),"actual_ms":y,"predicted_ms":pred})
        frames.append(f);summary.append({"target":target,**c.metric(y,pred)})
        for matrix_id,g in f.groupby("matrix_id",sort=True):per_matrix.append({"target":target,"matrix_id":matrix_id,**c.metric(g.actual_ms,g.predicted_ms)})
        for (matrix_id,bs),g in f.groupby(["matrix_id","block_size"],sort=True):per_block.append({"target":target,"matrix_id":matrix_id,"block_size":int(bs),**c.metric(g.actual_ms,g.predicted_ms)})
        manifest["targets"][target]={"rows":len(f),"matrices":sorted(f.matrix_id.unique()),"features":c.FEATURES[target],"model":str(model_path),"model_sha256":sha256(model_path)}
    tasks=pd.concat(frames,ignore_index=True);tasks.to_csv(out/"task_predictions.csv.gz",index=False,compression="gzip")
    pd.DataFrame(summary).to_csv(out/"task_metrics_by_kernel.csv",index=False);pd.DataFrame(per_matrix).to_csv(out/"task_metrics_by_kernel_matrix.csv",index=False);pd.DataFrame(per_block).to_csv(out/"task_metrics_by_kernel_matrix_block.csv",index=False)
    schemes=tasks.groupby(["matrix_id","block_size"],as_index=False).agg(actual_ms=("actual_ms","sum"),predicted_ms=("predicted_ms","sum"),tasks=("task_id","size"))
    schemes.to_csv(out/"scheme_predictions.csv",index=False)
    scheme_summary=[{"scope":"all",**c.metric(schemes.actual_ms,schemes.predicted_ms)}];selection=[]
    for matrix_id,g in schemes.groupby("matrix_id",sort=True):
        scheme_summary.append({"scope":matrix_id,**c.metric(g.actual_ms,g.predicted_ms)})
        oracle=g.loc[g.actual_ms.idxmin()];chosen=g.loc[g.predicted_ms.idxmin()]
        selection.append({"matrix_id":matrix_id,"oracle_block_size":int(oracle.block_size),"oracle_actual_ms":oracle.actual_ms,"predicted_best_block_size":int(chosen.block_size),"selected_actual_ms":chosen.actual_ms,"selected_predicted_ms":chosen.predicted_ms,"regret_percent":(chosen.actual_ms/oracle.actual_ms-1)*100,"spearman":g.actual_ms.rank().corr(g.predicted_ms.rank())})
    pd.DataFrame(scheme_summary).to_csv(out/"scheme_metrics.csv",index=False);pd.DataFrame(selection).to_csv(out/"scheme_selection.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__=="__main__":main()
