#!/usr/bin/env python3
"""Cumulative A..E feature-group ablation for Roofline+HGB residual v4.2."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import nnls
from feature_sets_v4_2 import FEATURES_BY_KERNEL,BASELINE_FIELDS
from run_v4_1_flops_bytes_ablation import values,metric

TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense");SEED=20260806
STAGES=("A","A+B","A+B+C","A+B+C+D","A+B+C+D+E")

def feature_groups(target, operation_only=False):
 fs=FEATURES_BY_KERNEL[target]
 return {
  'A':[BASELINE_FIELDS[0]] if operation_only else BASELINE_FIELDS,
  'B':[f for f in fs if f=='replay_main_dimension' or f.startswith('replay_nnz_')],
  'C':[f for f in fs if f.endswith('_nnz_mean') or f.endswith('_nnz_max')],
  'D':[f for f in fs if f.endswith('_nnz_cv')],
  'E':[f for f in fs if f.endswith('_empty_fraction')],
 }

def roof(ops,byt,y,tr,operation_only=False):
 po=ops[tr][ops[tr]>0];so=max(float(np.median(po)),1)
 if operation_only:D=np.c_[np.ones(len(y)),ops/so]
 else:
  pb=byt[tr][byt[tr]>0];sb=max(float(np.median(pb)),1);D=np.c_[np.ones(len(y)),ops/so,byt/sb]
 c,_=nnls(D[tr],y[tr]);return np.maximum(D@c,1e-9)
def model(seed):return HistGradientBoostingRegressor(max_iter=250,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.,random_state=seed)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--fold-map',required=True);ap.add_argument('--output',required=True);ap.add_argument('--a-operation-only',action='store_true');a=ap.parse_args();src,out=Path(a.input_dir),Path(a.output)
 if out.exists() and any(out.iterdir()):raise SystemExit(f'output must be empty: {out}')
 out.mkdir(parents=True);fm0=pd.read_csv(a.fold_map);fm=dict(zip(fm0.matrix_id.astype(str),fm0.cv_fold.astype(int)));rows=[];foldrows=[];preds=[];manifest={'schema':'v4.2_cumulative_groups','model':'nonnegative analytical baseline + HGB log-residual','group_order':['A','B','C','D','E'],'a_operation_only':a.a_operation_only,'targets':{}};wall=time.perf_counter()
 for ti,target in enumerate(TARGETS):
  groups=feature_groups(target,a.a_operation_only);fs=FEATURES_BY_KERNEL[target];req=set(['task_id','matrix_id','block_size','label_time_ms']+BASELINE_FIELDS+fs);d=pd.read_csv(src/f'{target}.model_ready_v4.csv',usecols=lambda c:c in req,low_memory=False);y=pd.to_numeric(d.label_time_ms,errors='coerce').to_numpy(float);ok=np.isfinite(y)&(y>0);d=d.loc[ok].reset_index(drop=True);y=y[ok];fid=d.matrix_id.astype(str).map(fm).to_numpy(int);ops=pd.to_numeric(d[BASELINE_FIELDS[0]],errors='coerce').fillna(0).to_numpy(float);byt=pd.to_numeric(d[BASELINE_FIELDS[1]],errors='coerce').fillna(0).to_numpy(float)
  manifest['targets'][target]={'groups':groups}
  cumulative=[]
  for si,stage in enumerate(STAGES):
   if si>0:cumulative+=groups[chr(ord('A')+si)]
   X=values(d,cumulative) if cumulative else None;oof=np.empty(len(y));boof=np.empty(len(y))
   for fold in range(1,6):
    tr=fid!=fold;te=fid==fold;base=roof(ops,byt,y,tr,a.a_operation_only);boof[te]=base[te]
    if cumulative:
     z=np.log1p(y[tr])-np.log1p(base[tr]);m=model(SEED+ti*100+si*10+fold).fit(X[tr],z);oof[te]=np.expm1(np.log1p(base[te])+m.predict(X[te]))
    else:oof[te]=base[te]
    foldrows.append({'target':target,'stage':stage,'fold':fold,'residual_features':len(cumulative),**metric(y[te],oof[te])})
   r={'target':target,'stage':stage,'residual_features':len(cumulative),'total_original_features':len(groups['A'])+len(cumulative),**metric(y,oof)};rows.append(r);preds.append(pd.DataFrame({'target':target,'stage':stage,'task_id':d.task_id,'matrix_id':d.matrix_id,'block_size':d.block_size,'cv_fold':fid,'actual_ms':y,'predicted_ms':np.maximum(oof,0)}));print(target,stage,len(cumulative),f"MAPE={r['MAPE']:.3f}",flush=True)
 res=pd.DataFrame(rows);res['MAPE_improvement_from_previous']=res.groupby('target').MAPE.diff().mul(-1);res['MAPE_improvement_vs_A']=res.groupby('target').MAPE.transform('first')-res.MAPE;res.to_csv(out/'cumulative_group_metrics.csv',index=False);pd.DataFrame(foldrows).to_csv(out/'fold_metrics.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(out/'oof_predictions.csv.gz',index=False,compression='gzip')
 plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':10});fig,ax=plt.subplots(figsize=(7.4,4.2))
 for target,q in res.groupby('target',sort=False):ax.plot(q.stage,q.MAPE,marker='o',lw=1.8,label=target.replace('_','-'))
 ax.set_xlabel('Cumulative feature groups');ax.set_ylabel('OOF MAPE (%)');ax.grid(True,lw=.4,alpha=.4);ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(out/'cumulative_group_mape.pdf');fig.savefig(out/'cumulative_group_mape.png',dpi=300);plt.close(fig)
 manifest['wall_seconds']=time.perf_counter()-wall;(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
