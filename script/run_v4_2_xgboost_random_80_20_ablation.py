#!/usr/bin/env python3
"""Same-seed task-random 80:20 cumulative A..E XGBoost ablation."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from feature_sets_v4_2 import FEATURES_BY_KERNEL,BASELINE_FIELDS
from run_v4_1_flops_bytes_ablation import values,metric
from run_v4_2_paper_order_max_first_ablation import groups

TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense");STAGES=("A","A+B","A+B+C","A+B+C+D","A+B+C+D+E");SEED=20260807
PARAMS=dict(objective='reg:squarederror',n_estimators=500,learning_rate=.05,max_depth=8,
 min_child_weight=20,subsample=.8,colsample_bytree=.8,reg_alpha=0.,reg_lambda=1.,gamma=0.,
 tree_method='hist',max_bin=256,n_jobs=8)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();src,out=Path(a.input_dir),Path(a.output)
 if out.exists() and any(out.iterdir()):raise SystemExit(f'output must be empty: {out}')
 out.mkdir(parents=True);rows=[];preds=[];timing=[];manifest={'schema':'v4.2_xgboost_task_random_80_20_cumulative','seed':SEED,'split':'same task-random 80:20 indices for every stage within a kernel','xgboost_version':__import__('xgboost').__version__,'xgboost_params':PARAMS,'group_order':['A dimensions+input NNZ','B operation_count_v2','C maximum NNZ','D mean+derived std NNZ','E empty fraction'],'targets':{}};wall=time.perf_counter()
 for ti,target in enumerate(TARGETS):
  g,derived=groups(target);raw=FEATURES_BY_KERNEL[target]+[BASELINE_FIELDS[0]];req=set(['task_id','matrix_id','block_size','label_time_ms']+raw);d=pd.read_csv(src/f'{target}.model_ready_v4.csv',usecols=lambda c:c in req,low_memory=False)
  for new,(mean,cv) in derived.items():d[new]=pd.to_numeric(d[mean],errors='coerce').fillna(0)*pd.to_numeric(d[cv],errors='coerce').fillna(0)
  y=pd.to_numeric(d.label_time_ms,errors='coerce').to_numpy(float);ok=np.isfinite(y)&(y>0);d=d.loc[ok].reset_index(drop=True);y=y[ok];idx=np.arange(len(d));tr,te=train_test_split(idx,test_size=.2,random_state=SEED,shuffle=True);manifest['targets'][target]={'n_total':len(d),'n_train':len(tr),'n_test':len(te),'groups':g,'derived_std':derived,'train_test_matrix_overlap':len(set(d.iloc[tr].matrix_id.astype(str))&set(d.iloc[te].matrix_id.astype(str)))};cumulative=[]
  for si,stage in enumerate(STAGES):
   cumulative+=g[chr(ord('A')+si)];X=values(d,cumulative);m=XGBRegressor(**PARAMS,random_state=SEED+ti*10);t=time.perf_counter();m.fit(X[tr],np.log1p(y[tr]),verbose=False);fit=time.perf_counter()-t;t=time.perf_counter();p=np.maximum(np.expm1(m.predict(X[te])),0);predict=time.perf_counter()-t;r={'target':target,'stage':stage,'features':len(cumulative),'n_train':len(tr),'n_test':len(te),**metric(y[te],p)};rows.append(r);timing.append({'target':target,'stage':stage,'fit_seconds':fit,'predict_seconds':predict,'predict_us_per_sample':predict/len(te)*1e6});preds.append(pd.DataFrame({'target':target,'stage':stage,'task_id':d.iloc[te].task_id,'matrix_id':d.iloc[te].matrix_id,'block_size':d.iloc[te].block_size,'actual_ms':y[te],'predicted_ms':p}));print(target,stage,len(cumulative),f"MAPE={r['MAPE']:.3f}",flush=True)
 res=pd.DataFrame(rows);res['MAPE_improvement_from_previous']=res.groupby('target').MAPE.diff().mul(-1);res['MAPE_improvement_vs_A']=res.groupby('target').MAPE.transform('first')-res.MAPE;res.to_csv(out/'cumulative_group_metrics.csv',index=False);pd.DataFrame(timing).to_csv(out/'timing.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(out/'test_predictions.csv.gz',index=False,compression='gzip')
 plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':10});fig,ax=plt.subplots(figsize=(7.4,4.2))
 for target,q in res.groupby('target',sort=False):ax.plot(q.stage,q.MAPE,marker='o',lw=1.8,label=target.replace('_','-'))
 ax.set_xlabel('Cumulative feature groups');ax.set_ylabel('Random 20% test MAPE (%)');ax.grid(True,lw=.4,alpha=.4);ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(out/'xgboost_cumulative_mape.pdf');fig.savefig(out/'xgboost_cumulative_mape.png',dpi=300);plt.close(fig);manifest['wall_seconds']=time.perf_counter()-wall;(out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=list)+'\n')
if __name__=='__main__':main()
