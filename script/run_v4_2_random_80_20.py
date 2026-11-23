#!/usr/bin/env python3
"""Task-random 80:20 evaluation of the latest paper-order A+B+C+D+E features."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from feature_sets_v4_2 import FEATURES_BY_KERNEL,BASELINE_FIELDS
from run_v4_1_flops_bytes_ablation import values,metric
from run_v4_2_paper_order_max_first_ablation import groups

TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense");SEED=20260807
def model():return HistGradientBoostingRegressor(max_iter=250,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.,random_state=SEED)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();src,out=Path(a.input_dir),Path(a.output)
 if out.exists() and any(out.iterdir()):raise SystemExit(f'output must be empty: {out}')
 out.mkdir(parents=True);rows=[];preds=[];manifest={'schema':'v4.2_task_random_80_20','seed':SEED,'split':'independent random 80:20 within each kernel','model':'direct HGB on log1p(time)','targets':{}};wall=time.perf_counter()
 for target in TARGETS:
  g,derived=groups(target);features=g['A']+g['B']+g['C']+g['D']+g['E'];raw=FEATURES_BY_KERNEL[target]+[BASELINE_FIELDS[0]];req=set(['task_id','matrix_id','block_size','label_time_ms']+raw);d=pd.read_csv(src/f'{target}.model_ready_v4.csv',usecols=lambda c:c in req,low_memory=False)
  for new,(mean,cv) in derived.items():d[new]=pd.to_numeric(d[mean],errors='coerce').fillna(0)*pd.to_numeric(d[cv],errors='coerce').fillna(0)
  y=pd.to_numeric(d.label_time_ms,errors='coerce').to_numpy(float);ok=np.isfinite(y)&(y>0);d=d.loc[ok].reset_index(drop=True);y=y[ok];idx=np.arange(len(d));tr,te=train_test_split(idx,test_size=.2,random_state=SEED,shuffle=True);X=values(d,features);t=time.perf_counter();m=model().fit(X[tr],np.log1p(y[tr]));fit=time.perf_counter()-t;t=time.perf_counter();p=np.maximum(np.expm1(m.predict(X[te])),0);pred=time.perf_counter()-t
  train_m=set(d.iloc[tr].matrix_id.astype(str));test_m=set(d.iloc[te].matrix_id.astype(str));r={'target':target,'n_total':len(d),'n_train':len(tr),'n_test':len(te),'train_matrices':len(train_m),'test_matrices':len(test_m),'overlap_matrices':len(train_m&test_m),'features':len(features),'fit_seconds':fit,'predict_seconds':pred,**metric(y[te],p)};rows.append(r);preds.append(pd.DataFrame({'target':target,'task_id':d.iloc[te].task_id,'matrix_id':d.iloc[te].matrix_id,'block_size':d.iloc[te].block_size,'actual_ms':y[te],'predicted_ms':p}));manifest['targets'][target]={'features':features,'derived_std':derived};print(target,r,flush=True)
 pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(out/'test_predictions.csv.gz',index=False,compression='gzip');manifest['wall_seconds']=time.perf_counter()-wall;(out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=list)+'\n')
if __name__=='__main__':main()
