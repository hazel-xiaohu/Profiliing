#!/usr/bin/env python3
"""Matrix-grouped five-fold validation of v4.1 vs pruned v4.2 residual features."""
import argparse,time,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import nnls
from run_v4_1_flops_bytes_ablation import metric,values
from feature_sets_v4 import FEATURES_BY_KERNEL as V41,BASELINE_FIELDS
from feature_sets_v4_2 import FEATURES_BY_KERNEL as V42
TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense");SEED=20260806
def roof(ops,byt,y,tr):
 s=np.array([max(np.median(ops[tr][ops[tr]>0]),1),max(np.median(byt[tr][byt[tr]>0]),1)]);D=np.c_[np.ones(len(y)),ops/s[0],byt/s[1]];c,_=nnls(D[tr],y[tr]);return np.maximum(D@c,1e-9)
def model(seed):return HistGradientBoostingRegressor(max_iter=250,learning_rate=.06,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.,random_state=seed)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--fold-map',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.input_dir);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 fm0=pd.read_csv(a.fold_map);fm=dict(zip(fm0.matrix_id.astype(str),fm0.cv_fold.astype(int)));rows=[];foldrows=[];manifest={'schema':'v4.2_pruning_validation','model':'Roofline+HGB residual','targets':{}};wall=time.perf_counter()
 for ti,t in enumerate(TARGETS):
  feats=sorted(set(V41[t]+V42[t]));req=set(['matrix_id','label_time_ms']+BASELINE_FIELDS+feats);d=pd.read_csv(src/f'{t}.model_ready_v4.csv',usecols=lambda c:c in req,low_memory=False);y=pd.to_numeric(d.label_time_ms,errors='coerce').to_numpy(float);ok=np.isfinite(y)&(y>0);d=d.loc[ok].reset_index(drop=True);y=y[ok];fid=d.matrix_id.astype(str).map(fm).to_numpy(int);ops=pd.to_numeric(d[BASELINE_FIELDS[0]],errors='coerce').fillna(0).to_numpy(float);byt=pd.to_numeric(d[BASELINE_FIELDS[1]],errors='coerce').fillna(0).to_numpy(float)
  manifest['targets'][t]={'v4.1_features':V41[t],'v4.2_features':V42[t],'removed':[f for f in V41[t] if f not in V42[t]]}
  for vi,(version,fs) in enumerate((('v4.1',V41[t]),('v4.2',V42[t]))):
   X=values(d,fs);oof=np.empty(len(y))
   for fold in range(1,6):
    tr=fid!=fold;te=fid==fold;base=roof(ops,byt,y,tr);z=np.log1p(y[tr])-np.log1p(base[tr]);m=model(SEED+ti*100+vi*10+fold).fit(X[tr],z);oof[te]=np.expm1(np.log1p(base[te])+m.predict(X[te]));foldrows.append({'target':t,'version':version,'fold':fold,**metric(y[te],oof[te])})
   r={'target':t,'version':version,'n_features':len(fs),**metric(y,oof)};rows.append(r);print(t,version,len(fs),f"MAPE={r['MAPE']:.3f}",flush=True)
 res=pd.DataFrame(rows);base=res[res.version=='v4.1'].set_index('target').MAPE;res['MAPE_delta_vs_v4_1']=res.apply(lambda r:r.MAPE-base[r.target],axis=1);res.to_csv(out/'model_comparison.csv',index=False);pd.DataFrame(foldrows).to_csv(out/'fold_metrics.csv',index=False);manifest['wall_seconds']=time.perf_counter()-wall;(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
