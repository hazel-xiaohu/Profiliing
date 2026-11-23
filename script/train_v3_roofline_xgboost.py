#!/usr/bin/env python3
"""Matrix-grouped five-fold Roofline + XGBoost residual training."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor
from feature_sets_v3 import BASELINE_FIELDS, FEATURES_BY_KERNEL

TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense","SSSSM_sparse")
PARAMS=dict(objective='reg:squarederror',n_estimators=500,learning_rate=.05,
 max_depth=8,min_child_weight=20,subsample=.8,colsample_bytree=.8,
 reg_alpha=0.,reg_lambda=1.,gamma=0.,tree_method='hist',max_bin=256,n_jobs=8)

def args():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--fold-map',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20260803);return p.parse_args()
def metrics(y,p):
 p=np.maximum(np.asarray(p,float),0);y=np.asarray(y,float);ape=np.abs(p-y)/np.maximum(np.abs(y),1e-6)*100
 return {'n_test':len(y),'MAPE':float(ape.mean()),'MdAPE':float(np.median(ape)),'P90_APE':float(np.percentile(ape,90)),'P95_APE':float(np.percentile(ape,95)),'WAPE':float(np.abs(p-y).sum()/np.maximum(np.abs(y).sum(),1e-12)*100),'RMSE_ms':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}
def roof(ops,byt,y,tr):
 po=ops[tr][ops[tr]>0];pb=byt[tr][byt[tr]>0];scale=np.array([max(float(np.median(po)),1),max(float(np.median(pb)),1)])
 design=np.column_stack([np.ones(len(y)),ops/scale[0],byt/scale[1]]);coef,_=nnls(design[tr],y[tr]);return scale,coef,np.maximum(design@coef,1e-9)
def main():
 a=args();src=Path(a.input_dir);out=Path(a.output)
 if out.exists() and any(out.iterdir()):raise SystemExit(f'output must be empty: {out}')
 (out/'models').mkdir(parents=True);(out/'predictions').mkdir();folds=pd.read_csv(a.fold_map);fm=dict(zip(folds.matrix_id.astype(str),folds.cv_fold.astype(int)))
 comparisons=[];foldrows=[];timing=[];manifest={'schema':'roofline_xgboost_v3','seed':a.seed,'cv':'shared matrix-grouped 5-fold','xgboost_params':PARAMS,'targets':{}};total0=time.perf_counter()
 for ti,target in enumerate(TARGETS):
  features=FEATURES_BY_KERNEL[target];required=set(['task_id','matrix_id','block_size','label_time_ms','hardware_id']+features+BASELINE_FIELDS)
  d=pd.read_csv(src/f'{target}.model_ready_v3.csv',usecols=lambda c:c in required,low_memory=False);d['label_time_ms']=pd.to_numeric(d.label_time_ms,errors='coerce');d=d[np.isfinite(d.label_time_ms)&(d.label_time_ms>0)].reset_index(drop=True)
  if set(d.hardware_id.astype(str))!={'H100_SXM_80GB'}:raise RuntimeError(f'{target}: hardware mismatch')
  d['cv_fold']=d.matrix_id.astype(str).map(fm)
  if d.cv_fold.isna().any() or d.cv_fold.nunique()!=5:raise RuntimeError(f'{target}: invalid folds')
  X=d[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float);y=d.label_time_ms.to_numpy(float);ops=pd.to_numeric(d.replay_operation_count_v2,errors='coerce').fillna(0).to_numpy(float);byt=pd.to_numeric(d.replay_base_bytes_est,errors='coerce').fillna(0).to_numpy(float);fid=d.cv_fold.to_numpy(int)
  poof=np.empty(len(y));boof=np.empty(len(y))
  for fold in range(1,6):
   tr=fid!=fold;te=fid==fold;scale,coef,base=roof(ops,byt,y,tr);boof[te]=base[te];z=np.log1p(y[tr])-np.log1p(base[tr]);m=XGBRegressor(**PARAMS,random_state=a.seed+ti*10+fold)
   t=time.perf_counter();m.fit(X[tr],z,verbose=False);fit=time.perf_counter()-t;t=time.perf_counter();poof[te]=np.maximum(np.expm1(np.log1p(base[te])+m.predict(X[te])),0);pred=time.perf_counter()-t
   foldrows.append({'target':target,'fold':fold,'model':'Roofline_XGBoost_residual',**metrics(y[te],poof[te])});timing.append({'target':target,'fold':fold,'fit_seconds':fit,'predict_seconds':pred,'n_train':int(tr.sum()),'n_test':int(te.sum())})
  comparisons.append({'target':target,'model':'Analytical_Roofline',**metrics(y,boof)});comparisons.append({'target':target,'model':'Roofline_XGBoost_residual',**metrics(y,poof)})
  scale,coef,base=roof(ops,byt,y,np.ones(len(y),bool));z=np.log1p(y)-np.log1p(base);final=XGBRegressor(**PARAMS,random_state=a.seed+ti);t=time.perf_counter();final.fit(X,z,verbose=False);final_s=time.perf_counter()-t
  joblib.dump({'schema':'roofline_xgboost_v3','target':target,'hardware_id':'H100_SXM_80GB','features':features,'baseline_scale':scale,'baseline_coef':coef,'residual_definition':'log1p(time)-log1p(roofline)','xgboost_params':PARAMS,'model':final},out/'models'/f'{target}.roofline_xgboost_v3.joblib')
  pd.DataFrame({'task_id':d.task_id,'matrix_id':d.matrix_id,'block_size':d.block_size,'cv_fold':fid,'actual_ms':y,'roofline_ms':boof,'roofline_xgboost_ms':poof}).to_csv(out/'predictions'/f'{target}.oof.csv',index=False)
  manifest['targets'][target]={'rows':len(d),'matrices':int(d.matrix_id.nunique()),'features':features,'final_fit_seconds':final_s,'baseline_scale':scale.tolist(),'baseline_coef':coef.tolist()};print(target,comparisons[-1],flush=True)
 pd.DataFrame(comparisons).to_csv(out/'model_comparison.csv',index=False);pd.DataFrame(foldrows).to_csv(out/'fold_metrics.csv',index=False);pd.DataFrame(timing).to_csv(out/'timing.csv',index=False);manifest['wall_seconds']=time.perf_counter()-total0;(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
