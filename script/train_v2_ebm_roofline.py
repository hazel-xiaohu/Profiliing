#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score
from interpret.glassbox import ExplainableBoostingRegressor
from train_v2_ablation import SETS

def metrics(y,p):
 ape=np.abs(p-y)/np.maximum(np.abs(y),1e-6)*100
 return {'MAPE':float(ape.mean()),'MdAPE':float(np.median(ape)),'P90_APE':float(np.percentile(ape,90)),'P95_APE':float(np.percentile(ape,95)),'RMSE_ms':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)),'n_test':len(y)}
def main():
 a=argparse.ArgumentParser(); a.add_argument('--input',required=True); a.add_argument('--hgb-dir',required=True); a.add_argument('--output',required=True); a.add_argument('--seed',type=int,default=20260728); z=a.parse_args()
 out=Path(z.output); out.mkdir(parents=True,exist_ok=True); features=SETS['F_full']; cols=set(['matrix_id','kernel_type','time_kernel_ms','forced_ssssm_variant']+features); cols.discard('forced_variant_code')
 df=pd.read_csv(z.input,usecols=lambda x:x in cols,low_memory=False); df['forced_variant_code']=(df.forced_ssssm_variant=='dense').astype(float); df.time_kernel_ms=pd.to_numeric(df.time_kernel_ms,errors='coerce'); df=df[np.isfinite(df.time_kernel_ms)&(df.time_kernel_ms>0)]
 splits=json.loads((Path(z.hgb_dir)/'splits.json').read_text()); results=[]
 for kernel in ('GETRF','GESSM','TSTRF','SSSSM'):
  d=df[df.kernel_type==kernel].copy(); tr=d.matrix_id.isin(splits[kernel]['train_matrices']).to_numpy(); te=d.matrix_id.isin(splits[kernel]['test_matrices']).to_numpy(); X=d[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float); y=d.time_kernel_ms.to_numpy(float)
  ebm=ExplainableBoostingRegressor(feature_names=features,max_bins=128,max_interaction_bins=32,interactions=10,outer_bags=4,inner_bags=0,learning_rate=.04,max_rounds=2500,early_stopping_rounds=100,min_samples_leaf=20,n_jobs=4,random_state=z.seed)
  ebm.fit(X[tr],np.log1p(y[tr])); ep=np.maximum(np.expm1(ebm.predict(X[te])),0); results.append({'kernel':kernel,'model':'EBM_full',**metrics(y[te],ep)}); joblib.dump({'model':ebm,'features':features,'target_transform':'log1p','kernel':kernel,'schema':'v2'},out/f'{kernel}.ebm_v2.joblib')
  ops=d.replay_operation_count_v2.to_numpy(float); byt=d.replay_base_bytes_est.to_numpy(float); scale=np.array([max(np.median(ops[tr][ops[tr]>0]),1),max(np.median(byt[tr][byt[tr]>0]),1)])
  Z=np.column_stack([np.ones(len(d)),ops/scale[0],byt/scale[1]]); coef,_=nnls(Z[tr],y[tr]); base=np.maximum(Z@coef,1e-9); residual=np.log1p(y)-np.log1p(base)
  rr=HistGradientBoostingRegressor(learning_rate=.07,max_iter=250,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1,random_state=z.seed); rr.fit(X[tr],residual[tr]); rp=np.maximum(np.expm1(np.log1p(base[te])+rr.predict(X[te])),0); results.append({'kernel':kernel,'model':'Roofline_residual_HGB','baseline_intercept_ms':coef[0],'baseline_ms_per_scaled_ops':coef[1],'baseline_ms_per_scaled_bytes':coef[2],**metrics(y[te],rp)}); joblib.dump({'residual_model':rr,'features':features,'baseline_coef':coef,'baseline_scale':scale,'kernel':kernel,'schema':'v2'},out/f'{kernel}.roofline_residual_v2.joblib')
  pd.DataFrame({'actual_ms':y[te],'ebm_ms':ep,'roofline_residual_ms':rp}).to_csv(out/f'{kernel}.comparison_predictions.csv',index=False); print(results[-2],flush=True); print(results[-1],flush=True)
 pd.DataFrame(results).to_csv(out/'model_comparison.csv',index=False)
if __name__=='__main__': main()
