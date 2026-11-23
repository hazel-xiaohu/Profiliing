#!/usr/bin/env python3
"""Paired grouped-5-fold audit of full-v3 versus Partition-light features."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from run_v3_shared_split_experiments import (FEATURES_BY_KERNEL,feature_matrix,
    fit_roofline,hgb,metrics)
TARGET_MODELS={"GETRF":"Roofline_residual_HGB","GESSM":"HGB",
               "TSTRF":"HGB","SSSSM_dense":"Roofline_residual_HGB"}
def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',required=True);p.add_argument('--fold-map',required=True);p.add_argument('--light-model-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20260729);p.add_argument('--bootstrap',type=int,default=10000);return p.parse_args()
def predict(kind,x,y,ops,byt,folds,seed):
 oof=np.empty(len(y));timing=[]
 for fold in range(1,6):
  tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);started=time.perf_counter()
  if kind=='HGB':
   model=hgb(seed+fold);model.fit(x[tr],np.log1p(y[tr]));pred=np.maximum(np.expm1(model.predict(x[te])),0)
  else:
   scale,coef,base=fit_roofline(ops,byt,y,tr);res=np.log1p(y[tr])-np.log1p(base[tr]);model=hgb(seed+fold);model.fit(x[tr],res);pred=np.maximum(np.expm1(np.log1p(base[te])+model.predict(x[te])),0)
  oof[te]=pred;timing.append({'fold':fold,'seconds':time.perf_counter()-started,'n_train':len(tr),'n_test':len(te)})
 return oof,timing
def main():
 a=args();src=Path(a.input_dir);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);fm=pd.read_csv(a.fold_map);fold_map=dict(zip(fm.matrix_id.astype(str),fm.cv_fold.astype(int)));summary=[];fold_rows=[];matrix_rows=[];timing=[];rng=np.random.default_rng(a.seed)
 for ti,(target,kind) in enumerate(TARGET_MODELS.items()):
  full=FEATURES_BY_KERNEL[target];light=joblib.load(Path(a.light_model_dir)/f'{target}.roofline_residual_light.joblib')['features'];required=set(['matrix_id','label_time_ms','replay_operation_count_v2','replay_base_bytes_est']+full);d=pd.read_csv(src/f'{target}.model_ready_v3.csv',usecols=lambda c:c in required,low_memory=False);d['matrix_id']=d.matrix_id.astype(str);fold=d.matrix_id.map(fold_map).to_numpy(int);y=pd.to_numeric(d.label_time_ms,errors='coerce').to_numpy(float);ops=pd.to_numeric(d.replay_operation_count_v2,errors='coerce').fillna(0).to_numpy(float);byt=pd.to_numeric(d.replay_base_bytes_est,errors='coerce').fillna(0).to_numpy(float)
  preds={}
  for label,features in [('full_v3',full),('partition_light',light)]:
   x=feature_matrix(d,features);pred,times=predict(kind,x,y,ops,byt,fold,a.seed+ti*100);preds[label]=pred;row={'target':target,'model':kind,'feature_set':label,'n_features':len(features),**metrics(y,pred)};summary.append(row)
   for z in times:timing.append({'target':target,'model':kind,'feature_set':label,**z})
   for f in range(1,6):
    q=fold==f;fold_rows.append({'target':target,'model':kind,'feature_set':label,'fold':f,'n_features':len(features),**metrics(y[q],pred[q])})
  per=[]
  for matrix,idx in d.groupby('matrix_id').indices.items():
   idx=np.asarray(idx);mf=metrics(y[idx],preds['full_v3'][idx]);ml=metrics(y[idx],preds['partition_light'][idx]);delta=ml['MAPE']-mf['MAPE'];per.append(delta);matrix_rows.append({'target':target,'matrix_id':matrix,'fold':int(fold[idx[0]]),'full_MAPE':mf['MAPE'],'light_MAPE':ml['MAPE'],'delta_light_minus_full_MAPE':delta,'light_wins':int(delta<=0)})
  per=np.asarray(per);boot=np.mean(rng.choice(per,(a.bootstrap,len(per)),replace=True),axis=1);full_m=summary[-2];light_m=summary[-1];summary.append({'target':target,'model':kind,'feature_set':'paired_evidence','n_features':len(light),'MAPE':light_m['MAPE']-full_m['MAPE'],'MdAPE':light_m['MdAPE']-full_m['MdAPE'],'P95_APE':light_m['P95_APE']-full_m['P95_APE'],'RMSE_ms':light_m['RMSE_ms']-full_m['RMSE_ms'],'R2':light_m['R2']-full_m['R2'],'matrix_mean_delta_MAPE':float(per.mean()),'matrix_median_delta_MAPE':float(np.median(per)),'matrix_bootstrap_ci95_low':float(np.percentile(boot,2.5)),'matrix_bootstrap_ci95_high':float(np.percentile(boot,97.5)),'light_matrix_wins':int(np.sum(per<=0)),'matrices':len(per)})
 pd.DataFrame(summary).to_csv(out/'model_and_paired_metrics.csv',index=False);pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv',index=False);pd.DataFrame(matrix_rows).to_csv(out/'matrix_paired_metrics.csv',index=False);pd.DataFrame(timing).to_csv(out/'timing.csv',index=False)
 evidence=pd.DataFrame(summary);lines=['# Full-v3 vs Partition-light grouped-5-fold audit','', 'Positive delta means the light feature set is worse.','', '| Target | Model | Full/light features | Full MAPE | Light MAPE | Delta | Matrix bootstrap 95% CI | Light wins |','|---|---|---:|---:|---:|---:|---:|---:|']
 for target,kind in TARGET_MODELS.items():
  q=evidence[evidence.target.eq(target)];f=q[q.feature_set.eq('full_v3')].iloc[0];l=q[q.feature_set.eq('partition_light')].iloc[0];e=q[q.feature_set.eq('paired_evidence')].iloc[0];lines.append(f"| {target} | {kind} | {int(f.n_features)}/{int(l.n_features)} | {f.MAPE:.3f}% | {l.MAPE:.3f}% | {e.MAPE:+.3f} pp | [{e.matrix_bootstrap_ci95_low:+.3f}, {e.matrix_bootstrap_ci95_high:+.3f}] pp | {int(e.light_matrix_wins)}/{int(e.matrices)} |")
 (out/'report.md').write_text('\n'.join(lines)+'\n');(out/'manifest.json').write_text(json.dumps({'cv':'shared matrix-grouped 5-fold','seed':a.seed,'bootstrap':a.bootstrap,'no_features_deleted':True},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
