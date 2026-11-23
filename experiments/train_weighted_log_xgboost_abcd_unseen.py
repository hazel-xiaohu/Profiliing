#!/usr/bin/env python3
"""Select a scheme-normalized weighted log-time XGBoost by matrix-grouped OOF.

Weights are computed only from profiling labels.  Unseen matrices are never
used for weight/model selection or fitting.
"""
from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd

CONTRACT=Path('/output/Partition/script/train_export_direct_log_xgboost_abcd.py')

def contract():
 spec=importlib.util.spec_from_file_location('v42_contract',CONTRACT)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def load(src,target,c):
 req={'task_id','matrix_id','block_size','label_time_ms','hardware_id',*c.FEATURES[target]}
 for mean,cv in c.DERIVED[target].values():req|={mean,cv}
 d=pd.read_csv(src/f'{target}.model_ready_v4.csv',usecols=lambda x:x in req,low_memory=False)
 for new,(mean,cv) in c.DERIVED[target].items():
  d[new]=pd.to_numeric(d[mean],errors='coerce').fillna(0)*pd.to_numeric(d[cv],errors='coerce').fillna(0)
 y=pd.to_numeric(d.label_time_ms,errors='coerce').to_numpy(float);ok=np.isfinite(y)&(y>0)
 d=d.loc[ok].reset_index(drop=True);y=y[ok]
 if set(d.hardware_id.astype(str))!={'H100_SXM_80GB'}:raise RuntimeError(f'{target}: hardware mismatch')
 return d,y,c.values(d,c.FEATURES[target])

def weight_tables(frames,alphas):
 all_labels=pd.concat([pd.DataFrame({'matrix_id':d.matrix_id.astype(str),'block_size':pd.to_numeric(d.block_size).astype(int),'y':y}) for d,y,_ in frames.values()],ignore_index=True)
 result={}
 for alpha in alphas:
  z=all_labels.assign(v=np.power(all_labels.y,alpha)).groupby(['matrix_id','block_size']).v.sum()
  result[alpha]=z
 return result

def weights(d,y,den,alpha):
 keys=pd.MultiIndex.from_arrays([d.matrix_id.astype(str),pd.to_numeric(d.block_size).astype(int)])
 w=np.power(y,alpha)/den.reindex(keys).to_numpy(float)
 if not np.isfinite(w).all() or (w<=0).any():raise RuntimeError('invalid weights')
 return w/w.mean()

def pair_accuracy(g):
 a=g.actual_ms.to_numpy();p=g.predicted_ms.to_numpy();correct=total=0
 for i in range(len(g)):
  for j in range(i+1,len(g)):
   if a[i]==a[j]:continue
   correct+=int(np.sign(a[i]-a[j])==np.sign(p[i]-p[j]));total+=1
 return correct/max(total,1)

def scheme_summary(tasks,c,scope):
 schemes=tasks.groupby(['matrix_id','block_size'],as_index=False).agg(actual_ms=('actual_ms','sum'),predicted_ms=('predicted_ms','sum'),tasks=('task_id','size'))
 selections=[]
 for matrix_id,g in schemes.groupby('matrix_id',sort=True):
  oracle=g.loc[g.actual_ms.idxmin()];chosen=g.loc[g.predicted_ms.idxmin()]
  selections.append({'scope':scope,'matrix_id':matrix_id,'oracle_block_size':int(oracle.block_size),'predicted_best_block_size':int(chosen.block_size),'oracle_actual_ms':float(oracle.actual_ms),'selected_actual_ms':float(chosen.actual_ms),'regret_percent':float((chosen.actual_ms/oracle.actual_ms-1)*100),'spearman':float(g.actual_ms.rank().corr(g.predicted_ms.rank())),'pairwise_accuracy':pair_accuracy(g)})
 sel=pd.DataFrame(selections)
 m=c.metric(schemes.actual_ms,schemes.predicted_ms)
 summary={'scope':scope,**m,'matrices':len(sel),'oracle_accuracy':float((sel.oracle_block_size==sel.predicted_best_block_size).mean()),'mean_spearman':float(sel.spearman.mean()),'mean_pairwise_accuracy':float(sel.pairwise_accuracy.mean()),'mean_regret':float(sel.regret_percent.mean()),'median_regret':float(sel.regret_percent.median()),'p95_regret':float(sel.regret_percent.quantile(.95))}
 return schemes,sel,summary

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train-dir',required=True);ap.add_argument('--unseen-dir',required=True);ap.add_argument('--fold-map',required=True);ap.add_argument('--output',required=True);ap.add_argument('--alphas',default='0.5,1.0');a=ap.parse_args()
 src=Path(a.train_dir);unseen=Path(a.unseen_dir);out=Path(a.output)
 if out.exists() and any(out.iterdir()):raise SystemExit(f'output must be empty: {out}')
 out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir();c=contract();alphas=tuple(float(x) for x in a.alphas.split(','));started=time.time()
 fm=pd.read_csv(a.fold_map);foldmap=dict(zip(fm.matrix_id.astype(str),fm.cv_fold.astype(int)))
 train={t:load(src,t,c) for t in c.TARGETS};denoms=weight_tables(train,alphas)
 candidate_summaries=[];candidate_selections=[];candidate_schemes=[];oof_task_metrics=[];oof_frames_by_alpha={}
 for ai,alpha in enumerate(alphas):
  frames=[]
  for ti,target in enumerate(c.TARGETS):
   d,y,X=train[target];fid=d.matrix_id.astype(str).map(foldmap)
   if fid.isna().any() or set(fid.astype(int))!={1,2,3,4,5}:raise RuntimeError(f'{target}: invalid fold map')
   fid=fid.to_numpy(int);w=weights(d,y,denoms[alpha],alpha);pred=np.empty(len(y),float)
   for fold in range(1,6):
    tr=fid!=fold;te=fid==fold
    model=c.XGBRegressor(**c.PARAMS,random_state=c.SEED+ai*10000+ti*1000+fold)
    model.fit(X[tr],np.log1p(y[tr]),sample_weight=w[tr],verbose=False)
    pred[te]=np.maximum(np.expm1(model.predict(X[te])),0)
   f=pd.DataFrame({'alpha':alpha,'target':target,'task_id':d.task_id.astype(str),'matrix_id':d.matrix_id.astype(str),'block_size':pd.to_numeric(d.block_size).astype(int),'actual_ms':y,'predicted_ms':pred})
   frames.append(f);oof_task_metrics.append({'alpha':alpha,'target':target,**c.metric(y,pred)})
  tasks=pd.concat(frames,ignore_index=True);oof_frames_by_alpha[alpha]=tasks
  schemes,sel,summary=scheme_summary(tasks,c,f'oof_alpha_{alpha:g}')
  schemes.insert(0,'alpha',alpha);sel.insert(0,'alpha',alpha);summary['alpha']=alpha
  candidate_schemes.append(schemes);candidate_selections.append(sel);candidate_summaries.append(summary)
  print('OOF',alpha,summary,flush=True)
 summary_frame=pd.DataFrame(candidate_summaries).sort_values(['mean_regret','p95_regret','MAPE','alpha'])
 best=float(summary_frame.iloc[0].alpha)
 print('SELECTED_ALPHA',best,flush=True)

 # Freeze one deploy model per kernel on all profiling25 rows using the OOF-selected alpha.
 external_frames=[];external_task_metrics=[]
 for ti,target in enumerate(c.TARGETS):
  d,y,X=train[target];w=weights(d,y,denoms[best],best)
  model=c.XGBRegressor(**c.PARAMS,random_state=c.SEED+50000+ti*1000)
  model.fit(X,np.log1p(y),sample_weight=w,verbose=False)
  model.save_model(out/'models'/f'{target}.weighted_log_xgboost_ABCD_alpha_{best:g}.json')
  u,uy,uX=load(unseen,target,c)
  overlap=set(d.matrix_id.astype(str))&set(u.matrix_id.astype(str))
  if overlap:raise RuntimeError(f'{target}: train/unseen overlap {overlap}')
  up=np.maximum(np.expm1(model.predict(uX)),0)
  f=pd.DataFrame({'alpha':best,'target':target,'task_id':u.task_id.astype(str),'matrix_id':u.matrix_id.astype(str),'block_size':pd.to_numeric(u.block_size).astype(int),'actual_ms':uy,'predicted_ms':up})
  external_frames.append(f);external_task_metrics.append({'alpha':best,'target':target,**c.metric(uy,up)})
 external_tasks=pd.concat(external_frames,ignore_index=True)
 external_schemes,external_sel,external_summary=scheme_summary(external_tasks,c,f'unseen_alpha_{best:g}')

 pd.DataFrame(oof_task_metrics).to_csv(out/'oof_task_metrics.csv',index=False)
 pd.concat(candidate_schemes,ignore_index=True).to_csv(out/'oof_scheme_predictions.csv',index=False)
 pd.concat(candidate_selections,ignore_index=True).to_csv(out/'oof_scheme_selection_by_matrix.csv',index=False)
 summary_frame.to_csv(out/'oof_alpha_selection.csv',index=False)
 oof_frames_by_alpha[best].to_csv(out/'selected_alpha_oof_task_predictions.csv.gz',index=False,compression='gzip')
 external_tasks.to_csv(out/'unseen_task_predictions.csv.gz',index=False,compression='gzip')
 pd.DataFrame(external_task_metrics).to_csv(out/'unseen_task_metrics.csv',index=False)
 external_schemes.to_csv(out/'unseen_scheme_predictions.csv',index=False);external_sel.to_csv(out/'unseen_scheme_selection.csv',index=False)
 manifest={'schema':'weighted_log_xgboost_v4.2_ABCD_matrix_oof_select_v1','target':'log1p(label_time_ms)','prediction':'expm1(raw_prediction)','features':'log1p(max(raw_feature,0))','weight_formula':'w_q = T_q^alpha / sum_{r in same matrix+NB scheme, all kernels}(T_r^alpha), then normalized to mean 1 within kernel','candidate_alphas':alphas,'selection_rule':'minimum OOF mean scheme regret; ties by P95 regret, scheme MAPE, alpha','selected_alpha':best,'oof_selected_summary':summary_frame.iloc[0].to_dict(),'unseen_summary':external_summary,'unseen_used_for_selection_or_fit':False,'train_dir':str(src.resolve()),'unseen_dir':str(unseen.resolve()),'fold_map':str(Path(a.fold_map).resolve()),'wall_seconds':time.time()-started}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=float)+'\n')
 print('UNSEEN',external_summary,flush=True);print(external_sel.to_string(index=False),flush=True);print('wrote',out,flush=True)

if __name__=='__main__':main()
