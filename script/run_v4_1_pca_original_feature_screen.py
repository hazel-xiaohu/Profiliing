#!/usr/bin/env python3
"""Unsupervised PCA/correlation screening of original features, confirmed by grouped CV."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from run_v4_1_flops_bytes_ablation import TARGETS,FEATURES,OPS,BYTES,SEED,values,metric
from run_v4_1_feature_evidence import model

THRESHOLDS=(.90,.95,.98)
# No feature is protected during redundancy screening. Operations and bytes
# compete with all other original dimensions under the same rule.
MANDATORY=()

def components(corr,threshold):
 n=len(corr); seen=set(); ans=[]
 for i in range(n):
  if i in seen: continue
  stack=[i]; seen.add(i); comp=[]
  while stack:
   u=stack.pop(); comp.append(u)
   for v in np.flatnonzero(np.abs(corr[u])>=threshold):
    v=int(v)
    if v!=u and v not in seen: seen.add(v); stack.append(v)
  ans.append(sorted(comp))
 return ans

def select_unsupervised(X,names,threshold):
 Xs=StandardScaler().fit_transform(X); p=PCA().fit(Xs); cum=np.cumsum(p.explained_variance_ratio_); k=int(np.searchsorted(cum,.95)+1)
 # Variance-weighted contribution to the retained 95% PCA subspace.
 communality=((p.components_[:k].T**2)*p.explained_variance_[:k]).sum(axis=1)
 corr=np.corrcoef(Xs,rowvar=False); comps=components(corr,threshold); keep=set(); audit=[]
 for cid,comp in enumerate(comps):
  forced=[i for i in comp if names[i] in MANDATORY]
  chosen=set(forced)
  remaining=[i for i in comp if i not in forced]
  if remaining: chosen.add(max(remaining,key=lambda i:(communality[i],names[i])))
  keep.update(chosen)
  for i in comp: audit.append({'cluster':cid,'feature':names[i],'communality':communality[i],'selected':int(i in chosen),'mandatory':int(names[i] in MANDATORY),'cluster_size':len(comp)})
 return [n for i,n in enumerate(names) if i in keep],audit,k

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--fold-map',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 src,out=Path(a.input_dir),Path(a.output)
 if out.exists() and any(out.iterdir()):raise SystemExit(f'output must be empty: {out}')
 out.mkdir(parents=True);fm0=pd.read_csv(a.fold_map);fm=dict(zip(fm0.matrix_id.astype(str),fm0.cv_fold.astype(int)))
 results=[];audits=[];selections=[];manifest={'schema':'pca_original_feature_screen_v4.1','thresholds':THRESHOLDS,'mandatory':MANDATORY,'targets':{},'selection_uses_labels':False};wall=time.perf_counter()
 for ti,target in enumerate(TARGETS):
  names=FEATURES[target]+[OPS,BYTES];req=set(['matrix_id','label_time_ms']+names);d=pd.read_csv(src/f'{target}.model_ready_v4.csv',usecols=lambda c:c in req,low_memory=False)
  y=pd.to_numeric(d.label_time_ms,errors='coerce').to_numpy(float);ok=np.isfinite(y)&(y>0);d=d.loc[ok].reset_index(drop=True);y=y[ok];fid=d.matrix_id.astype(str).map(fm).to_numpy(int);X=values(d,names)
  variants={'All':names};pca_k=None
  for th in THRESHOLDS:
   selected,audit,k=select_unsupervised(X,names,th);variants[f'PCA_corr_{th:.2f}']=selected;pca_k=k
   for r in audit:r.update({'target':target,'threshold':th});audits.extend(audit)
  manifest['targets'][target]={'original_features':len(names),'pca_components_95':pca_k}
  for vi,(variant,selected) in enumerate(variants.items()):
   idx=[names.index(f) for f in selected];oof=np.empty(len(y));fit_sum=0.;pred_sum=0.
   for fold in range(1,6):
    tr=fid!=fold;te=fid==fold;t=time.perf_counter();m=model(SEED+ti*1000+vi*10+fold).fit(X[tr][:,idx],np.log1p(y[tr]));fit_sum+=time.perf_counter()-t
    t=time.perf_counter();oof[te]=np.expm1(m.predict(X[te][:,idx]));pred_sum+=time.perf_counter()-t
   row={'target':target,'variant':variant,'n_features':len(selected),**metric(y,oof),'fit_seconds':fit_sum,'predict_seconds':pred_sum};results.append(row)
   for f in names:selections.append({'target':target,'variant':variant,'feature':f,'selected':int(f in selected)})
   print(target,variant,len(selected),f"MAPE={row['MAPE']:.3f}",flush=True)
 res=pd.DataFrame(results);base=res[res.variant=='All'].set_index('target').MAPE;res['MAPE_delta_points']=res.apply(lambda r:r.MAPE-base[r.target],axis=1);res['accepted_delta_le_0.2']=((res.variant!='All')&(res.MAPE_delta_points<=.2)).astype(int);res.to_csv(out/'cv_confirmation.csv',index=False)
 # Smallest accepted set; ties favor lower MAPE.
 chosen=[]
 for target,q in res[res['accepted_delta_le_0.2']==1].groupby('target'):
  r=q.sort_values(['n_features','MAPE']).iloc[0];chosen.append(r)
 pd.DataFrame(chosen).to_csv(out/'recommended_variants.csv',index=False)
 pd.DataFrame(audits).to_csv(out/'redundancy_clusters.csv',index=False);pd.DataFrame(selections).to_csv(out/'selected_features.csv',index=False)
 manifest['wall_seconds']=time.perf_counter()-wall;(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
