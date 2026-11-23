#!/usr/bin/env python3
"""Train EBM and latency-aware Roofline+EBM-residual models with grouped CV."""
import argparse,csv,json,math
from pathlib import Path
import joblib,numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
import train_kernel_time_models_phase1 as p1

SPECS=(("GETRF","GETRF.csv","time_kernel_ms"),("GESSM","GESSM.csv","time_kernel_ms"),("TSTRF","TSTRF.csv","time_kernel_ms"),("SSSSM_sparse","SSSSM.csv","sparse_time_kernel_ms"),("SSSSM_dense","SSSSM.csv","dense_time_kernel_ms"))

def arguments():
 p=argparse.ArgumentParser(description="Train EBM and Roofline-residual models")
 p.add_argument("training_set");p.add_argument("--output",required=True);p.add_argument("--folds",type=int,default=5);p.add_argument("--epsilon-ms",type=float,default=1e-4);p.add_argument("--seed",type=int,default=20260727);p.add_argument("--interactions",type=int,default=8);p.add_argument("--max-rounds",type=int,default=2000);p.add_argument("--outer-bags",type=int,default=4);p.add_argument("--kernels",nargs='*',choices=[x[0] for x in SPECS]);return p.parse_args()
def ebm(args):
 return ExplainableBoostingRegressor(feature_names=p1.FEATURES,feature_types=['continuous']*len(p1.FEATURES),max_bins=256,interactions=args.interactions,outer_bags=args.outer_bags,inner_bags=0,learning_rate=.03,max_rounds=args.max_rounds,early_stopping_rounds=100,early_stopping_tolerance=1e-5,min_samples_leaf=10,max_leaves=3,n_jobs=-2,random_state=args.seed)
def roof_arrays(rows):
 f=np.array([p1.number(r,'flops')/1e9 for r in rows]);b=np.array([p1.number(r,'estimated_bytes')/1e9 for r in rows]);l=np.ones(len(rows));return f,b,l
def roof_predict(par,f,b,l):
 base,extra,compute,bw=par;return base+extra*np.maximum(l-1,0)+np.maximum(f/compute,b/bw)
def roof_fit(f,b,l,y,w,eps):
 variable=np.unique(l).size>1;initial=np.array([max(np.percentile(y,10)*.5,1e-4),1e-3,10.,1.]);active=np.array([1,variable,1,1],dtype=bool);fixed=initial.copy();fixed[1]=0 if not variable else fixed[1]
 def expand(theta):
  z=fixed.copy();z[active]=np.exp(theta);return z
 def residual(theta):return np.sqrt(w/np.mean(w))*(np.log(roof_predict(expand(theta),f,b,l)+eps)-np.log(y+eps))
 result=least_squares(residual,np.log(initial[active]),bounds=(math.log(1e-9),math.log(1e6)),loss='soft_l1',f_scale=.1,max_nfev=5000)
 return expand(result.x)
def metric(y,p,w):
 p=np.maximum(p,0);a=np.abs(p-y);r=a/np.maximum(y,1e-9);s=2*a/np.maximum(y+p,1e-9)
 return {'samples':len(y),'mae_ms':float(a.mean()),'rmse_ms':float(np.sqrt(np.mean((p-y)**2))),'mape_percent':float(r.mean()*100),'median_ape_percent':float(np.median(r)*100),'p90_ape_percent':float(np.percentile(r,90)*100),'wape_percent':float(a.sum()/y.sum()*100),'smape_percent':float(s.mean()*100),'weighted_mape_percent':float(np.average(r,weights=w)*100),'weighted_smape_percent':float(np.average(s,weights=w)*100),'r2':float(r2_score(y,p))}
def formula(par):return f"T_roof_ms={par[0]:.8g}+{par[1]:.8g}*max(launches-1,0)+max(F_GFLOP/{par[2]:.8g},B_GB/{par[3]:.8g})"
def write_pred(path,rows,y,e,r,fold):
 fields=['task_id','matrix_id','block_size','repeat_index','actual_time_kernel_ms','ebm_predicted_ms','roofline_residual_predicted_ms','cv_fold']
 with path.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
  for row,a,b,c,d in zip(rows,y,e,r,fold):w.writerow({'task_id':row.get('task_id',''),'matrix_id':row['matrix_id'],'block_size':row['block_size'],'repeat_index':row['repeat_index'],'actual_time_kernel_ms':a,'ebm_predicted_ms':b,'roofline_residual_predicted_ms':c,'cv_fold':d})
def main():
 args=arguments();src=Path(args.training_set).resolve();out=Path(args.output).resolve()
 if out.exists() and any(out.iterdir()):raise SystemExit(f'output is not empty: {out}')
 out.mkdir(parents=True,exist_ok=True);wanted=set(args.kernels or [x[0] for x in SPECS]);results=[];selector={}
 for name,file,target in SPECS:
  if name not in wanted:continue
  rows=p1.read_rows(src/file);x=p1.design(rows);y=p1.target(rows,target);w=p1.weights(rows);groups=np.array([r['matrix_id'] for r in rows]);f,b,l=roof_arrays(rows);folds=min(args.folds,len(np.unique(groups)));split=GroupKFold(folds);ep=np.empty_like(y);rp=np.empty_like(y);fid=np.empty(len(y),int);outer_roof=[]
  for fold,(train,test) in enumerate(split.split(x,groups=groups),1):
   direct=ebm(args);direct.fit(x[train],np.log(y[train]+args.epsilon_ms),sample_weight=w[train]);ep[test]=np.maximum(np.exp(direct.predict(x[test]))-args.epsilon_ms,0)
   par=roof_fit(f[train],b[train],l[train],y[train],w[train],args.epsilon_ms);base_train=roof_predict(par,f[train],b[train],l[train]);base_test=roof_predict(par,f[test],b[test],l[test]);residual=np.log(y[train]+args.epsilon_ms)-np.log(base_train+args.epsilon_ms);correction=ebm(args);correction.fit(x[train],residual,sample_weight=w[train]);rp[test]=np.maximum(np.exp(np.log(base_test+args.epsilon_ms)+correction.predict(x[test]))-args.epsilon_ms,0);fid[test]=fold;outer_roof.append({'fold':fold,'parameters':par.tolist(),'formula':formula(par)})
  direct=ebm(args);direct.fit(x,np.log(y+args.epsilon_ms),sample_weight=w);par=roof_fit(f,b,l,y,w,args.epsilon_ms);base=roof_predict(par,f,b,l);correction=ebm(args);correction.fit(x,np.log(y+args.epsilon_ms)-np.log(base+args.epsilon_ms),sample_weight=w)
  joblib.dump({'estimator':direct,'features':p1.FEATURES,'target':target,'target_transform':f'log(time_ms+{args.epsilon_ms})'},out/f'{name}_EBM.joblib')
  joblib.dump({'roofline_parameters':par,'roofline_formula':formula(par),'residual_estimator':correction,'features':p1.FEATURES,'target':target,'prediction':'exp(log(T_roof+epsilon)+EBM_residual)-epsilon'},out/f'{name}_RooflineResidual.joblib')
  em=metric(y,ep,w);rm=metric(y,rp,w);em.update({'model':f'{name}_EBM','matrices':len(np.unique(groups)),'folds':folds});rm.update({'model':f'{name}_RooflineResidual','matrices':len(np.unique(groups)),'folds':folds,'full_roofline_parameters':par.tolist(),'full_roofline_formula':formula(par),'outer_roofline':outer_roof});results.extend([em,rm]);write_pred(out/f'{name}_cv_predictions.csv',rows,y,ep,rp,fid)
  if name.startswith('SSSSM_'):selector[name]=(rows,y,ep,rp,w,fid)
  print(name,'EBM MAPE',em['mape_percent'],'Residual MAPE',rm['mape_percent'],flush=True)
 if {'SSSSM_sparse','SSSSM_dense'}<=selector.keys():
  sr,sy,se,srpred,sw,sf=selector['SSSSM_sparse'];dr,dy,de,drpred,dw,df=selector['SSSSM_dense'];
  if len(sr)!=len(dr) or not np.array_equal(sf,df):raise RuntimeError('SSSSM rows/folds differ')
  def sel(label,dense_choice):
   actual_dense=dy<sy;chosen=np.where(dense_choice,dy,sy);best=np.minimum(sy,dy);reg=(chosen-best)/np.maximum(best,1e-9);return {'model':label,'samples':len(sy),'selection_accuracy_percent':float(np.mean(dense_choice==actual_dense)*100),'weighted_selection_accuracy_percent':float(np.average(dense_choice==actual_dense,weights=sw)*100),'mean_regret_percent':float(reg.mean()*100),'weighted_mean_regret_percent':float(np.average(reg,weights=sw)*100),'p90_regret_percent':float(np.percentile(reg,90)*100)}
  results.extend([sel('SSSSM_EBM_selector',de<se),sel('SSSSM_RooflineResidual_selector',drpred<srpred)])
 (out/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(json.dumps([{k:v for k,v in z.items() if k not in ('outer_roofline',)} for z in results],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
