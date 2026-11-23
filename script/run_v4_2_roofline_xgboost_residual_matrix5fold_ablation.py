#!/usr/bin/env python3
"""Matrix-grouped five-fold Roofline + XGBoost log-residual cumulative A..E ablation."""
from __future__ import annotations
import argparse, json, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from xgboost import XGBRegressor
from feature_sets_v4_2 import FEATURES_BY_KERNEL, BASELINE_FIELDS
from run_v4_1_flops_bytes_ablation import values, metric
from run_v4_2_paper_order_max_first_ablation import groups

TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense")
STAGES=("A","A+B","A+B+C","A+B+C+D","A+B+C+D+E")
SEED=20260812
PARAMS=dict(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
    tree_method="hist",
    max_bin=256,
    n_jobs=8,
)

def arguments():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--fold-map', required=True)
    ap.add_argument('--direct-dir', default=None, help='Optional direct XGBoost output dir for side-by-side comparison.')
    ap.add_argument('--output', required=True)
    return ap.parse_args()

def fit_roofline(ops, byt, y, train):
    po=ops[train & (ops>0)]
    pb=byt[train & (byt>0)]
    scale=np.array([max(float(np.median(po)),1.0), max(float(np.median(pb)),1.0)])
    design=np.column_stack([np.ones(len(y)), ops/scale[0], byt/scale[1]])
    coef,_=nnls(design[train], y[train])
    base=np.maximum(design @ coef, 1e-9)
    return scale, coef, base

def matrix_metrics(frame):
    rows=[]
    for matrix_id, part in frame.groupby('matrix_id', sort=True):
        rows.append({'matrix_id':matrix_id, **metric(part.actual_ms, part.predicted_ms)})
    tab=pd.DataFrame(rows)
    return tab, {'macro_MAPE':float(tab.MAPE.mean()), 'macro_MdAPE':float(tab.MdAPE.mean()), 'worst_matrix_MAPE':float(tab.MAPE.max())}

def make_model(ti:int, si:int=0, fold:int=0):
    return XGBRegressor(**PARAMS, random_state=SEED + ti*1000 + si*10 + fold)

def main():
    a=arguments(); src=Path(a.input_dir); out=Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'output must be empty: {out}')
    out.mkdir(parents=True)
    (out/'deploy_models').mkdir()
    shutil.copy2(a.fold_map, out/'shared_matrix_folds.csv')
    ff=pd.read_csv(a.fold_map)
    counts=ff.groupby('cv_fold').size().sort_index().tolist()
    if counts != [5,5,5,5,5]:
        raise RuntimeError(f'expected 25-matrix folds 5/5/5/5/5, got {counts}')
    fmap=dict(zip(ff.matrix_id.astype(str), ff.cv_fold.astype(int)))
    rows=[]; foldrows=[]; matrix_rows=[]; pred_frames=[]; timing=[]
    manifest={'schema':'h100_profile25_v4.2_roofline_xgboost_log_residual_matrix5fold_ablation_v1',
              'formula':'log1p(T)=log1p(T_roofline)+f_XGB(x); T_roofline is a nonnegative affine function of operation_count_v2 and base_bytes_est fitted on each training fold.',
              'seed':SEED,'cv':'shared matrix-grouped 5-fold','xgboost_params':PARAMS,
              'input_dir':str(src.resolve()),'fold_map':str(Path(a.fold_map).resolve()),'targets':{}}
    wall=time.perf_counter()
    for ti,target in enumerate(TARGETS):
        fgroups, derived=groups(target)
        raw=FEATURES_BY_KERNEL[target]+BASELINE_FIELDS
        req=set(['task_id','matrix_id','block_size','label_time_ms','hardware_id']+raw)
        d=pd.read_csv(src/f'{target}.model_ready_v4.csv', usecols=lambda c:c in req, low_memory=False)
        for new,(mean,cv) in derived.items():
            d[new]=pd.to_numeric(d[mean],errors='coerce').fillna(0)*pd.to_numeric(d[cv],errors='coerce').fillna(0)
        y=pd.to_numeric(d.label_time_ms, errors='coerce').to_numpy(float)
        ok=np.isfinite(y)&(y>0)
        d=d.loc[ok].reset_index(drop=True); y=y[ok]
        fid=d.matrix_id.astype(str).map(fmap)
        if fid.isna().any() or set(fid.astype(int))!={1,2,3,4,5}:
            missing=sorted(d.loc[fid.isna(),'matrix_id'].astype(str).unique().tolist())
            raise RuntimeError(f'{target}: invalid fold coverage, missing={missing}')
        fid=fid.to_numpy(int)
        ops=pd.to_numeric(d[BASELINE_FIELDS[0]],errors='coerce').fillna(0).to_numpy(float)
        byt=pd.to_numeric(d[BASELINE_FIELDS[1]],errors='coerce').fillna(0).to_numpy(float)
        cumulative=[]
        manifest['targets'][target]={'rows':int(len(d)), 'matrices':sorted(d.matrix_id.astype(str).unique().tolist()),
                                     'groups':fgroups, 'derived_std':{k:{'mean':v[0], 'cv':v[1], 'formula':'std=mean*cv'} for k,v in derived.items()}}
        for si,stage in enumerate(STAGES):
            cumulative += fgroups[chr(ord('A')+si)]
            X=values(d, cumulative)
            poof=np.empty(len(y)); boof=np.empty(len(y))
            stage_t0=time.perf_counter()
            for fold in range(1,6):
                train=fid!=fold; test=fid==fold
                scale,coef,base=fit_roofline(ops,byt,y,train)
                boof[test]=base[test]
                residual=np.log1p(y[train])-np.log1p(base[train])
                fit_t0=time.perf_counter(); model=make_model(ti,si,fold)
                model.fit(X[train], residual, verbose=False)
                fit_s=time.perf_counter()-fit_t0
                pred_t0=time.perf_counter()
                poof[test]=np.maximum(np.expm1(np.log1p(base[test])+model.predict(X[test])),0)
                pred_s=time.perf_counter()-pred_t0
                foldrows.append({'target':target,'stage':stage,'fold':fold,'features':len(cumulative),'n_train':int(train.sum()),'n_test':int(test.sum()),**metric(y[test],poof[test])})
                timing.append({'target':target,'stage':stage,'fold':fold,'fit_seconds':fit_s,'predict_seconds':pred_s,'n_train':int(train.sum()),'n_test':int(test.sum())})
            total=metric(y,poof); roof_m=metric(y,boof)
            frame=pd.DataFrame({'matrix_id':d.matrix_id,'actual_ms':y,'predicted_ms':poof})
            per_matrix,macro=matrix_metrics(frame)
            rows.append({'target':target,'stage':stage,'features':len(cumulative),**total,**macro,
                         'roofline_MAPE':roof_m['MAPE'],'roofline_MdAPE':roof_m['MdAPE'],'roofline_P95_APE':roof_m['P95_APE'],'roofline_R2':roof_m['R2'],
                         'stage_wall_seconds':time.perf_counter()-stage_t0})
            per_matrix.insert(0,'stage',stage); per_matrix.insert(0,'target',target); matrix_rows.append(per_matrix)
            pred_frames.append(pd.DataFrame({'target':target,'stage':stage,'task_id':d.task_id,'matrix_id':d.matrix_id,'block_size':d.block_size,'cv_fold':fid,'actual_ms':y,'roofline_ms':boof,'predicted_ms':poof}))
            print(target, stage, len(cumulative), f"MAPE={total['MAPE']:.3f}", f"MdAPE={total['MdAPE']:.3f}", f"P95={total['P95_APE']:.3f}", f"R2={total['R2']:.4f}", flush=True)
        # deploy model with final stage on all data
        final_features=list(cumulative)
        Xf=values(d, final_features)
        scale,coef,base=fit_roofline(ops,byt,y,np.ones(len(y),dtype=bool))
        residual=np.log1p(y)-np.log1p(base)
        deploy=make_model(ti,len(STAGES),0); deploy.fit(Xf,residual,verbose=False)
        deploy.save_model(out/'deploy_models'/f'{target}.roofline_xgboost_log_residual_A_B_C_D_E.json')
        manifest['targets'][target]['deploy_model']={'path':str((out/'deploy_models'/f'{target}.roofline_xgboost_log_residual_A_B_C_D_E.json').resolve()),'features':final_features,'baseline_scale':scale.tolist(),'baseline_coef':coef.tolist()}
    summary=pd.DataFrame(rows)
    summary['MAPE_improvement_from_previous']=summary.groupby('target').MAPE.diff().mul(-1)
    summary['MAPE_improvement_vs_A']=summary.groupby('target').MAPE.transform('first')-summary.MAPE
    summary.to_csv(out/'roofline_xgboost_cumulative_group_metrics.csv',index=False)
    pd.DataFrame(foldrows).to_csv(out/'fold_metrics.csv',index=False)
    pd.concat(matrix_rows,ignore_index=True).to_csv(out/'per_matrix_metrics.csv',index=False)
    pd.DataFrame(timing).to_csv(out/'timing.csv',index=False)
    pd.concat(pred_frames,ignore_index=True).to_csv(out/'oof_predictions.csv.gz',index=False,compression='gzip')
    if a.direct_dir:
        direct=pd.read_csv(Path(a.direct_dir)/'xgboost_cumulative_group_metrics.csv')
        comp=summary.merge(direct, on=['target','stage','features'], suffixes=('_roofline_residual','_direct'))
        keep=['target','stage','features','MAPE_roofline_residual','MAPE_direct','MdAPE_roofline_residual','MdAPE_direct','P95_APE_roofline_residual','P95_APE_direct','R2_roofline_residual','R2_direct']
        comp[keep].to_csv(out/'roofline_residual_vs_direct.csv',index=False)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':10})
    fig,ax=plt.subplots(figsize=(7.4,4.2))
    for target,q in summary.groupby('target',sort=False):
        ax.plot(q.stage,q.MAPE,marker='o',lw=1.8,label=target.replace('_','-'))
    ax.set_xlabel('Cumulative feature groups')
    ax.set_ylabel('OOF MAPE (%)')
    ax.grid(True,lw=.4,alpha=.4); ax.legend(frameon=False,ncol=2); fig.tight_layout()
    fig.savefig(out/'roofline_xgboost_cumulative_group_mape.pdf')
    fig.savefig(out/'roofline_xgboost_cumulative_group_mape.png',dpi=300)
    plt.close(fig)
    manifest['wall_seconds']=time.perf_counter()-wall
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
