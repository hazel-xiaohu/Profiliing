#!/usr/bin/env python3
"""Matrix-grouped five-fold direct XGBoost cumulative A..E ablation for v4.2 features."""
from __future__ import annotations
import argparse, json, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from feature_sets_v4_2 import FEATURES_BY_KERNEL, BASELINE_FIELDS
from run_v4_1_flops_bytes_ablation import values, metric
from run_v4_2_paper_order_max_first_ablation import groups

TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense")
STAGES=("A","A+B","A+B+C","A+B+C+D","A+B+C+D+E")
SEED=20260811
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--fold-map', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--allow-nonempty-output', action='store_true')
    return p.parse_args()

def matrix_metrics(frame):
    rows=[]
    for matrix_id, part in frame.groupby('matrix_id', sort=True):
        rows.append({'matrix_id': matrix_id, **metric(part.actual_ms, part.predicted_ms)})
    table=pd.DataFrame(rows)
    return table, {
        'macro_MAPE': float(table.MAPE.mean()),
        'macro_MdAPE': float(table.MdAPE.mean()),
        'worst_matrix_MAPE': float(table.MAPE.max()),
    }

def make_model(target_index:int, stage_index:int=0, fold:int=0):
    return XGBRegressor(**PARAMS, random_state=SEED + target_index*1000 + stage_index*10 + fold)

def main():
    a=arguments(); src=Path(a.input_dir); out=Path(a.output)
    if out.exists() and any(out.iterdir()) and not a.allow_nonempty_output:
        raise SystemExit(f'output must be empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'deploy_models').mkdir(exist_ok=True)
    shutil.copy2(a.fold_map, out/'shared_matrix_folds.csv')
    fold_frame=pd.read_csv(a.fold_map)
    counts=fold_frame.groupby('cv_fold').size().sort_index().tolist()
    if counts != [5,5,5,5,5]:
        raise RuntimeError(f'expected 25-matrix folds 5/5/5/5/5, got {counts}')
    fold_map=dict(zip(fold_frame.matrix_id.astype(str), fold_frame.cv_fold.astype(int)))
    rows=[]; foldrows=[]; matrix_rows=[]; pred_frames=[]; timing=[]
    manifest={
        'schema':'h100_profile25_v4.2_direct_xgboost_matrix5fold_ablation_v1',
        'seed':SEED,
        'cv':'shared matrix-grouped 5-fold; all tasks from one matrix stay in the same fold',
        'xgboost_params':PARAMS,
        'target_kernels':TARGETS,
        'stages':STAGES,
        'group_order':[
            'A: block dimensions and nonzero counts of input blocks',
            'B: floating-point operation count',
            'C: max nonzeros per row/column',
            'D: mean and derived std of nonzeros per row/column',
            'E: empty row/column fraction',
        ],
        'input_dir':str(src.resolve()),
        'fold_map':str(Path(a.fold_map).resolve()),
        'targets':{},
    }
    wall=time.perf_counter()
    for ti,target in enumerate(TARGETS):
        feature_groups, derived = groups(target)
        raw_features = FEATURES_BY_KERNEL[target] + [BASELINE_FIELDS[0]]
        req=set(['task_id','matrix_id','block_size','label_time_ms','hardware_id'] + raw_features)
        d=pd.read_csv(src/f'{target}.model_ready_v4.csv', usecols=lambda c:c in req, low_memory=False)
        for new,(mean,cv) in derived.items():
            d[new]=pd.to_numeric(d[mean], errors='coerce').fillna(0) * pd.to_numeric(d[cv], errors='coerce').fillna(0)
        y=pd.to_numeric(d.label_time_ms, errors='coerce').to_numpy(float)
        ok=np.isfinite(y) & (y>0)
        d=d.loc[ok].reset_index(drop=True); y=y[ok]
        mapped=d.matrix_id.astype(str).map(fold_map)
        if mapped.isna().any() or set(mapped.astype(int)) != {1,2,3,4,5}:
            missing=sorted(d.loc[mapped.isna(),'matrix_id'].astype(str).unique().tolist())
            raise RuntimeError(f'{target}: invalid fold coverage, missing={missing}')
        fid=mapped.to_numpy(int)
        cumulative=[]
        manifest['targets'][target]={
            'rows':int(len(d)),
            'matrices':sorted(d.matrix_id.astype(str).unique().tolist()),
            'groups':feature_groups,
            'derived_std':{k:{'mean':v[0], 'cv':v[1], 'formula':'std=mean*cv'} for k,v in derived.items()},
        }
        for si,stage in enumerate(STAGES):
            cumulative += feature_groups[chr(ord('A')+si)]
            X=values(d, cumulative)
            oof=np.empty(len(y), dtype=float)
            stage_t0=time.perf_counter()
            for fold in range(1,6):
                train=fid!=fold; test=fid==fold
                fit_t0=time.perf_counter()
                model=make_model(ti, si, fold)
                model.fit(X[train], np.log1p(y[train]), verbose=False)
                fit_s=time.perf_counter()-fit_t0
                pred_t0=time.perf_counter()
                oof[test]=np.maximum(np.expm1(model.predict(X[test])), 0)
                pred_s=time.perf_counter()-pred_t0
                fm=metric(y[test], oof[test])
                foldrows.append({'target':target,'stage':stage,'fold':fold,'features':len(cumulative),'n_train':int(train.sum()),'n_test':int(test.sum()),**fm})
                timing.append({'target':target,'stage':stage,'fold':fold,'fit_seconds':fit_s,'predict_seconds':pred_s,'n_train':int(train.sum()),'n_test':int(test.sum())})
            total=metric(y,oof)
            frame=pd.DataFrame({'matrix_id':d.matrix_id, 'actual_ms':y, 'predicted_ms':oof})
            per_matrix, macro = matrix_metrics(frame)
            rows.append({'target':target,'stage':stage,'features':len(cumulative),**total,**macro,'stage_wall_seconds':time.perf_counter()-stage_t0})
            per_matrix.insert(0,'stage',stage); per_matrix.insert(0,'target',target)
            matrix_rows.append(per_matrix)
            pred_frames.append(pd.DataFrame({'target':target,'stage':stage,'task_id':d.task_id,'matrix_id':d.matrix_id,'block_size':d.block_size,'cv_fold':fid,'actual_ms':y,'predicted_ms':oof}))
            print(target, stage, len(cumulative), f"MAPE={total['MAPE']:.3f}", f"MdAPE={total['MdAPE']:.3f}", f"P95={total['P95_APE']:.3f}", f"R2={total['R2']:.4f}", flush=True)
        # Train deploy model on the full profiling set using the final A+B+C+D+E feature set.
        final_features=list(cumulative)
        X_final=values(d, final_features)
        deploy=make_model(ti, len(STAGES), 0)
        deploy.fit(X_final, np.log1p(y), verbose=False)
        deploy.save_model(out/'deploy_models'/f'{target}.xgboost_A_B_C_D_E.json')
        manifest['targets'][target]['deploy_model']={
            'path':str((out/'deploy_models'/f'{target}.xgboost_A_B_C_D_E.json').resolve()),
            'features':final_features,
            'trained_on_all_profile25_rows':int(len(y)),
            'note':'deployment model trained after OOF evaluation; do not use it to report generalization on the same data',
        }
    summary=pd.DataFrame(rows)
    summary['MAPE_improvement_from_previous']=summary.groupby('target').MAPE.diff().mul(-1)
    summary['MAPE_improvement_vs_A']=summary.groupby('target').MAPE.transform('first')-summary.MAPE
    summary.to_csv(out/'xgboost_cumulative_group_metrics.csv', index=False)
    pd.DataFrame(foldrows).to_csv(out/'fold_metrics.csv', index=False)
    pd.concat(matrix_rows, ignore_index=True).to_csv(out/'per_matrix_metrics.csv', index=False)
    pd.DataFrame(timing).to_csv(out/'timing.csv', index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(out/'oof_predictions.csv.gz', index=False, compression='gzip')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':10})
    fig,ax=plt.subplots(figsize=(7.4,4.2))
    for target,q in summary.groupby('target',sort=False):
        ax.plot(q.stage, q.MAPE, marker='o', lw=1.8, label=target.replace('_','-'))
    ax.set_xlabel('Cumulative feature groups')
    ax.set_ylabel('OOF MAPE (%)')
    ax.grid(True,lw=.4,alpha=.4)
    ax.legend(frameon=False,ncol=2)
    fig.tight_layout()
    fig.savefig(out/'xgboost_cumulative_group_mape.pdf')
    fig.savefig(out/'xgboost_cumulative_group_mape.png',dpi=300)
    plt.close(fig)
    manifest['wall_seconds']=time.perf_counter()-wall
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__':
    main()
