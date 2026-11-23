#!/usr/bin/env python3
"""PCA + supervised HGB feature evidence on frozen H100 v4.1 datasets."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from run_v4_1_flops_bytes_ablation import TARGETS, FEATURES, OPS, BYTES, SEED, values, metric


def model(seed):
    return HistGradientBoostingRegressor(loss='squared_error', max_iter=250, learning_rate=.06,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1., random_state=seed)


def groups(features):
    nnz=[c for c in features if c.startswith('replay_nnz_')]
    stats=[c for c in features if c not in nnz and c not in (OPS,BYTES,'replay_main_dimension')]
    return {'FLOPs':[OPS], 'Bytes':[BYTES], 'FLOPs_Bytes':[OPS,BYTES],
            'NNZ':nnz, 'Row_or_col_structure':stats, 'Size':['replay_main_dimension']}


def mape(y,p): return float((np.abs(np.maximum(p,0)-y)/np.maximum(np.abs(y),1e-6)*100).mean())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--fold-map',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    src,out=Path(a.input_dir),Path(a.output)
    if out.exists() and any(out.iterdir()): raise SystemExit(f'output must be empty: {out}')
    (out/'figures').mkdir(parents=True)
    fm0=pd.read_csv(a.fold_map); fm=dict(zip(fm0.matrix_id.astype(str),fm0.cv_fold.astype(int)))
    pca_summary=[]; loadings=[]; permutation=[]; ablation=[]; timing=[]
    manifest={'schema':'h100_v4.1_feature_evidence','seed':SEED,'cv':'matrix-grouped 5-fold','shap':'not run: package unavailable','targets':{}}
    wall=time.perf_counter()
    for ti,target in enumerate(TARGETS):
        feats=FEATURES[target]+[OPS,BYTES]; req=set(['task_id','matrix_id','label_time_ms','hardware_id']+feats)
        d=pd.read_csv(src/f'{target}.model_ready_v4.csv',usecols=lambda c:c in req,low_memory=False)
        y=pd.to_numeric(d.label_time_ms,errors='coerce').to_numpy(float); ok=np.isfinite(y)&(y>0); d=d.loc[ok].reset_index(drop=True); y=y[ok]
        fid=d.matrix_id.astype(str).map(fm)
        if fid.isna().any() or set(fid.astype(int))!={1,2,3,4,5}: raise RuntimeError(f'{target}: bad folds')
        fid=fid.to_numpy(int); X=values(d,feats)
        manifest['targets'][target]={'rows':len(d),'matrices':int(d.matrix_id.nunique()),'features':feats,'groups':groups(feats)}

        # PCA describes the standardized log-feature geometry, never label importance.
        Xs=StandardScaler().fit_transform(X); p=PCA().fit(Xs); cum=np.cumsum(p.explained_variance_ratio_)
        n90=int(np.searchsorted(cum,.90)+1); n95=int(np.searchsorted(cum,.95)+1)
        pca_summary.append({'target':target,'n_features':len(feats),'pc90':n90,'pc95':n95,
                            'pc1_variance_pct':p.explained_variance_ratio_[0]*100,
                            'pc2_variance_pct':p.explained_variance_ratio_[1]*100,
                            'effective_rank_95_fraction':n95/len(feats)})
        for j in range(len(feats)):
            for k in range(len(feats)):
                loadings.append({'target':target,'feature':feats[j],'component':k+1,
                                 'loading':p.components_[k,j],'explained_variance_ratio':p.explained_variance_ratio_[k]})
        show=min(8,len(feats)); fig,ax=plt.subplots(figsize=(8,max(4,.32*len(feats))))
        sns.heatmap(pd.DataFrame(p.components_[:show].T,index=feats,columns=[f'PC{i+1}' for i in range(show)]),center=0,cmap='vlag',ax=ax)
        ax.set_title(f'{target}: standardized log-feature PCA loadings'); fig.tight_layout(); fig.savefig(out/'figures'/f'{target}_pca_loadings.pdf'); plt.close(fig)
        fig,ax=plt.subplots(figsize=(max(6,.42*len(feats)),max(5,.38*len(feats))))
        sns.heatmap(pd.DataFrame(Xs,columns=feats).corr(),vmin=-1,vmax=1,center=0,cmap='vlag',ax=ax)
        ax.set_title(f'{target}: feature correlation'); fig.tight_layout(); fig.savefig(out/'figures'/f'{target}_correlation.pdf'); plt.close(fig)

        # Strict held-out permutation: train once per fold, permute test columns only.
        base_oof=np.empty(len(y)); perm_oof={f:np.empty(len(y)) for f in feats}; rng=np.random.default_rng(SEED+ti)
        for fold in range(1,6):
            tr=fid!=fold; te=fid==fold; t0=time.perf_counter(); mdl=model(SEED+ti*100+fold).fit(X[tr],np.log1p(y[tr])); fit=time.perf_counter()-t0
            t0=time.perf_counter(); base_oof[te]=np.expm1(mdl.predict(X[te])); pred=time.perf_counter()-t0
            for j,f in enumerate(feats):
                xp=X[te].copy(); xp[:,j]=xp[rng.permutation(len(xp)),j]; perm_oof[f][te]=np.expm1(mdl.predict(xp))
            timing.append({'target':target,'stage':'permutation','fold':fold,'fit_seconds':fit,'predict_base_seconds':pred})
        base_mape=mape(y,base_oof)
        for f in feats:
            pm=mape(y,perm_oof[f]); permutation.append({'target':target,'feature':f,'baseline_MAPE':base_mape,'permuted_MAPE':pm,'MAPE_increase_points':pm-base_mape,'relative_increase_pct':(pm-base_mape)/base_mape*100})

        # Grouped deletion ablation with identical folds and HGB hyperparameters.
        g=groups(feats); variants={'All':feats}
        for name,removed in g.items(): variants[f'Drop_{name}']=[f for f in feats if f not in removed]
        for vi,(name,cols) in enumerate(variants.items()):
            idx=[feats.index(f) for f in cols]; oof=np.empty(len(y)); fit_sum=0.; pred_sum=0.
            for fold in range(1,6):
                tr=fid!=fold; te=fid==fold; t0=time.perf_counter(); mdl=model(SEED+ti*1000+vi*10+fold).fit(X[tr][:,idx],np.log1p(y[tr])); fit_sum+=time.perf_counter()-t0
                t0=time.perf_counter(); oof[te]=np.expm1(mdl.predict(X[te][:,idx])); pred_sum+=time.perf_counter()-t0
            ablation.append({'target':target,'variant':name,'removed_group':'None' if name=='All' else name[5:],
                             'n_features':len(cols),**metric(y,oof),'fit_seconds':fit_sum,'predict_seconds':pred_sum})
            print(target,name,f'MAPE={mape(y,oof):.3f}',flush=True)

        # Compact permutation chart.
        q=pd.DataFrame([r for r in permutation if r['target']==target]).sort_values('MAPE_increase_points').tail(12)
        fig,ax=plt.subplots(figsize=(7,max(3,.32*len(q)))); ax.barh(q.feature,q.MAPE_increase_points); ax.axvline(0,color='black',lw=.7)
        ax.set_xlabel('OOF MAPE increase after held-out permutation (points)'); ax.set_title(f'{target}: HGB permutation importance'); fig.tight_layout(); fig.savefig(out/'figures'/f'{target}_permutation_importance.pdf'); plt.close(fig)

    ps=pd.DataFrame(pca_summary); ps.to_csv(out/'pca_summary.csv',index=False); pd.DataFrame(loadings).to_csv(out/'pca_loadings.csv',index=False)
    pi=pd.DataFrame(permutation); pi.to_csv(out/'permutation_importance.csv',index=False)
    ab=pd.DataFrame(ablation); ab.to_csv(out/'grouped_ablation.csv',index=False); pd.DataFrame(timing).to_csv(out/'timing.csv',index=False)
    # Evidence summary relative to the independently refitted All variant.
    base=ab[ab.variant=='All'].set_index('target').MAPE
    ev=ab[ab.variant!='All'].copy(); ev['MAPE_increase_vs_All']=ev.apply(lambda r:r.MAPE-base[r.target],axis=1); ev.to_csv(out/'grouped_ablation_effects.csv',index=False)
    manifest['wall_seconds']=time.perf_counter()-wall; (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
