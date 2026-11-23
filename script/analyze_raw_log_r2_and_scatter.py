#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense")

def metrics(y,p):
    y=np.asarray(y,float); p=np.maximum(np.asarray(p,float),0)
    ape=np.abs(p-y)/np.maximum(np.abs(y),1e-6)*100
    return {
        'n':int(len(y)),
        'MAPE':float(np.mean(ape)),
        'MdAPE':float(np.median(ape)),
        'P95_APE':float(np.percentile(ape,95)),
        'R2_raw':float(r2_score(y,p)),
        'R2_log':float(r2_score(np.log1p(y),np.log1p(p))),
    }

def load_direct(direct_dir:Path, stage:str):
    d=pd.read_csv(direct_dir/'oof_predictions.csv.gz')
    d=d[d.stage.eq(stage)].copy()
    d['model']='Direct XGBoost'
    d=d.rename(columns={'predicted_ms':'pred_ms'})
    return d[['model','target','stage','task_id','matrix_id','block_size','actual_ms','pred_ms']]

def load_resid(resid_dir:Path, stage:str):
    d=pd.read_csv(resid_dir/'oof_predictions.csv.gz')
    d=d[d.stage.eq(stage)].copy()
    d['model']='Roofline+XGBoost residual'
    d=d.rename(columns={'predicted_ms':'pred_ms'})
    return d[['model','target','stage','task_id','matrix_id','block_size','actual_ms','pred_ms','roofline_ms']]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--direct-dir',required=True)
    ap.add_argument('--residual-dir',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--stage',default='A+B+C+D')
    a=ap.parse_args()
    direct_dir=Path(a.direct_dir); resid_dir=Path(a.residual_dir); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    direct=load_direct(direct_dir,a.stage)
    resid=load_resid(resid_dir,a.stage)
    frames=[direct,resid]
    rows=[]; matrix_rows=[]
    for df in frames:
        model=df.model.iloc[0]
        for target,q in df.groupby('target',sort=False):
            row={'model':model,'target':target,'stage':a.stage,**metrics(q.actual_ms,q.pred_ms)}
            rows.append(row)
            for matrix_id,mq in q.groupby('matrix_id',sort=True):
                matrix_rows.append({'model':model,'target':target,'matrix_id':matrix_id,'stage':a.stage,**metrics(mq.actual_ms,mq.pred_ms)})
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'r2_raw_log_summary.csv',index=False)
    pd.DataFrame(matrix_rows).to_csv(out/'r2_raw_log_by_matrix.csv',index=False)
    comp=summary.pivot(index='target',columns='model',values=['MAPE','MdAPE','P95_APE','R2_raw','R2_log'])
    comp.to_csv(out/'r2_raw_log_model_comparison_wide.csv')
    allp=pd.concat([direct.assign(roofline_ms=np.nan),resid],ignore_index=True,sort=False)
    allp.to_csv(out/'combined_oof_predictions_stage.csv.gz',index=False,compression='gzip')
    # plot predicted vs measured log-log: rows=model, columns=kernel
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':9})
    block_sizes=sorted(allp.block_size.dropna().unique())
    cmap=plt.get_cmap('viridis',len(block_sizes))
    color={bs:cmap(i) for i,bs in enumerate(block_sizes)}
    models=['Direct XGBoost','Roofline+XGBoost residual']
    fig,axes=plt.subplots(len(models),len(TARGETS),figsize=(13.2,6.2),sharex=False,sharey=False)
    for r,model in enumerate(models):
        for c,target in enumerate(TARGETS):
            ax=axes[r,c]
            q=allp[(allp.model.eq(model))&(allp.target.eq(target))]
            for bs,bq in q.groupby('block_size',sort=True):
                ax.scatter(bq.actual_ms,bq.pred_ms,s=5,alpha=0.35,color=color[bs],edgecolors='none')
            lo=max(min(q.actual_ms.min(),q.pred_ms[q.pred_ms>0].min())*0.8,1e-4)
            hi=max(q.actual_ms.max(),q.pred_ms.max())*1.2
            ax.plot([lo,hi],[lo,hi],color='black',lw=.8,ls='--')
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
            ax.grid(True,which='both',lw=.25,alpha=.35)
            if r==0: ax.set_title(target.replace('_','-'))
            if c==0: ax.set_ylabel(model+'\nPredicted ms')
            if r==len(models)-1: ax.set_xlabel('Measured ms')
    handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=color[bs],markersize=5,label=str(int(bs))) for bs in block_sizes]
    fig.legend(handles=handles,title='Block size',loc='upper center',bbox_to_anchor=(0.5,1.02),ncol=8,frameon=False)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out/'predicted_vs_measured_loglog_by_kernel_blocksize.png',dpi=300)
    fig.savefig(out/'predicted_vs_measured_loglog_by_kernel_blocksize.pdf')
    plt.close(fig)
    # residual variant also plot roofline baseline vs measured for context
    fig,axes=plt.subplots(1,len(TARGETS),figsize=(13.2,3.2))
    for c,target in enumerate(TARGETS):
        ax=axes[c]
        q=resid[resid.target.eq(target)]
        for bs,bq in q.groupby('block_size',sort=True):
            ax.scatter(bq.actual_ms,bq.roofline_ms,s=5,alpha=0.35,color=color[bs],edgecolors='none')
        lo=max(min(q.actual_ms.min(),q.roofline_ms[q.roofline_ms>0].min())*0.8,1e-4)
        hi=max(q.actual_ms.max(),q.roofline_ms.max())*1.2
        ax.plot([lo,hi],[lo,hi],color='black',lw=.8,ls='--')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_title(target.replace('_','-')); ax.grid(True,which='both',lw=.25,alpha=.35)
        if c==0: ax.set_ylabel('Roofline baseline ms')
        ax.set_xlabel('Measured ms')
    fig.legend(handles=handles,title='Block size',loc='upper center',bbox_to_anchor=(0.5,1.08),ncol=8,frameon=False)
    fig.tight_layout(rect=[0,0,1,0.92])
    fig.savefig(out/'roofline_baseline_vs_measured_loglog_by_kernel_blocksize.png',dpi=300)
    fig.savefig(out/'roofline_baseline_vs_measured_loglog_by_kernel_blocksize.pdf')
    plt.close(fig)
    manifest={'stage':a.stage,'direct_dir':str(direct_dir.resolve()),'residual_dir':str(resid_dir.resolve()),'outputs':['r2_raw_log_summary.csv','r2_raw_log_by_matrix.csv','predicted_vs_measured_loglog_by_kernel_blocksize.png','roofline_baseline_vs_measured_loglog_by_kernel_blocksize.png']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('output',out)
if __name__=='__main__': main()
