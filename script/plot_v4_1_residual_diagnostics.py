#!/usr/bin/env python3
"""OOF analytical-baseline and residual-correction diagnostic figures."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt

TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense")
STRUCTURE={
 "GETRF":"replay_input_0_col_nnz_max",
 "GESSM":"replay_input_0_col_nnz_max",
 "TSTRF":"replay_input_0_row_nnz_max",
 "SSSSM_dense":"replay_input_0_col_nnz_max",
}
LABELS={
 "GETRF":r"max column NNZ of $A$",
 "GESSM":r"max column NNZ of $A$",
 "TSTRF":r"max row NNZ of $A$",
 "SSSSM_dense":r"max column NNZ of $A$",
}

def args():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--prediction-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--max-points',type=int,default=20000);return p.parse_args()

def binned(x,y,n=20):
 ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 edges=np.unique(np.quantile(x,np.linspace(0,1,n+1)))
 if len(edges)<3:return np.array([]),np.array([])
 bid=np.clip(np.digitize(x,edges[1:-1]),0,len(edges)-2);xx=[];yy=[]
 for i in range(len(edges)-1):
  q=bid==i
  if q.any():xx.append(np.median(x[q]));yy.append(np.median(y[q]))
 return np.asarray(xx),np.asarray(yy)

def main():
 a=args();data,preds,out=Path(a.data_dir),Path(a.prediction_dir),Path(a.output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(20260806);frames=[]
 for target in TARGETS:
  p=pd.read_csv(preds/f'{target}.oof.csv');f=STRUCTURE[target]
  d=pd.read_csv(data/f'{target}.model_ready_v4.csv',usecols=['task_id','matrix_id','block_size',f],low_memory=False)
  q=p.merge(d,on=['task_id','matrix_id','block_size'],how='left',validate='one_to_one')
  q['target']=target;q['structure_feature']=f;q['structure_value']=pd.to_numeric(q[f],errors='coerce')
  q['baseline_log_residual']=np.log1p(q.actual_ms)-np.log1p(q.roofline_ms)
  q['corrected_log_residual']=np.log1p(q.actual_ms)-np.log1p(q.roofline_xgboost_ms)
  frames.append(q)
 allq=pd.concat(frames,ignore_index=True);allq.to_csv(out/'oof_residual_diagnostics.csv.gz',index=False,compression='gzip')
 plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':10,'axes.titlesize':11})

 # Figure 1: measured time versus strictly out-of-fold analytical baseline.
 fig,axs=plt.subplots(2,2,figsize=(7.8,6.4),constrained_layout=True)
 for ax,target in zip(axs.flat,TARGETS):
  q=allq[allq.target==target];take=q if len(q)<=a.max_points else q.iloc[rng.choice(len(q),a.max_points,replace=False)]
  ax.scatter(take.roofline_ms/1000.,take.actual_ms/1000.,s=5,alpha=.25,edgecolors='none',rasterized=True)
  lo=max(min(q.roofline_ms.min(),q.actual_ms.min())/1000.,1e-9);hi=max(q.roofline_ms.max(),q.actual_ms.max())/1000.;ax.plot([lo,hi],[lo,hi],'k--',lw=1,label='ideal $y=x$')
  ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.grid(True,which='both',lw=.35,alpha=.35);ax.set_title(target.replace('_','-'))
  ax.set_xlabel('Analytical-baseline prediction (s)');ax.set_ylabel('Measured kernel time (s)')
 fig.savefig(out/'actual_vs_analytical_baseline_oof.pdf',dpi=300);fig.savefig(out/'actual_vs_analytical_baseline_oof.png',dpi=300);plt.close(fig)

 # Figure 2: systematic baseline residual and remaining post-correction error.
 fig,axs=plt.subplots(2,2,figsize=(8.4,6.4),constrained_layout=True)
 for ax,target in zip(axs.flat,TARGETS):
  q=allq[allq.target==target].copy();q=q[np.isfinite(q.structure_value)&(q.structure_value>=0)];take=q if len(q)<=a.max_points else q.iloc[rng.choice(len(q),a.max_points,replace=False)]
  x=np.log1p(take.structure_value.to_numpy(float));yb=take.baseline_log_residual.to_numpy(float);yc=take.corrected_log_residual.to_numpy(float)
  ax.scatter(x,yb,s=4,alpha=.10,color='#d95f02',edgecolors='none',rasterized=True,label='analytical baseline')
  ax.scatter(x,yc,s=4,alpha=.10,color='#1b9e77',edgecolors='none',rasterized=True,label='after residual correction')
  xb,mb=binned(np.log1p(q.structure_value.to_numpy(float)),q.baseline_log_residual.to_numpy(float));xc,mc=binned(np.log1p(q.structure_value.to_numpy(float)),q.corrected_log_residual.to_numpy(float))
  ax.plot(xb,mb,color='#d95f02',lw=2);ax.plot(xc,mc,color='#1b9e77',lw=2);ax.axhline(0,color='black',lw=.8,ls='--');ax.grid(True,lw=.35,alpha=.35);ax.set_title(target.replace('_','-'))
  ax.set_xlabel('log(1 + '+LABELS[target]+')');ax.set_ylabel(r'log residual: $\log(1+T)-\log(1+\hat T)$')
 handles,labels=ax.get_legend_handles_labels()
 fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False);fig.savefig(out/'residual_vs_structure_before_after_oof.pdf',dpi=300);fig.savefig(out/'residual_vs_structure_before_after_oof.png',dpi=300);plt.close(fig)

 summary=allq.groupby('target').agg(samples=('task_id','size'),baseline_log_residual_median=('baseline_log_residual','median'),baseline_log_residual_abs_median=('baseline_log_residual',lambda x:np.median(np.abs(x))),corrected_log_residual_median=('corrected_log_residual','median'),corrected_log_residual_abs_median=('corrected_log_residual',lambda x:np.median(np.abs(x)))).reset_index();summary.to_csv(out/'residual_diagnostic_summary.csv',index=False)
if __name__=='__main__':main()
