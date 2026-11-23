#!/usr/bin/env python3
"""Plot H100 v4.1 task samples in intensity-throughput and intensity-time spaces."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.patches import Patch

TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense")
OPS="replay_operation_count_v2"; BYTES="replay_base_bytes_est"; TIME="label_time_ms"

def args():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--output',required=True);return p.parse_args()

def main():
 a=args();src=Path(a.input_dir);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 frames=[]
 for target in TARGETS:
  d=pd.read_csv(src/f'{target}.model_ready_v4.csv',usecols=['task_id','matrix_id','block_size',OPS,BYTES,TIME],low_memory=False)
  for c in ['block_size',OPS,BYTES,TIME]:d[c]=pd.to_numeric(d[c],errors='coerce')
  d=d[np.isfinite(d[OPS])&np.isfinite(d[BYTES])&np.isfinite(d[TIME])&(d[OPS]>0)&(d[BYTES]>0)&(d[TIME]>0)].copy()
  d['kernel']=target;d['operation_intensity_ops_per_byte']=d[OPS]/d[BYTES];d['time_s']=d[TIME]/1000.;d['throughput_ops_per_s']=d[OPS]/d.time_s
  frames.append(d)
 points=pd.concat(frames,ignore_index=True);points.to_csv(out/'roofline_scatter_points.csv.gz',index=False,compression='gzip')
 plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':10,'axes.titlesize':11})
 bs=points.block_size.to_numpy(float);vmin=max(float(np.nanmin(bs[bs>0])),1);vmax=float(np.nanmax(bs));norm=LogNorm(vmin=vmin,vmax=vmax);cmap='viridis'
 specs=[('throughput_ops_per_s','Achieved throughput (operations/s)','intensity_vs_throughput'),('time_s','Measured kernel time (s)','intensity_vs_time')]
 for ycol,ylabel,stem in specs:
  fig,axs=plt.subplots(2,2,figsize=(8.2,6.3),sharex=False,sharey=False,constrained_layout=True)
  artist=None
  for ax,target in zip(axs.flat,TARGETS):
   q=points[points.kernel==target];artist=ax.scatter(q.operation_intensity_ops_per_byte,q[ycol],c=q.block_size,s=5,alpha=.28,cmap=cmap,norm=norm,edgecolors='none',rasterized=True)
   ax.set_xscale('log');ax.set_yscale('log');ax.set_title(target.replace('_','-'));ax.grid(True,which='both',lw=.35,alpha=.35)
   ax.set_xlabel('Operational intensity (operations/byte)');ax.set_ylabel(ylabel)
  cb=fig.colorbar(artist,ax=axs.ravel().tolist(),pad=.02,shrink=.92);cb.set_label('Block size')
  fig.savefig(out/f'{stem}.pdf',dpi=300);fig.savefig(out/f'{stem}.png',dpi=300);plt.close(fig)

 # Operations versus time. Nearby byte counts share one discrete logarithmic
 # byte band, and the boundaries are common to all four kernels.
 positive_bytes=points[BYTES].to_numpy(float)
 log_lo=float(np.floor(np.log10(positive_bytes.min())*2)/2)
 log_hi=float(np.ceil(np.log10(positive_bytes.max())*2)/2)
 boundaries=np.power(10.0,np.linspace(log_lo,log_hi,9))
 byte_cmap=plt.get_cmap('viridis',8)
 byte_norm=BoundaryNorm(boundaries,ncolors=byte_cmap.N,clip=True)
 fig,axs=plt.subplots(2,2,figsize=(9.4,6.3),constrained_layout=True)
 for ax,target in zip(axs.flat,TARGETS):
  q=points[points.kernel==target]
  ax.scatter(q[OPS],q.time_s,c=q[BYTES],s=5,alpha=.28,cmap=byte_cmap,norm=byte_norm,edgecolors='none',rasterized=True)
  ax.set_xscale('log');ax.set_yscale('log');ax.set_title(target.replace('_','-'));ax.grid(True,which='both',lw=.35,alpha=.35)
  ax.set_xlabel('Operation count');ax.set_ylabel('Measured kernel time (s)')
 def compact(v):
  if v>=1e9:return f'{v/1e9:.1g} GB'
  if v>=1e6:return f'{v/1e6:.1g} MB'
  if v>=1e3:return f'{v/1e3:.1g} KB'
  return f'{v:.1g} B'
 handles=[Patch(facecolor=byte_cmap(i),label=f'{compact(boundaries[i])}–{compact(boundaries[i+1])}') for i in range(8)]
 fig.legend(handles=handles,title='Estimated bytes band',loc='center right',bbox_to_anchor=(1.0,.5),frameon=False)
 fig.set_constrained_layout_pads(w_pad=7/72,h_pad=2/72,wspace=.08,hspace=.08)
 fig.savefig(out/'operations_vs_time_colored_by_bytes.pdf',dpi=300);fig.savefig(out/'operations_vs_time_colored_by_bytes.png',dpi=300);plt.close(fig)
 pd.DataFrame({'lower_bytes':boundaries[:-1],'upper_bytes':boundaries[1:]}).to_csv(out/'byte_color_bins.csv',index=False)

 # Bytes versus time, with nearby operation counts assigned to a common
 # discrete logarithmic band shared by all four kernels.
 positive_ops=points[OPS].to_numpy(float)
 ops_log_lo=float(np.floor(np.log10(positive_ops.min())*2)/2)
 ops_log_hi=float(np.ceil(np.log10(positive_ops.max())*2)/2)
 ops_boundaries=np.power(10.0,np.linspace(ops_log_lo,ops_log_hi,9))
 ops_cmap=plt.get_cmap('plasma',8)
 ops_norm=BoundaryNorm(ops_boundaries,ncolors=ops_cmap.N,clip=True)
 fig,axs=plt.subplots(2,2,figsize=(9.4,6.3),constrained_layout=True)
 for ax,target in zip(axs.flat,TARGETS):
  q=points[points.kernel==target]
  ax.scatter(q[BYTES],q.time_s,c=q[OPS],s=5,alpha=.28,cmap=ops_cmap,norm=ops_norm,edgecolors='none',rasterized=True)
  ax.set_xscale('log');ax.set_yscale('log');ax.set_title(target.replace('_','-'));ax.grid(True,which='both',lw=.35,alpha=.35)
  ax.set_xlabel('Estimated bytes');ax.set_ylabel('Measured kernel time (s)')
 def compact_ops(v):
  if v>=1e12:return f'{v/1e12:.1g} T'
  if v>=1e9:return f'{v/1e9:.1g} G'
  if v>=1e6:return f'{v/1e6:.1g} M'
  if v>=1e3:return f'{v/1e3:.1g} K'
  return f'{v:.1g}'
 ops_handles=[Patch(facecolor=ops_cmap(i),label=f'{compact_ops(ops_boundaries[i])}–{compact_ops(ops_boundaries[i+1])}') for i in range(8)]
 fig.legend(handles=ops_handles,title='Operation-count band',loc='center right',bbox_to_anchor=(1.0,.5),frameon=False)
 fig.set_constrained_layout_pads(w_pad=7/72,h_pad=2/72,wspace=.08,hspace=.08)
 fig.savefig(out/'bytes_vs_time_colored_by_operations.pdf',dpi=300);fig.savefig(out/'bytes_vs_time_colored_by_operations.png',dpi=300);plt.close(fig)
 pd.DataFrame({'lower_operations':ops_boundaries[:-1],'upper_operations':ops_boundaries[1:]}).to_csv(out/'operation_color_bins.csv',index=False)
 summary=points.groupby('kernel').agg(samples=('task_id','size'),matrices=('matrix_id','nunique'),intensity_min=('operation_intensity_ops_per_byte','min'),intensity_median=('operation_intensity_ops_per_byte','median'),intensity_max=('operation_intensity_ops_per_byte','max'),throughput_median=('throughput_ops_per_s','median'),time_s_median=('time_s','median')).reset_index()
 summary.to_csv(out/'roofline_scatter_summary.csv',index=False)

if __name__=='__main__':main()
