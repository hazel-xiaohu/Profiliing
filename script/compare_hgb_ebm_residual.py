#!/usr/bin/env python3
"""Compare phase-1 HGB, EBM, and Roofline-residual models."""
import argparse,json
from pathlib import Path
import joblib,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['font.family']='DejaVu Sans';matplotlib.rcParams['axes.unicode_minus']=False
ORDER=['GETRF','GESSM','TSTRF','SSSSM_sparse','SSSSM_dense']
def args():
 p=argparse.ArgumentParser();p.add_argument('hgb_metrics');p.add_argument('new_metrics');p.add_argument('--models',required=True);p.add_argument('--output',required=True);return p.parse_args()
def main():
 a=args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);h={x['model']:x for x in json.loads(Path(a.hgb_metrics).read_text())};n={x['model']:x for x in json.loads(Path(a.new_metrics).read_text())};records=[]
 for k in ORDER:
  for family,key in [('HGB',k),('EBM',k+'_EBM'),('RooflineResidual',k+'_RooflineResidual')]:
   m=h[key] if family=='HGB' else n[key];records.append({'kernel':k,'family':family,**{q:m[q] for q in ('samples','mape_percent','median_ape_percent','p90_ape_percent','wape_percent','r2')}})
 selectors=[h['SSSSM_independent_time_selector'],n['SSSSM_EBM_selector'],n['SSSSM_RooflineResidual_selector']]
 data={'comparison_scope':'identical 20 matrices, 100 matrix/block strata, sample weights, 5-fold matrix-grouped CV','kernel_metrics':records,'selectors':selectors,'recommendation':{'relative_time_and_selector':'HGB','absolute_scale_and_long_tasks':'RooflineResidual','interpretability':'EBM or RooflineResidual'}}
 (out/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 x=np.arange(len(ORDER));width=.25;families=['HGB','EBM','RooflineResidual'];colors=['#4477AA','#CC6677','#228833']
 fig,axes=plt.subplots(1,3,figsize=(16,4.8))
 for ax,metric,title in zip(axes,['mape_percent','wape_percent','r2'],['MAPE (%)','WAPE (%)','R2']):
  for i,(fam,c) in enumerate(zip(families,colors)):
   vals=[next(r[metric] for r in records if r['kernel']==k and r['family']==fam) for k in ORDER];ax.bar(x+(i-1)*width,vals,width,label=fam,color=c)
  ax.set_xticks(x,ORDER,rotation=20);ax.set_title(title);ax.grid(axis='y',alpha=.22)
 axes[0].legend();fig.suptitle('HGB vs EBM vs Roofline+EBM residual (identical grouped-CV scope)');fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'model_metric_comparison.png',dpi=180,bbox_inches='tight');fig.savefig(out/'model_metric_comparison.pdf',bbox_inches='tight');plt.close(fig)
 # EBM global importance
 importance={};fig,axes=plt.subplots(2,3,figsize=(15,9))
 for ax,k in zip(axes.flat,ORDER):
  artifact=joblib.load(Path(a.models)/f'{k}_EBM.joblib');model=artifact['estimator'];names=list(model.term_names_);scores=np.asarray(model.term_importances());idx=np.argsort(scores)[-12:];importance[k]=[{"term":names[i],"importance":float(scores[i])} for i in idx[::-1]];ax.barh(np.arange(len(idx)),scores[idx],color='#4477AA');ax.set_yticks(np.arange(len(idx)),[names[i] for i in idx],fontsize=7);ax.set_title(k);ax.grid(axis='x',alpha=.2)
 axes.flat[-1].axis('off');fig.suptitle('EBM global term importance');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'ebm_global_importance.png',dpi=180,bbox_inches='tight');fig.savefig(out/'ebm_global_importance.pdf',bbox_inches='tight');plt.close(fig);(out/'ebm_global_importance.json').write_text(json.dumps(importance,ensure_ascii=False,indent=2)+'\n')
 lines=['# HGB、EBM 与 Roofline-residual 公平对比','',data['comparison_scope'],'','| Kernel | Model | MAPE | Median APE | P90 APE | WAPE | R2 |','|---|---|---:|---:|---:|---:|---:|']
 for r in records:lines.append(f"| {r['kernel']} | {r['family']} | {r['mape_percent']:.2f}% | {r['median_ape_percent']:.2f}% | {r['p90_ape_percent']:.2f}% | {r['wape_percent']:.2f}% | {r['r2']:.4f} |")
 lines+=['','## SSSSM selector','','| Model | Accuracy | Mean regret | Weighted mean regret | P90 regret |','|---|---:|---:|---:|---:|']
 for s in selectors:lines.append(f"| {s['model']} | {s['selection_accuracy_percent']:.2f}% | {s['mean_regret_percent']:.2f}% | {s.get('weighted_mean_regret_percent',float('nan')):.2f}% | {s['p90_regret_percent']:.2f}% |")
 lines+=['','## 结论','','- 相对时间误差和 SSSSM 部署选择：HGB 最优。','- 长任务绝对尺度、WAPE/R2：部分 kernel 的 Roofline-residual 更优。','- 可解释性：EBM 提供单特征/交互贡献，Roofline-residual 保留物理基线。','- 当前不建议用 EBM 或 Roofline-residual 全面替换 HGB；可按目标采用双模型报告。','','![metrics](model_metric_comparison.png)','','![importance](ebm_global_importance.png)','']
 (out/'README.md').write_text('\n'.join(lines))
 print(json.dumps(data,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
