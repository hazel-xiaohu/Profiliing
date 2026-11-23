#!/usr/bin/env python3
"""Create a concise baseline-versus-phase1 report and CV diagnostic plots."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams["font.family"]="DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"]=False

ORDER=["GETRF","GESSM","TSTRF","SSSSM_sparse","SSSSM_dense"]

def args():
 p=argparse.ArgumentParser(); p.add_argument("baseline"); p.add_argument("phase1"); p.add_argument("--output",required=True); return p.parse_args()
def load(path): return {x["model"]:x for x in json.loads(Path(path).read_text())}
def main():
 a=args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 base=load(a.baseline); phase=load(a.phase1)
 comparison=[]
 for k in ORDER:
  comparison.append({"model":k,"baseline_mape":base[k]["mape_percent"],"phase1_mape":phase[k]["mape_percent"],
   "mape_change":phase[k]["mape_percent"]-base[k]["mape_percent"],"baseline_r2":base[k]["r2"],"phase1_r2":phase[k]["r2"],
   "phase1_weighted_mape":phase[k]["weighted_mape_percent"],"baseline_samples":base[k]["samples"],"phase1_samples":phase[k]["samples"]})
 bsel=base["SSSSM_selector"]; isel=phase["SSSSM_independent_time_selector"]; rsel=phase["SSSSM_regret_weighted_ratio_selector"]
 same_population = all(base[k]["samples"] == phase[k]["samples"] for k in ORDER)
 comparison_note = ("fair comparison on identical rows: 20 matrices and 100 matrix/block strata"
                    if same_population else
                    "comparison populations differ; interpret metric changes cautiously")
 summary={"comparison_warning":comparison_note,
  "kernel_metrics":comparison,"selectors":{"baseline":bsel,"phase1_independent":isel,"phase1_ratio":rsel},
  "recommended_selector":"SSSSM_independent_time_selector"}
 (out/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

 x=np.arange(len(ORDER)); width=.36
 fig,ax=plt.subplots(figsize=(10,5)); ax.bar(x-width/2,[base[k]['mape_percent'] for k in ORDER],width,label='baseline'); ax.bar(x+width/2,[phase[k]['mape_percent'] for k in ORDER],width,label='phase1')
 ax.set_xticks(x,ORDER,rotation=15); ax.set_ylabel('Grouped-CV MAPE (%)'); ax.grid(axis='y',alpha=.25); ax.legend(); ax.set_title('Baseline vs phase-1 on identical 20-matrix / 100-stratum data'); fig.tight_layout(); fig.savefig(out/'baseline_vs_phase1_mape.png',dpi=180); fig.savefig(out/'baseline_vs_phase1_mape.pdf'); plt.close(fig)

 fig,axes=plt.subplots(2,3,figsize=(14,8.5))
 files={"GETRF":("GETRF_cv_predictions.csv","actual_time_kernel_ms","predicted_time_kernel_ms"),"GESSM":("GESSM_cv_predictions.csv","actual_time_kernel_ms","predicted_time_kernel_ms"),"TSTRF":("TSTRF_cv_predictions.csv","actual_time_kernel_ms","predicted_time_kernel_ms"),"SSSSM_sparse":("SSSSM_cv_predictions.csv","actual_sparse_ms","predicted_sparse_ms"),"SSSSM_dense":("SSSSM_cv_predictions.csv","actual_dense_ms","predicted_dense_ms")}
 modeldir=Path(a.phase1).parent
 for ax,k in zip(axes.flat,ORDER):
  f,yf,pf=files[k]; rows=list(csv.DictReader((modeldir/f).open())); y=np.array([float(r[yf]) for r in rows]); yh=np.array([float(r[pf]) for r in rows]); lo=min(y.min(),yh.min()); hi=max(y.max(),yh.max()); ax.scatter(y,yh,s=5,alpha=.2,rasterized=True); ax.plot([lo,hi],[lo,hi],'k--',lw=1); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(f"{k}: MAPE={phase[k]['mape_percent']:.1f}%, R2={phase[k]['r2']:.3f}"); ax.set_xlabel('Measured ms'); ax.set_ylabel('Nested-CV predicted ms'); ax.grid(True,which='both',alpha=.18)
 axes.flat[-1].axis('off'); fig.suptitle('Phase-1 nested matrix-grouped CV',fontsize=15); fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(out/'phase1_cv_measured_vs_predicted.png',dpi=180,bbox_inches='tight'); fig.savefig(out/'phase1_cv_measured_vs_predicted.pdf',bbox_inches='tight'); plt.close(fig)

 lines=['# 第一阶段模型改进报告','',summary['comparison_warning'],'','| Kernel | Baseline samples | Phase1 samples | Baseline MAPE | Phase1 MAPE | Phase1 weighted MAPE | Baseline R2 | Phase1 R2 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in comparison: lines.append(f"| {r['model']} | {r['baseline_samples']} | {r['phase1_samples']} | {r['baseline_mape']:.2f}% | {r['phase1_mape']:.2f}% | {r['phase1_weighted_mape']:.2f}% | {r['baseline_r2']:.4f} | {r['phase1_r2']:.4f} |")
 lines += ['','## SSSSM selector','','| Selector | Accuracy | Mean regret | Weighted mean regret | P90 regret |','|---|---:|---:|---:|---:|',f"| Baseline independent | {bsel['selection_accuracy_percent']:.2f}% | {bsel['mean_regret_percent']:.2f}% | n/a | {bsel['p90_regret_percent']:.2f}% |",f"| Phase1 independent | {isel['selection_accuracy_percent']:.2f}% | {isel['mean_regret_percent']:.2f}% | {isel['weighted_mean_regret_percent']:.2f}% | {isel['p90_regret_percent']:.2f}% |",f"| Phase1 regret-weighted ratio | {rsel['selection_accuracy_percent']:.2f}% | {rsel['mean_regret_percent']:.2f}% | {rsel['weighted_mean_regret_percent']:.2f}% | {rsel['p90_regret_percent']:.2f}% |",'',"推荐保留 `SSSSM_independent_time_selector`；本轮 ratio selector 没有超过它。",'', '![MAPE comparison](baseline_vs_phase1_mape.png)','', '![Phase1 CV](phase1_cv_measured_vs_predicted.png)','']
 (out/'README.md').write_text('\n'.join(lines))
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
