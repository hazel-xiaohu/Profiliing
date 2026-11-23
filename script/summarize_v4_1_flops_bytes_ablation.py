#!/usr/bin/env python3
"""Create compact tables, figures, and a reproducibility report for the v4.1 ablation."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',required=True); a=p.parse_args()
    root=Path(a.experiment); d=pd.read_csv(root/'model_comparison.csv'); t=pd.read_csv(root/'timing.csv')
    targets=['GETRF','GESSM','TSTRF','SSSSM_dense']
    key=d[d.model.isin(['Ridge','HGB','EBM']) & d.feature_group.isin(['Structure_only','FLOPs_Bytes','All_minus_FB','All'])]
    key.pivot_table(index=['target','model'],columns='feature_group',values='MAPE').reset_index().to_csv(root/'paper_mape_table.csv',index=False)
    best=d.loc[d.groupby('target').MAPE.idxmin(),['target','model','feature_group','MAPE','MdAPE','P95_APE','R2']]
    best.to_csv(root/'best_by_kernel.csv',index=False)
    evidence=pd.read_csv(root/'flops_bytes_evidence.csv')

    plt.rcParams.update({'font.family':'serif','font.size':10})
    fig,ax=plt.subplots(figsize=(7.2,3.5))
    q=evidence[evidence.model.isin(['HGB','EBM'])].copy()
    x=np.arange(len(targets)); width=.34
    for j,m in enumerate(['HGB','EBM']):
        z=q[q.model==m].set_index('target').loc[targets]
        ax.bar(x+(j-.5)*width,z.relative_MAPE_reduction_pct,width,label=m)
    ax.axhline(0,color='black',lw=.7); ax.set_xticks(x,targets); ax.set_ylabel('MAPE reduction after adding FLOPs/Bytes (%)')
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(root/'flops_bytes_mape_reduction.pdf'); fig.savefig(root/'flops_bytes_mape_reduction.png',dpi=220); plt.close(fig)

    hgb=d[d.model=='HGB'].pivot(index='target',columns='feature_group',values='MAPE').loc[targets]
    roof=d[d.model.isin(['Analytical_Roofline','Roofline_HGB_residual'])].pivot(index='target',columns='model',values='MAPE').loc[targets]
    ev=evidence[evidence.model.isin(['HGB','EBM'])]
    improvements=int((ev.MAPE_reduction_points>0).sum())
    total=len(ev)
    timing=t.groupby(['target','model','feature_group'],as_index=False).agg(fit_seconds=('fit_seconds','sum'),predict_seconds=('predict_seconds','sum'),predict_us_per_sample=('predict_us_per_sample','mean'))
    timing.to_csv(root/'timing_summary.csv',index=False)
    manifest=json.loads((root/'manifest.json').read_text())
    lines=[
      '# H100 v4.1 FLOPs/Bytes cross-model ablation', '',
      '## Experimental protocol','',
      '- Data: the frozen H100 `model_ready_v4_1_n1_n8_balanced_gt` sets.',
      '- Samples: GETRF 8,265; GESSM 25,996; TSTRF 25,996; SSSSM-dense 124,739, from 24 matrices.',
      '- Validation: the same deterministic matrix-grouped five-fold split used by the formal v4.1 model; no matrix occurs in both train and test within a fold.',
      '- Targets: positive measured kernel time (`label_time_ms`). All task features are available before target-kernel execution.',
      '- Model families: Ridge, direct HGB, additive EBM, nonnegative analytical Roofline, and Roofline+HGB residual.',
      '- Feature groups: size; structure; FLOPs; bytes; FLOPs+bytes; all v4.1 structural features without FLOPs/bytes; and all features.', '',
      '## Main result','',
      f'Adding FLOPs and bytes to the full structural representation reduced OOF MAPE in {improvements}/{total} nonlinear model/kernel comparisons (HGB and EBM).',
      '', '| Kernel | HGB Structure | HGB F+B | HGB All−F/B | HGB All | Roofline | Roofline+HGB |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for k in targets:
        lines.append(f"| {k} | {hgb.loc[k,'Structure_only']:.2f} | {hgb.loc[k,'FLOPs_Bytes']:.2f} | {hgb.loc[k,'All_minus_FB']:.2f} | {hgb.loc[k,'All']:.2f} | {roof.loc[k,'Analytical_Roofline']:.2f} | {roof.loc[k,'Roofline_HGB_residual']:.2f} |")
    lines += ['', 'All entries are out-of-fold MAPE (%).', '', '## Interpretation','',
      'The experiment supports using FLOPs and estimated bytes as the physical baseline variables, but it does **not** support claiming that they alone dominate sparse-kernel prediction. FLOPs+bytes alone are substantially weaker than structure-aware HGB for GETRF, GESSM, and TSTRF. Conversely, adding them to structural features consistently improves nonlinear models, with the strongest HGB gain for SSSSM-dense. The analytical Roofline baseline is deliberately low-dimensional and interpretable; its large standalone error shows that irregular sparse structure and implementation effects must be learned by the residual model.', '',
      'Therefore the defensible claim is: *FLOPs and bytes encode the first-order compute and data-movement demand, while pre-execution sparse-structure descriptors explain kernel-specific deviations from that physical baseline.* It would be too strong to call FLOPs/bytes the universally most predictive standalone feature group.', '',
      'Ridge is retained as a linear stress test; its sensitivity to correlated count features and inability to represent nonlinear regime changes makes it unsuitable as the principal evidence. EBM is an additive, interpretable nonlinear check. HGB and the already frozen XGBoost residual are the high-capacity tree checks.', '',
      '## Reproduction','', '```bash',
      'python3 script/run_v4_1_flops_bytes_ablation.py \\',
      '  --input-dir artifacts/platforms/H100_SXM_80GB/model_ready_v4_1_n1_n8_balanced_gt \\',
      '  --fold-map artifacts/online_safe_v3/experiments_grouped_5fold_v3/splits/shared_matrix_folds_v3.csv \\',
      f'  --output {root}', '```', '', f"Total experiment wall time: {manifest['wall_seconds']:.1f} s."]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
