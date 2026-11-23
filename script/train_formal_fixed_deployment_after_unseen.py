#!/usr/bin/env python3
"""Stage 3 gate: retrain fixed full-data deployment models only after unseen validation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from xgboost import XGBRegressor
from compare_h100_rtx_xgboost_matched_80_20 import TARGETS,read_target
from run_v4_1_flops_bytes_ablation import values
from run_v4_2_paper_order_max_first_ablation import groups

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--h100',required=True);p.add_argument('--rtx',required=True)
 p.add_argument('--stage1',required=True);p.add_argument('--stage2-h100',required=True)
 p.add_argument('--stage2-rtx',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 stage1,out=Path(a.stage1),Path(a.output)
 if out.exists() and any(out.iterdir()):raise SystemExit(f'output must be empty: {out}')
 validations={'H100':Path(a.stage2_h100),'RTX_PRO_6000':Path(a.stage2_rtx)}
 for platform,path in validations.items():
  m=json.loads((path/'manifest.json').read_text())
  if m.get('platform')!=platform:raise RuntimeError(f'stage2 platform mismatch: {platform}')
  if set(m.get('targets',{}))!=set(TARGETS):raise RuntimeError(f'incomplete stage2: {platform}')
  if any(v.get('overlap') for v in m['targets'].values()):raise RuntimeError(f'nonzero overlap: {platform}')
  if not (path/'unseen_task_and_macro_metrics.csv').is_file():raise RuntimeError(f'missing metrics: {platform}')
 config=json.loads((stage1/'frozen_stage1_configuration.json').read_text());params=config['fixed_xgboost_params']
 roots={'H100':Path(a.h100),'RTX_PRO_6000':Path(a.rtx)};out.mkdir(parents=True)
 manifest={'schema':'formal_fixed_deployment_v1','fixed_xgboost_params':params,
  'external_validation_data_included':False,'reported_accuracy_source':'stage1 OOF and stage2 unseen','platforms':{}}
 for platform,root in roots.items():
  manifest['platforms'][platform]={}
  for ti,target in enumerate(TARGETS):
   features=config['platforms'][platform][target]['features'];_,derived=groups(target)
   d=read_target(root,target,features,derived);x=pd.DataFrame(values(d,features),columns=features);y=d.label_time_ms.to_numpy(float)
   model=XGBRegressor(**params,random_state=config['seed']+ti*100);model.fit(x,np.log1p(y),verbose=False)
   name=f'{target}.{platform}.deployment.json';model.save_model(out/name)
   manifest['platforms'][platform][target]={'rows':len(d),'features':features,'model':name}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
