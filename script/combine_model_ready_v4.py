#!/usr/bin/env python3
"""Combine disjoint model-ready-v4 roots and validate identity/platform."""
import argparse, json
from pathlib import Path
import pandas as pd

TARGETS=("GETRF","GESSM","TSTRF","SSSSM_dense","SSSSM_sparse")
KEY=["task_id","matrix_id","kernel_type","level","block_row","block_col",
     "update_k","rank","block_size","process_rows","process_cols"]

p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True)
p.add_argument('--output',required=True);p.add_argument('--hardware-id',required=True)
p.add_argument('--block-sizes',nargs='+',type=int,required=True)
p.add_argument('--targets',nargs='+',choices=TARGETS,default=list(TARGETS))
p.add_argument('--exclude-matrices',nargs='*',default=[])
p.add_argument('--schema-version',choices=('v4','v4.1'),default='v4')
a=p.parse_args();out=Path(a.output)
if out.exists() and any(out.iterdir()):raise SystemExit(f'output must be empty: {out}')
out.mkdir(parents=True,exist_ok=True);audit={'schema':'combined_model_ready_v4','inputs':a.inputs,'hardware_id':a.hardware_id,'excluded_matrices':sorted(set(a.exclude_matrices)),'requested_targets':a.targets,'targets':{}}
for target in a.targets:
 frames=[]
 for root in map(Path,a.inputs):
  f=root/f'{target}.model_ready_v4.csv';d=pd.read_csv(f,low_memory=False);d['_source_root']=str(root.resolve());frames.append(d)
 cols=list(frames[0].columns)
 for i,d in enumerate(frames[1:],1):
  if set(d.columns)!=set(cols):raise RuntimeError(f'{target}: schema mismatch')
  frames[i]=d[cols]
 data=pd.concat(frames,ignore_index=True)
 data=data[~data.matrix_id.astype(str).isin(a.exclude_matrices)].copy()
 data=data[pd.to_numeric(data.block_size,errors='coerce').isin(a.block_sizes)].copy()
 if not data.replay_feature_schema_version.eq(a.schema_version).all():raise RuntimeError(f'{target}: expected schema {a.schema_version}')
 found=set(data.hardware_id.astype(str));
 if found!={a.hardware_id}:raise RuntimeError(f'{target}: hardware mismatch {found}')
 dup=data.duplicated(KEY,keep=False)
 if dup.any():raise RuntimeError(f'{target}: {int(dup.sum())} duplicate identity rows')
 sizes=sorted(pd.to_numeric(data.block_size).astype(int).unique().tolist())
 if sizes!=sorted(set(a.block_sizes)):raise RuntimeError(f'{target}: block sizes {sizes}')
 data.sort_values(['matrix_id','block_size','level','task_id'],kind='stable').to_csv(out/f'{target}.model_ready_v4.csv',index=False)
 audit['targets'][target]={'rows':len(data),'matrices':int(data.matrix_id.nunique()),'block_sizes':sizes,'rows_by_source':{str(Path(r).resolve()):len(frames[i]) for i,r in enumerate(a.inputs)}}
(out/'combine_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
