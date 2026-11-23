#!/usr/bin/env python3
"""Strictly join CPU feature replay catalogs to successful GPU task catalogs."""
import argparse, csv, json
from collections import Counter
from pathlib import Path

KERNELS=("GETRF","GESSM","TSTRF","SSSSM")
CONFIG=("matrix_id","kernel_type","level","block_row","block_col","update_k","rank",
        "block_size","process_rows","process_cols")
NNZ=("nnz_input_0","nnz_input_1","nnz_input_2","nnz_output_0","nnz_output_1")
HASH_V2="canonical_structure_hash_v2"
HASH_V1="structure_hash"
PROVENANCE=("forced_ssssm_variant","repeat_index")


def args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--campaign',required=True)
 p.add_argument('--replay',required=True)
 p.add_argument('--matrix-list',required=True)
 p.add_argument('--output',required=True)
 return p.parse_args()

def read_rows(path):
 with path.open(newline='') as f: return list(csv.DictReader(f))

def key(row, variant, repeat):
 return tuple([row['task_id']]+[row[x] for x in CONFIG]+[variant,str(repeat)])

def main():
 a=args(); campaign=Path(a.campaign).resolve(); replay=Path(a.replay).resolve(); out=Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=True)
 wanted={Path(x.strip()).resolve() for x in Path(a.matrix_list).read_text().splitlines() if x.strip()}
 gpu={}; gpu_dups=[]; matrix_slugs=set()
 for marker in sorted((campaign/'raw').glob('*/bs_*/*/repeat_*/_SUCCESS.json')):
  meta=json.loads(marker.read_text()); matrix=Path(meta['matrix']).resolve()
  if matrix not in wanted: continue
  variant=meta['forced_ssssm_variant']; repeat=meta['repeat_index']; matrix_slugs.add(marker.parents[3].name)
  for kernel, names in meta['catalogs'].items():
   if kernel not in KERNELS: continue
   for name in names:
    path=Path(name)
    if not path.exists(): path=marker.parent/'catalog'/Path(name).name
    for row in read_rows(path):
     k=key(row,variant,repeat)
     if k in gpu: gpu_dups.append(k)
     gpu[k]=(row,meta)
 replay_base={}; replay_dups=[]
 for slug in sorted(matrix_slugs):
  for path in sorted((replay/slug).glob('bs_*/catalog/*.catalog.csv')):
   for row in read_rows(path):
    base=tuple([row['task_id']]+[row[x] for x in CONFIG])
    if base in replay_base: replay_dups.append(base)
    replay_base[base]=row
 rejected=[]; merged=[]; missing=Counter(); mismatch=Counter(); hash_checks=Counter()
 if gpu_dups: rejected.extend({'reason':'duplicate_gpu_key','key':repr(k)} for k in gpu_dups)
 if replay_dups: rejected.extend({'reason':'duplicate_replay_key','key':repr(k)} for k in replay_dups)
 for k,(grow,meta) in gpu.items():
  base=k[:-2]; rrow=replay_base.get(base)
  if rrow is None:
   missing[grow['kernel_type']]+=1; rejected.append({'reason':'missing_replay','key':repr(k)}); continue
  hash_field = HASH_V2 if grow.get(HASH_V2) and rrow.get(HASH_V2) else HASH_V1
  hash_checks[hash_field]+=1
  bad=[hash_field] if grow.get(hash_field)!=rrow.get(hash_field) else []
  bad.extend(f for f in NNZ if grow.get(f)!=rrow.get(f))
  if bad:
   mismatch[grow['kernel_type']]+=1
   rejected.append({'reason':'structure_mismatch','key':repr(k),'fields':'|'.join(bad),
                    **{f'gpu_{f}':grow.get(f,'') for f in bad},**{f'replay_{f}':rrow.get(f,'') for f in bad}})
   continue
  result=dict(grow)
  result.update({f'replay_{name}':value for name,value in rrow.items() if name not in CONFIG and name!='task_id'})
  result.update({'forced_ssssm_variant':meta['forced_ssssm_variant'],'repeat_index':str(meta['repeat_index']),
                 'matrix_path':meta['matrix'],'gpu_run_id':meta['run_id'],
                 'hardware_id':meta.get('hardware_id','UNSPECIFIED'),
                 'gpu_model':meta.get('gpu_model','')})
  merged.append(result)
 gpu_bases={k[:-2] for k in gpu}
 orphan=sum(base not in gpu_bases for base in replay_base)
 def write(name,rows):
  path=out/name
  fields=[]
  for row in rows:
   for x in row:
    if x not in fields: fields.append(x)
  with path.open('w',newline='') as f:
   if fields:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
  return path
 write('merged_tasks.csv',merged); write('rejected_tasks.csv',rejected)
 for kernel in KERNELS: write(f'{kernel}.csv',[x for x in merged if x['kernel_type']==kernel])
 summary={'gpu_rows':len(gpu),'replay_base_rows':len(replay_base),'accepted_rows':len(merged),'rejected_rows':len(rejected),
          'gpu_duplicate_keys':len(gpu_dups),'replay_duplicate_keys':len(replay_dups),'orphan_replay_rows':orphan,
          'missing_replay_by_kernel':dict(missing),'structure_mismatch_by_kernel':dict(mismatch),
          'hash_checks':dict(hash_checks),
          'hardware_ids':sorted({meta.get('hardware_id','UNSPECIFIED') for _,meta in gpu.values()}),
          'strict_pass':not rejected and orphan==0}
 (out/'merge_audit.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
 if not summary['strict_pass']: raise SystemExit(2)
if __name__=='__main__': main()
