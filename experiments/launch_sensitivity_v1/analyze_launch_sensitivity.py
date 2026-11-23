#!/usr/bin/env python3
"""Strict paired analysis for launch_sensitivity_v1 output (stdlib only)."""
import argparse, csv, json, math, statistics
from collections import defaultdict
from pathlib import Path

KERNELS=("GETRF","GESSM","TSTRF","SSSSM")
KEY_FIELDS=("kernel_type","task_id","matrix_id","kernel_variant","level","block_row","block_col","update_k","rank","block_size","process_rows","process_cols")
CHECK_FIELDS=("structure_hash","nnz_input_0","nnz_input_1","nnz_input_2","nnz_output_0","nnz_output_1")

def num(x):
    try: return float(x)
    except (TypeError,ValueError): return math.nan

def percentile(xs,p):
    if not xs:return math.nan
    ys=sorted(xs); q=(len(ys)-1)*p; lo=int(q); hi=min(lo+1,len(ys)-1)
    return ys[lo]+(ys[hi]-ys[lo])*(q-lo)

def ranks(xs):
    order=sorted(range(len(xs)),key=lambda i:xs[i]); out=[0.0]*len(xs); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and xs[order[j]]==xs[order[i]]:j+=1
        r=(i+j-1)/2+1
        for k in range(i,j):out[order[k]]=r
        i=j
    return out

def spearman(x,y):
    if len(x)<2:return math.nan
    a,b=ranks(x),ranks(y); ma=statistics.mean(a); mb=statistics.mean(b)
    top=sum((u-ma)*(v-mb) for u,v in zip(a,b)); da=sum((u-ma)**2 for u in a); db=sum((v-mb)**2 for v in b)
    return top/math.sqrt(da*db) if da and db else math.nan

def read_config(root,config,kernel):
    path=root/"configs"/config/"measurements"/f"{kernel}.csv"
    if not path.exists():return []
    with path.open(newline="") as f:return list(csv.DictReader(f))

def key(row):return tuple(row.get(f,"") for f in KEY_FIELDS)

def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:
        t=num(r.get("time_kernel_ms"))
        if math.isfinite(t) and t>0:groups[key(r)].append((t,r))
    return {k:(statistics.median(vv for vv,_ in vs),vs[0][1],len(vs)) for k,vs in groups.items()}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("root"); p.add_argument("--baseline",default="baseline"); p.add_argument("--output")
    a=p.parse_args(); root=Path(a.root).resolve(); out=Path(a.output).resolve() if a.output else root/"analysis"; out.mkdir(parents=True,exist_ok=True)
    configs=sorted(p.name for p in (root/"configs").iterdir() if p.is_dir())
    if a.baseline not in configs:raise SystemExit(f"missing baseline config: {a.baseline}")
    summary=[]; rejects=[]; paired=[]
    for kernel in KERNELS:
        base=aggregate(read_config(root,a.baseline,kernel))
        for config in configs:
            if config==a.baseline:continue
            other=aggregate(read_config(root,config,kernel)); xb=[]; xo=[]; changes=[]; ratios=[]
            for k in sorted(set(base)&set(other)):
                bt,br,bn=base[k]; ot,orr,on=other[k]
                mismatch=[f for f in CHECK_FIELDS if br.get(f,"")!=orr.get(f,"")]
                if mismatch:
                    rejects.append({"kernel":kernel,"config":config,"task_key":"|".join(k),"reason":"mismatch:"+",".join(mismatch)})
                    continue
                xb.append(bt);xo.append(ot);ratios.append(bt/ot);changes.append(abs(ot/bt-1)*100)
                paired.append({"kernel":kernel,"config":config,"task_id":br.get("task_id",""),"matrix_id":br.get("matrix_id",""),"block_size":br.get("block_size",""),"baseline_median_ms":bt,"config_median_ms":ot,"baseline_over_config_speedup":bt/ot,"absolute_change_percent":abs(ot/bt-1)*100,"baseline_repeats":bn,"config_repeats":on})
            missing_base=len(set(other)-set(base));missing_config=len(set(base)-set(other))
            summary.append({"kernel":kernel,"config":config,"paired_tasks":len(xb),"missing_in_baseline":missing_base,"missing_in_config":missing_config,"rejected_structure_or_nnz":sum(r["kernel"]==kernel and r["config"]==config for r in rejects),"median_speedup":statistics.median(ratios) if ratios else math.nan,"median_absolute_change_percent":statistics.median(changes) if changes else math.nan,"p95_absolute_change_percent":percentile(changes,.95),"spearman_runtime_ranking":spearman(xb,xo)})
    def write(name,rows,fields):
        with (out/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write("sensitivity_summary.csv",summary,list(summary[0]) if summary else ["kernel","config"])
    write("paired_task_times.csv",paired,list(paired[0]) if paired else ["kernel","config"])
    write("rejected_pairs.csv",rejects,["kernel","config","task_key","reason"])
    report={"schema":"launch_sensitivity_analysis_v1","baseline":a.baseline,"configs":configs,"summary":summary,"rejected_pairs":len(rejects),"interpretation":{"runtime_ranking_target":"Spearman >= 0.90","decision_validation":"Run the same Partition candidate set with each fitted model; report Top-1/Top-2 agreement and regret. Runtime ranking alone does not prove decision stability."}}
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=True)+"\n")
    print(f"wrote {out}")

if __name__=="__main__":main()
