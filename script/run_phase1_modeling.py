#!/usr/bin/env python3
"""Run the no-GPU phase-1 PanguLU model improvement workflow."""
import argparse,subprocess,sys
from pathlib import Path

def args():
 p=argparse.ArgumentParser(description="Build, train, and compare phase-1 models without GPU reruns")
 p.add_argument("campaign",nargs="?",default="profiling_suite_001"); p.add_argument("--output",required=True)
 p.add_argument("--seed",type=int,default=20260727); p.add_argument("--outer-folds",type=int,default=5); p.add_argument("--inner-folds",type=int,default=3); return p.parse_args()
def run(cmd): print('+',' '.join(map(str,cmd)),flush=True); subprocess.run(cmd,check=True)
def main():
 a=args(); scripts=Path(__file__).resolve().parent; out=Path(a.output).expanduser().resolve()
 if out.exists(): raise SystemExit(f"output already exists: {out}")
 training=out/'training_set'; models=out/'models'; baseline=out/'baseline_same_coverage'; analysis=out/'analysis'
 run([sys.executable,scripts/'build_phase1_training_sets.py',a.campaign,'--output',training,'--seed',str(a.seed)])
 run([sys.executable,scripts/'train_kernel_time_models_phase1.py',training,'--output',models,'--outer-folds',str(a.outer_folds),'--inner-folds',str(a.inner_folds),'--seed',str(a.seed)])
 run([sys.executable,scripts/'train_kernel_time_models.py',training,'--output',baseline,'--folds',str(a.outer_folds),'--seed',str(a.seed)])
 run([sys.executable,scripts/'analyze_phase1_models.py',baseline/'metrics.json',models/'metrics.json','--output',analysis])
 print(f"Phase-1 workflow complete: {out}")
if __name__=='__main__': main()
