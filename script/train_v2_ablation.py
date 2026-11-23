#!/usr/bin/env python3
"""Matrix-grouped HGB ablation for strictly accepted replay-v2/GPU tasks."""
import argparse,json,math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error,r2_score

BASE=['replay_main_dimension','replay_block_rows','replay_block_cols','replay_aspect_ratio','replay_min_dimension','replay_max_dimension','replay_nnz_input_0','replay_nnz_input_1','replay_nnz_input_2','replay_nnz_output_0','replay_nnz_output_1']
ARITH=['replay_num_macs','replay_num_divisions','replay_operation_count_v2']
IRREG=['replay_row_nnz_mean','replay_row_nnz_max','replay_row_nnz_cv','replay_row_nnz_max_mean_ratio','replay_row_empty_fraction','replay_col_nnz_mean','replay_col_nnz_max','replay_col_nnz_cv','replay_col_nnz_max_mean_ratio','replay_col_empty_fraction','replay_work_mean','replay_work_max','replay_work_cv','replay_work_max_mean_ratio','replay_active_work_fraction']
MEM=['replay_density_input_0','replay_density_input_1','replay_density_input_2','replay_value_bytes_est','replay_index_bytes_est','replay_pointer_bytes_est','replay_output_write_bytes_est','replay_dense_workspace_bytes_est','replay_base_bytes_est','replay_index_value_byte_ratio','replay_reuse_factor_est']
LOCAL=['replay_block_csc_index_gap_mean','replay_block_csc_index_gap_cv','replay_block_csc_consecutive_fraction','replay_derived_csr_index_gap_mean','replay_derived_csr_index_gap_cv','replay_derived_csr_consecutive_fraction']
PAR=['replay_active_rows','replay_active_cols','replay_ssssm_bin_1','replay_ssssm_bin_2','replay_ssssm_bin_3','replay_ssssm_bin_4','replay_ssssm_bin_5','replay_ssssm_bin_6','replay_ssssm_bin_7','replay_ssssm_bin_8','replay_ssssm_bin_9','replay_ssssm_bin_10','replay_ssssm_nonempty_bins','replay_ssssm_max_bin_fraction','replay_ssssm_sparse_launches_est','forced_variant_code']
SETS={'A_size_nnz':BASE,'B_arithmetic':BASE+ARITH,'C_irregularity':BASE+ARITH+IRREG,'D_memory':BASE+ARITH+IRREG+MEM,'E_locality':BASE+ARITH+IRREG+MEM+LOCAL,'F_full':BASE+ARITH+IRREG+MEM+LOCAL+PAR}

def metric(y,p):
 e=np.abs(p-y); ape=e/np.maximum(np.abs(y),1e-6)*100
 return {'n_test':len(y),'MAPE':float(np.mean(ape)),'MdAPE':float(np.median(ape)),'P90_APE':float(np.percentile(ape,90)),'P95_APE':float(np.percentile(ape,95)),'RMSE_ms':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--seed',type=int,default=20260728); a=ap.parse_args()
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 cols=set(['matrix_id','kernel_type','time_kernel_ms','forced_ssssm_variant']+sum(SETS.values(),[])); cols.discard('forced_variant_code')
 df=pd.read_csv(a.input,usecols=lambda x:x in cols,low_memory=False)
 df['forced_variant_code']=(df['forced_ssssm_variant']=='dense').astype(np.float64)
 df['time_kernel_ms']=pd.to_numeric(df['time_kernel_ms'],errors='coerce')
 df=df[np.isfinite(df.time_kernel_ms)&(df.time_kernel_ms>0)].copy()
 results=[]; splits={}
 for kernel in ('GETRF','GESSM','TSTRF','SSSSM'):
  kd=df[df.kernel_type==kernel].copy(); groups=kd.matrix_id.astype(str)
  tr,te=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=a.seed).split(kd,groups=groups))
  train_m=sorted(set(groups.iloc[tr])); test_m=sorted(set(groups.iloc[te])); splits[kernel]={'train_matrices':train_m,'test_matrices':test_m,'n_train':len(tr),'n_test':len(te)}
  for stage,features in SETS.items():
   X=kd[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(np.float64)
   y=kd.time_kernel_ms.to_numpy(np.float64); model=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.07,max_iter=250,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1.0,random_state=a.seed)
   model.fit(X[tr],np.log1p(y[tr])); pred=np.expm1(model.predict(X[te])); pred=np.maximum(pred,0)
   row={'kernel':kernel,'ablation':stage,'n_train':len(tr),'n_features':len(features),**metric(y[te],pred)}; results.append(row); print(row,flush=True)
   if stage=='F_full':
    joblib.dump({'model':model,'features':features,'target_transform':'log1p','kernel':kernel,'schema':'v3'},out/f'{kernel}.hgb_v3.joblib')
    pd.DataFrame({'matrix_id':groups.iloc[te].to_numpy(),'actual_ms':y[te],'predicted_ms':pred}).to_csv(out/f'{kernel}.predictions.csv',index=False)
 pd.DataFrame(results).to_csv(out/'ablation_metrics.csv',index=False); (out/'splits.json').write_text(json.dumps(splits,indent=2)+'\n'); (out/'feature_groups.json').write_text(json.dumps(SETS,indent=2)+'\n')
if __name__=='__main__': main()
