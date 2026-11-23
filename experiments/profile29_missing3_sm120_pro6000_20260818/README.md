# RTX PRO 6000 profile29 missing-three campaign

This incremental campaign collects the three matrices absent from the RTX
PRO 6000 training data but present in the frozen H100 profile29 membership:
`atmosmodd`, `kkt_power`, and `language`.

The GPU contract is CUDA 12.8, PTX 8.7, and `sm_120`.  The task protocol is
dense SSSSM only, block sizes 2048 through 16384 in increments of 2048, one
measurement and no warmup.  Existing RTX labels for the other 26 profile29
members are reused; `nd24k` is excluded when the final profile29 set is built.
