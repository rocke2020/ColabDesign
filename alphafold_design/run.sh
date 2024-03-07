export NUMEXPR_MAX_THREADS=8
# run disulfide_design
file=alphafold_design/disulfide_design.py
gpu=2
export CUDA_VISIBLE_DEVICES=$gpu
log_filename=$file-gpu$gpu
log_filename=$file
nohup python $file \
    --args_file alphafold_design/args/demo_binder.yml \
    > $log_filename.log 2>&1 &