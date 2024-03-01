# python utils/os_util.py 2>&1  </dev/null | tee utils/os_util.log
# 
file=utils_comm/cluster_seqs.py
nohup python $file > $file.log 2>&1 &