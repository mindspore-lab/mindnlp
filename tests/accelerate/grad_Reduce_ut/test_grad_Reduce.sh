# mpirun -n 8 -H 127.0.0.1:8 --output-filename bak/log_output_mpirun_single/log_ python test_grad_Reduce.py
msrun --worker_num=4 --local_worker_num=4 --master_port=8123 --log_dir=bak/msrun_log --join=True --cluster_time_out=100 test_grad_Reduce.py
# msrun --worker_num=8 --local_worker_num=8 --master_port=8123 --log_dir=bak/msrun_log --join=True --cluster_time_out=100 test_grad_Reduce.py