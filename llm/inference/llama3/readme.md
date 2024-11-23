## Run distributed (pipeline parallel)

### use msrun (recommend)

`msrun` is a MindSpore defined launcher for multi-process parallel execution, which can get best performance, you can use it by the command below:

```bash
msrun --worker_num=2 --local_worker_num=2 --master_port=8118 --join=True run_llama3_distributed.py
```

if you use Ascend NPU with Kunpeng CPU, you should bind-core to get better performance

```bash
msrun --worker_num=2 --local_worker_num=2 --master_port=8118 --join=True --bind_core=True run_llama3_distributed.py
```

### use mpirun

`mpirun` controls several aspects of program execution in Open MPI, you can use it by the command below:

```bash
mpirun -n 2 python run_llama3_distributed.py
```

if you use Ascend NPU with Kunpeng CPU, you should bind-core to get better performance:

```bash
mpirun --bind-to numa -n 2 python run_llama3_distributed.py
```

