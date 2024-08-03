## Run ChatGLM2-6b inference

### GPU/CPU

```bash
python cli_demo.py
```

### Ascend

Since Kunpeng CPU will change execute processor core automatically, we should bind the python process to fixed cpu core.

```bash
lscpu
# find the information like below:
# NUMA node0 CPU(s):               0-23
# NUMA node1 CPU(s):               24-47
# NUMA node2 CPU(s):               48-71
# NUMA node3 CPU(s):               72-95
# NUMA node4 CPU(s):               96-119
# NUMA node5 CPU(s):               120-143
# NUMA node6 CPU(s):               144-167
# NUMA node7 CPU(s):               168-191

# we choose node0 to bind
taskset -c 0-23 python cli_demo.py
```