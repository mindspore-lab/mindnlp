## How to run

### prepare

```bash
pip install mindnlp==0.5.0
```

upgrade transformers >= 4.55.0

```bash
pip install transformers>=4.55.0
```

### single-card

```bash
python gpt_oss_standalone.py
```

### multi-card

```bash
mpirun -n 2 --map-by numa python gpt_oss_multiprocess.py
```