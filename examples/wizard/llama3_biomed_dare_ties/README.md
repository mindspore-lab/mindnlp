# Llama3-8B-merge-biomed — DARE-TIES 医学合并复现

使用 MindNLP Wizard 合并引擎，复现
[lighteternal/Llama3-merge-biomed-8b](https://huggingface.co/lighteternal/Llama3-merge-biomed-8b)
的 DARE-TIES 合并实验。

**合并后模型已上传至 HuggingFace**：
[chenjingshen/Llama3-8B-merge-biomed-wizard](https://huggingface.co/chenjingshen/Llama3-8B-merge-biomed-wizard)

## 目录结构

```
llama3_biomed_dare_ties/
├── README.md                 # 本文档
├── requirements.txt          # 环境依赖（锁定版本）
├── dare_ties_biomed.yaml     # DARE-TIES 合并配方
├── download_models.py        # 模型下载脚本
├── run_merge.sh              # 合并执行脚本
├── run_eval.sh               # 评测执行脚本
├── run_lm_eval.py            # lm-eval MindSpore 后端入口
├── models/                   # 下载的源模型（自动创建，已 gitignore）
└── output/                   # 合并产物与评测结果（自动创建，已 gitignore）
    ├── merged/               # 合并后的模型文件
    ├── eval/                 # 评测日志与 JSON 结果
    └── datasets/             # HuggingFace 数据集缓存
```

## 合并配置

| 配置项 | 值 |
|--------|------|
| 合并方法 | DARE-TIES |
| 基座模型 | meta-llama/Meta-Llama-3-8B-Instruct |
| 输出精度 | bfloat16 |
| int8_mask | true |

### 合并模型列表

| 模型 | HuggingFace 链接 | density | weight | 需要令牌 |
|------|------------------|---------|--------|----------|
| Meta-Llama-3-8B-Instruct (base) | [NousResearch/Meta-Llama-3-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct) | 1.0 | 1.0 | 否 |
| Meta-Llama-3-8B-Instruct (delta) | 同上 | 0.60 | 0.5 | 否 |
| Hermes-2-Pro-Llama-3-8B | [NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) | 0.55 | 0.1 | 否 |
| Llama3-OpenBioLLM-8B | [aaditya/Llama3-OpenBioLLM-8B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B) | 0.55 | 0.4 | 否 |

> 注：本示例使用的模型均为 NousResearch 镜像或开放模型，**不需要**申请 Meta Llama 访问权限。

### 评测数据集

评测设置与 Open LLM Leaderboard v1 一致。所有数据集由 `lm-eval-harness` 自动从 HuggingFace 下载。

| 数据集 | HuggingFace 来源 | few-shot | 需要令牌 |
|--------|------------------|----------|----------|
| ARC Challenge | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | 25-shot | 否 |
| HellaSwag | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | 10-shot | 否 |
| Winogrande | [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) | 5-shot | 否 |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 5-shot | 否 |
| MMLU (6 个医学子集) | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | 5-shot | 否 |

数据集默认缓存到 `output/datasets/`（可通过 `HF_DATASETS_CACHE` 环境变量自定义）。

## 环境要求

| 项目 | 版本 |
|------|------|
| Python | >= 3.9, < 3.12（实测 3.11） |
| MindSpore | 2.7.1 |
| CANN | >= 8.1.RC1（实测 8.2.RC2） |
| NPU Driver | 25.3.rc1.2 |
| 硬件 | Ascend 910B2（合并可在 CPU 运行，评测需 NPU） |

**Ascend 环境**：`run_merge.sh` 和 `run_eval.sh` 会自动检测并 source CANN toolkit 与 driver。
如遇 `Unsupported device target Ascend` 报错，请根据实际安装位置手动设置：

```bash
# 设置 ASCEND_HOME 为你机器上 CANN 的实际安装目录（默认 /usr/local/Ascend）
export ASCEND_HOME=/usr/local/Ascend   # 按实际路径修改
source ${ASCEND_HOME}/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/driver/lib64/driver:$LD_LIBRARY_PATH
```

## 快速复现

### 0. 配置令牌

虽然本示例的模型和数据集均不需要令牌，但 HuggingFace 镜像或下载大文件时可能需要。
两种方式任选其一：

```bash
# 方式 1: 环境变量
export HF_TOKEN="hf_xxxxxxxxxxxx"

# 方式 2: 令牌文件（推荐，脚本会自动读取）
# 编辑 examples/wizard/.hf_token，将令牌粘贴到文件中（非注释行）
```

令牌获取地址：https://huggingface.co/settings/tokens

### 1. 安装依赖

```bash
cd mindnlp  # 项目根目录

# 使用本示例锁定的依赖版本
pip install -r examples/wizard/llama3_biomed_dare_ties/requirements.txt

# 或分步安装
pip install -r requirements/requirements.txt         # MindNLP 核心
pip install -r requirements/wizard-requirements.txt   # Wizard 模块
pip install lm_eval datasets                          # 评测
```

### 2. 下载模型

```bash
# 下载三个源模型到 llama3_biomed_dare_ties/models/ 目录
python examples/wizard/llama3_biomed_dare_ties/download_models.py

# 或指定自定义下载目录
python examples/wizard/llama3_biomed_dare_ties/download_models.py \
    --output-dir /data/hf_models

# 国内用户可使用镜像加速
HF_ENDPOINT=https://hf-mirror.com \
    python examples/wizard/llama3_biomed_dare_ties/download_models.py
```

> 三个模型合计约 48GB，请确保磁盘空间充足。

### 3. 执行合并

```bash
# 使用脚本（推荐）
bash examples/wizard/llama3_biomed_dare_ties/run_merge.sh

# 自定义输出路径
bash examples/wizard/llama3_biomed_dare_ties/run_merge.sh /path/to/output

# 或直接调用 Wizard CLI
python -m mindnlp.wizard.merge.scripts.run_yaml \
    examples/wizard/llama3_biomed_dare_ties/dare_ties_biomed.yaml \
    ./output/Llama3-merge-biomed-8b \
    --copy-tokenizer --write-model-card
```

合并产物约 16GB（4 个 safetensors 分片 + config.json + tokenizer）。

### 4. 评测

```bash
# 使用脚本（依次运行 10 个数据集，约 12 小时）
bash examples/wizard/llama3_biomed_dare_ties/run_eval.sh ./output/Llama3-merge-biomed-8b

# 或单独评测某个数据集
python examples/wizard/llama3_biomed_dare_ties/run_lm_eval.py \
    --model mindspore \
    --model_args pretrained=./output/Llama3-merge-biomed-8b,dtype=bfloat16 \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --batch_size 1
```

### 跳过合并：直接使用已合并模型

如果不想自己执行合并，可以直接从 HuggingFace 下载已合并的模型：

```bash
# 使用 huggingface-cli
huggingface-cli download chenjingshen/Llama3-8B-merge-biomed-wizard \
    --local-dir ./output/Llama3-merge-biomed-8b

# 或使用 Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('chenjingshen/Llama3-8B-merge-biomed-wizard',
                  local_dir='./output/Llama3-merge-biomed-8b')
"
```

然后直接跳到第 4 步执行评测。

## 评测结果

| 数据集 | 指标 | Wizard 合并 | Llama3-8B-Instruct | OpenBioLLM-8B |
|--------|------|-------------|---------------------|---------------|
| **ARC Challenge** | Accuracy | **59.73%** | 57.17% | 55.38% |
| | Norm. Accuracy | **64.59%** | 60.75% | 58.62% |
| **HellaSwag** | Accuracy | 62.26% | **62.59%** | 61.83% |
| | Norm. Accuracy | 81.35% | **81.53%** | 80.76% |
| **Winogrande** | Accuracy | **76.01%** | 74.51% | 70.88% |
| **GSM8K** | Accuracy | **70.81%** | 68.69% | 10.15% |
| **MMLU-Anatomy** | Accuracy | 71.11% | **72.59%** | 69.62% |
| **MMLU-Clinical Knowledge** | Accuracy | **77.74%** | **77.83%** | 60.38% |
| **MMLU-College Biology** | Accuracy | **80.56%** | **81.94%** | 79.86% |
| **MMLU-College Medicine** | Accuracy | 68.21% | 63.58% | **70.52%** |
| **MMLU-Medical Genetics** | Accuracy | 82.00% | 80.00% | 80.00% |
| **MMLU-Prof. Medicine** | Accuracy | **77.57%** | 71.69% | **77.94%** |

### 综合平均（10 项主指标）

| 模型 | 平均分 |
|------|--------|
| **Wizard 合并** | **75.00%** |
| Llama3-8B-Instruct | 73.31% |
| OpenBioLLM-8B | 65.87% |

- Wizard 合并以 **75.00%** 的 10 项平均分，超越 Llama3-8B-Instruct（+1.68%）和 OpenBioLLM-8B（+9.12%）
- 4 项通用指标平均（ARC norm / HellaSwag norm / Winogrande / GSM8K）：Wizard **73.19%** vs Instruct 71.37% vs BioLLM 55.10%
- 6 项 MMLU 医学子集平均：Wizard **76.20%** vs Instruct 74.61% vs BioLLM 73.05%
- 10 项中 Wizard 在 6 项胜出 Instruct、8 项胜出 BioLLM
- 验证了 DARE-TIES 合并方法同时提升通用能力与专业领域知识的有效性

## 实际运行环境

| 项目 | 版本 |
|------|------|
| 硬件 | Ascend 910B2 |
| MindSpore | 2.7.1 |
| Python | 3.11.15 |
| CANN | 8.2.RC2 |
| NPU Driver | 25.3.rc1.2 |
| transformers | 4.55.0 |
| lm_eval | 0.4.11 |

## 参考

- [lighteternal/Llama3-merge-biomed-8b](https://huggingface.co/lighteternal/Llama3-merge-biomed-8b) — 原始合并模型
- [chenjingshen/Llama3-8B-merge-biomed-wizard](https://huggingface.co/chenjingshen/Llama3-8B-merge-biomed-wizard) — Wizard 复现模型
- [Language Models are Super Mario](https://arxiv.org/abs/2311.03099) — DARE 论文
- [Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) — TIES 论文
