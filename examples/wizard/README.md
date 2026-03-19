# Wizard Merge 示例

本目录包含 MindNLP Wizard 模型合并引擎的配方模板与端到端复现案例。

## 目录结构

```
examples/wizard/
├── README.md                           # 本文档
├── .hf_token                           # HuggingFace 令牌文件（需自行填写，已 gitignore）
├── .gitignore
├── linear_merge.yaml                   # 配方模板：线性加权合并
├── ties_merge.yaml                     # 配方模板：TIES 合并
├── slerp_merge.yaml                    # 配方模板：球面线性插值
└── llama3_biomed_dare_ties/            # 端到端示例：Llama3-8B 医学 DARE-TIES 合并
    ├── README.md                       # 示例详细文档
    ├── requirements.txt                # 环境依赖（锁定版本）
    ├── dare_ties_biomed.yaml           # DARE-TIES 合并配方
    ├── download_models.py              # 模型下载脚本
    ├── run_merge.sh                    # 合并执行脚本
    ├── run_eval.sh                     # 评测执行脚本
    └── run_lm_eval.py                  # lm-eval MindSpore 后端入口
```

## 环境要求

| 项目 | 版本 |
|------|------|
| Python | >= 3.9, < 3.12（实测 3.11.15） |
| MindSpore | 2.7.1 |
| CANN | >= 8.1.RC1（实测 8.2.RC2） |
| NPU Driver | 25.3.rc1.2 |
| 硬件 | Ascend 910B2（合并可在 CPU 运行，评测需 NPU） |

**Ascend 环境初始化**：`run_merge.sh` 和 `run_eval.sh` 会自动检测并 source CANN toolkit
与 NPU driver 路径。如果自动检测失败（报 `Unsupported device target Ascend`），
请根据实际安装位置手动设置，示例：

```bash
# 设置 ASCEND_HOME 为你机器上 CANN 的实际安装目录（默认 /usr/local/Ascend）
export ASCEND_HOME=/usr/local/Ascend   # 按实际路径修改
source ${ASCEND_HOME}/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/driver/lib64/driver:$LD_LIBRARY_PATH
```

> 脚本已支持通过 `ASCEND_HOME` 环境变量指定 CANN 安装路径，未设置时默认 `/usr/local/Ascend/`。
> 该错误的根因是 `set_env.sh` 未将 `libascend_hal.so` 所在的 driver 目录加入 `LD_LIBRARY_PATH`。

安装依赖：

```bash
cd mindnlp  # 项目根目录

# 方式 1: 安装 MindNLP 基础依赖 + Wizard 模块兼容依赖
# 适合开发、调试、运行配方模板
pip install -r requirements/requirements.txt
pip install -r requirements/wizard-requirements.txt

# 方式 2: 直接安装端到端示例锁定的完整依赖
# 适合严格复现 llama3_biomed_dare_ties 示例结果
pip install -r examples/wizard/llama3_biomed_dare_ties/requirements.txt
```

说明：

- `requirements/wizard-requirements.txt` 提供的是 Wizard 模块的兼容依赖范围，不锁定精确版本。
- `examples/wizard/llama3_biomed_dare_ties/requirements.txt` 是示例验证通过的锁定环境。
- 如已按方式 2 安装，一般不需要再单独执行 `pip install lm_eval datasets`。

## HuggingFace 令牌配置

部分场景（镜像下载、gated 模型）需要 HuggingFace 令牌。两种方式任选其一：

```bash
# 方式 1: 环境变量
export HF_TOKEN="hf_xxxxxxxxxxxx"

# 方式 2: 令牌文件（推荐，所有脚本自动读取）
# 编辑 examples/wizard/.hf_token，将令牌粘贴到非注释行
```

令牌获取地址：https://huggingface.co/settings/tokens

> `.hf_token` 已被 `.gitignore` 排除，不会提交到版本库。

---

## 一、配方模板

以下为基础合并配方模板，可直接运行或作为自定义配方的起点。

### linear — 线性加权合并

```yaml
# linear_merge.yaml
models:
  - model: Qwen/Qwen2.5-7B
    parameters:
      weight: 1.0
  - model: Qwen/Qwen2.5-7B-Instruct
    parameters:
      weight: 0.3
merge_method: linear
dtype: float16
```

```bash
python -m mindnlp.wizard.merge.scripts.run_yaml \
    examples/wizard/linear_merge.yaml ./output/linear_merged
```

### ties — TIES (Trim, Elect Sign & Merge)

```yaml
# ties_merge.yaml
models:
  - model: Qwen/Qwen2.5-7B-Instruct
    parameters:
      density: 0.5
      weight: 1.0
  - model: Qwen/Qwen2.5-7B-Math
    parameters:
      density: 0.5
      weight: 0.5
merge_method: ties
base_model: Qwen/Qwen2.5-7B
parameters:
  normalize: true
dtype: bfloat16
```

```bash
python -m mindnlp.wizard.merge.scripts.run_yaml \
    examples/wizard/ties_merge.yaml ./output/ties_merged
```

### slerp — 球面线性插值

```yaml
# slerp_merge.yaml
models:
  - model: meta-llama/Llama-3-8B
  - model: meta-llama/Llama-3-8B-Instruct
merge_method: slerp
base_model: meta-llama/Llama-3-8B
parameters:
  t: 0.5
dtype: bfloat16
```

```bash
python -m mindnlp.wizard.merge.scripts.run_yaml \
    examples/wizard/slerp_merge.yaml ./output/slerp_merged
```

### 通用选项

```bash
# 输出为 MindSpore ckpt 格式（默认 safetensors）
python -m mindnlp.wizard.merge.scripts.run_yaml \
    recipe.yaml ./output --output-format ckpt

# 复制 tokenizer 到输出目录
python -m mindnlp.wizard.merge.scripts.run_yaml \
    recipe.yaml ./output --copy-tokenizer

# 生成模型卡
python -m mindnlp.wizard.merge.scripts.run_yaml \
    recipe.yaml ./output --write-model-card
```

### 自定义配方

配方 YAML 基本结构：

```yaml
models:
  - model: <HuggingFace 模型 ID 或本地路径>
    parameters:
      weight: <权重值>
      density: <密度值，TIES/DARE 专用>
merge_method: <方法名>
base_model: <基准模型，TIES/DARE/SLERP 需要>
parameters:
  normalize: true
dtype: <float16 | bfloat16 | float32>
```

Wizard 支持 21 种合并方法：linear, slerp, ties, dare_ties, dare_linear, passthrough,
model_stock, nearswap, nuslerp, multislerp, generalized_task_arithmetic, arcee_fusion,
karcher, sce, ram, rectify_embed 等。完整说明见
[`src/mindnlp/wizard/README.md`](../../src/mindnlp/wizard/README.md)。

---

## 二、端到端示例：Llama3-8B 医学合并

使用 DARE-TIES 方法复现
[lighteternal/Llama3-merge-biomed-8b](https://huggingface.co/lighteternal/Llama3-merge-biomed-8b)，
覆盖**模型下载 → 合并 → 评测**全流程。

**合并后模型**：[chenjingshen/Llama3-8B-merge-biomed-wizard](https://huggingface.co/chenjingshen/Llama3-8B-merge-biomed-wizard)

### 合并配置

| 配置项 | 值 |
|--------|------|
| 合并方法 | DARE-TIES |
| 基座模型 | meta-llama/Meta-Llama-3-8B-Instruct |
| 输出精度 | bfloat16 |
| int8_mask | true |

### 所需模型

| 模型 | HuggingFace 链接 | density | weight | 需要令牌 |
|------|------------------|---------|--------|----------|
| Meta-Llama-3-8B-Instruct (base) | [NousResearch/Meta-Llama-3-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct) | 1.0 | 1.0 | 否 |
| Meta-Llama-3-8B-Instruct (delta) | 同上 | 0.60 | 0.5 | 否 |
| Hermes-2-Pro-Llama-3-8B | [NousResearch/Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) | 0.55 | 0.1 | 否 |
| Llama3-OpenBioLLM-8B | [aaditya/Llama3-OpenBioLLM-8B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B) | 0.55 | 0.4 | 否 |

> 使用 NousResearch 镜像，**不需要**申请 Meta Llama 访问权限。三个模型合计约 48GB。

### 评测数据集

所有数据集由 `lm-eval-harness` 自动从 HuggingFace 下载，缓存到 `output/datasets/`
（可通过 `HF_DATASETS_CACHE` 自定义）。

| 数据集 | HuggingFace 来源 | few-shot | 需要令牌 |
|--------|------------------|----------|----------|
| ARC Challenge | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | 25-shot | 否 |
| HellaSwag | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | 10-shot | 否 |
| Winogrande | [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) | 5-shot | 否 |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 5-shot | 否 |
| MMLU (6 个医学子集) | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | 5-shot | 否 |

### 快速复现

```bash
# 步骤 1: 下载模型（约 48GB）
python examples/wizard/llama3_biomed_dare_ties/download_models.py
# 国内镜像: HF_ENDPOINT=https://hf-mirror.com python ...

# 步骤 2: 执行合并
bash examples/wizard/llama3_biomed_dare_ties/run_merge.sh

# 步骤 3: 评测（依次运行 10 个数据集，约 12 小时）
bash examples/wizard/llama3_biomed_dare_ties/run_eval.sh
```

**跳过合并**——直接下载已合并模型后评测：

```bash
huggingface-cli download chenjingshen/Llama3-8B-merge-biomed-wizard \
    --local-dir ./output/Llama3-merge-biomed-8b

bash examples/wizard/llama3_biomed_dare_ties/run_eval.sh ./output/Llama3-merge-biomed-8b
```

### 评测结果

评测设置与 Open LLM Leaderboard v1 一致：

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

**综合平均（10 项主指标）**：

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

---

## 相关链接

- [Wizard Merge 模块文档](../../src/mindnlp/wizard/README.md)
- [MindNLP 主仓库](https://github.com/mindspore-lab/mindnlp)
- [MergeKit 原始仓库](https://github.com/arcee-ai/mergekit)
- [Wizard 复现模型 (HuggingFace)](https://huggingface.co/chenjingshen/Llama3-8B-merge-biomed-wizard)
