## Open-R1 基于 MindNLP 的完全复现

### 仓库用途
- 本仓库用于在 MindSpore + MindNLP 环境中完全复现 DeepSeek-R1 / Open-R1 的训练与推理流程。
- 目标是在尽量对齐 Hugging Face Transformers / TRL 的接口与训练流程的前提下，提供可直接运行的复现方案与脚本。

### 快速开始
- 启动监督微调（SFT）训练：

```bash
bash sh/sft.sh
```

- 说明：
  - 脚本会调用仓库内的 `src/mind_openr1/sft.py` 并加载配置（参见 `src/mind_openr1/configs.py`）。
  - 训练日志与权重等产物默认输出到 `trainer_output/` 与 `logs/`（可在脚本或配置中修改）。

### 目录结构（节选）
- `sh/`：运行脚本（监督微调：`sft.sh`；如需扩展，可在此目录新增脚本）。
- `src/mind_openr1/`：核心源码与配置：
  - `sft.py`：监督微调入口。
  - `configs.py`：训练与模型/数据相关配置。
  - `rewards.py`、`grpo.py`：与强化学习相关模块。
  - `utils/`：数据、评估、回调、日志等辅助模块。
- `data/`（如存在）：数据相关目录（按需准备）。
- `logs/`、`trainer_output/`：训练日志与输出目录。

### 环境与依赖
- 建议版本：MindSpore 2.6、MindNLP 0.5.0rc2、Python 3.10+。
- MindNLP 请参考官方文档安装。

### 数据准备
- 请准备符合监督微调（SFT）需求的数据集，并在 `configs.py` 或脚本参数中填入数据路径与格式。
- 如需自定义数据加载/预处理，可在 `src/mind_openr1/utils/data.py` 中扩展，或在 `sft.py` 内接入自定义 Dataset。

### 训练配置与输出
- 训练超参（batch size、学习率、训练步数、保存/评估间隔等）可在 `configs.py` 中修改。
- 运行中会在 `logs/` 输出日志，在 `trainer_output/`（或你自定义的路径）保存权重/检查点。

### 与 TRL 的兼容性说明
为让 MindNLP 在训练环节尽量对齐/兼容 TRL 与部分 Transformers 训练器行为，本仓库对 `mindnlp/mindtorch` 的若干底层组件做了必要的调整（详见文末“附录：源码改动（为 TRL 兼容所做）”）：
- 调整 autograd 接口，支持“仅对张量输入求导、手动回填参数梯度”等场景。
- 完善模块 Hook 与 `state_dict` 行为，便于与上层训练器/加速器协同。
- 提供 `autograd.graph` 的最小 API 以兼容现有调用路径。
- 在 CPU 后端缺失场景提供稳健回退。
- 在 `Trainer` 的 `training_step` 中对梯度累积/分布式做安全处理以对齐常见训练器行为。


---

## 附录：源码改动（为 TRL 兼容所做）

以下内容为相对 `origin/master` 的本地修改汇总，旨在说明为 TRL 兼容所做的变更。

### 概览
- 变更基线：分支 `master` 跟踪 `origin/master`，无额外提交差异；改动均为未提交的本地修改
- 修改文件与规模（插入/删除）：
  - `mindnlp/transformers/trainer.py`: +46 / -4
  - `mindtorch/_apis/cpu.py`: +7 / -1
  - `mindtorch/autograd/__init__.py`: +1 / -0
  - `mindtorch/autograd/function.py`: +154 / -126
  - `mindtorch/nn/modules/module.py`: +2393 / -2373
- 新增文件：
  - `mindtorch/autograd/graph.py`

---

### 详细改动与位置

#### 1) mindnlp/transformers/trainer.py
- 变更要点：
  - 引入 `_mindspore_grad_enabled` 开关。
  - 将原先直接对 `forward_fn` 做 `value_and_grad` 的做法，改为对 `inputs` 张量键进行扁平化，仅以张量参数参与求导，避免将 `dict` 作为求导输入。
  - 使用 `attach_grads=False` 获取梯度后，手动回填到 `param.grad`，并与梯度累积、分布式场景相容。
  - 在不走自定义求导路径时回退为原始 `compute_loss` 流程。

- 位置（hunk）：

```diff
@@ -88,9 +88,51 @@ def training_step(
 
         return loss, loss_true
 
-    if not hasattr(self, 'grad_fn'):
-        self.grad_fn = autograd.value_and_grad(forward_fn, model.trainable_params(), has_aux=True)
+    if not hasattr(self, '_mindspore_grad_enabled'):
+        self._mindspore_grad_enabled = True
+
+    if self._mindspore_grad_enabled:
+        # 仅传入张量参数，避免将 dict 作为 grad 输入
+        input_keys = tuple(sorted(k for k, v in inputs.items() if hasattr(v, "shape")))
+
+        def forward_fn_flat(*flat_tensors):
+            local_inputs = {}
+            # 重建 inputs，仅包含张量键值；非张量保持原值
+            for k in inputs:
+                if k in input_keys:
+                    # 对应位置映射
+                    idx = input_keys.index(k)
+                    local_inputs[k] = flat_tensors[idx]
+                else:
+                    local_inputs[k] = inputs[k]
+
+            with self.compute_loss_context_manager():
+                loss = self.compute_loss(model, local_inputs, num_items_in_batch=num_items_in_batch)
+
+            if self.args.n_gpu > 1:
+                loss = loss.mean()
 
-    loss_scaled, (loss_true,) = self.grad_fn(inputs, num_items_in_batch)
+            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
+                loss = loss / self.current_gradient_accumulation_steps
+
+            if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
+                loss = loss / self.accelerator.gradient_accumulation_steps
+
+            return loss, loss
+
+        weights = model.trainable_params()
+        flat_args = tuple(inputs[k] for k in input_keys)
+        grad_fn = autograd.value_and_grad(forward_fn_flat, weights, has_aux=True, attach_grads=False)
+        (loss_scaled, loss_true), grads = grad_fn(*flat_args)
+
+        # 回填梯度，供优化器使用
+        for param, grad in zip(weights, grads):
+            if getattr(param, 'grad', None) is None:
+                param.grad = mindtorch.tensor(grad, device=param.device)
+            else:
+                param.grad += mindtorch.tensor(grad, device=param.device)
+
+        return loss_true
 
-    return loss_true
+    loss_scaled = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
+    return loss_scaled
```

---

#### 2) mindtorch/_apis/cpu.py
- 变更要点：
  - `empty` 新增 CPU 后端未实现时的回退逻辑，优雅降级为 `numpy.empty` 并封装为 `mindtorch.Tensor`。

- 位置（hunk）：

```diff
@@ -9,7 +9,13 @@ from .._op_prim.cpu import legacy
 
 empty_op = Empty().set_device('CPU')
 def empty(size, dtype):
-    return empty_op(size, dtype=dtype, device='CPU')
+    try:
+        return empty_op(size, dtype=dtype, device='CPU')
+    except RuntimeError as err:  # pragma: no cover - fallback path depends on runtime
+        if 'Not implement' not in str(err):
+            raise
+        # MindSpore 默认 CPU backend 未实现 Empty 原语，退回到 numpy 实现
+        return mindtorch.Tensor.from_numpy(np.empty(size, mindtorch.dtype2np[dtype]))
```

---

#### 3) mindtorch/autograd/__init__.py
- 变更要点：
  - 新增导出 `saved_tensors_hooks` 与 `current_hooks`，与 PyTorch 接口对齐。

- 位置（hunk）：

```diff
@@ -2,3 +2,4 @@
 from .node import Node
 from .function import Function, value_and_grad
 from .grad_mode import no_grad, enable_grad, inference_mode
+from .graph import saved_tensors_hooks, current_hooks
```

---

#### 4) mindtorch/autograd/function.py
- 变更要点：
  - 重写 `value_and_grad`：
    - 当 `attach_grads=True` 时，采用 MindSpore 的 `value_and_grad` 并安全合并到 `param.grad`。
    - 当 `attach_grads=False` 时，显式构建/运行 PyNative 求导图，返回 `(values, grads)`。
    - 对参数集合进行缓存/清零以避免跨次调用的梯度污染。

- 位置（hunk，节选）：

```diff
@@ -1,126 +1,154 @@
-"""functional autograd"""
-...
-def value_and_grad(fn, params_or_argnums, has_aux=False, attach_grads=True):
-    grad_fn = mindspore.value_and_grad(fn, None, tuple(params_or_argnums), has_aux)
-    if attach_grads:
-        def new_grad_fn(*args, **kwargs):
-            values, grads = grad_fn(*args, **kwargs)
-            for param, grad in zip(params_or_argnums, grads):
-                grad = mindtorch.tensor(grad, device=param.device)
-                if param.grad is None:
-                    param.grad = grad
-                else:
-                    param.grad += grad
-            return values
-        return new_grad_fn
-    return grad_fn
+"""functional autograd"""
+...
+def value_and_grad(fn, params_or_argnums, has_aux=False, attach_grads=True):
+    params = tuple(params_or_argnums)
+    # Fast path: let MindSpore wrap gradients when we want autoupdate of .grad
+    if attach_grads:
+        grad_fn = mindspore.value_and_grad(fn, None, params, has_aux)
+        def new_grad_fn(*args, **kwargs):
+            attached_params = getattr(new_grad_fn, 'attached_params', None)
+            if attached_params is not params:
+                if attached_params is not None:
+                    for param in attached_params:
+                        if param.grad is not None:
+                            param.grad = mindtorch.zeros_like(param.grad.detach())
+                new_grad_fn.attached_params = params
+            values, grads = grad_fn(*args, **kwargs)
+            for param, grad in zip(params, grads):
+                grad = mindtorch.tensor(grad, device=param.device)
+                if param.grad is None:
+                    param.grad = grad
+                else:
+                    updated_grad = mindtorch.zeros_like(param.grad, device=param.device)
+                    updated_grad.copy_(param.grad)
+                    updated_grad += grad
+                    param.grad = updated_grad
+            return values
+        return new_grad_fn
+
+    # Stable path for MindSpore PyNative: explicitly build and run grad graph
+    def value_and_grad_f(*args, **kwargs):
+        fn_ = fn
+        _pynative_executor.set_grad_flag(True)
+        _pynative_executor.new_graph(fn_, *args, **kwargs)
+        values = fn_(*args, **kwargs)
+        _pynative_executor.end_graph(fn_, values, *args, **kwargs)
+
+        run_args = args
+        if kwargs:
+            run_args = args + tuple(kwargs.values())
+
+        grads = _pynative_executor.grad(fn_, grad_, params, None, *run_args)
+        return values, grads
```

---

#### 5) mindtorch/nn/modules/module.py
- 变更要点：
  - 大幅对齐/扩展 `Module` API 与全局/实例级 Hook 机制（buffer/parameter/module 注册 hook、forward/forward_pre hooks、backward hooks 等）。
  - 伴随大量内部工具与 `state_dict`/`load_state_dict` 行为增强。
  - 改动范围覆盖全文件（行范围见 hunk 头）。

- 位置（hunk 概览）：

```diff
@@ -1,2373 +1,2393 @@
```

- 完整 diff 已保存：`final/mindnlp/open_r1/module.diff`

---

#### 6) 新增：mindtorch/autograd/graph.py
- 目的：
  - 提供最小可用的 `saved_tensors_hooks`/`current_hooks` API（线程本地栈维护），与 `torch.autograd.graph` 接口对齐；当前不直接接入 MindSpore 梯度记录流水线，但可安全作为上下文管理器使用。

- 文件片段：

```1:24:/Users/guojialiang/code/kaiyuanzhixia/mindnlp/mindtorch/autograd/graph.py
"""Autograd graph utilities.

This module provides a minimal ``saved_tensors_hooks`` implementation so that
``mindtorch.autograd.graph`` exposes the same API surface as
``torch.autograd.graph``. The current implementation focuses on API
compatibility and keeps a thread-local stack of the registered hooks. The
stored hooks are not yet wired into MindSpore's gradient recording pipeline,
but existing code can safely enter/exit the context manager without raising
``ImportError``.
"""

from __future__ import annotations

from contextlib import ContextDecorator
from typing import Any, Callable, List, Optional, Tuple
import threading

PackHook = Callable[[Any], Any]
UnpackHook = Callable[[Any], Any]
```

---

### 复现实验/对比方法
- 列出本地相对上游差异（文件与统计）：
  - `git diff --stat origin/master...master`
- 查看具体文件差异：
  - `git diff -- mindnlp/transformers/trainer.py`
  - `git diff -- mindtorch/_apis/cpu.py`
  - `git diff -- mindtorch/autograd/__init__.py`
  - `git diff -- mindtorch/autograd/function.py`
  - `git diff -- mindtorch/nn/modules/module.py`
- 导出 `module.py` 全量 diff（已导出）：
  - `git diff --no-color -- mindtorch/nn/modules/module.py > final/mindnlp/open_r1/module.diff`
