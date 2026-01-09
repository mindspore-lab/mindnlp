from typing import Union, Any, Optional

from transformers.training_args import OptimizerNames
from accelerate.utils import DistributedType

import mindtorch
from mindtorch import nn, autograd

def training_step(
    self,
    model: nn.Module,
    inputs: dict[str, Union[mindtorch.Tensor, Any]],
    num_items_in_batch: Optional[mindtorch.Tensor] = None,
) -> mindtorch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (`nn.Module`):
            The model to train.
        inputs (`dict[str, Union[mindtorch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.

    Return:
        `mindtorch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()
    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        self.optimizer.train()

    inputs = self._prepare_inputs(inputs)
    
    # 使用标准 PyTorch backward 而不是 MindSpore autograd（避免 PEFT 兼容性问题）
    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

    if self.args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training

    # 标准 backward 流程
    kwargs = {}
    if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        kwargs["learning_rate"] = self._get_learning_rate()

    if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
        loss = loss / self.current_gradient_accumulation_steps

    if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
        kwargs["scale_wrt_gas"] = False

    learning_rate = kwargs.get("learning_rate")
    
    if self.use_apex:
        raise RuntimeError('mindspore not support apex.')
    else:
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.deepspeed_engine_wrapped.backward(loss, **kwargs)
        elif self.accelerator.distributed_type == DistributedType.MEGATRON_LM:
            return loss.detach()
        elif self.accelerator.scaler is not None:
            self.accelerator.scaler.scale(loss).backward(**kwargs)
        elif learning_rate is not None and self.has_lomo_optimizer:
            self.accelerator.lomo_backward(loss, learning_rate)
        else:
            loss.backward(**kwargs)

    return loss.detach()
