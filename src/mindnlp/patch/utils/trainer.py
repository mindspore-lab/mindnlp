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



    def forward_fn(inputs, num_items_in_batch):
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            raise RuntimeError('mindspore not support apex.')
        else:
            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            learning_rate = kwargs.get("learning_rate")


            loss_true = loss

            if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.accelerator.gradient_accumulation_steps

            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                self.accelerator.deepspeed_engine_wrapped.backward(loss, **kwargs)
            elif self.accelerator.distributed_type == DistributedType.MEGATRON_LM:
                return
            elif self.accelerator.scaler is not None:
                loss = self.accelerator.scaler.scale(loss).backward(**kwargs)
            elif learning_rate is not None and self.has_lomo_optimizer:
                self.accelerator.lomo_backward(loss, learning_rate)
            else:
                loss.backward(**kwargs)

        return loss, loss_true

    if not hasattr(self, 'grad_fn'):
        self.grad_fn = autograd.value_and_grad(forward_fn, model.trainable_params(), has_aux=True)

    loss_scaled, (loss_true,) = self.grad_fn(inputs, num_items_in_batch)

    return loss_true
