"""Trainer CallBacks Module."""
from typing import Optional, Union

import mindspore

from mindnlp.engine.callbacks import TrainerCallback
from ...transformers.modeling_utils import PreTrainedModel




class SyncRefModelCallback(TrainerCallback):
    """Sync Reference Model in training."""
    def __init__(
        self,
        ref_model: Union[PreTrainedModel, mindspore.nn.Cell],
        accelerator: Optional[None],
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_target_model(model, target_model, alpha):
        for target_param, copy_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.mul_(1.0 - alpha).add_(copy_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model(model, target_model, alpha):
        """sync the target model with training model."""
        # deepspeed_plugin = AcceleratorState().deepspeed_plugin
        # if deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3:
        #     with deepspeed.zero.GatheredParameters(
        #         list(model.parameters()) + list(target_model.parameters()), modifier_rank=0
        #     ):
        #         if deepspeed.comm.get_rank() == 0:
        #             SyncRefModelCallback._sync_target_model(model, target_model, alpha)
        # else:
        SyncRefModelCallback._sync_target_model(model, target_model, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            # if self.accelerator:
            #     model = self.accelerator.unwrap_model(model)
            self.sync_target_model(model, self.ref_model, args.ref_model_mixup_alpha)
