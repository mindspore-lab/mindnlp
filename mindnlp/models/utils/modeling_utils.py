# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindNLP Model Utils"""
# pylint: disable=C0412
# pylint: disable=C0103

from typing import Optional
import os
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.nn import CrossEntropyLoss
from .activations import get_activation
from ...abc.backbones.pretrained import PretrainedConfig

XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

try:
    from mindspore.nn import Identity
except ImportError:
    # Older MindSpore compatibility
    class Identity(nn.Cell):
        r"""A placeholder identity operator that is argument-insensitive."""

        def __init__(self):
            super().__init__()

        def construct(self, hidden_states):
            """
            Return hidden value
            """
            return hidden_states


class SequenceSummary(nn.Cell):
    """
    GPT2DoubleHeadsModel class that self.multiple_choice_head
    """

    def __init__(self, config):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Dense(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation = get_activation(activation_string) if activation_string else Identity()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def construct(self, hidden_states: Tensor, cls_index: Optional[Tensor] = None) -> Tensor:
        if self.summary_type == "last":
            output = hidden_states[:, -1, :]
        elif self.summary_type == "first":
            output = hidden_states[:, 0, :]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = ops.fill(
                    mindspore.int64,
                    hidden_states[..., :1, :].shape,
                    hidden_states.shape[-2] - 1,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = ops.BroadcastTo(shape=((-1,) * (cls_index.ndim - 1) + (hidden_states.shape[-1],)))(cls_index)
            output = hidden_states.gather_elements(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output

class PoolerStartLogits(nn.Cell):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, 1)

    def construct(
        self, hidden_states: Tensor, p_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            `torch.FloatTensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)

        #TODO : get_parameter_dtype(self)

        # if p_mask is not None:
        #     if get_parameter_dtype(self) == torch.float16:
        #         x = x * (1 - p_mask) - 65500 * p_mask
        #     else:
        #         x = x * (1 - p_mask) - 1e30 * p_mask

        return x

class SQuADHead(nn.Cell):
    r"""
    A SQuAD head inspired by XLNet.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config):
        super().__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def construct(
        self,
        hidden_states: Tensor,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        cls_index: Optional[Tensor] = None,
        is_impossible: Optional[Tensor] = None,
        p_mask: Optional[Tensor] = None,
        return_dict: bool = False,
    ) -> Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                Final hidden states of the model on the sequence tokens.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Positions of the first token for the labeled span.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Positions of the last token for the labeled span.
            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.
            is_impossible (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Whether the question has a possible answer in the paragraph or not.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
        """

        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                try:
                    if x is not None and x.ndim > 1:
                        x = ops.squeeze(x,axis=-1)
                except:ValueError


            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 
                # so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            return (total_loss,)

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.shape
            start_log_probs = ops.softmax(start_logits,axis=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = ops.topk(
                start_log_probs, self.start_n_top
            )  # shape (bsz, start_n_top)
            start_top_index_exp = ops.BroadcastTo(shape
                                                  =(-1, -1, hsz))(start_top_index.unsqueeze(-1))  
                                                  # shape (bsz, start_n_top, hsz)
            start_states = ops.gather_elements(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = ops.BroadcastTo(shape
                                           =(-1, slen, -1, -1))(start_states.unsqueeze(1))  
                                                  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = ops.softmax(end_logits, axis=1)  # shape (bsz, slen, start_n_top)
            end_top_log_probs, end_top_index = ops.topk(
                end_log_probs, self.end_n_top
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = ops.matmul(hidden_states,start_log_probs)

            # start_states = ops.einsum("blh,bl->bh", hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

            return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)


class PoolerEndLogits(nn.Cell):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense_0 = nn.Dense(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dense_1 = nn.Dense(config.hidden_size, 1)

    def construct(
        self,
        hidden_states: Tensor,
        start_states: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        p_mask: Optional[Tensor] = None,
    ) -> Tensor:

        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = ops.BroadcastTo(shape
                                              =(-1, -1, hsz))(start_positions[:, None, None])  # shape (bsz, 1, hsz)
            start_states = ops.gather_elements(hidden_states,-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = ops.BroadcastTo(shape=(-1, slen, -1))(start_states)  # shape (bsz, slen, hsz)

        x = self.dense_0(ops.cat([hidden_states, start_states], axis=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)
       #TODO : get_parameter_dtype(self)
        # if p_mask is not None:
        #     if get_parameter_dtype(self) == mindspore.float16:
        #         x = x * (1 - p_mask) - 65500 * p_mask
        #     else:
        #         x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerAnswerClass(nn.Cell):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Dense(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Dense(config.hidden_size, 1, has_bias=False)

    def construct(
        self,
        hidden_states: Tensor,
        start_states: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        cls_index: Optional[Tensor] = None,
    ) -> Tensor:

        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = ops.BroadcastTo(shape
                                              =(-1, -1, hsz))(start_positions[:, None, None])  # shape (bsz, 1, hsz)
            start_states = ops.gather_elements(hidden_states,-2, start_positions).squeeze(-2)  # shape (bsz, hsz)
        if cls_index is not None:
            cls_index = ops.BroadcastTo(shape=(-1, -1, hsz))(cls_index[:, None, None])  # shape (bsz, 1, hsz)
            cls_token_state = ops.gather_elements(hidden_states,-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(ops.cat([start_states, cls_token_state], axis=-1))
        x = self.activation(x)
        x = ops.squeeze(self.dense_1(x),axis=-1)

        return x
