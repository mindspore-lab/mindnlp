# Copyright 2022 Huawei Technologies Co., Ltd
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name
# pylint: disable=assignment-from-none
# pylint: disable=logging-fstring-interpolation
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-boolean-expressions
# pylint: disable=unused-argument
# pylint: disable=attribute-defined-outside-init
# pylint: disable=self-cls-assignment
# pylint: disable=no-name-in-module
# pylint: disable=global-statement
"""
Abstract class for Pretrained models.
"""
import os
import gc
import re
import json
import collections
from dataclasses import dataclass
from typing import Union, Optional, Tuple, Callable, Dict, List
from contextlib import contextmanager
from tqdm.autonotebook import tqdm

import numpy as np
import mindspore
from mindspore import load_checkpoint, save_checkpoint
from mindspore import nn, ops, Tensor, Parameter
from mindspore._c_expression import Tensor as Tensor_

from mindnlp.configs import PT_WEIGHTS_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME, PT_WEIGHTS_INDEX_NAME, \
    SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from mindnlp.utils.download import is_remote_url, download_url, cached_file, get_checkpoint_shard_files
from mindnlp.utils import convert_file_size_to_int, logging, ModelOutput, is_safetensors_available
from mindnlp._legacy.functional import arange
from mindnlp.utils.serialization import load
from mindnlp.injection import set_global_fp16

from .generation import GenerationMixin
from .configuration_utils import PretrainedConfig
from .generation.configuration_utils import GenerationConfig
from .activations import get_activation

logger = logging.get_logger(__name__)

_init_weights = True
init_func = mindspore.common.initializer.initializer


@contextmanager
def no_init_weights(_skip_init, _enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    global _init_weights
    old_init_weights = _init_weights

    if _enable:
        _init_weights = False
        # # Save the original initialization functions
        replace_references(mindspore.common.initializer.initializer, _skip_init, ignore_vars=['init_func'])

    try:
        yield
    finally:
        _init_weights = old_init_weights
        if _enable:
            # # Restore the original initialization functions
            replace_references(mindspore.common.initializer.initializer, init_func, ignore_vars=['init_func'])


def set_initialized_submodules(model: nn.Cell, state_dict_keys):
    """
    Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state
    dict.
    """
    def fix_weight_norm_loaded_keys(loaded_keys: Dict[str, Tensor]) -> List[str]:
        ''' if both `weight_g` and `weight_v` are loaded, pretend `weight` to be loaded :) '''
        patched_keys = []
        for key in loaded_keys:
            if not key.endswith('_g'):
                continue
            key_base = key[:-len('_g')]
            key_v = key_base + '_v'
            if key_v in loaded_keys:
                patched_keys.append(key_base)
        return patched_keys

    not_initialized_submodules = {}
    for module_name, module in model.cells_and_names():
        if module_name:
            loaded_keys = {k.replace(f"{module_name}.", "") for k in state_dict_keys if k.startswith(f"{module_name}.")}
        else:
            loaded_keys = set(state_dict_keys)
        params_keys = {k.replace(f"{module_name}.", "") for k in module.parameters_dict(recurse=False).keys()}
        # NOTE: monkey patching weight_norm
        loaded_keys.update(fix_weight_norm_loaded_keys(loaded_keys))

        if loaded_keys.issuperset(params_keys):
            module._is_initialized = True
        else:
            not_initialized_submodules[module_name] = module

    return not_initialized_submodules


class CellUtilMixin:
    """
    A few utilities to be used as a mixin.
    """

    @property
    def dtype(self) -> mindspore.TensorType:
        """
        `mindspore.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask):
        """create_extended_attention_mask_for_decoder"""
        batch_size, seq_length = input_shape
        seq_ids = arange(seq_length)
        causal_mask = ops.tile((seq_ids[None, None, :]).astype(mindspore.int32),\
         (batch_size, seq_length, 1)) <= seq_ids[None, :, None] # mindspore 2.0
        # causal_mask = Tensor(np.tile(seq_ids[None, None, :].asnumpy(), (batch_size, seq_length, 1))) \
        #     <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        causal_mask = causal_mask.astype(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = ops.concat(
                [
                    ops.ones((batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`mindspore.Tensor`): An attention mask.

        Returns:
            `mindspore.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.ndim == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.ndim == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            encoder_extended_attention_mask = encoder_attention_mask
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.astype(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) \
            * Tensor(np.finfo(mindspore.dtype_to_nptype(self.dtype)).min)

        return encoder_extended_attention_mask

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], dtype = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable
            #   to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = CellUtilMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.astype(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) \
            * Tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min)
        return extended_attention_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`mindspore.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `mindspore.Tensor` with shape `[num_hidden_layers x batch x
            num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.expand_dims(-1)
        else:
            head_mask = ()
            for _ in range(num_hidden_layers):
                head_mask += (None,)

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.expand_dims(0).expand_dims(0).expand_dims(-1).expand_dims(-1)
            head_mask = head_mask.broadcast_to(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.expand_dims(1).expand_dims(-1)\
                .expand_dims(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.astype(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask


class PreTrainedModel(nn.Cell, CellUtilMixin, GenerationMixin):
    """
    Abstract class for Pretrained models
    """
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    _keys_to_ignore_on_save = None

    _tied_weights_keys = None

    _keep_in_fp32_modules = None

    supports_recompute = False

    def __init__(self, config):
        super().__init__(config)
        self._check_and_unset_acl()
        # Save config in model
        self.config = config
        self.name_or_path = config.name_or_path
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    def _check_and_unset_acl(self):
        if "MS" in str(self.__class__.__name__) and \
            'MS_DEV_FORCE_ACL' in os.environ:
            del os.environ['MS_DEV_FORCE_ACL']

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).

        """
        self.init_weights()

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        model = cls(config, **kwargs)

        return model

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        if _init_weights:
            # Initialize weights
            if getattr(self, 'apply', None):
                self.apply(self._initialize_weights)
            else:
                for _, cell in self.name_cells().items():
                    self._initialize_weights(cell)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

    def prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
                layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

        self.base_model._prune_heads(heads_to_prune)

    def _init_weights(self, cell):
        """
        Initialize the weights. This method should be overridden by derived class and is
        the only initialization method that will be called when loading a checkpoint
        using `from_pretrained`. Any attempt to initialize outside of this function
        will be useless as the torch.nn.init function are all replaced with skip.
        """

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_initialized", False):
            return
        self._init_weights(module)
        module._is_initialized = True

    @property
    def base_model(self):
        """
        to get base_model
        """
        return getattr(self, self.base_model_prefix, self)

    def get_input_embeddings(self) -> "nn.Cell":
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Cell`: A mindspore cell mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        raise NotImplementedError

    def set_input_embeddings(self, new_embeddings: nn.Cell):
        """
        Set model's input embeddings.

        Args:
            value (:obj:`nn.Cell`): A mindspore cell mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.set_input_embeddings(new_embeddings)
        raise NotImplementedError

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        resize the model position embeddings if necessary
        """
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__}"
        )

    def get_output_embeddings(self):
        """ Get model's output embeddings
            Return None if the model doesn't have output embeddings
        """
        return None  # Overwrite for models with output embeddings

    def set_output_embeddings(self, new_embeddings: nn.Cell):
        """
        Set model's output embeddings.

        Args:
            value (:obj:`nn.Cell`): A mindspore cell mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.set_output_embeddings(new_embeddings)
        raise NotImplementedError

    def get_position_embeddings(self):
        """
        get the model position embeddings if necessary
        """
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__}"
        )

    def tie_weights(self):
        """
        Make sure we are sharing the input and output embeddings.
        If you need this feature,
        you need to get it yourself output Add the output you need to add to the embeddings function_ Embedding layer,
        otherwise you cannot
        """
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(
                    output_embeddings, self.get_input_embeddings())

        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder, self.base_model_prefix)

        for _, cell in self.cells_and_names():
            if hasattr(cell, "_tie_weights"):
                cell._tie_weights()

    @staticmethod
    def _tie_encoder_decoder_weights(encoder: nn.Cell, decoder: nn.Cell, base_model_prefix: str):
        """tie encoder decoder weights"""
        uninitialized_encoder_weights: List[str] = []
        if decoder.__class__ != encoder.__class__:
            logger.info(
                f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder"
                " weights are correctly initialized."
            )

        def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Cell,
            encoder_pointer: nn.Cell,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            depth=0,
        ):
            assert isinstance(decoder_pointer, nn.Cell) and isinstance(
                encoder_pointer, nn.Cell
            ), f"{decoder_pointer} and {encoder_pointer} have to be of type nn.Module"
            if hasattr(decoder_pointer, "weight"):
                assert hasattr(encoder_pointer, "weight")
                encoder_pointer.weight = decoder_pointer.weight
                encoder_pointer._params['weight'] = decoder_pointer.weight
                if hasattr(decoder_pointer, "bias"):
                    assert hasattr(encoder_pointer, "bias")
                    encoder_pointer.bias = decoder_pointer.bias
                    encoder_pointer._params['bias'] = decoder_pointer.bias
                return

            encoder_cells = encoder_pointer._cells
            decoder_cells = decoder_pointer._cells
            if len(decoder_cells) > 0:
                assert (
                    len(encoder_cells) > 0
                ), f"Encoder cell {encoder_pointer} does not match decoder cell {decoder_pointer}"

                all_encoder_weights = {module_name + "/" + sub_name for sub_name in encoder_cells.keys()}
                encoder_layer_pos = 0
                for name, _ in decoder_cells.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(decoder_cells[decoder_name], type(encoder_cells[encoder_name])) and len(
                            encoder_cells
                        ) != len(decoder_cells):
                            # this can happen if the name corresponds to the position in a list module list of layers
                            # in this case the decoder has added a cross-attention that the encoder does not have
                            # thus skip this step and subtract one layer pos from encoder
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_cells:
                        continue
                    elif depth > 500:
                        raise ValueError(
                            "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is"
                            " a circular dependency between two or more `nn.Cell` of your model."
                        )
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(
                        decoder_cells[decoder_name],
                        encoder_cells[encoder_name],
                        module_name + "/" + name,
                        uninitialized_encoder_weights,
                        depth=depth + 1,
                    )
                    all_encoder_weights.remove(module_name + "/" + encoder_name)

                uninitialized_encoder_weights += list(all_encoder_weights)

        # tie weights recursively
        tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights)
        if len(uninitialized_encoder_weights) > 0:
            logger.warning(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone module weights depending of weither we are using or not
        """
        if hasattr(output_embeddings, 'weight'):
            output_embeddings.weight = input_embeddings.weight
            output_embeddings._params['weight'] = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            if output_embeddings.weight.shape[0] == output_embeddings.bias.shape[0]:
                pass
            else:
                # instantial a new Parameter since mindspore.Parameter do not support assign_value with different shape
                replace_references(output_embeddings.bias, Parameter(ops.pad(
                    output_embeddings.bias.data,
                    (0, output_embeddings.weight.shape[0] -
                    output_embeddings.bias.shape[0]),
                    "constant",
                    0,
                ), name=output_embeddings.bias.name, requires_grad=output_embeddings.bias.requires_grad))

        if hasattr(output_embeddings, "out_channels") and hasattr(input_embeddings, "vocab_size"):
            output_embeddings.out_channels = input_embeddings.vocab_size

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds
        # Update base model and current model config
        self.config.vocab_size = model_embeds.weight.shape[0]
        self.vocab_size = model_embeds.weight.shape[0]
        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)

        self.set_input_embeddings(new_embeddings)
        self.get_input_embeddings().weight.data_sync(True)
        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            new_num_tokens = new_embeddings.weight.shape[0]
        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(
                old_lm_head, new_num_tokens)
            self.set_output_embeddings(new_lm_head)
            self.get_output_embeddings().weight.data_sync(True)


        return self.get_input_embeddings()

    def resize_tokenizer_embeddings(self, new_num_tokens):
        """
        Obtain a new embedding layer or use the original one without updating it.
        """
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``mindspore.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            logger.info(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.shape
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[
                                                                      :num_tokens_to_copy, :]
        return new_embeddings

    def _get_resized_lm_head(
        self, old_lm_head: nn.Dense, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ) -> nn.Dense:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`nn.Dense`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `nn.Dense` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.

        Return:
            `nn.Dense`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """
        if new_num_tokens is None:
            return old_lm_head

        old_num_tokens, old_lm_head_dim = (
            old_lm_head.weight.shape if not transposed else old_lm_head.weight.T.shape
        )

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        if not isinstance(old_lm_head, nn.Dense):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Dense}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Dense}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_lm_head = nn.Dense(
            *new_lm_head_shape,
            has_bias=has_new_lm_head_bias,
        )

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        self._copy_lm_head_original_to_resized(
            new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
        )

        return new_lm_head

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]


    @classmethod
    def load(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
             *args, **kwargs):
        """
        Load a pre-trained checkpoint from a pre-trained model file or url,
        download and cache the pre-trained model file if model name in model list.

        Params:
            pretrained_model_name_or_path:
        """
        return cls.from_pretrained(pretrained_model_name_or_path, args, kwargs)

    @classmethod
    def from_pretrained(    # pylint: disable=too-many-locals
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        use_safetensors: bool = None,
        **kwargs,
    ):
        """from_pretrained"""
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        _ = kwargs.pop("from_pt", True)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        _fast_init = kwargs.pop("_fast_init", True)
        output_loading_info = kwargs.pop("output_loading_info", False)
        subfolder = kwargs.pop("subfolder", "")
        variant = kwargs.pop("variant", None)
        ms_dtype = kwargs.pop("ms_dtype", None)
        _ = kwargs.pop('low_cpu_mem_usage', None)
        revision = kwargs.pop('revision', 'main')

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False

        is_sharded = False
        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, PT_WEIGHTS_NAME)
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, PT_WEIGHTS_NAME)
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
                ):
                    # Load from a MindSpore checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                    )
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                    )
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(PT_WEIGHTS_INDEX_NAME, variant))
                ):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(PT_WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
                ):
                    # Load from a sharded MindSpore checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif use_safetensors:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {PT_WEIGHTS_NAME},"
                        f" found in directory {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                if use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(WEIGHTS_NAME, variant)

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_missing_entries": False,
                        'revision': revision,
                        "token": token
                    }
                    # try safetensors
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
                    use_safetensors = resolved_archive_file is not None

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                            use_safetensors = True

                    if resolved_archive_file is None:
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True

                    if resolved_archive_file is None:
                        filename = _add_variant(PT_WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    if resolved_archive_file is None and filename == _add_variant(PT_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(PT_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True

                    if resolved_archive_file is None:
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(SAFE_WEIGHTS_NAME, variant)}, {_add_variant(PT_WEIGHTS_NAME, variant)}"
                        )
                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as exc:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        ", make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                        f" {_add_variant(PT_WEIGHTS_NAME, variant)}."
                    ) from exc

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        if is_sharded:
            # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                subfolder=subfolder,
                revision=revision
            )

        if pretrained_model_name_or_path is None and state_dict is None:
            raise ValueError("the argument 'pretrained_model_name_or_path' should be "
                             "a string of model name or checkpoint path, but got 'None'.")

        config.name_or_path = pretrained_model_name_or_path
        # Instantiate model.
        if ms_dtype is None or ms_dtype == 'auto':
            ms_dtype = config.ms_dtype

        if ms_dtype is None:
            ms_dtype = mindspore.float32

        use_fp16 = False
        usage_dtype = mindspore.dtype_to_nptype(ms_dtype)
        if ms_dtype == mindspore.bfloat16:
            ms_dtype = mindspore.float16
            usage_dtype = np.float16
            use_fp16 = True

        def empty_initializer(init, shape=None, dtype=mindspore.float32):
            if not isinstance(shape, (tuple, list)):
                shape = (shape,)
            if dtype in (mindspore.float16, mindspore.float32) \
                and ms_dtype is not None:
                dtype = ms_dtype
            return Tensor_(shape=shape, dtype=dtype)

        with no_init_weights(empty_initializer, _fast_init):
            model = cls(config, *model_args, **model_kwargs)

        if ms_dtype != mindspore.float32:
            set_global_fp16(False)

        if is_sharded:
            converted_filenames = resolved_archive_file

        # tie the model weights before retrieving the state_dict
        model.tie_weights()


        ptrs = collections.defaultdict(list)
        for name, tensor in model.parameters_dict().items():
            id_tensor = id(tensor)
            ptrs[id_tensor].append(name)

        # These are all the pointers of shared tensors.
        tied_params = [names for _, names in ptrs.items() if len(names) > 1]

        def load_ckpt(resolved_archive_file):
            if 'ckpt' not in resolved_archive_file:
                if use_safetensors:
                    from safetensors.numpy import load_file
                    origin_state_dict = load_file(resolved_archive_file)
                    if use_fp16:
                        logger.warning_once("MindSpore do not support bfloat16 dtype, we will automaticlly convert to float16")

                    state_dict = {k: Parameter(v.astype(np.float32).astype(usage_dtype)) for k, v in origin_state_dict.items()}
                else:
                    state_dict = load(resolved_archive_file)
            else:
                try:
                    state_dict = load_checkpoint(str(resolved_archive_file))
                except Exception as exc:
                    raise OSError(
                        f"Unable to load weights from mindspore checkpoint file '{resolved_archive_file}'. "
                    ) from exc

            new_state_dict = {}
            for key, value in state_dict.items():
                key = key.replace('gamma', 'weight').replace('beta', 'bias').replace('embedding_table', 'weight')
                value.name = value.name.replace('gamma', 'weight').replace('beta', 'bias')\
                    .replace('embedding_table', 'weight')
                new_state_dict[key] = value
            return new_state_dict

        keys_missing = list(model.parameters_dict().keys())
        param_id_set = set()

        use_keep_in_fp32_modules = False
        if model._keep_in_fp32_modules:
            use_keep_in_fp32_modules = True

        remove_prefix_from_model = None
        add_prefix_to_model = None

        def fix_weight_norm_missing_keys(state_dict_keys: dict, keys_missing:List[str]) -> List[str]:
            ''' if both `weight_g` and `weight_v` are loaded, key `weight` is not missing :) '''
            non_missing_keys = []
            for key in keys_missing:
                if f'{key}_g' in state_dict_keys and f'{key}_v' in state_dict_keys:
                    non_missing_keys.append(key)
            return non_missing_keys

        def load_param_into_net(model: nn.Cell, param_dict: dict, prefix: str):
            state_dict_keys = list(param_dict.keys())
            keep_in_fp32_modules = model._keep_in_fp32_modules
            keys_unexpected = list(param_dict.keys())

            has_prefix_module = any(s.startswith(prefix) for s in keys_unexpected)
            expects_prefix_module = any(s.startswith(prefix) for s in keys_missing)

            nonlocal remove_prefix_from_model
            nonlocal add_prefix_to_model
            remove_prefix_from_model = not has_prefix_module and expects_prefix_module
            add_prefix_to_model = has_prefix_module and not expects_prefix_module

            for pname_in_net, param in model.parameters_and_names():
                if add_prefix_to_model:
                    param_name = prefix + '.' + pname_in_net
                elif remove_prefix_from_model:
                    param_name = pname_in_net.replace(f'{prefix}.', '')
                else:
                    param_name = pname_in_net

                if id(param) in param_id_set:
                    # for tied params
                    if param_name in keys_unexpected:
                        keys_unexpected.remove(param_name)
                    continue

                new_param = param_dict.pop(param_name, None)

                if new_param is not None:
                    use_replace = False
                    if new_param.shape != param.shape:
                        if not ignore_mismatched_sizes:
                            raise RuntimeError(f'The shape of parameter `{param.name} is {param.shape}, but got mismatch parameter'
                                            f' `{param_name} with shape {new_param.shape} in checkpoint, '
                                            f'\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method.')
                        logger.warning(f'The shape of parameter `{param.name} is {param.shape}, but got mismatch parameter'
                                        f' `{param_name} with shape {new_param.shape} in checkpoint, ')
                        continue

                    if use_keep_in_fp32_modules and \
                        any(module_to_keep_in_fp32 in pname_in_net.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules):
                        new_param = new_param.astype(mindspore.float32)
                    elif ms_dtype and param.dtype in (mindspore.float32, mindspore.float16):
                        new_param = new_param.astype(ms_dtype)

                    if new_param.dtype != param.dtype or new_param.shape != param.shape:
                        use_replace = True

                    if use_replace:
                        if isinstance(new_param, Parameter):
                            new_param.name = param.name
                            new_param.requires_grad = param.requires_grad
                            replace_references(param, new_param)
                        else:
                            replace_references(param, Parameter(new_param, requires_grad=param.requires_grad, name=param.name))
                    else:
                        param.set_data(new_param)
                    keys_unexpected.remove(param_name)
                    keys_missing.remove(pname_in_net)
                    param_id_set.add(id(param))
                else:
                    # fix missing value parameter dtype cast.
                    if ms_dtype and ms_dtype != param.dtype:
                        new_param = param.astype(ms_dtype)
                        replace_references(param, Parameter(new_param, name=param.name, requires_grad=param.requires_grad))

            # NOTE: monkey patching weight_norm
            for key in fix_weight_norm_missing_keys(state_dict_keys, keys_missing):
                keys_missing.remove(key)

            return keys_unexpected, keys_missing

        all_keys_unexpected = None
        if state_dict is None:
            if is_sharded:
                all_keys_unexpected = []
                for name in tqdm(converted_filenames, desc="Loading checkpoint shards"):
                    state_dict = load_ckpt(name)
                    keys_unexpected, keys_missing = load_param_into_net(model, state_dict, cls.base_model_prefix)
                    all_keys_unexpected.extend(keys_unexpected)
                    del state_dict
                    gc.collect()
                loaded_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                state_dict = load_ckpt(resolved_archive_file)
                loaded_keys = list(state_dict.keys())
                all_keys_unexpected, keys_missing = load_param_into_net(model, state_dict, cls.base_model_prefix)
        else:
            loaded_keys = list(state_dict.keys())
            all_keys_unexpected, keys_missing = load_param_into_net(model, state_dict, cls.base_model_prefix)

        loaded_add_keys = []
        for group in tied_params:
            missing_in_group = [k for k in keys_missing if k in group]
            if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
                loaded_add_keys = [k for k in keys_missing if k in missing_in_group]
                keys_missing = [k for k in keys_missing if k not in missing_in_group]

        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                keys_missing = [k for k in keys_missing if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                all_keys_unexpected = [k for k in all_keys_unexpected if re.search(pat, k) is None]

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # retrieve unintialized modules and initialize before maybe overriding that with the pretrained weights.
        if _fast_init:
            if not ignore_mismatched_sizes:
                if remove_prefix_from_model:
                    _loaded_keys = [f"{cls.base_model_prefix}.{k}" for k in loaded_keys]
                elif add_prefix_to_model:
                    _loaded_keys = [k[len(cls.base_model_prefix) + 1 :] for k in loaded_keys]
                else:
                    _loaded_keys = loaded_keys

                _loaded_keys += loaded_add_keys
                _ = set_initialized_submodules(model, _loaded_keys)
            else:
                _ = dict(model.cells_and_names())

            model.apply(model._initialize_weights)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.set_train(False)

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    subfolder=subfolder,
                    revision=revision,
                    **kwargs,
                )
            except OSError:
                logger.warning(
                    "Generation config file not found, using a generation config created from the model config."
                )

        if output_loading_info:
            loading_info = {
                "missing_keys": keys_missing,
                "unexpected_keys": all_keys_unexpected,
            }
            return model, loading_info

        if all_keys_unexpected:
            logger.warning(f'The following parameters in checkpoint files are not loaded:\n'
                           f'{all_keys_unexpected}')
        if keys_missing:
            logger.warning(f'The following parameters in models are missing parameter:\n'
                           f'{keys_missing}')
        return model

    def save(self, save_dir):
        """ Save a model and its configuration file to a directory, so that
            it can be re-loaded using the `:func:`PreTrainedModel.from_pretrained`` class method.

            Arguments:
                save_dir: directory to which to save.
        """
        if os.path.isfile(save_dir):
            logger.error(f"Provided path ({save_dir}) should be a directory, not a file")
            return
        os.makedirs(save_dir, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.cell if hasattr(self, "cell") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
        save_checkpoint(model_to_save, output_model_file)

        logger.info(f"Model weights saved in {output_model_file}")

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
        # Alternativelly, the model can also have a custom `generate` function.
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True

    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """
        if (attention_mask is not None) or (self.config.pad_token_id is None):
            return

        # Check only the first and last input IDs to reduce overhead.
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            warn_string = (
                "We strongly recommend passing in an `attention_mask` since your input_ids may be padded."
            )

            # If the pad token is equal to either BOS, EOS, or SEP, we do not know whether the user should use an
            # attention_mask or not. In this case, we should still show a warning because this is a rare case.
            if (
                (self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id)
                or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id)
                or (self.config.sep_token_id is not None and self.config.sep_token_id == self.config.pad_token_id)
            ):
                warn_string += (
                    f"\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical "
                    f"to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), "
                    f"or the `sep_token_id` ({self.config.sep_token_id}), and your input is not padded."
                )

            logger.warning(warn_string)

    def parameters_and_names(self, name_prefix='', expand=True):
        """
        fix ignore tied weights
        """
        cells = []
        if expand:
            cells = self.cells_and_names(name_prefix=name_prefix)
        else:
            cells.append((name_prefix, self))

        for cell_name, cell in cells:
            params = cell._params.items()
            for par_name, par in params:
                if par is not None and par.inited_param is not None:
                    par = par.inited_param
                if par is not None:
                    par_new_name = par_name
                    if cell_name:
                        par_new_name = cell_name + '.' + par_new_name

                    yield par_new_name, par


    def num_parameters(self, only_trainable=False):
        """return parameters count"""
        total = 0
        param_set = set()
        for param in self.get_parameters():
            param_id = id(param)
            if param_id not in param_set and (only_trainable or param.requires_grad):
                total += param.size
            param_set.add(param_id)
        return total


    def trainable_params(self, recurse=True):
        """
        fix duplicated weights
        """
        return list(set(filter(lambda x: x.requires_grad, self.get_parameters(expand=recurse))))

    def check_names_and_refresh_name(self):
        """
        fix ignore tied weights
        """
        if not hasattr(self, "_params"):
            return
        all_name = dict(self.parameters_and_names()).keys()

        if len(set(all_name)) < len(all_name):
            self.update_parameters_name()
            self.check_names()

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = mindspore.save_checkpoint,
        max_shard_size: Union[int, str] = "5GB",
        variant: Optional[str] = None,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~PreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
                without CPU OOM issues.

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
                keys of the state dict of adapters needs to be pre-pended with `base_model.model`. Advanced users can
                disable this behaviours by setting `save_peft_format` to `False`.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)


        # Only save the model itself if we are using distributed training
        model_to_save = self

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.ms_dtype = str(dtype).lower()

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save the config
        if is_main_process:
            model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)


        # Save the model
        if state_dict is None:
            state_dict = model_to_save.parameters_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]

        # Shard the model if it is too big.
        weights_name = _add_variant(WEIGHTS_NAME, variant)

        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, shard in shards.items():
            save_function(shard, os.path.join(save_directory, shard_file))

        if index is None:
            path_to_weights = os.path.join(save_directory, _add_variant(WEIGHTS_NAME, variant))
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    def enable_recompute(self):
        """Activates recompute (aka gradient checkpointing) for the current model."""
        if not self.supports_recompute:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        for _, cell in self.cells_and_names():
            if hasattr(cell, "_set_recompute"):
                cell._set_recompute()

    def check_names(self):
        pass

def get_parameter_dtype(parameter: Union[nn.Cell, GenerationMixin, "ModuleUtilsMixin"]):
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None
    for t in parameter.get_parameters():
        last_dtype = t.dtype
        if ops.is_floating_point(t):
            return t.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype

    return last_dtype

def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name

def shard_checkpoint(
    state_dict: Dict[str, mindspore.Tensor], max_shard_size: Union[int, str] = "10GB", weights_name: str = WEIGHTS_NAME
):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger than `max_shard_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, torch.Tensor]`): The state dictionary of a model to save.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
        weights_name (`str`, *optional*, defaults to `"pytorch_model.bin"`):
            The name of the model save file.
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = [{}]
    last_block_size = 0
    total_size = 0
    storage_id_to_block = {}

    for key, weight in state_dict.items():
        # when bnb serialization is used the weights in the state dict can be strings
        # check: https://github.com/huggingface/transformers/pull/24416 for more details
        if isinstance(weight, str):
            continue
        storage_id = id(weight)

        # If a `weight` shares the same underlying storage as another tensor, we put `weight` in the same `block`
        if storage_id in storage_id_to_block:
            block_id = storage_id_to_block[storage_id]
            sharded_state_dicts[block_id][key] = weight
            continue

        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        # If this weight is going to tip up over the maximal size, we split, but only if we have put at least one
        # weight in the current shard.
        if last_block_size + weight_size > max_shard_size and len(sharded_state_dicts[-1]) > 0:
            sharded_state_dicts.append({})
            last_block_size = 0

        sharded_state_dicts[-1][key] = weight
        last_block_size += weight_size
        total_size += weight_size
        storage_id_to_block[storage_id] = len(sharded_state_dicts) - 1

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".ckpt", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.ckpt")
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index

def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(torch.float32)
    4
    ```
    """
    if dtype == mindspore.bool_:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


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
        self, hidden_states: mindspore.Tensor, p_mask: Optional[mindspore.Tensor] = None
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (`mindspore.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            `mindspore.Tensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == mindspore.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


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
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dense_1 = nn.Dense(config.hidden_size, 1)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        start_states: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        p_mask: Optional[mindspore.Tensor] = None,
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`mindspore.Tensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            p_mask (`mindspore.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `mindspore.Tensor`: The end logits for SQuAD.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].broadcast_to((-1, -1, hsz))  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather_elements(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.broadcast_to((-1, slen, -1))  # shape (bsz, slen, hsz)

        x = self.dense_0(ops.cat([hidden_states, start_states], axis=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == mindspore.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

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
        hidden_states: mindspore.Tensor,
        start_states: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        cls_index: Optional[mindspore.Tensor] = None,
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`mindspore.Tensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            cls_index (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `mindspore.Tensor`: The SQuAD 2.0 answer class.
        """
        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].broadcast_to((-1, -1, hsz))  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather_elements(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].broadcast_to((-1, -1, hsz))  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather_elements(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(ops.cat([start_states, cls_token_state], axis=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x

@dataclass
class SquadHeadOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a [`~modeling_utils.SQuADHead`].

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`mindspore.Tensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`mindspore.Tensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`mindspore.Tensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        end_top_index (`mindspore.Tensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`mindspore.Tensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.

    """

    loss: Optional[mindspore.Tensor] = None
    start_top_log_probs: Optional[mindspore.Tensor] = None
    start_top_index: Optional[mindspore.Tensor] = None
    end_top_log_probs: Optional[mindspore.Tensor] = None
    end_top_index: Optional[mindspore.Tensor] = None
    cls_logits: Optional[mindspore.Tensor] = None


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
        hidden_states: mindspore.Tensor,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        cls_index: Optional[mindspore.Tensor] = None,
        is_impossible: Optional[mindspore.Tensor] = None,
        p_mask: Optional[mindspore.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[SquadHeadOutput, Tuple[mindspore.Tensor]]:
        """
        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                Final hidden states of the model on the sequence tokens.
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Positions of the first token for the labeled span.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Positions of the last token for the labeled span.
            cls_index (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.
            is_impossible (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Whether the question has a possible answer in the paragraph or not.
            p_mask (`mindspore.Tensor` of shape `(batch_size, seq_len)`, *optional*):
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
                if x is not None and x.ndim > 1:
                    x = x.squeeze(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            start_loss = ops.cross_entropy(start_logits, start_positions)
            end_loss = ops.cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                cls_loss = ops.binary_cross_entropy_with_logits(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            return SquadHeadOutput(loss=total_loss) if return_dict else (total_loss,)

        # during inference, compute the end logits based on beam search
        _, slen, hsz = hidden_states.shape
        start_log_probs = ops.softmax(start_logits, axis=-1)  # shape (bsz, slen)

        start_top_log_probs, start_top_index = ops.topk(
            start_log_probs, self.start_n_top, dim=-1
        )  # shape (bsz, start_n_top)
        start_top_index_exp = start_top_index.unsqueeze(-1).broadcast_to((-1, -1, hsz))  # shape (bsz, start_n_top, hsz)
        start_states = ops.gather_elements(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
        start_states = start_states.unsqueeze(1).broadcast_to((-1, slen, -1, -1))  # shape (bsz, slen, start_n_top, hsz)

        hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
            start_states
        )  # shape (bsz, slen, start_n_top, hsz)
        p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
        end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
        end_log_probs = ops.softmax(end_logits, axis=1)  # shape (bsz, slen, start_n_top)

        end_top_log_probs, end_top_index = ops.topk(
            end_log_probs, self.end_n_top, dim=1
        )  # shape (bsz, end_n_top, start_n_top)
        end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
        end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

        start_states = ops.einsum("blh,bl->bh", hidden_states, start_log_probs)
        cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

        if not return_dict:
            return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)
        return SquadHeadOutput(
            start_top_log_probs=start_top_log_probs,
            start_top_index=start_top_index,
            end_top_log_probs=end_top_log_probs,
            end_top_index=end_top_index,
            cls_logits=cls_logits,
        )


class SequenceSummary(nn.Cell):
    """
    GPTDoubleHeadsModel and GPT2DoubleHeadsModel class that self.multiple_choice_head
    """

    def __init__(self, config):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = nn.Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Dense(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation = get_activation(activation_string) if activation_string else nn.Identity()

        self.first_dropout = nn.Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(p=config.summary_first_dropout)

        self.last_dropout = nn.Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(p=config.summary_last_dropout)

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
                cls_index = cls_index.expand_dims(-1).expand_dims(-1)
                cls_index = cls_index.broadcast_to((-1,) * (cls_index.ndim - 1) + (hidden_states.shape[-1],))
            output = hidden_states.gather_elements(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        else:
            output = hidden_states

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output

def replace_references(old_obj, new_obj, ignore_vars=None):
    """use replace_references instead of Tensor.set_data due to mindspore errors."""
    # Get all objects referring to old_obj
    if ignore_vars is None:
        ignore_vars = []
    referrers = gc.get_referrers(old_obj)
    # Replace references
    for referrer in referrers:
        if isinstance(referrer, dict):
            # If the reference is in a dictionary
            for key, value in referrer.items():
                if value is old_obj:
                    if key not in ignore_vars:
                        referrer[key] = new_obj
        elif isinstance(referrer, list):
            # If the reference is in a list or tuple
            index = referrer.index(old_obj)
            referrer[index] = new_obj
        elif isinstance(referrer, tuple):
            pass
        elif hasattr(referrer, '__dict__'):
            # If the reference is in the __dict__ of an object
            for key, value in referrer.__dict__.items():
                if value is old_obj:
                    setattr(referrer, key, new_obj)
