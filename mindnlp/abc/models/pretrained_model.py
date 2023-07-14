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
# pylint: disable=E1128
# pylint: disable=C0103
# pylint: disable=C0412
# pylint: disable=W0642
# pylint: disable=W0212
# pylint: disable=W0201
"""
Abstract class for Pretrained models.
"""
import os
import gc
from typing import Union, Optional
from tqdm.autonotebook import tqdm

from mindspore import nn, ops
from mindspore import log as logger
from mindspore.train.serialization import save_checkpoint

from mindnlp.configs import HF_MODEL_URL_BASE, DEFAULT_ROOT
from mindnlp.utils.download import cached_path, get_checkpoint_shard_files
from mindnlp.abc.configs import PreTrainedConfig, GenerationConfig
from mindnlp.abc.mixins import CellUtilMixin, GenerationMixin
from mindnlp.utils import less_min_pynative_first
if less_min_pynative_first:
    from mindspore import load_checkpoint
else:
    from mindnlp._legacy.utils import load_checkpoint

_init_weights = True
WEIGHTS_NAME = "mindspore.ckpt"
WEIGHTS_INDEX_NAME = "mindspore.ckpt.index.json"
HF_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"


class PreTrainedModel(nn.Cell, CellUtilMixin, GenerationMixin):
    """
    Abstract class for Pretrained models
    """
    config_class = None
    pretrained_model_archive_map = {}
    base_model_prefix = ""
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        # Save config in model
        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).

        """
        self.init_weights()

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

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        self._init_weights(module)

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

        for cell in self.cells():
            if hasattr(cell, "_tie_weights"):
                cell._tie_weights()

    @staticmethod
    def _tie_encoder_decoder_weights(encoder: nn.Cell, decoder: nn.Cell, base_model_prefix: str):
        """tie encoder decoder weights"""

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """ Tie or clone module weights depending of weither we are using or not
        """
        output_embeddings.weight = input_embeddings.embedding_table

        if output_embeddings.has_bias:
            output_embeddings.bias.set_data(ops.pad(
                output_embeddings.bias.data,
                (0, output_embeddings.weight.shape[0] -
                 output_embeddings.bias.shape[0]),
                "constant",
                0,
            ))
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_channels = input_embeddings.vocab_size

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
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
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # if word embeddings are not tied, make sure that lm head is resized as well

        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(
                old_lm_head, new_num_tokens)
            self.set_output_embeddings(new_lm_head)

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

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
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
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.embedding_table.shape
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.embedding_table.data[:num_tokens_to_copy, :] = old_embeddings.embedding_table.data[
            :num_tokens_to_copy, :]

        return new_embeddings

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
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """from_pretrained"""
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", os.path.join(DEFAULT_ROOT, 'models'))
        from_pt = kwargs.pop("from_pt", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)

        is_sharded = False
        # Load config if we don't provide a configuration
        if not isinstance(config, PreTrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                from_pt=from_pt,
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

        folder_name = None
        # Load model
        if pretrained_model_name_or_path is not None:
            folder_name = pretrained_model_name_or_path
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map and not from_pt:
                archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            elif os.path.isdir(pretrained_model_name_or_path):
                archive_file = os.path.join(
                    pretrained_model_name_or_path, "mindspore_model.ckpt")
            elif os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif from_pt:
                archive_file = HF_MODEL_URL_BASE.format(
                    pretrained_model_name_or_path)
            else:
                raise ValueError(
                    f'not found model of {pretrained_model_name_or_path}.')

            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    proxies=proxies,
                    folder_name=folder_name
                )[0]

                if resolved_archive_file is None:
                    base_url = '/'.join(archive_file.split('/')[:-1])
                    archive_file = base_url + '/' + HF_WEIGHTS_INDEX_NAME if from_pt else \
                        base_url + '/' + WEIGHTS_INDEX_NAME

                    resolved_archive_file = str(cached_path(
                        archive_file,
                        cache_dir=cache_dir,
                        proxies=proxies,
                        folder_name=folder_name
                    )[0])

                    if resolved_archive_file is not None:
                        cached_filenames, _ = get_checkpoint_shard_files(
                            pretrained_model_name_or_path=folder_name,
                            index_filename=resolved_archive_file,
                            cache_dir=cache_dir,
                            url=base_url,
                            proxies=proxies,
                            subfolder=folder_name
                        )
                        is_sharded = True
                    else:
                        raise EnvironmentError(f"Couldn't reach server at '{archive_file}' to download pretrained weights.")

            except EnvironmentError as exc:
                raise exc

            if resolved_archive_file == archive_file:
                logger.info("loading weights file %s", archive_file)
            else:
                logger.info("loading weights file %s from cache at %s",
                            archive_file, resolved_archive_file)
        else:
            raise ValueError("the argument 'pretrained_model_name_or_path' should be "
                             "a string of model name or checkpoint path, but got 'None'.")

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)


        if from_pt:
            if is_sharded:
                converted_filenames = []
                for name in cached_filenames:
                    converted = cls.convert_torch_to_mindspore(
                        str(name), prefix=cls.base_model_prefix)
                    converted_filenames.append(converted)
            else:
                resolved_archive_file = cls.convert_torch_to_mindspore(
                    str(resolved_archive_file), prefix=cls.base_model_prefix)
        else:
            if is_sharded:
                converted_filenames = cached_filenames

        def load_ckpt(resolved_archive_file):
            try:
                state_dict = load_checkpoint(str(resolved_archive_file))
            except Exception as exc:
                raise OSError(
                    f"Unable to load weights from mindspore checkpoint file '{resolved_archive_file}'. "
                ) from exc
            return state_dict

        def load_param_into_net(model: nn.Cell, param_dict: dict, prefix: str):
            not_loaded = list(param_dict.keys())
            for _, param in model.parameters_and_names():
                if param.name in param_dict:
                    param_name = param.name
                else:
                    param_name = prefix + '.' + param.name
                new_param = param_dict.get(param_name)
                if new_param is not None:
                    param.set_dtype(new_param.dtype)
                    param.assign_value(new_param)
                    not_loaded.remove(param_name)

            return not_loaded

        if state_dict is None:
            if is_sharded:
                not_loaded = []
                for name in tqdm(converted_filenames, desc="Loading checkpoint shards"):
                    state_dict = load_ckpt(name)
                    not_loaded.extend(load_param_into_net(model, state_dict, cls.base_model_prefix))
                    del state_dict
                    gc.collect()
            else:
                state_dict = load_ckpt(resolved_archive_file)
                not_loaded = load_param_into_net(model, state_dict, cls.base_model_prefix)

        if not_loaded:
            logger.warning(f'The following parameters in checkpoint files are not loaded:\n'
                           f'{not_loaded}')

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


    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation
        if "GenerationMixin" in str(self.prepare_inputs_for_generation.__func__):
            return False
        return True
