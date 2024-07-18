# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""
Image/Text processor class for CLIP
"""

import warnings

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class CLIPProcessor(ProcessorMixin):
    r"""
    Constructs a CLIP processor which wraps a CLIP image processor and a CLIP tokenizer into a single processor.

    [`CLIPProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`CLIPTokenizerFast`]. See the
    [`~CLIPProcessor.__call__`] and [`~CLIPProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        """
        Initialize a CLIPProcessor object.
        
        Args:
            self (object): The instance of the class.
            image_processor (object, optional): An image processor object used for processing images. 
                If not provided, it can be passed as part of the kwargs parameter.
            tokenizer (object): A tokenizer object used for tokenizing text inputs.
        
        Returns:
            None.
        
        Raises:
            ValueError: If either `image_processor` or `tokenizer` is not specified.
            FutureWarning: If the deprecated argument `feature_extractor` is used,
                a warning is issued recommending to use `image_processor` instead.
        """
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to CLIPTokenizerFast's [`~CLIPTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

                - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
                - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
                `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
                `None`).
                - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **kwargs)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data={**image_features}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        This method, 'model_input_names', is a property of the 'CLIPProcessor' class.
        It returns a list of unique model input names derived from the tokenizer and image processor model input names.

        Args:
            self: An instance of the 'CLIPProcessor' class.

        Returns:
            The method returns a list of unique model input names derived from the tokenizer and image processor model input names.

        Raises:
            No exceptions are explicitly raised by this method.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @property
    def feature_extractor_class(self):
        """
        This method returns the image processor class used for extracting features in the CLIPProcessor class.

        Args:
            self: An instance of the CLIPProcessor class.

        Returns:
            None

        Raises:
            FutureWarning: If the method is called, a FutureWarning will be raised to inform the user that
                `feature_extractor_class` is deprecated and will be removed in v5. It is recommended to use
                `image_processor_class` instead.

        Note:
            The returned image processor class is responsible for extracting features from images in the CLIPProcessor.

        Example:
            ```python
            >>> clip_processor = CLIPProcessor()
            >>> clip_processor.feature_extractor_class
            <class 'image_processor.ImageProcessor'>
            ```
        """
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        """
        This method is deprecated and will be removed in v5. Use `image_processor` instead.
        
        Args:
            self: An instance of the CLIPProcessor class.
        
        Returns:
            None.
        
        Raises:
            FutureWarning: This method raises a FutureWarning to alert users that it is deprecated and will be removed in v5.
        """
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor

__all__ = ['CLIPProcessor']
