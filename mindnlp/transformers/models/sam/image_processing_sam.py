# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for SAM."""
import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from mindspore.ops._primitive_cache import _get_cache_prim

from ....configs import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ....utils import (
    TensorType,
    is_mindspore_available,
    logging,
    requires_backends,
)


if is_mindspore_available():
    import mindspore
    from mindnlp.core import ops
    from mindnlp.core.nn import functional as F


logger = logging.get_logger(__name__)


class SamImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SAM image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"longest_edge" -- 1024}`):
            Size of the output image after resizing. Resizes the longest edge of the image to match
            `size["longest_edge"]` while maintaining the aspect ratio. Can be overridden by the `size` parameter in the
            `preprocess` method.
        mask_size (`dict`, *optional*, defaults to `{"longest_edge" -- 256}`):
            Size of the output segmentation map after resizing. Resizes the longest edge of the image to match
            `size["longest_edge"]` while maintaining the aspect ratio. Can be overridden by the `mask_size` parameter
            in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Wwhether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to the specified `pad_size`. Can be overridden by the `do_pad` parameter in the
            `preprocess` method.
        pad_size (`dict`, *optional*, defaults to `{"height" -- 1024, "width" -- 1024}`):
            Size of the output image after padding. Can be overridden by the `pad_size` parameter in the `preprocess`
            method.
        mask_pad_size (`dict`, *optional*, defaults to `{"height" -- 256, "width" -- 256}`):
            Size of the output segmentation map after padding. Can be overridden by the `mask_pad_size` parameter in
            the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        mask_size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        pad_size: int = None,
        mask_pad_size: int = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        """
        Initializes an instance of the SamImageProcessor class.
        
        Args:
            self: The instance of the class.
            do_resize (bool): Determines whether resizing of images should be performed. Defaults to True.
            size (Dict[str, int]): The desired size of the images. Defaults to {'longest_edge': 1024}.
                The size can be specified as a dictionary with keys 'longest_edge' or 'height' and 'width'.
                If not provided as a dictionary, it is converted to a dictionary with the 'longest_edge' key.
            mask_size (Dict[str, int]): The desired size of the segmentation masks. Defaults to {'longest_edge': 256}.
                The size can be specified as a dictionary with keys 'longest_edge' or 'height' and 'width'.
                If not provided as a dictionary, it is converted to a dictionary with the 'longest_edge' key.
            resample (PILImageResampling): The resampling method to use during image resizing.
                Defaults to PILImageResampling.BILINEAR.
            do_rescale (bool): Determines whether rescaling of pixel values should be performed. Defaults to True.
            rescale_factor (Union[int, float]): The factor to divide pixel values by during rescaling.
                Defaults to 1 / 255.
            do_normalize (bool): Determines whether normalization of pixel values should be performed.
                Defaults to True.
            image_mean (Optional[Union[float, List[float]]]): The mean values to subtract from pixel values
                during normalization. Defaults to None, which uses the IMAGENET_DEFAULT_MEAN.
            image_std (Optional[Union[float, List[float]]]): The standard deviation values to divide pixel values
                by during normalization. Defaults to None, which uses the IMAGENET_DEFAULT_STD.
            do_pad (bool): Determines whether padding of images should be performed. Defaults to True.
            pad_size (int): The desired size of the padded images. Defaults to None,
                which uses {'height': 1024, 'width': 1024}. The size can be specified as a single integer, representing
                both height and width.
            mask_pad_size (int): The desired size of the padded segmentation masks. Defaults to None,
                which uses {'height': 256, 'width': 256}. The size can be specified as a single integer,
                representing both height and width.
            do_convert_rgb (bool): Determines whether conversion to RGB color space should be performed. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the parent class forwardor.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(**kwargs)
        size = size if size is not None else {"longest_edge": 1024}
        size = get_size_dict(max_size=size, default_to_square=False) if not isinstance(size, dict) else size

        pad_size = pad_size if pad_size is not None else {"height": 1024, "width": 1024}
        pad_size = get_size_dict(pad_size, default_to_square=True)

        mask_size = mask_size if mask_size is not None else {"longest_edge": 256}
        mask_size = (
            get_size_dict(max_size=mask_size, default_to_square=False)
            if not isinstance(mask_size, dict)
            else mask_size
        )

        mask_pad_size = mask_pad_size if mask_pad_size is not None else {"height": 256, "width": 256}
        mask_pad_size = get_size_dict(mask_pad_size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.mask_size = mask_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.mask_pad_size = mask_pad_size
        self.do_convert_rgb = do_convert_rgb
        self._valid_processor_keys = [
            "images",
            "segmentation_maps",
            "do_resize",
            "size",
            "mask_size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_pad",
            "pad_size",
            "mask_pad_size",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    def pad_image(
        self,
        image: np.ndarray,
        pad_size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Pad an image to `(pad_size["height"], pad_size["width"])` with zeros to the right and bottom.

        Args:
            image (`np.ndarray`):
                Image to pad.
            pad_size (`Dict[str, int]`):
                Size of the output image after padding.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the image. Can be either "channels_first" or "channels_last". If `None`, the
                `data_format` of the `image` will be used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        output_height, output_width = pad_size["height"], pad_size["width"]
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        pad_width = output_width - input_width
        pad_height = output_height - input_height

        padded_image = pad(
            image,
            ((0, pad_height), (0, pad_width)),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return padded_image

    def _get_preprocess_shape(self, old_shape: Tuple[int, int], longest_edge: int):
        """
        Compute the output size given input size and target long side length.
        """
        oldh, oldw = old_shape
        scale = longest_edge * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"longest_edge": int}` specifying the size of the output image. The longest
                edge of the image will be resized to the specified size, while the other edge will be resized to
                maintain the aspect ratio.
            resample:
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:

                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:

                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "longest_edge" not in size:
            raise ValueError(f"The `size` dictionary must contain the key `longest_edge`. Got {size.keys()}")
        input_size = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = self._get_preprocess_shape(input_size, size["longest_edge"])
        return resize(
            image,
            size=(output_height, output_width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _preprocess(
        self,
        image: ImageInput,
        do_resize: bool,
        do_rescale: bool,
        do_normalize: bool,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        rescale_factor: Optional[float] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        '''
        This method preprocesses the input image according to the specified operations such as resizing, rescaling,
        normalization, and padding.

        Args:
            self: The instance of the SamImageProcessor class.
            image (ImageInput): The input image to be preprocessed.
            do_resize (bool): A flag indicating whether to perform resizing on the input image.
            do_rescale (bool): A flag indicating whether to perform rescaling on the input image.
            do_normalize (bool): A flag indicating whether to perform normalization on the input image.
            size (Optional[Dict[str, int]]): The target size for resizing the image in the format
                {'width': int, 'height': int}. Default is None.
            resample (PILImageResampling): The resampling filter to be used during image resizing. Default is None.
            rescale_factor (Optional[float]): The factor by which the image should be rescaled. Default is None.
            image_mean (Optional[Union[float, List[float]]]): The mean value to be used for image normalization.
                It can be a single float value or a list of float values, depending on the input_data_format.
                Default is None.
            image_std (Optional[Union[float, List[float]]]):
                The standard deviation value to be used for image normalization.
                It can be a single float value or a list of float values, depending on the input_data_format.
                Default is None.
            do_pad (Optional[bool]): A flag indicating whether to perform padding on the input image. Default is None.
            pad_size (Optional[Dict[str, int]]): The size of the padding to be applied in the format
                {'top': int, 'bottom': int, 'left': int, 'right': int}. Default is None.
            input_data_format (Optional[Union[str, ChannelDimension]]): The data format of the input image,
                e.g., 'channels_first' or 'channels_last'. Default is None.

        Returns:
            Tuple[ImageInput, Tuple[int, int, int]]: The preprocessed image and the reshaped input size in the format
                (image, (height, width, channels)).

        Raises:
            ValueError: If the input_data_format is invalid or not supported.
            TypeError: If the input_data_format is not a string or ChannelDimension.
        '''
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
        reshaped_input_size = get_image_size(image, channel_dim=input_data_format)

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        if do_pad:
            image = self.pad_image(image=image, pad_size=pad_size, input_data_format=input_data_format)

        return image, reshaped_input_size

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        This method preprocesses the input image with various transformations and returns the processed image,
        original size, and reshaped input size.

        Args:
            self: The instance of the SamImageProcessor class.
            image (ImageInput): The input image to be preprocessed.
            do_resize (Optional[bool]): A flag indicating whether to resize the image. Defaults to None.
            size (Optional[Dict[str, int]]): A dictionary containing the target width and height for resizing the image.
                Defaults to None.
            resample (PILImageResampling): The resampling filter to be used during image resizing.
            do_rescale (Optional[bool]): A flag indicating whether to rescale the image. Defaults to None.
            rescale_factor (Optional[float]): The factor by which to rescale the image. Defaults to None.
            do_normalize (Optional[bool]): A flag indicating whether to normalize the image. Defaults to None.
            image_mean (Optional[Union[float, List[float]]]): The mean values to be used for image normalization.
                Defaults to None.
            image_std (Optional[Union[float, List[float]]]): The standard deviation values to be used for
                image normalization. Defaults to None.
            do_pad (Optional[bool]): A flag indicating whether to pad the image. Defaults to None.
            pad_size (Optional[Dict[str, int]]): A dictionary containing the padding width and height.
                Defaults to None.
            do_convert_rgb (Optional[bool]): A flag indicating whether to convert the image to RGB format.
                Defaults to None.
            data_format (Optional[Union[str, ChannelDimension]]): The desired data format for the processed image.
            input_data_format (Optional[Union[str, ChannelDimension]]): The input data format of the image.

        Returns:
            Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]: A tuple containing the processed image as a numpy array,
                the original size of the input image, and the reshaped input size after preprocessing.

        Raises:
            None
        """
        image = to_numpy_array(image)

        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            image = convert_to_rgb(image)

        # All transformations expect numpy arrays.
        image = to_numpy_array(image)

        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        original_size = get_image_size(image, channel_dim=input_data_format)

        image, reshaped_input_size = self._preprocess(
            image=image,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            pad_size=pad_size,
            input_data_format=input_data_format,
        )

        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        return image, original_size, reshaped_input_size

    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: Optional[bool] = None,
        mask_size: Dict[str, int] = None,
        do_pad: Optional[bool] = None,
        mask_pad_size: Optional[Dict[str, int]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Method to preprocess a segmentation mask.

        Args:
            self: The instance of the SamImageProcessor class.
            segmentation_map (ImageInput): The input segmentation map to be preprocessed.
            do_resize (Optional[bool]): Flag indicating whether resizing should be performed. Default is None.
            mask_size (Dict[str, int]): Dictionary containing the target size for the mask after resizing.
            do_pad (Optional[bool]): Flag indicating whether padding should be applied. Default is None.
            mask_pad_size (Optional[Dict[str, int]]): Dictionary containing the padding size for the mask.
            input_data_format (Optional[Union[str, ChannelDimension]]): Format of the input data. Default is None.

        Returns:
            np.ndarray: The preprocessed segmentation map as a NumPy array.
            original_size: The size of the original segmentation map.

        Raises:
            None
        """
        segmentation_map = to_numpy_array(segmentation_map)

        # Add channel dimension if missing - needed for certain transformations
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)

        original_size = get_image_size(segmentation_map, channel_dim=input_data_format)

        segmentation_map, _ = self._preprocess(
            image=segmentation_map,
            do_resize=do_resize,
            size=mask_size,
            resample=PILImageResampling.NEAREST,
            do_rescale=False,
            do_normalize=False,
            do_pad=do_pad,
            pad_size=mask_pad_size,
            input_data_format=input_data_format,
        )

        # Remove extra channel dimension if added for processing
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        segmentation_map = segmentation_map.astype(np.int64)

        return segmentation_map, original_size

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        mask_size: Optional[Dict[str, int]] = None,
        resample: Optional["PILImageResampling"] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        mask_pad_size: Optional[Dict[str, int]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            segmentation_maps (`ImageInput`, *optional*):
                Segmentation map to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Controls the size of the image after `resize`. The longest edge of the image is resized to
                `size["longest_edge"]` whilst preserving the aspect ratio.
            mask_size (`Dict[str, int]`, *optional*, defaults to `self.mask_size`):
                Controls the size of the segmentation map after `resize`. The longest edge of the image is resized to
                `size["longest_edge"]` whilst preserving the aspect ratio.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image pixel values by rescaling factor.
            rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to apply to the image pixel values.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image.
            pad_size (`Dict[str, int]`, *optional*, defaults to `self.pad_size`):
                Controls the size of the padding applied to the image. The image is padded to `pad_size["height"]` and
                `pad_size["width"]` if `do_pad` is set to `True`.
            mask_pad_size (`Dict[str, int]`, *optional*, defaults to `self.mask_pad_size`):
                Controls the size of the padding applied to the segmentation map. The image is padded to
                `mask_pad_size["height"]` and `mask_pad_size["width"]` if `do_pad` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:

                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `mindspore.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:

                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:

                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(max_size=size, default_to_square=False) if not isinstance(size, dict) else size
        mask_size = mask_size if mask_size is not None else self.mask_size
        mask_size = (
            get_size_dict(max_size=mask_size, default_to_square=False)
            if not isinstance(mask_size, dict)
            else mask_size
        )
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        pad_size = pad_size if pad_size is not None else self.pad_size
        pad_size = get_size_dict(pad_size, default_to_square=True)
        mask_pad_size = mask_pad_size if mask_pad_size is not None else self.mask_pad_size
        mask_pad_size = get_size_dict(mask_pad_size, default_to_square=True)
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_list_of_images(images)

        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "mindspore.Tensor, tf.Tensor or jax.ndarray."
            )

        if segmentation_maps is not None:
            segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)

            if not valid_images(segmentation_maps):
                raise ValueError(
                    "Invalid segmentation map type. Must be of type PIL.Image.Image, numpy.ndarray, "
                    "mindspore.Tensor, tf.Tensor or jax.ndarray."
                )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            size_divisibility=pad_size,  # Here _preprocess needs do_pad and pad_size.
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        images, original_sizes, reshaped_input_sizes = zip(
            *(
                self._preprocess_image(
                    image=img,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    do_pad=do_pad,
                    pad_size=pad_size,
                    do_convert_rgb=do_convert_rgb,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for img in images
            )
        )

        data = {
            "pixel_values": images,
            "original_sizes": original_sizes,
            "reshaped_input_sizes": reshaped_input_sizes,
        }

        if segmentation_maps is not None:
            segmentation_maps, original_mask_sizes = zip(
                *(
                    self._preprocess_mask(
                        segmentation_map=mask,
                        do_resize=do_resize,
                        mask_size=mask_size,
                        do_pad=do_pad,
                        mask_pad_size=mask_pad_size,
                        input_data_format=input_data_format,
                    )
                    for mask in segmentation_maps
                )
            )

            # masks should start out the same size as input images
            assert all(
                original_im_size == original_mask_size
                for original_im_size, original_mask_size in zip(original_sizes, original_mask_sizes)
            ), "Segmentation maps should be the same size as input images."

            data["labels"] = segmentation_maps

        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_masks(
        self,
        masks,
        original_sizes,
        reshaped_input_sizes,
        mask_threshold=0.0,
        binarize=True,
        pad_size=None,
        return_tensors="ms",
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[List[mindspore.Tensor], List[np.ndarray], List[tf.Tensor]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[mindspore.Tensor, tf.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            reshaped_input_sizes (`Union[mindspore.Tensor, tf.Tensor, List[Tuple[int,int]]]`):
                The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
            return_tensors (`str`, *optional*, defaults to `"ms"`):
                If `"ms"`, return PyTorch tensors. If `"tf"`, return TensorFlow tensors.

        Returns:
            (`Union[mindspore.Tensor, tf.Tensor]`): Batched masks in batch_size, num_channels, height, width) format, where
            (height, width) is given by original_size.
        """
        if return_tensors == "ms":
            return self._post_process_masks_ms(
                masks=masks,
                original_sizes=original_sizes,
                reshaped_input_sizes=reshaped_input_sizes,
                mask_threshold=mask_threshold,
                binarize=binarize,
                pad_size=pad_size,
            )
        else:
            raise ValueError("return_tensors must be 'ms'.")

    def _post_process_masks_ms(
        self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[List[mindspore.Tensor], List[np.ndarray]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[mindspore.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            reshaped_input_sizes (`Union[mindspore.Tensor, List[Tuple[int,int]]]`):
                The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.

        Returns:
            (`mindspore.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)
            is given by original_size.
        """
        requires_backends(self, ["mindspore"])
        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size["height"], pad_size["width"])
        if isinstance(original_sizes, (mindspore.Tensor, np.ndarray)):
            original_sizes = original_sizes.tolist()
        if isinstance(reshaped_input_sizes, (mindspore.Tensor, np.ndarray)):
            reshaped_input_sizes = reshaped_input_sizes.tolist()
        output_masks = []
        for i, original_size in enumerate(original_sizes):
            if isinstance(masks[i], np.ndarray):
                masks[i] = mindspore.Tensor(masks[i], dtype=mindspore.float32)
            elif not isinstance(masks[i], mindspore.Tensor):
                raise ValueError("Input masks should be a list of `mindspore.tensors` or a list of `np.ndarray`")
            interpolated_mask = F.interpolate(masks[i], target_image_size, mode="bilinear", align_corners=False)
            interpolated_mask = interpolated_mask[..., : reshaped_input_sizes[i][0], : reshaped_input_sizes[i][1]]
            interpolated_mask = F.interpolate(interpolated_mask, original_size, mode="bilinear", align_corners=False)
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(interpolated_mask)

        return output_masks

    def post_process_for_mask_generation(
        self, all_masks, all_scores, all_boxes, crops_nms_thresh, return_tensors="ms"
    ):
        """
        Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.

        Args:
            all_masks (`Union[List[mindspore.Tensor], List[tf.Tensor]]`):
                List of all predicted segmentation masks
            all_scores (`Union[List[mindspore.Tensor], List[tf.Tensor]]`):
                List of all predicted iou scores
            all_boxes (`Union[List[mindspore.Tensor], List[tf.Tensor]]`):
                List of all bounding boxes of the predicted masks
            crops_nms_thresh (`float`):
                Threshold for NMS (Non Maximum Suppression) algorithm.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `mindspore.Tensor`. If `tf`, returns `tf.Tensor`.
        """
        if return_tensors == "ms":
            return _postprocess_for_mg(all_masks, all_scores, all_boxes, crops_nms_thresh)

    def generate_crop_boxes(
        self,
        image,
        target_size,
        crop_n_layers: int = 0,
        overlap_ratio: float = 512 / 1500,
        points_per_crop: Optional[int] = 32,
        crop_n_points_downscale_factor: Optional[List[int]] = 1,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        return_tensors: str = "ms",
    ):
        """
        Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

        Args:
            image (`np.array`):
                Input original image
            target_size (`int`):
                Target size of the resized image
            crop_n_layers (`int`, *optional*, defaults to 0):
                If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where
                each layer has 2**i_layer number of image crops.
            overlap_ratio (`float`, *optional*, defaults to 512/1500):
                Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            points_per_crop (`int`, *optional*, defaults to 32):
                Number of points to sample from each crop.
            crop_n_points_downscale_factor (`List[int]`, *optional*, defaults to 1):
                The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `mindspore.Tensor`. If `tf`, returns `tf.Tensor`.
        """
        crop_boxes, points_per_crop, cropped_images, input_labels = _generate_crop_boxes(
            image,
            target_size,
            crop_n_layers,
            overlap_ratio,
            points_per_crop,
            crop_n_points_downscale_factor,
            input_data_format,
        )
        if return_tensors == "ms":
            crop_boxes = mindspore.tensor(crop_boxes)
            points_per_crop = mindspore.tensor(points_per_crop)
            # cropped_images stays as np
            input_labels = mindspore.tensor(input_labels)
        else:
            raise ValueError("return_tensors must be 'ms'.")
        return crop_boxes, points_per_crop, cropped_images, input_labels

    def filter_masks(
        self,
        masks,
        iou_scores,
        original_size,
        cropped_box_image,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        mask_threshold=0,
        stability_score_offset=1,
        return_tensors="ms",
    ):
        """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`Union[mindspore.Tensor, tf.Tensor]`):
                Input masks.
            iou_scores (`Union[mindspore.Tensor, tf.Tensor]`):
                List of IoU scores.
            original_size (`Tuple[int,int]`):
                Size of the orginal image.
            cropped_box_image (`np.array`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the iou scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `mindspore.Tensor`. If `tf`, returns `tf.Tensor`.
        """
        if return_tensors == "ms":
            return self._filter_masks(
                masks=masks,
                iou_scores=iou_scores,
                original_size=original_size,
                cropped_box_image=cropped_box_image,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                mask_threshold=mask_threshold,
                stability_score_offset=stability_score_offset,
            )
        elif return_tensors == "tf":
            return self._filter_masks_tf(
                masks=masks,
                iou_scores=iou_scores,
                original_size=original_size,
                cropped_box_image=cropped_box_image,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                mask_threshold=mask_threshold,
                stability_score_offset=stability_score_offset,
            )

    def _filter_masks(
        self,
        masks,
        iou_scores,
        original_size,
        cropped_box_image,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        mask_threshold=0,
        stability_score_offset=1,
    ):
        """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`mindspore.Tensor`):
                Input masks.
            iou_scores (`mindspore.Tensor`):
                List of IoU scores.
            original_size (`Tuple[int,int]`):
                Size of the orginal image.
            cropped_box_image (`np.array`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the iou scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.

        """
        requires_backends(self, ["torch"])
        original_height, original_width = original_size
        iou_scores = iou_scores.flatten(start_dim=0, end_dim=1)
        masks = masks.flatten(start_dim=0, end_dim=1)

        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError("masks and iou_scores must have the same batch size.")

        batch_size = masks.shape[0]

        keep_mask = ops.ones(batch_size, dtype=mindspore.bool_)

        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)

        # compute stability score
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)

        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]

        # binarize masks
        masks = masks > mask_threshold
        converted_boxes = _batched_mask_to_box(masks)

        keep_mask = ~_is_box_near_crop_edge(
            converted_boxes, cropped_box_image, [0, 0, original_width, original_height]
        )

        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]

        masks = _pad_masks(masks, cropped_box_image, original_height, original_width)
        # conversion to rle is necessary to run non-maximum suppresion
        masks = _mask_to_rle(masks)

        return masks, scores, converted_boxes


def _compute_stability_score(masks: "mindspore.Tensor", mask_threshold: float, stability_score_offset: int):
    '''
    Compute stability score based on given masks, threshold, and offset.

    Args:
        masks (mindspore.Tensor): A tensor containing masks.
        mask_threshold (float): The threshold value to consider for masks.
        stability_score_offset (int): An offset value to adjust stability score calculations.

    Returns:
        None.

    Raises:
        None.
    '''
    # One mask is always contained inside the other.
    # Save memory by preventing unnecesary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + stability_score_offset)).sum(-1, dtype=mindspore.int16).sum(-1, dtype=mindspore.int32)
    )
    unions = (masks > (mask_threshold - stability_score_offset)).sum(-1, dtype=mindspore.int16).sum(-1, dtype=mindspore.int32)
    stability_scores = intersections / unions
    return stability_scores


def _build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def _normalize_coordinates(
    target_size: int, coords: np.ndarray, original_size: Tuple[int, int], is_bounding_box=False
) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the original image size in (height, width)
    format.
    """
    old_height, old_width = original_size

    scale = target_size * 1.0 / max(old_height, old_width)
    new_height, new_width = old_height * scale, old_width * scale
    new_width = int(new_width + 0.5)
    new_height = int(new_height + 0.5)

    coords = deepcopy(coords).astype(float)

    if is_bounding_box:
        coords = coords.reshape(-1, 2, 2)

    coords[..., 0] = coords[..., 0] * (new_width / old_width)
    coords[..., 1] = coords[..., 1] * (new_height / old_height)

    if is_bounding_box:
        coords = coords.reshape(-1, 4)

    return coords


def _generate_crop_boxes(
    image,
    target_size: int,  # Is it tuple here?
    crop_n_layers: int = 0,
    overlap_ratio: float = 512 / 1500,
    points_per_crop: Optional[int] = 32,
    crop_n_points_downscale_factor: Optional[List[int]] = 1,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

    Args:
        image (Union[`numpy.ndarray`, `PIL.Image`, `mindspore.Tensor`]):
            Image to generate crops for.
        target_size (`int`):
            Size of the smallest crop.
        crop_n_layers (`int`, *optional*):
            If `crops_n_layers>0`, mask prediction will be run again on crops of the image. Sets the number of layers
            to run, where each layer has 2**i_layer number of image crops.
        overlap_ratio (`int`, *optional*):
            Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the
            image length. Later layers with more crops scale down this overlap.
        points_per_crop (`int`, *optional*):
            Number of points to sample per crop.
        crop_n_points_downscale_factor (`int`, *optional*):
            The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred.
    """
    if isinstance(image, list):
        raise ValueError("Only one image is allowed for crop generation.")
    image = to_numpy_array(image)
    original_size = get_image_size(image, input_data_format)

    points_grid = []
    for i in range(crop_n_layers + 1):
        n_points = int(points_per_crop / (crop_n_points_downscale_factor**i))
        points_grid.append(_build_point_grid(n_points))

    crop_boxes, layer_idxs = _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size)

    cropped_images, point_grid_per_crop = _generate_crop_images(
        crop_boxes, image, points_grid, layer_idxs, target_size, original_size, input_data_format
    )
    crop_boxes = np.array(crop_boxes)
    crop_boxes = crop_boxes.astype(np.float32)
    points_per_crop = np.array([point_grid_per_crop])
    points_per_crop = np.transpose(points_per_crop, axes=(0, 2, 1, 3))

    input_labels = np.ones_like(points_per_crop[:, :, :, 0], dtype=np.int64)

    return crop_boxes, points_per_crop, cropped_images, input_labels


def _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size):
    """
    Generates 2 ** (layers idx + 1) crops for each crop_n_layers. Crops are in the XYWH format : The XYWH format
    consists of the following required indices:

    - X: X coordinate of the top left of the bounding box
    - Y: Y coordinate of the top left of the bounding box
    - W: width of the bounding box
    - H: height of the bounding box
    """
    crop_boxes, layer_idxs = [], []
    im_height, im_width = original_size
    short_side = min(im_height, im_width)

    # Original image
    crop_boxes.append([0, 0, im_width, im_height])
    layer_idxs.append(0)
    for i_layer in range(crop_n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_width = int(math.ceil((overlap * (n_crops_per_side - 1) + im_width) / n_crops_per_side))
        crop_height = int(math.ceil((overlap * (n_crops_per_side - 1) + im_height) / n_crops_per_side))

        crop_box_x0 = [int((crop_width - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_height - overlap) * i) for i in range(n_crops_per_side)]

        for left, top in product(crop_box_x0, crop_box_y0):
            box = [left, top, min(left + crop_width, im_width), min(top + crop_height, im_height)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def _generate_crop_images(
    crop_boxes, image, points_grid, layer_idxs, target_size, original_size, input_data_format=None
):
    """
    Takes as an input bounding boxes that are used to crop the image. Based in the crops, the corresponding points are
    also passed.
    """
    cropped_images = []
    total_points_per_crop = []
    for i, crop_box in enumerate(crop_boxes):
        left, top, right, bottom = crop_box

        channel_dim = infer_channel_dimension_format(image, input_data_format)
        if channel_dim == ChannelDimension.LAST:
            cropped_im = image[top:bottom, left:right, :]
        else:
            cropped_im = image[:, top:bottom, left:right]

        cropped_images.append(cropped_im)

        cropped_im_size = get_image_size(cropped_im, channel_dim)
        points_scale = np.array(cropped_im_size)[None, ::-1]

        points = points_grid[layer_idxs[i]] * points_scale
        normalized_points = _normalize_coordinates(target_size, points, original_size)
        total_points_per_crop.append(normalized_points)

    return cropped_images, total_points_per_crop


def _pad_masks(masks, crop_box: List[int], orig_height: int, orig_width: int):
    """
    This function pads the input masks based on the provided crop box and original image dimensions.

    Args:
        masks (List): List of masks to be padded.
        crop_box (List[int]): A list containing the coordinates [left, top, right, bottom] of the crop box.
        orig_height (int): The original height of the image.
        orig_width (int): The original width of the image.

    Returns:
        None: The function does not return a value; it modifies the input masks in place.

    Raises:
        ValueError: If the crop_box coordinates are invalid or if the original dimensions are inconsistent.
    """
    left, top, right, bottom = crop_box
    if left == 0 and top == 0 and right == orig_width and bottom == orig_height:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_width - (right - left), orig_height - (bottom - top)
    pad = (left, pad_x - left, top, pad_y - top)
    return ops.pad(masks, pad, value=0)


def _is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0):
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    crop_box_torch = mindspore.tensor(crop_box, dtype=mindspore.float32)
    orig_box_torch = mindspore.tensor(orig_box, dtype=mindspore.float32)

    left, top, _, _ = crop_box
    offset = mindspore.tensor([[left, top, left, top]])
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    boxes = (boxes + offset).float()

    near_crop_edge = ops.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = ops.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = ops.logical_and(near_crop_edge, ~near_image_edge)
    return ops.any(near_crop_edge, dim=1)


def _batched_mask_to_box(masks: "mindspore.Tensor"):
    """
    Computes the bounding boxes around the given input masks. The bounding boxes are in the XYXY format which
    corresponds the following required indices:

    - LEFT: left hand side of the bounding box
    - TOP: top of the bounding box
    - RIGHT: right of the bounding box
    - BOTTOM: bottom of the bounding box

    Return [0,0,0,0] for an empty mask. For input shape channel_1 x channel_2 x ... x height x width, the output shape
    is channel_1 x channel_2 x ... x 4.

    Args:
        masks (`mindspore.Tensor` of shape `(batch, nb_mask, height, width)`)
    """
    # torch.max below raises an error on empty inputs, just skip in this case

    if ops.numel(masks) == 0:
        return ops.zeros(*masks.shape[:-2], 4)

    # Normalize shape to Cxheightxwidth
    shape = masks.shape
    height, width = shape[-2:]

    # Get top and bottom edges
    in_height, _ = ops.max(masks, dim=-1)
    in_height_coords = in_height * ops.arange(height)[None, :]
    bottom_edges, _ = ops.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + height * (~in_height)
    top_edges, _ = ops.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = ops.max(masks, dim=-2)
    in_width_coords = in_width * ops.arange(width)[None, :]
    right_edges, _ = ops.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + width * (~in_width)
    left_edges, _ = ops.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = ops.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    out = out.reshape(*shape[:-2], 4)
    return out


def _mask_to_rle(input_mask: "mindspore.Tensor"):
    """
    Encodes masks the run-length encoding (RLE), in the format expected by pycoco tools.
    """
    # Put in fortran order and flatten height and width
    batch_size, height, width = input_mask.shape
    input_mask = input_mask.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = input_mask[:, 1:] ^ input_mask[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(batch_size):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1] + 1
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if input_mask[i, 0] == 0 else [0]
        counts += [cur_idxs[0].item()] + btw_idxs.tolist() + [height * width - cur_idxs[-1]]
        out.append({"size": [height, width], "counts": counts})
    return out


def _rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    height, width = rle["size"]
    mask = np.empty(height * width, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity = not parity
    mask = mask.reshape(width, height)
    return mask.transpose()  # Reshape to original shape


def _postprocess_for_mg(rle_masks, iou_scores, mask_boxes, amg_crops_nms_thresh=0.7):
    """
    Perform NMS (Non Maximum Suppression) on the outputs.

    Args:
        rle_masks (`mindspore.Tensor`):
            binary masks in the RLE format
        iou_scores (`mindspore.Tensor` of shape (nb_masks, 1)):
            iou_scores predicted by the model
        mask_boxes (`mindspore.Tensor`):
            The bounding boxes corresponding to segmentation masks
        amg_crops_nms_thresh (`float`, *optional*, defaults to 0.7):
            NMS threshold.
    """
    keep_by_nms = batched_nms(
        boxes=mask_boxes.float(),
        scores=iou_scores,
        idxs=ops.zeros(mask_boxes.shape[0]),
        iou_threshold=amg_crops_nms_thresh,
    )

    iou_scores = iou_scores[keep_by_nms]
    rle_masks = [rle_masks[i] for i in keep_by_nms]
    mask_boxes = mask_boxes[keep_by_nms]
    masks = [_rle_to_mask(rle) for rle in rle_masks]

    return masks, iou_scores, rle_masks, mask_boxes

def batched_nms(
    boxes: mindspore.Tensor,
    scores: mindspore.Tensor,
    idxs: mindspore.Tensor,
    iou_threshold: float,
) -> mindspore.Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
            in decreasing order of scores
    """
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    if boxes.numel() > (4000 if mindspore.get_context('device_target') == "CPU" else 20000):
        return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
    else:
        return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)


def _batched_nms_coordinate_trick(
    boxes: mindspore.Tensor,
    scores: mindspore.Tensor,
    idxs: mindspore.Tensor,
    iou_threshold: float,
) -> mindspore.Tensor:
    """
    Performs non-maximum suppression (NMS) on a batch of bounding boxes using the coordinate trick.
    
    Args:
        boxes (mindspore.Tensor): A tensor containing the bounding boxes coordinates.
            Its shape should be (N, 4), where N is the number of boxes and each box is represented by
            (x_min, y_min, x_max, y_max).
        scores (mindspore.Tensor): A tensor containing the confidence scores for each bounding box.
            Its shape should be (N,).
        idxs (mindspore.Tensor): A tensor containing the indices of the bounding boxes. Its shape should be (N,).
        iou_threshold (float): The intersection over union (IoU) threshold used for NMS. Boxes with IoU higher
            than this threshold will be suppressed.
    
    Returns:
        mindspore.Tensor: A tensor containing the indices of the boxes to keep after NMS.
            Its shape is (M,), where M is the number of boxes to keep.
    
    Raises:
        None.
    """
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return ops.zeros((0,), dtype=mindspore.int64)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + mindspore.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def _batched_nms_vanilla(
    boxes: mindspore.Tensor,
    scores: mindspore.Tensor,
    idxs: mindspore.Tensor,
    iou_threshold: float,
) -> mindspore.Tensor:
    """
    Args:
        boxes (mindspore.Tensor): A tensor containing bounding boxes coordinates of shape (N, 4)
            where N is the number of boxes. Each row represents a box in the format (x_min, y_min, x_max, y_max).
        scores (mindspore.Tensor): A tensor containing confidence scores associated with each bounding box in 'boxes'.
        idxs (mindspore.Tensor): A tensor containing class indices corresponding to each bounding box in 'boxes'.
        iou_threshold (float): The intersection over union (IoU) threshold used for non-maximum suppression.
            It is a float value between 0 and 1.
    
    Returns:
        mindspore.Tensor: A tensor containing the indices of the selected boxes after applying non-maximum
            suppression based on the provided 'iou_threshold'.
    
    Raises:
        TypeError: If the input arguments are not of the expected types.
        ValueError: If 'boxes' and 'scores' have mismatched shapes, or if 'iou_threshold' is not within
            the valid range [0, 1].
    """
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = ops.zeros_like(scores, dtype=mindspore.bool_)
    for class_id in ops.unique(idxs):
        curr_indices = ops.nonzero(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = ops.nonzero(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]

def nms(boxes: mindspore.Tensor, scores: mindspore.Tensor, iou_threshold: float):
    """
    Performs non-maximum suppression (NMS) on a set of bounding boxes.
    
    Args:
        boxes (mindspore.Tensor): A tensor of shape (N, 4) representing the coordinates of the N bounding boxes. 
            Each bounding box is defined by four values: (x_min, y_min, x_max, y_max).
        scores (mindspore.Tensor): A tensor of shape (N,) representing the scores associated with each bounding box.
        iou_threshold (float): The Intersection over Union (IoU) threshold used for NMS. 
            Bounding boxes with IoU greater than or equal to this threshold will be suppressed.
    
    Returns:
        mindspore.Tensor: A tensor containing the indices of the selected bounding boxes after NMS. 
            The shape of the returned tensor is (M,), where M is the number of selected bounding boxes.
    
    Raises:
        TypeError: If any of the input arguments are not of the expected type.
        ValueError: If the shape of 'boxes' and 'scores' tensors are incompatible or if 'iou_threshold'
            is not within the valid range.
    """
    box_with_score = ops.stack((boxes, scores))
    _, _, selected_mask = _get_cache_prim(mindspore.ops.NMSWithMask)(iou_threshold)(box_with_score)
    return ops.nonzero(selected_mask).reshape(-1)

__all__ = ['SamImageProcessor']
