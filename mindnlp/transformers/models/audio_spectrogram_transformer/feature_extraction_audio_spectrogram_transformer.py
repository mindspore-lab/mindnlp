# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Feature extractor class for Audio Spectrogram Transformer.
"""

from typing import List, Optional, Union

import numpy as np

from mindnlp.utils import TensorType, is_mindspore_available, logging
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature


if is_mindspore_available():
    import mindspore


logger = logging.get_logger(__name__)


class ASTFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Audio Spectrogram Transformer (AST) feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using TorchAudio if installed or using numpy
    otherwise, pads/truncates them to a fixed length and normalizes them using a mean and standard deviation.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of Mel-frequency bins.
        max_length (`int`, *optional*, defaults to 1024):
            Maximum length to which to pad/truncate the extracted features.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the log-Mel features using `mean` and `std`.
        mean (`float`, *optional*, defaults to -4.2677393):
            The mean value used to normalize the log-Mel features. Uses the AudioSet mean by default.
        std (`float`, *optional*, defaults to 4.5689974):
            The standard deviation value used to normalize the log-Mel features. Uses the AudioSet standard deviation
            by default.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~ASTFeatureExtractor.__call__`] should return `attention_mask`.
    """
    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        num_mel_bins=128,
        max_length=1024,
        padding_value=0.0,
        do_normalize=True,
        mean=-4.2677393,
        std=4.5689974,
        return_attention_mask=False,
        **kwargs,
    ):
        """
        Initializes an instance of the ASTFeatureExtractor class.
        
        Args:
            self: The instance of the class.
            feature_size (int, optional): The size of the input features. Defaults to 1.
            sampling_rate (int, optional): The sampling rate of the audio. Defaults to 16000.
            num_mel_bins (int, optional): The number of mel bins for mel filtering. Defaults to 128.
            max_length (int, optional): The maximum length of the input. Defaults to 1024.
            padding_value (float, optional): The value used for padding sequences. Defaults to 0.0.
            do_normalize (bool, optional): Whether to normalize the input features. Defaults to True.
            mean (float, optional): The mean value for input feature normalization. Defaults to -4.2677393.
            std (float, optional): The standard deviation value for input feature normalization. Defaults to 4.5689974.
            return_attention_mask (bool, optional): Whether to return attention mask. Defaults to False.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std
        self.return_attention_mask = return_attention_mask

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,
            num_mel_filters=self.num_mel_bins,
            min_frequency=20,
            max_frequency=sampling_rate // 2,
            sampling_rate=sampling_rate,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )

        self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))
        self.window = window_function(400, "hann", periodic=False)

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        # waveform = waveform * (2**15)  # Kaldi compliance: 16-bit signed integers
        waveform = np.squeeze(waveform)
        fbank = spectrogram(
            waveform,
            self.window,
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            preemphasis=0.97,
            mel_filters=self.mel_filters,
            log_mel="log",
            mel_floor=1.192092955078125e-07,
            remove_dc_offset=True,
        ).T

        fbank = mindspore.Tensor.from_numpy(fbank)

        n_frames = fbank.shape[0]
        difference = max_length - n_frames

        # pad or truncate, depending on difference
        if difference > 0:
            pad_module = mindspore.nn.ZeroPad2d((0, 0, 0, difference))
            fbank = pad_module(fbank)
        elif difference < 0:
            fbank = fbank[0:max_length, :]

        fbank = fbank.numpy()

        return fbank

    def normalize(self, input_values: np.ndarray) -> np.ndarray:
        """
        Normalize the input values using the mean and standard deviation stored in the ASTFeatureExtractor instance.
        
        Args:
            self (ASTFeatureExtractor): An instance of the ASTFeatureExtractor class.
                It holds the mean and standard deviation values necessary for normalization.
            input_values (np.ndarray): A NumPy array containing the input values to be normalized.
                The shape of the array must be compatible with the mean and standard deviation arrays stored in the ASTFeatureExtractor instance.
        
        Returns:
            np.ndarray: A NumPy array with the normalized values.
                The normalization is performed by subtracting the mean value and dividing by twice the standard deviation value.
        
        Raises:
            None
        """
        return (input_values - (self.mean)) / (self.std * 2)

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # extract fbank features and pad/truncate to max_length
        features = [self._extract_fbank_features(waveform, max_length=self.max_length) for waveform in raw_speech]

        # convert into BatchFeature
        padded_inputs = BatchFeature({"input_values": features})

        # make sure list is in array format
        input_values = padded_inputs.get("input_values")
        if isinstance(input_values[0], list):
            padded_inputs["input_values"] = [np.asarray(feature, dtype=np.float32) for feature in input_values]

        # normalization
        if self.do_normalize:
            padded_inputs["input_values"] = [self.normalize(feature) for feature in input_values]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs

__all__ = ["ASTFeatureExtractor"]
