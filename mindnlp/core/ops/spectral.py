"""spectral"""
from mindspore import ops
# stft
def stft(input, n_fft, hop_length=None, win_length=None,
         window=None, center=True, pad_mode='reflect',
         normalized=False, onesided=None, return_complex=None):
    return ops.stft(input, n_fft, hop_length, win_length, window,
                    center, pad_mode.upper(), normalized, onesided, return_complex)

# istft


# bartlett_window


# blackman_window


# hamming_window


# hann_window
def hann_window(window_length, periodic=True, *, dtype=None):
    return ops.hann_window(window_length, periodic, dtype=dtype)

# kaiser_window
