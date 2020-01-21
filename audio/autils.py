import numpy as np
import librosa.util as librosa_util
from scipy.signal import get_window

import torch

import sys
sys.path.insert(0, '../')

EPSILON = 1e-8


def frames_to_wave(frames, hop_length=256):
    frames = frames.tolist()
    fl = len(frames[0])
    wave = frames[0]
    for frame in frames[1:]:
        l = len(wave)
        for i, sample in enumerate(frame[:-hop_length]):
            idx = l - (fl - hop_length) + i
            wave[idx] += sample
            wave[idx] / 2
        for sample in frame[-hop_length:]:
            wave.append(sample)
    return wave


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    PARAMS
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function. By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output

    RETURNS
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x