from librosa.filters import mel as librosa_mel_fn

import torch
import numpy as np

from .stft import STFT

import sys
sys.path.insert(0, '../')

EPSILON = 1e-8


class MelTransformer(torch.nn.Module):
    """
    Class providing interface for audio-->melspectrogram transformation.
    """
    def __init__(self,
                 filter_length=1024,
                 hop_length=256,
                 win_length=1024,
                 n_mel_channels=80,
                 sampling_rate=22050,
                 mel_fmin=0.0, mel_fmax=8000.0,
                 dynamic_range_compression='nvidia'):
        super(MelTransformer, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.dynamic_range_compression = dynamic_range_compression
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.tensor(mel_basis, dtype=torch.float32)
        self.register_buffer('mel_basis', mel_basis)

    def _custom_spectral_normalize(self, magnitudes, clip_min=1e-5):
        return ((((magnitudes.clamp(min=clip_min) + 1).log() / 6 - 1) * 4).exp() + 1).log()

    def _custom_spectral_de_normalize(self, magnitudes):
        output = (magnitudes.clamp(min=1e-8).exp() - 1 + np.e ** (-4)).log()
        return ((output / 4 + 1 + EPSILON) * 6).exp() - 1
    
    def _nvidia_spectral_normalize(self, magnitudes, C=1):
        return (magnitudes.clamp(min=1e-5) * C).log()
    
    def _nvidia_spectral_de_normalize(self, magnitudes, C=1):
        return magnitudes.exp() / C

    def spectral_normalize(self, magnitudes):
        if self.dynamic_range_compression == 'nvidia':
            return self._nvidia_spectral_normalize(magnitudes)
        return self._custom_spectral_normalize(magnitudes)

    def spectral_de_normalize(self, magnitudes):
        if self.dynamic_range_compression == 'nvidia':
            return self._nvidia_spectral_de_normalize(magnitudes)
        return self._custom_spectral_de_normalize(magnitudes)

    def transform(self, y, normalization_const=32768.0):
        """Computes mel-spectrograms from a batch of waves

        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        
        if y.data.min() <= -1 and y.data.max() >= 1:
            y_norm = y / normalization_const
        else:
            y_norm = y
        
        magnitudes, _ = self.stft_fn.transform(y_norm)
        magnitudes = magnitudes.data
        mel_output = self.mel_basis.matmul(magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        mel_output = mel_output.squeeze(0)
        return mel_output

    def forward(self, y, normalization_const=32768.0):
        return self.transform(y, normalization_const)
