# Copyright (c) Ivan Vovk, 2019-2020. All rights reserved.

import librosa
import numpy as np
from tqdm import tqdm

import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

try:
    import jax.numpy as npjax
    from jax import jit
    from jax.lib import xla_bridge
    if xla_bridge.get_backend().platform == 'cpu':
        print('No GPU available. All computations will be ran on CPU, but using JIT-compiler.')
    JAX_BUILD_INSTALLED = True
except ImportError as e:
    print('No JAX installed. All computations will be ran with the standard CPU-backed `numpy` package.')
    JAX_BUILD_INSTALLED = False

from .utils import frames_to_wave


class DMD(object):
    """
    Class for Dynamic Mode Decomposition (DMD) computation from (Schmid & Sesterhenn, 2008).
    """
    def __init__(
        self,
        svd_rank=80,
        opt='projected',
        lag=1,
        delta_t=1,
        use_jax_opt=True
    ):
        """
        Constructor function.
        :param svd_rank: rank of snapshots for computing Truncated Singular Value Decomposition
        :param opt: how to calculate modes - by default or project on denoised data
        :param lag: which shift to choose for linear dynamic system X1 = A_tilde @ X0
        :param delta_t: time difference between 2 snapshots
        :param use_jax_opt: enable JAX numerical algebra optimizations (jit, running on GPU), if available
        """
        self.svd_rank = svd_rank
        self.opt = opt
        self.lag = lag
        self.delta_t = delta_t
        self.use_jax_opt = use_jax_opt & JAX_BUILD_INSTALLED
        if self.use_jax_opt:
            self._DOT = jit(lambda X, Y: X @ Y)
            self._ELEMWISE_DOT = jit(lambda X, y: X * y)
            self._COMPUTE_SVD = jit(npjax.linalg.svd)
        else:
            self._DOT = lambda X, Y: np.dot(X, Y)
            self._ELEMWISE_DOT = lambda X, y: X * y
            self._COMPUTE_SVD = np.linalg.svd
        
        self.time_series_size = None
        self.modes = None
        self.eigvals = None
        self.fitted = False
        
        self.method = None
        
    def fit(self, X):
        """
        Fits DMD to the given snaphots X. After fitting one can reconstruct data with `modes`, `eigvals` and `b` parameters.
        :param X: snaphots of shape (M, N), where M is the size of a single snapshot and N is the number of such snapshots
        """
        # split data
        X0, X1 = X[:, :-self.lag], X[:, self.lag:]
        self.method = 'shapshots' if X0.shape[0] > X0.shape[1] else 'svd'
        self.time_series_size = X.shape[-1] 
        
        _rank = min(X0.shape) if self.svd_rank == -1 else self.svd_rank
        U = None
        if self.method == 'svd':
            U, s, V = self._COMPUTE_SVD(X0)
            U, s, V = U[:, :_rank], s[:_rank], V[:_rank].conj().T
        elif self.method == 'shapshots':
            s, V = np.linalg.eig(self._DOT(X0.T, X0))
            s, V = np.sqrt(s[:_rank]), V[:, :_rank]
            U = self._ELEMWISE_DOT(self._DOT(X0, V), 1/s)
            
        # restore linear operator of dynamics
        operator = self._ELEMWISE_DOT(self._DOT(self._DOT(U.T.conj(), X1), V), 1/s)
        
        # perform eigen decomposition
        self.eigvals, self.modes = np.linalg.eig(operator)
        
        # obtain modes
        if self.opt == 'projected':
            self.modes = self._DOT(U, self.modes)
        elif self.opt == 'exact':
            self.modes = self._DOT(self._ELEMWISE_DOT(self._DOT(X1, V), 1/s), self.modes)
            
        self.b = np.linalg.lstsq(self.modes, X0.T[0], rcond=None)[0]
        self.fitted = True
        
    def fit_reconstruct(self, X):
        """
        Fits DMD to the given snapshots X and calculates it's reconstruction.
        :param X: snaphots of shape (M, N), where M is the size of a single snapshot and N is the number of such snapshots
        """
        self.fit(X)
        return self.reconstruct()
    
    def reconstruct(self):
        """
        Reconstructs data with fitted `modes`, `eigvals` and `b` parameters.
        :return reconstructed snapshots
        """
        if not self.fitted:
            raise RuntimeError('Model is not fitted yet.')
        return self._DOT(
            self._DOT(
                self.modes,
                np.diag(self.eigvals)
            ),
            self._ELEMWISE_DOT(
                np.vander(self.eigvals,
                          N=self.time_series_size,
                          increasing=True),
                self.b.reshape(-1, 1)
            )
        )
    
    def nparams(self):
        """
        Calculates number of parameters to store in order to be able to reconstruct snapshots.
        :return parameters number
        """
        return int(np.multiply(*self.modes.shape) \
            + self.eigvals.shape[0] + self.b.shape[0])

    def frequencies(self):
        """
        Calculates ordinary frequencies from angular ones.
        :return ordinary frequencies
        """
        return np.log(self.eigvals).imag / (2 * np.pi * self.delta_t)

    
class STDMD(object):
    """Short-time Dynamic Mode Decomposition class. Similar to ST Fourier Transformation, but DMD is applied instead of FFT."""
    def __init__(
        self,
        frame_length=2048,
        hop_length=256,
        n_channels=80,
        opt='projected',
        sampling_rate=22050,
        lag=1,
        n_jobs=-1,
        use_jax_opt=True,
        verbose=True
    ):
        """
        Constructor function.
        :param frame_length: length of the sliding window to which DMD will be applied (for each timestep)
        :param hop_length: length of the slider hop to calculate next window
        :param n_channels: number of frequencies to be stored. E.g. storing from (0, n_channels)
        :param opt: how to calculate modes - by default or project on denoised data
        :param sampling_rate: how much timesteps in one second
        :param lag: which shift to choose for linear dynamic system X1 = A_tilde @ X0
        :param n_jobs: how much processes to parallelize computation into
        :param use_jax_opt: enable JAX numerical algebra optimizations (jit, running on GPU), if available
        :param verbose: use crossbar to show process compliteness
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_channels = n_channels
        self.fl = 2*self.n_channels
        self.hl = 1
        self.opt = opt
        self.sampling_rate = sampling_rate
        self.lag = lag
        _n_cpus = mp.cpu_count()
        self.n_jobs = _n_cpus if n_jobs == -1 or n_jobs >= _n_cpus else n_jobs
        self.use_jax_opt = use_jax_opt & JAX_BUILD_INSTALLED
        self.verbose = verbose
        
        self.rank = self.n_channels

    def _process_frame(self, frame):
        """
        Applies DMD to a single frame and returns frequency-amplitude vector
        :param frame: windowed-sequence to be processed
        :return vector, i-th item of which corresponds to the DMD amplitude at i-th frequency
        """
        # framing into snapshots
        subframes = librosa.util.frame(
            frame, frame_length=self.fl, hop_length=self.hl, axis=0
        ).astype(np.float32).T

        # creating and fitting a DMD
        dmd = DMD(
            svd_rank=self.rank,
            opt=self.opt,
            lag=self.lag,
            delta_t=self.hl / self.sampling_rate,
            use_jax_opt = self.use_jax_opt
        )
        dmd.fit(subframes)

        # calculating ordinary frequencies and ordering returning amplitudes
        _frame = np.zeros(shape=self.n_channels)
        freqs = dmd.frequencies()
        positive_mask = freqs >= 0
        freqs = freqs[positive_mask]
        sorted_idx = np.argsort(freqs)[:self.n_channels]
        freqs = freqs[sorted_idx]
        freqs = np.round(freqs / (self.sampling_rate / 1.95) * self.n_channels).astype(int)
        fs = []
        for f in freqs:
            if f not in fs:
                fs.append(f)
            else:
                if f + 1 <= self.n_channels:
                    fs.append(f + 1)
                else:
                    fs.append(f)
        magnitudes = np.abs(dmd.b[positive_mask][sorted_idx])
        _frame[fs] = magnitudes
        return _frame

    def fit_transform(self, y):
        """
        Fits STDMD and calculates dmdgram on a fly.
        :param y: time-series sequence
        :return DMDgram: spectrogram-like sequence representation
        """
        frames = librosa.util.frame(
            y, frame_length=self.frame_length,
            hop_length=self.hop_length, axis=0
        ).astype(np.float32)
        
        with ThreadPool(processes=self.n_jobs) as pool:
            dmdgram = list(tqdm(pool.imap(self._process_frame, frames), total=len(frames))) \
                if self.verbose else pool.map(self._process_frame, frames)
        return np.array(dmdgram, dtype=np.float32)
