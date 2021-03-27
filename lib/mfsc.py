from typing import Union, Sequence
import math
import numpy as np

AnyArray = Union[int, float, Sequence[int], Sequence[float], np.array]

def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

# matches wav2letter's TriFilterbank layout
def trifilter(sr: float, n_mel: int, n_fft: int) -> np.array:
    filterlen = (n_fft >> 1) + 1
    warp_max = hz_to_mel(sr / 2)
    dwarp = warp_max / (n_mel + 1)
    f = [mel_to_hz(i * dwarp) * (filterlen - 1) * 2.0 / sr
        for i in range(0, n_mel + 2)]
    weights = np.zeros((filterlen, n_mel))
    for i in range(filterlen):
        for j in range(n_mel):
            hislope = (i - f[j]) / (f[j+1] - f[j])
            loslope = (f[j+2] - i) / (f[j + 2] - f[j + 1])
            weights[i][j] = max(min(hislope, loslope), 0)
    return weights

class Mfsc:
    def __init__(self, sr: int, n_mel: int, frame_size_ms: int, frame_stride_ms: int,
                 *, mel_floor: float=1.0, preem_coeff: float=0.97):
        self.n_mel = n_mel
        self.preem_coeff = preem_coeff
        self.mel_floor = mel_floor

        self.frame_size   = int(1e-3 * frame_size_ms * sr)
        self.frame_stride = int(1e-3 * frame_stride_ms * sr)
        n_fft = 1 << math.ceil(math.log(self.frame_size, 2))

        self.n_fft = n_fft
        self.window = np.hamming(self.frame_size).astype(np.float32)
        self.trifilter = trifilter(sr, n_mel, n_fft).astype(np.float32)
        self.filterlen = (n_fft >> 1) + 1

    def apply(self, samples: AnyArray) -> np.array:
        samples = np.asarray(samples, dtype=np.float32)
        frames = self.frame_signal(samples)
        frames = frames * 32768.0 # HTK scaling to int range
        P = self.power_spectrum(frames)
        T = np.log(np.maximum(P @ self.trifilter, self.mel_floor))
        N = self.normalize(T)
        return N

    def frame_signal(self, samples: np.array) -> np.array:
        samples = np.asarray(samples, dtype=np.float32)
        if len(samples) < self.frame_size:
            return np.array([])
        shape = (int(1 + np.floor((len(samples) - self.frame_size) / self.frame_stride)), self.frame_size)
        element = samples.strides[0]
        strides = (element * self.frame_stride, element)
        return np.lib.stride_tricks.as_strided(samples, shape, strides)

    def power_spectrum(self, frames: np.array) -> np.array:
        # pre-emphasis
        if self.preem_coeff > 0:
            frames[:,0] *= (1 - self.preem_coeff)
            frames[:,1:] -= frames[:,:-1] * self.preem_coeff
        out = np.fft.rfft(frames * self.window, self.n_fft)
        return np.abs(out)

    def normalize(self, frames: np.array) -> np.array:
        mean = np.mean(frames)
        std = np.std(frames)
        if std > 0:
            return (frames - mean) / std
        else:
            return frames - mean