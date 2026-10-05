"""Signal processing for FinalSpark 4x8 MEA raw recordings.

Raw HDF5 layout (verified on the platform): datasets electrode_0..electrode_31, each (n_rows, 1280)
float32 in µV at 30 kHz, already high-pass filtered by the platform; dataset 'triggers' of the same
shape, uint16, bit k set while trigger key k is high.

Spikes (census, no stimulation): raw µV -> 300-3000 Hz band-pass -> negative threshold at
-k x robust noise -> common-mode rejection.
"""
from __future__ import annotations

import h5py
import numpy as np
from scipy import signal

FS = 30000
N_CH = 32
BLOCK_SIZE = 8
BAND_HZ = (300.0, 3000.0)
_SOS = signal.butter(3, BAND_HZ, btype="band", fs=FS, output="sos")


def block_of(ch: int) -> int:
    """Physical block (= organoid site) of an electrode: 0-7 -> 0, 8-15 -> 1, ..."""
    return ch // BLOCK_SIZE


def samples(ms_value: float) -> int:
    return int(round(ms_value * FS / 1000.0))


# --------------------------------------------------------------------------------------- I/O

def load_raw(path):
    """Return (X, T): X float32 (32, n) in µV, T uint16 (n,) trigger bitmask."""
    with h5py.File(path, "r") as f:
        X = np.stack([f[f"electrode_{i}"][:].reshape(-1) for i in range(N_CH)]).astype(np.float32)
        T = f["triggers"][:].reshape(-1).astype(np.uint16)
    return X, T


def load_triggers(path) -> np.ndarray:
    """The trigger bitmask only (1/33 of the file: quick, also straight from the S3 mount)."""
    with h5py.File(path, "r") as f:
        return f["triggers"][:].reshape(-1).astype(np.uint16)


def trigger_edges(T: np.ndarray) -> list[tuple[int, int]]:
    """All rising edges as (sample, key), sorted by time."""
    edges = []
    for key in range(16):
        hi = (T & (1 << key)) != 0
        edges.extend((int(i), key) for i in np.flatnonzero(hi & np.r_[True, ~hi[:-1]]))
    return sorted(edges)


# ------------------------------------------------------------------------------ spikes

def bandpass(X: np.ndarray) -> np.ndarray:
    """Zero-phase 300-3000 Hz band-pass, channel by channel (bounded memory)."""
    Y = np.empty(X.shape, dtype=np.float32)
    for c in range(X.shape[0]):
        Y[c] = signal.sosfiltfilt(_SOS, X[c].astype(np.float64))
    return Y


def spike_noise(Y: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Quiroga's robust noise estimate, median(|y|)/0.6745, on the samples in `keep` only."""
    return np.array([np.median(np.abs(Y[c][keep])) / 0.6745 for c in range(Y.shape[0])])


def detect_spikes(Y: np.ndarray, noise: np.ndarray, k: float = 5.0, max_amp_uV: float = 500.0,
                  refractory_ms: float = 1.0) -> list[np.ndarray]:
    """Negative threshold crossings at -k*noise, aligned to the trough (within 0.5 ms).
    Events with |trough| > max_amp_uV (artifact residue) are dropped."""
    ref = samples(refractory_ms)
    look = samples(0.5)
    out = []
    for c in range(Y.shape[0]):
        y = Y[c]
        thr = -k * noise[c]
        idx = np.flatnonzero((y[1:-look] < thr) & (y[:-look - 1] >= thr)) + 1
        if len(idx) == 0:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        win = y[idx[:, None] + np.arange(look)[None, :]]
        tr = idx + np.argmin(win, axis=1)
        tr = tr[np.abs(y[tr]) < max_amp_uV]
        keep, last = [], -(10 ** 12)
        for i in tr:
            if i - last > ref:
                keep.append(i)
                last = i
        out.append(np.asarray(keep, dtype=np.int64))
    return out


def reject_common_mode(spikes: list[np.ndarray], tol_ms: float = 0.2, min_channels: int = 3,
                       min_blocks: int = 2) -> tuple[list[np.ndarray], int]:
    """Drop events that coincide (±tol) on >= min_channels channels spanning >= min_blocks blocks.
    The four organoids are not connected, so simultaneous cross-block 'spikes' are interference."""
    ch = np.concatenate([np.full(len(s), c) for c, s in enumerate(spikes)])
    t = np.concatenate(spikes) if len(ch) else np.zeros(0, dtype=np.int64)
    if len(t) == 0:
        return spikes, 0
    o = np.argsort(t, kind="stable")
    t, ch = t[o], ch[o]
    tol = samples(tol_ms)
    lo = np.searchsorted(t, t - tol, "left")
    hi = np.searchsorted(t, t + tol, "right")
    bad = np.zeros(len(t), bool)
    for i in np.flatnonzero(hi - lo >= min_channels):
        chs = np.unique(ch[lo[i]:hi[i]])
        if len(chs) >= min_channels and len(np.unique(chs // BLOCK_SIZE)) >= min_blocks:
            bad[lo[i]:hi[i]] = True
    return [t[(ch == c) & ~bad] for c in range(len(spikes))], int(bad.sum())


def waveforms(Y: np.ndarray, ch: int, idx: np.ndarray, pre_ms: float = 1.0, post_ms: float = 1.5,
              max_n: int = 500) -> np.ndarray:
    pre, post = samples(pre_ms), samples(post_ms)
    idx = idx[(idx >= pre) & (idx < Y.shape[1] - post)][:max_n]
    if len(idx) == 0:
        return np.zeros((0, pre + post), dtype=np.float32)
    return Y[ch][idx[:, None] + np.arange(-pre, post)[None, :]]


def expected_noise_rate(Y: np.ndarray, keep: np.ndarray, k: float) -> np.ndarray:
    """Rice's formula: crossings/s of level -k*sigma for Gaussian noise with the measured
    zero-crossing rate. What a channel with no neurons would report."""
    out = np.empty(Y.shape[0])
    for c in range(Y.shape[0]):
        y = Y[c][keep]
        z = np.count_nonzero((y[1:] < 0) & (y[:-1] >= 0)) / (len(y) / FS)
        out[c] = z * np.exp(-k * k / 2.0)
    return out
