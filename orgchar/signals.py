"""Signal processing for FinalSpark 4x8 MEA raw recordings.

Raw HDF5 layout (verified on the platform): datasets electrode_0..electrode_31, each (n_rows, 1280)
float32 in µV at 30 kHz, already high-pass filtered by the platform; dataset 'triggers' of the same
shape, uint16, bit k set while trigger key k is high.

Pipeline for spikes:  raw µV -> remove stimulation artifacts -> 300-3000 Hz band-pass
                       -> negative threshold at -k * robust noise -> common-mode rejection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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


def ms(n_samples):
    """Samples -> milliseconds (scalar or array)."""
    return np.asarray(n_samples, dtype=float) * 1000.0 / FS


def samples(ms_value: float) -> int:
    return int(round(ms_value * FS / 1000.0))


# --------------------------------------------------------------------------------------- I/O

def load_raw(path, t0_s: float | None = None, t1_s: float | None = None):
    """Return (X, T): X float32 (32, n) in µV, T uint16 (n,) trigger bitmask.
    Optional t0_s/t1_s crop (rounded to the file's 1280-sample rows)."""
    with h5py.File(path, "r") as f:
        row = f["electrode_0"].shape[1]
        r0 = 0 if t0_s is None else int(t0_s * FS) // row
        r1 = None if t1_s is None else int(np.ceil(t1_s * FS / row))
        X = np.stack([f[f"electrode_{i}"][r0:r1].reshape(-1) for i in range(N_CH)]).astype(np.float32)
        T = f["triggers"][r0:r1].reshape(-1).astype(np.uint16)
    return X, T


def trigger_edges(T: np.ndarray) -> list[tuple[int, int]]:
    """All rising edges as (sample, key), sorted by time."""
    edges = []
    for key in range(16):
        hi = (T & (1 << key)) != 0
        edges.extend((int(i), key) for i in np.flatnonzero(hi & np.r_[True, ~hi[:-1]]))
    return sorted(edges)


# ------------------------------------------------------------------------- artifact removal

@dataclass
class ArtifactGroup:
    """Pulses expected to produce the same artifact: same stimulating electrode, same waveform,
    same position in the train. `onsets` are trigger/pulse sample indices."""
    key: tuple
    stim_electrode: int
    onsets: np.ndarray
    length: int  # samples analysed after each onset (<= gap to the next pulse)


@dataclass
class ArtifactResult:
    X: np.ndarray                 # cleaned raw µV
    valid: np.ndarray             # bool (32, n): False where no spike may be detected
    blank_ms: dict = field(default_factory=dict)   # group key -> per-channel blank (ms after onset)
    template_used: dict = field(default_factory=dict)


def raw_noise(X: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Robust per-channel SD of the (already high-passed) raw signal."""
    out = np.empty(X.shape[0])
    for c in range(X.shape[0]):
        x = X[c] if keep is None else X[c][keep]
        out[c] = 1.4826 * np.median(np.abs(x - np.median(x)))
    return out


_LOW_SOS = signal.butter(2, 300.0, btype="low", fs=FS, output="sos")


def remove_artifacts(X: np.ndarray, groups: list[ArtifactGroup], noise: np.ndarray,
                     pre_ms: float = 0.2, floor_ms: float = 2.5, group_k: float = 1.0,
                     pulse_k: float = 6.0, min_reps_for_template: int = 8,
                     fallback_blank_ms: float = 10.0, margin_ms: float = 0.3) -> ArtifactResult:
    """Subtract a per-group median artifact template, then blank (linearly interpolate) each pulse
    until the artifact is over.

    The artifact is deterministic for a given electrode and waveform; spikes are jittered. So:
      * template = median across pulses (does not absorb jittered spikes), subtracted from each pulse;
      * 'artifact over' is judged on the residual BELOW 300 Hz, where artifact tails live and spikes
        carry little energy. Judging on the full band would mistake a real evoked response (which
        is pulse-to-pulse variability) for artifact and blank it away;
      * group blank = last time the <300 Hz residual spread across pulses exceeds group_k * noise;
      * per-pulse extension: a pulse whose own <300 Hz residual still exceeds pulse_k * noise later
        (an outlier artifact, e.g. first pulse of a train) is blanked for longer.
    Groups with too few pulses for a template get a fixed conservative blank instead.
    """
    X = X.copy()
    n = X.shape[1]
    valid = np.ones(X.shape, bool)
    pre = samples(pre_ms)
    margin = samples(margin_ms)
    floor = samples(floor_ms)
    res = ArtifactResult(X=X, valid=valid)
    for g in groups:
        on = np.asarray(g.onsets, dtype=np.int64)
        on = on[(on - pre >= 1) & (on + g.length < n - 1)]
        if len(on) == 0:
            continue
        L = g.length
        idx = on[:, None] + np.arange(-pre, L)[None, :]           # (reps, pre+L)
        if len(on) >= min_reps_for_template:
            S = X[:, idx]                                          # (32, reps, pre+L)
            tmpl = np.median(S, axis=1)                            # (32, pre+L)
            R = S - tmpl[:, None, :]
            Rl = signal.sosfiltfilt(_LOW_SOS, R.astype(np.float64), axis=2)
            spread = 1.4826 * np.median(np.abs(Rl), axis=1)        # (32, pre+L)
            blank = np.empty((N_CH, len(on)), dtype=np.int64)
            for c in range(N_CH):
                bad = np.flatnonzero(spread[c, pre:] > group_k * noise[c])
                b = max(floor, (bad[-1] + 1) if len(bad) else 0)
                blank[c] = b
                for i in range(len(on)):
                    out = np.flatnonzero(np.abs(Rl[c, i, pre + b:]) > pulse_k * noise[c])
                    if len(out):
                        blank[c, i] = b + out[-1] + 1
            X[:, idx] = R.astype(np.float32)
            res.template_used[g.key] = True
        else:
            blank = np.full((N_CH, len(on)), max(floor, samples(fallback_blank_ms)), dtype=np.int64)
            res.template_used[g.key] = False
        blank = np.minimum(blank, L)
        res.blank_ms[g.key] = np.median(blank, axis=1) * 1000.0 / FS
        for c in range(N_CH):
            for i, s in enumerate(on):
                a, e = s - pre, min(n - 1, s + int(blank[c, i]))
                X[c, a:e] = np.linspace(X[c, a - 1], X[c, e], e - a, dtype=np.float32)
                valid[c, max(0, a - margin):min(n, e + margin)] = False
    return res


# ------------------------------------------------------------------------------ spikes

def bandpass(X: np.ndarray) -> np.ndarray:
    """Zero-phase 300-3000 Hz band-pass, channel by channel (bounded memory)."""
    Y = np.empty(X.shape, dtype=np.float32)
    for c in range(X.shape[0]):
        Y[c] = signal.sosfiltfilt(_SOS, X[c].astype(np.float64))
    return Y


def spike_noise(Y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Quiroga's robust noise estimate, median(|y|)/0.6745, on valid samples only."""
    return np.array([np.median(np.abs(Y[c][valid[c]])) / 0.6745 for c in range(Y.shape[0])])


def detect_spikes(Y: np.ndarray, valid: np.ndarray, noise: np.ndarray, k: float = 5.0,
                  max_amp_uV: float = 500.0, refractory_ms: float = 1.0) -> list[np.ndarray]:
    """Negative threshold crossings at -k*noise, aligned to the trough (within 0.5 ms).
    Events outside `valid`, or with |trough| > max_amp_uV (artifact residue), are dropped."""
    ref = samples(refractory_ms)
    look = samples(0.5)
    out = []
    for c in range(Y.shape[0]):
        y = Y[c]
        thr = -k * noise[c]
        idx = np.flatnonzero((y[1:-look] < thr) & (y[:-look - 1] >= thr)) + 1
        idx = idx[valid[c, idx]]
        if len(idx) == 0:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        win = y[idx[:, None] + np.arange(look)[None, :]]
        tr = idx + np.argmin(win, axis=1)
        tr = tr[(np.abs(y[tr]) < max_amp_uV) & valid[c, tr]]
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


def expected_noise_rate(Y: np.ndarray, valid: np.ndarray, noise: np.ndarray, k: float) -> np.ndarray:
    """Rice's formula: crossings/s of level -k*sigma for Gaussian noise with the measured
    zero-crossing rate. What a channel with no neurons would report."""
    out = np.empty(Y.shape[0])
    for c in range(Y.shape[0]):
        y = Y[c][valid[c]]
        z = np.count_nonzero((y[1:] < 0) & (y[:-1] >= 0)) / (len(y) / FS)
        out[c] = z * np.exp(-k * k / 2.0)
    return out


def spikes_pipeline(X: np.ndarray, groups: list[ArtifactGroup] | None = None, k: float = 5.0,
                    quiet_mask: np.ndarray | None = None, noise_override: np.ndarray | None = None):
    """Full pipeline. Returns dict with cleaned/filtered data, valid mask, spikes, noise, blanks.
    quiet_mask: samples (True) usable to estimate noise, e.g. away from any stimulation."""
    rn = raw_noise(X, quiet_mask)
    art = remove_artifacts(X, groups or [], rn)
    Y = bandpass(art.X)
    est_mask = art.valid if quiet_mask is None else (art.valid & quiet_mask[None, :])
    noise = spike_noise(Y, est_mask) if noise_override is None else noise_override
    spk = detect_spikes(Y, art.valid, noise, k=k)
    spk, n_cm = reject_common_mode(spk)
    return dict(Y=Y, X_clean=art.X, valid=art.valid, spikes=spk, noise=noise, raw_noise=rn, blank_ms=art.blank_ms,
                template_used=art.template_used, n_common_mode=n_cm)
