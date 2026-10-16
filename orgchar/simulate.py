"""Synthetic FinalSpark recordings with known ground truth, written in the platform's HDF5 layout.

Used by the tests and by `orgchar ... --simulate` to exercise the full workflow without hardware.
The artifact model follows what fs647 showed on 8 Sep 2026: huge transient + slow tail on the
stimulated block, a short transient on the 'mirror' channels (31 - e) of another block, and ~5 %
pulse-to-pulse gain jitter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy import signal

from .signals import FS, N_CH, block_of, samples


@dataclass
class Organoid:
    noise_uV: float = 5.0
    spont_hz: dict = field(default_factory=lambda: {16: 2.0, 20: 4.0, 21: 2.0, 22: 3.0, 23: 2.5, 13: 1.0})
    # (stim, rec) -> (shape, a, b, c): mean extra spikes 10-210 ms after a train of n pulses;
    # 'long': a / (1 + exp(-b (n - c))), 'short': a / (1 + exp(b (n - c))), 'band': a exp(-(n - b)^2 / 2c^2)
    after_train: dict = field(default_factory=dict)
    seed: int = 1


def demo_organoid() -> Organoid:
    """The organoid `--simulate` runs on: one falling, one peaked and one rising after-train response
    (stim 5, 13, 21), so the screen and solve have something to find."""
    # sizes as on real tissue (fs675, Oct 2026: ranges of 40-100 evoked spikes)
    return Organoid(spont_hz={16: 2.0, 20: 4.0, 21: 2.0, 22: 3.0, 23: 2.5, 13: 1.0, 3: 3.0, 10: 3.0},
                    after_train={(5, 3): ("band", 30, 10, 3.5), (13, 10): ("short", 40, 0.6, 9),
                                 (21, 22): ("long", 35, 0.5, 10)})


def after_train_mean(shape, a, b, c, n):
    if shape == "long":
        return a / (1 + np.exp(-b * (n - c)))
    if shape == "short":
        return a / (1 + np.exp(b * (n - c)))
    return a * np.exp(-((n - b) ** 2) / (2 * c ** 2))


def _spike_shape(amp):
    t = np.arange(-30, 45) / FS * 1e3
    return (-amp * np.exp(-(t / 0.15) ** 2) + 0.4 * amp * np.exp(-((t - 0.6) / 0.35) ** 2)).astype(np.float32)


def _artifact(kind: str, amp_uA: float, phase_us: float, n: int) -> np.ndarray:
    t = np.arange(n) / FS * 1e3
    q = amp_uA * phase_us / 200.0          # charge relative to 2 µA x 100 µs
    a = np.zeros(n, np.float32)
    on = samples(0.25)
    if kind == "self":
        a[on:on + samples(1.0)] = 5000 * np.sign(np.sin(np.arange(samples(1.0)) / 3.0))
        a[on + samples(1.0):] = -2500 * q * np.exp(-(t[on + samples(1.0):] - 1.25) / 3.0)
    elif kind == "block":
        a[on:on + samples(0.6)] = 1500 * q * np.sign(np.sin(np.arange(samples(0.6)) / 2.0))
        a[on + samples(0.6):] = -350 * q * np.exp(-(t[on + samples(0.6):] - 0.85) / 1.8)
    elif kind == "mirror":
        a[on:on + samples(0.3)] = 800 * q * np.sign(np.sin(np.arange(samples(0.3)) / 2.0))
    return np.clip(a, -6300, 6300)


def record(org: Organoid, duration_s: float, pulses: list[tuple[int, int, float, float]],
           triggers: list[tuple[int, int]], path) -> None:
    """Write a synthetic recording.
    pulses: (sample, stim_electrode, amplitude_uA, phase_us) for every delivered pulse.
    triggers: (sample, key) digital marker edges (2 samples high)."""
    rng = np.random.default_rng(org.seed + int(duration_s * 1000) + len(pulses))
    n = int(duration_s * FS)
    n -= n % 1280
    sos = signal.butter(2, [250, 6000], btype="band", fs=FS, output="sos")
    X = signal.sosfilt(sos, rng.standard_normal((N_CH, n)), axis=1).astype(np.float32)
    X *= org.noise_uV / X.std(axis=1, keepdims=True)
    trains = {}
    for c, hz in org.spont_hz.items():  # bursty spontaneous firing (gamma-distributed intervals)
        t = np.cumsum(rng.gamma(0.5, 1.0 / (0.5 * hz), size=int(hz * duration_s * 3) + 10))
        trains[c] = list((t[t < duration_s - 0.01] * FS).astype(int))
    if org.after_train:  # trains = runs of pulses on one electrode less than 20 ms apart
        by_e = {}
        for s, e, *_ in sorted(pulses):
            runs = by_e.setdefault(e, [])
            if runs and s - runs[-1][-1] < samples(20):
                runs[-1].append(s)
            else:
                runs.append([s])
        for (se, rc), (shape, a, b, c) in org.after_train.items():
            for run in by_e.get(se, []):
                lam = after_train_mean(shape, a, b, c, len(run))
                for _ in range(rng.poisson(max(lam, 0.0))):
                    trains.setdefault(rc, []).append(run[-1] + samples(rng.uniform(12.0, 205.0)))
    for c, tr in trains.items():
        for s in tr:
            if 30 <= s < n - 45:
                X[c, s - 30:s + 45] += _spike_shape(rng.uniform(35, 70))
    L = samples(30)
    for s, e, amp, ph in pulses:
        g = rng.normal(1.0, 0.05)
        for c in range(N_CH):
            kind = "self" if c == e else ("block" if block_of(c) == block_of(e) else
                                         ("mirror" if abs(c - (31 - e)) <= 1 else None))
            if kind and s + L < n:
                X[c, s:s + L] += g * _artifact(kind, amp, ph, L)
    T = np.zeros(n, np.uint16)
    for s, key in triggers:
        T[s:s + 2] |= np.uint16(1 << key)
    with h5py.File(path, "w") as f:
        for c in range(N_CH):
            d = f.create_dataset(f"electrode_{c}", data=X[c].reshape(-1, 1280), chunks=(1, 1280))
            d.attrs["sampling_rate"] = np.uint32(FS)
        d = f.create_dataset("triggers", data=T.reshape(-1, 1280), chunks=(1, 1280))
        d.attrs["sampling_rate"] = np.uint32(FS)
