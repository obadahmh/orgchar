"""Census: which electrodes and organoids contain firing neurons. Raw HDF5 in, plain dict out
(JSON-serialisable); no hardware access here. The screen's analysis is in screen.py."""
from __future__ import annotations

import numpy as np

from . import signals as S

# Minimum evidence a channel must show in the census to count as containing firing neurons.
ACTIVE_MIN_RATE_HZ = 0.1
ACTIVE_MIN_SPIKES = 20
ACTIVE_MIN_SNR = 5.5


def census(raw_path, k: float = 5.0) -> dict:
    """Spontaneous activity, no stimulation. Answers: which channels/organoids contain firing neurons?"""
    X, T = S.load_raw(raw_path)
    edges = S.trigger_edges(T)
    quiet = np.ones(X.shape[1], bool)
    for s, _ in edges:  # a census has no stimulation, but ignore anything near a stray trigger
        quiet[max(0, s - S.samples(50)):s + S.FS] = False
    quiet[:S.samples(100)] = quiet[-S.samples(100):] = False
    Y = S.bandpass(X)
    noise = S.spike_noise(Y, quiet)
    spk, n_cm = S.reject_common_mode(S.detect_spikes(Y, noise, k=k))
    dur = quiet.sum() / S.FS
    noise_rate = S.expected_noise_rate(Y, quiet, k)
    chans = []
    for c in range(S.N_CH):
        sp = spk[c][quiet[spk[c]]]
        W = S.waveforms(Y, c, sp)
        mw = W.mean(0) if len(W) else np.zeros(W.shape[1])
        trough = float(-mw.min()) if len(W) else 0.0
        rate = len(sp) / dur
        active = (rate >= ACTIVE_MIN_RATE_HZ and len(sp) >= ACTIVE_MIN_SPIKES
                  and trough / noise[c] >= ACTIVE_MIN_SNR and rate > 10 * noise_rate[c])
        isi = np.diff(sp) / S.FS
        chans.append(dict(channel=c, block=S.block_of(c), noise_uV=float(noise[c]), spikes=int(len(sp)),
                          rate_hz=float(rate), noise_only_rate_hz=float(noise_rate[c]),
                          mean_trough_uV=trough, snr=float(trough / noise[c]) if noise[c] else 0.0,
                          burstiness_cv=float(isi.std() / isi.mean()) if len(isi) > 5 else None,
                          active=bool(active), waveform=mw.round(2).tolist()))
    blocks = []
    for b in range(4):
        cs = [c for c in chans if c["block"] == b]
        n_act = sum(c["active"] for c in cs)
        blocks.append(dict(block=b, channels=list(range(b * 8, b * 8 + 8)), active_channels=n_act,
                           total_rate_hz=float(sum(c["rate_hz"] for c in cs)),
                           verdict="LIVE" if n_act >= 1 else "SILENT"))
    return dict(step="census", raw=str(raw_path), duration_s=float(dur), threshold_k=k,
                common_mode_events_removed=n_cm, channels=chans, blocks=blocks,
                live_blocks=[b["block"] for b in blocks if b["verdict"] == "LIVE"],
                _spikes=spk, _dur=dur)
