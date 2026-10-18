"""The response readout: one function for every trial, in the screen and in solve (measure.py), on
windows cut from the continuous recording. Self-contained (numpy + scipy only).

Definition (method 'orgchar_bandpass_events_v1'), for one electrode and one stimulation train:
  1. post = voltage from the train's trigger onwards; the train itself and `blank_after_train_ms`
     after its last pulse are replaced by the first sample after that (no artifact edge left to
     filter);
  2. both the baseline (the `baseline_ms` before the trigger) and post are band-passed
     300-3000 Hz (zero phase), which removes the slow swings that dominate the raw voltage on
     some electrodes (hundreds of µV on fs666);
  3. events = negative crossings of the electrode's frozen threshold, at most one per
     `refractory_ms`;
  4. response = events in [train end + window[0], train end + window[1]]
              - baseline events scaled to the window's length.
No sham correction: a no-current trigger evokes nothing (tested many times).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

FS = 30000
METHOD = "orgchar_bandpass_events_v1"
SPEC = dict(method=METHOD, sample_rate_hz=FS, band_hz=[300.0, 3000.0], filter_order=3,
            blank_after_train_ms=4.0, response_window_ms_after_train=[10.0, 210.0],
            baseline_duration_ms=500.0, refractory_ms=1.0, threshold_noise_multiple=5.0)

_SOS = {}


def _sos(spec):
    key = (tuple(spec["band_hz"]), spec["filter_order"], spec["sample_rate_hz"])
    if key not in _SOS:
        _SOS[key] = signal.butter(spec["filter_order"], spec["band_hz"], btype="band",
                                  fs=spec["sample_rate_hz"], output="sos")
    return _SOS[key]


def bandpass(y, spec=SPEC):
    return signal.sosfiltfilt(_sos(spec), np.asarray(y, dtype=float))


def train_end_ms(n_pulses, ipi_ms, phase_us):
    """Time of the end of the last pulse, from the trigger (biphasic: two phases)."""
    return (max(int(n_pulses), 1) - 1) * float(ipi_ms) + 2 * float(phase_us) / 1000.0


def post_samples_needed(n_pulses, ipi_ms, phase_us, spec=SPEC):
    return int(math.ceil((train_end_ms(n_pulses, ipi_ms, phase_us) + spec["response_window_ms_after_train"][1])
                         * spec["sample_rate_hz"] / 1000)) + 1


def count_events(y, threshold_uV, refractory_ms, rate=FS):
    y = np.asarray(y)
    cross = np.flatnonzero((y[1:] < threshold_uV) & (y[:-1] >= threshold_uV)) + 1
    dead = max(1, int(math.ceil(rate * refractory_ms / 1000)))
    last, n, idx = -dead, 0, []
    for i in cross:
        if i - last >= dead:
            n += 1
            last = i
            idx.append(int(i))
    return n, idx


def measure_trial(baseline, post, n_pulses, ipi_ms, phase_us, threshold_uV, spec=SPEC, keep_trace=False):
    """baseline: the `baseline_duration_ms` of voltage before the trigger; post: voltage from the
    trigger on (at least post_samples_needed). Both 1-D, in µV, at spec['sample_rate_hz']."""
    rate = spec["sample_rate_hz"]
    baseline, post = np.asarray(baseline, dtype=float), np.asarray(post, dtype=float)
    nb = int(round(spec["baseline_duration_ms"] * rate / 1000))
    need = post_samples_needed(n_pulses, ipi_ms, phase_us, spec)
    if baseline.ndim != 1 or post.ndim != 1 or len(baseline) < nb or len(post) < need:
        raise ValueError("Incomplete baseline or response voltage window")
    if not (np.isfinite(baseline).all() and np.isfinite(post).all()):
        raise ValueError("Missing or nonfinite voltage; cannot interpret as zero activity")
    if not threshold_uV < 0:
        raise ValueError("Threshold must be negative")
    end = train_end_ms(n_pulses, ipi_ms, phase_us)
    blank = int(math.ceil((end + spec["blank_after_train_ms"]) * rate / 1000))
    p = post[:need].copy()
    p[:blank] = p[blank]
    fp = bandpass(p, spec)
    fb = bandpass(baseline[-nb:], spec)
    a, b = [int(math.ceil((end + ms) * rate / 1000)) for ms in spec["response_window_ms_after_train"]]
    resp, idx = count_events(fp[a:b], threshold_uV, spec["refractory_ms"], rate)
    base, base_idx = count_events(fb, threshold_uV, spec["refractory_ms"], rate)
    out = dict(response_events=resp, baseline_events=base,
               adjusted=resp - base * (b - a) / nb, window_samples=[a, b])
    if keep_trace:      # the filtered voltage and the crossings (window ones from the window's start)
        out.update(trace=fp[a:b].astype(np.float32), event_samples=idx, baseline=fb, post=fp,
                   baseline_samples=base_idx)
    return out


def noise_uV(y, spec=SPEC):
    """Robust noise of the band-passed voltage (median absolute deviation / 0.6745)."""
    f = bandpass(y, spec)
    return float(np.median(np.abs(f - np.median(f))) / 0.6745)


def threshold_from_noise(noise, spec=SPEC):
    return -spec["threshold_noise_multiple"] * float(noise)
