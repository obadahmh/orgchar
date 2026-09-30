"""What to send, and in which order. Plans are plain dicts saved as plan.json before anything runs.

A plan has segments. A segment = one stimulation-parameter upload + one continuous raw recording.
Each electrode keeps one waveform per upload, and an upload stops the board, so anything that
changes the waveform (e.g. amplitude) needs a new segment; anything that only changes timing
(e.g. number of pulses) is done with software-timed single-pulse triggers inside one segment.

Trigger key map inside a segment: stim electrodes get keys 0..14 in the order given; key 15 is the
SHAM key: a marker is recorded but no electrode is bound to it, so no current flows.
"""
from __future__ import annotations

import random

SHAM_KEY = 15
MAX_SEGMENT_S = 360.0          # keep raw files ~1.4 GB; larger files made stop_raw_recording time out

# Hard limits, checked before any upload. Defaults: the envelope used on this platform so far
# (2 µA x 100-200 µs); the vendor's own StimScan example uses 4 µA x 200 µs = 0.8 nC per phase.
LIMITS = dict(max_amplitude_uA=3.0, max_phase_us=200.0, max_charge_nC=0.4, max_pulses=20,
              min_ipi_ms=5.0, min_isi_s=1.0)


def check_setting(amplitude_uA, phase_us, n_pulses=1, ipi_ms=0.0, limits=LIMITS):
    q = amplitude_uA * phase_us / 1000.0
    problems = []
    if not 0 < amplitude_uA <= limits["max_amplitude_uA"]:
        problems.append(f"amplitude {amplitude_uA} µA outside (0, {limits['max_amplitude_uA']}]")
    if not 0 < phase_us <= limits["max_phase_us"]:
        problems.append(f"phase {phase_us} µs outside (0, {limits['max_phase_us']}]")
    if q > limits["max_charge_nC"] + 1e-9:
        problems.append(f"charge {q:.3f} nC/phase > {limits['max_charge_nC']}")
    if n_pulses > limits["max_pulses"]:
        problems.append(f"{n_pulses} pulses > {limits['max_pulses']}")
    if n_pulses > 1 and ipi_ms < limits["min_ipi_ms"]:
        problems.append(f"IPI {ipi_ms} ms < {limits['min_ipi_ms']}")
    if problems:
        raise ValueError("Stimulation outside safety limits: " + "; ".join(problems))


def _electrodes(stims, amplitude_uA, phase_us, polarity):
    if len(stims) > 15:
        raise ValueError("At most 15 stimulation electrodes per segment (key 15 is the sham key)")
    return {int(e): dict(key=i, amplitude_uA=amplitude_uA, phase_us=phase_us, polarity=polarity,
                         n_pulses=1, ipi_ms=0.0) for i, e in enumerate(stims)}


def _timeline(trials, isi_s, rng):
    """trials: list of lists of events (a train = several events). Assign times: the next trial
    starts isi_s[0]..isi_s[1] s after the previous trial's last pulse."""
    t, out = 1.0, []
    for tr in trials:
        t0 = t
        for ev in tr:
            out.append(dict(ev, t_s=round(t0 + ev.pop("dt_s", 0.0), 4)))
        t = out[-1]["t_s"] + rng.uniform(*isi_s)
    return out


def _split(events, electrodes, seg_prefix, max_s=MAX_SEGMENT_S):
    """Cut a long event list into segments at trial boundaries."""
    segs, cur, start = [], [], None
    for ev in events:
        if start is None:
            start = ev["t_s"]
        if ev["t_s"] - start > max_s and ev["pulse_index"] == 0:
            segs.append(cur)
            cur, start = [], ev["t_s"]
        cur.append(ev)
    if cur:
        segs.append(cur)
    out = []
    for i, evs in enumerate(segs):
        t0 = evs[0]["t_s"] - 1.0
        for ev in evs:
            ev["t_s"] = round(ev["t_s"] - t0, 4)
        out.append(dict(id=f"{seg_prefix}{i:02d}", electrodes=electrodes, events=evs,
                        duration_s=round(evs[-1]["t_s"] + 2.0, 1)))
    return out


def plan_census(duration_s=300.0):
    return dict(step="census", segments=[dict(id="seg00", electrodes={}, events=[], duration_s=duration_s)])


def plan_screen(blocks=None, electrodes=None, amplitude_uA=2.0, phase_us=100.0, polarity="negative-first",
                reps=30, sham_reps=30, isi_s=(1.0, 1.5), seed=0):
    """Single biphasic pulses on every electrode of each chosen block, interleaved at random with
    sham markers. One segment per block so each organoid gets its own upload and recording."""
    check_setting(amplitude_uA, phase_us)
    rng = random.Random(seed)
    groups = ([[e for e in electrodes if e // 8 == b] for b in sorted({e // 8 for e in electrodes})]
              if electrodes else [list(range(b * 8, b * 8 + 8)) for b in blocks])
    segs = []
    for stims in groups:
        el = _electrodes(stims, amplitude_uA, phase_us, polarity)
        trials = [[dict(key=el[e]["key"], kind="stim", stim_electrode=e, level=amplitude_uA, pulse_index=0)]
                  for e in stims for _ in range(reps)]
        trials += [[dict(key=SHAM_KEY, kind="sham", stim_electrode=-1, level=0.0, pulse_index=0)]
                   for _ in range(sham_reps)]
        rng.shuffle(trials)
        for i, tr in enumerate(trials):
            tr[0]["train_id"] = f"b{stims[0] // 8}t{i:04d}"
        evs = _timeline(trials, isi_s, rng)
        segs += _split(evs, el, f"block{stims[0] // 8}_")
    return dict(step="screen", amplitude_uA=amplitude_uA, phase_us=phase_us, polarity=polarity,
                reps=reps, sham_reps=sham_reps, segments=segs)


def plan_curve_pulses(stims, levels=(1, 2, 4, 6, 8, 10), ipi_ms=30.0, amplitude_uA=2.0, phase_us=100.0,
                      polarity="negative-first", reps=20, sham_reps=20, isi_s=(1.5, 2.0), seed=0):
    """Input = number of pulses (software-timed at ipi_ms), all levels interleaved at random in time."""
    check_setting(amplitude_uA, phase_us, max(levels), ipi_ms)
    rng = random.Random(seed)
    el = _electrodes(stims, amplitude_uA, phase_us, polarity)
    trials = []
    for e in stims:
        for n in levels:
            for _ in range(reps):
                trials.append([dict(key=el[e]["key"], kind="stim", stim_electrode=e, level=float(n), pulse_index=i,
                                    dt_s=i * ipi_ms / 1000.0) for i in range(int(n))])
    trials += [[dict(key=SHAM_KEY, kind="sham", stim_electrode=-1, level=0.0, pulse_index=0)] for _ in range(sham_reps)]
    rng.shuffle(trials)
    for i, tr in enumerate(trials):
        for ev in tr:
            ev["train_id"] = f"t{i:04d}"
    evs = _timeline(trials, isi_s, rng)
    return dict(step="curve", vary="pulses", levels=list(levels), ipi_ms=ipi_ms, amplitude_uA=amplitude_uA,
                phase_us=phase_us, polarity=polarity, reps=reps, segments=_split(evs, el, "seg"))


def plan_curve_amplitude(stims, levels=(0.5, 1.0, 1.5, 2.0, 3.0), phase_us=100.0, polarity="negative-first",
                         reps_per_pass=10, passes=3, sham_per_segment=4, isi_s=(1.0, 1.5), seed=0):
    """Input = current amplitude. One segment per (pass, level); level order reshuffled every pass so
    slow drift is spread over all levels instead of confounding them."""
    for a in levels:
        check_setting(a, phase_us)
    rng = random.Random(seed)
    segs = []
    for p in range(passes):
        order = list(levels)
        rng.shuffle(order)
        for a in order:
            el = _electrodes(stims, a, phase_us, polarity)
            trials = [[dict(key=el[e]["key"], kind="stim", stim_electrode=e, level=float(a), pulse_index=0)]
                      for e in stims for _ in range(reps_per_pass)]
            trials += [[dict(key=SHAM_KEY, kind="sham", stim_electrode=-1, level=0.0, pulse_index=0)]
                       for _ in range(sham_per_segment)]
            rng.shuffle(trials)
            for i, tr in enumerate(trials):
                tr[0]["train_id"] = f"p{p}a{a}t{i:03d}"
            evs = _timeline(trials, isi_s, rng)
            segs += _split(evs, el, f"pass{p}_amp{a:g}_")
    return dict(step="curve", vary="amplitude", levels=list(levels), phase_us=phase_us, polarity=polarity,
                reps=reps_per_pass * passes, segments=segs)


def plan_duration_s(plan, overhead_s=25.0):
    return sum(s["duration_s"] + (overhead_s if s["electrodes"] else 5.0) for s in plan["segments"])
