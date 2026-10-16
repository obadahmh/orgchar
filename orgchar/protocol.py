"""What to send, and in which order. Plans are plain dicts saved as plan.json before anything runs.

A plan has segments. A segment = one stimulation-parameter upload + one continuous raw recording.
Each electrode keeps one waveform per upload, including the pulse count of its hardware train, so
every train length gets its own segment.

Trigger key map inside a segment: stim electrodes get keys 0..14 in the order given; key 15 is the
SHAM key: a marker is recorded but no electrode is bound to it, so no current flows.

Train codes (optional, add_train_codes): ~100 ms before each train, two triggers on keys 8..14
(also bound to no electrode) spell the train's number in the segment, base 7 (49 codes). The
recording then says which train each pulse belongs to, instead of us inferring it from timing.
"""
from __future__ import annotations

import random

SHAM_KEY = 15
CODE_KEYS = tuple(range(8, 15))   # 7 keys, never bound to an electrode in a coded segment
CODE_LEAD_S = 0.100               # first code trigger this long before the train's first trigger
CODE_GAP_S = 0.010                # between the two code triggers
N_CODES = len(CODE_KEYS) ** 2

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


def _electrodes(stims, amplitude_uA, phase_us, polarity, n_pulses=1, ipi_ms=0.0):
    """n_pulses > 1: a hardware train (the board repeats the pulse every ipi_ms after one trigger),
    as the network delivers it."""
    if len(stims) > 15:
        raise ValueError("At most 15 stimulation electrodes per segment (key 15 is the sham key)")
    return {int(e): dict(key=i, amplitude_uA=amplitude_uA, phase_us=phase_us, polarity=polarity,
                         n_pulses=int(n_pulses), ipi_ms=float(ipi_ms) if n_pulses > 1 else 0.0)
            for i, e in enumerate(stims)}


def plan_census(duration_s=300.0):
    return dict(step="census", segments=[dict(id="seg00", electrodes={}, events=[], duration_s=duration_s)])


def plan_duration_s(plan):
    """Recording time plus the platform's overhead between recordings: about 50 s (upload, waits,
    disable) plus half the recording's length while the platform saves the file (fs666, 2 Oct 2026:
    85 s after a 41 s recording, 106 s after an 84 s one, before raw files stopped being copied)."""
    return sum(1.5 * s["duration_s"] + 50.0 for s in plan["segments"])


def train_code_keys(n: int) -> tuple[int, int]:
    """Keys spelling train number n (mod 49): two base-7 digits."""
    n %= N_CODES
    return CODE_KEYS[n // len(CODE_KEYS)], CODE_KEYS[n % len(CODE_KEYS)]


def decode_keys(k1: int, k2: int) -> int:
    return CODE_KEYS.index(k1) * len(CODE_KEYS) + CODE_KEYS.index(k2)


def add_train_codes(plan: dict) -> dict:
    """Precede every train (stim or sham) with its two code triggers. Train n of a segment gets
    code n mod 49; trains are >= 1 s apart, so neighbouring codes are never confused."""
    for seg in plan["segments"]:
        used = {v["key"] for v in seg["electrodes"].values()}
        if used & set(CODE_KEYS):
            raise ValueError(f"Segment {seg['id']} binds keys {sorted(used & set(CODE_KEYS))} to electrodes; "
                             "train codes need keys 8-14 free (at most 8 stimulating electrodes per segment)")
        starts = {}
        for ev in seg["events"]:
            starts.setdefault(ev["train_id"], ev["t_s"])
        prev_end, codes = 0.0, []
        for n, (tid, t0) in enumerate(starts.items()):
            t_code = round(t0 - CODE_LEAD_S, 4)
            if t_code - prev_end < 0.5:
                raise ValueError(f"No room for the code of train {tid}: previous trigger {t_code - prev_end:.3f} s before")
            for i, key in enumerate(train_code_keys(n)):
                codes.append(dict(key=key, kind="code", stim_electrode=-1, level=None, pulse_index=i - 2,
                                  train_id=tid, code=n % N_CODES, t_s=round(t_code + i * CODE_GAP_S, 4)))
            prev_end = max(e["t_s"] for e in seg["events"] if e["train_id"] == tid)
        seg["events"] = sorted(seg["events"] + codes, key=lambda e: e["t_s"])
        seg["train_codes"] = True
    plan["train_codes"] = True
    return plan


def plan_codetest(stims, amplitude_uA=2.0, phase_us=100.0, polarity="negative-first", n_codes=60,
                  n_positive=5, seed=0):
    """One-minute check that train codes are safe and readable. The stimulating electrodes' real
    parameters are uploaded exactly as in a run (keys 0..), but no electrode is triggered: every
    ~1 s a code (keys 8-14) is sent, followed by a sham marker (key 15), cycling through all 49 codes. At the end, n_positive single pulses on the first
    electrode show what a real stimulation artifact looks like in this recording (positive control)."""
    check_setting(amplitude_uA, phase_us)
    el = _electrodes(stims, amplitude_uA, phase_us, polarity)
    trials = [dict(key=SHAM_KEY, kind="sham", stim_electrode=-1, level=0.0, pulse_index=0) for _ in range(n_codes)]
    trials += [dict(key=el[stims[0]]["key"], kind="stim", stim_electrode=stims[0], level=amplitude_uA, pulse_index=0)
               for _ in range(n_positive)]
    evs = _spaced_timeline(trials, (1.0, 1.2), (1.0, 1.2), random.Random(seed))
    for i, ev in enumerate(evs):
        ev["train_id"] = f"c{i:03d}"
    plan = dict(step="codetest", amplitude_uA=amplitude_uA, phase_us=phase_us, polarity=polarity,
                segments=[dict(id="seg00", electrodes=el, events=evs, duration_s=round(evs[-1]["t_s"] + 2.0, 1))])
    return add_train_codes(plan)


def _spaced_timeline(trials, organoid_isi_s, spacing_s, rng, t0=1.0):
    """Order and time trials so that any two trains are spacing_s apart and the same organoid is
    stimulated at most every organoid_isi_s (shams deliver no current and only need spacing).
    trials: list of event dicts (one trigger each) with stim_electrode (-1 for sham)."""
    queue, ready, t, out = list(trials), {}, t0, []
    while queue:
        pick = next((i for i, ev in enumerate(queue)
                     if ev["stim_electrode"] < 0 or t >= ready.get(ev["stim_electrode"] // 8, 0.0)), None)
        if pick is None:
            t = min(ready[e["stim_electrode"] // 8] for e in queue if e["stim_electrode"] >= 0)
            continue
        ev = queue.pop(pick)
        out.append(dict(ev, t_s=round(t, 4)))
        if ev["stim_electrode"] >= 0:
            ready[ev["stim_electrode"] // 8] = t + rng.uniform(*organoid_isi_s)
        t += rng.uniform(*spacing_s)
    return out


SCREEN_COUNTS = (2, 10, 20)   # still tells falling, peaked and rising apart
MAX_UPLOAD_STIMS = 8          # keys 8-14 stay free for train codes


# One recording holds at most this much: larger raw files made stop_raw_recording time out (Sep 2026),
# and a 38-min FuncSolver recording (fs675, 14 Oct 2026, ~9 GB) was never saved to the S3 mount.
MAX_SEGMENT_S = 360.0


def plan_trials(trials, ipi_ms=10.0, amplitude_uA=2.0, phase_us=100.0, polarity="negative-first",
                organoid_isi_s=(4.0, 4.5), spacing_s=(1.0, 1.25), seed=0):
    """Segments that deliver an arbitrary list of trials (measure.measure).

    trials: (stim_electrode, n_pulses) pairs, one hardware train each. A train's length is part of
    the upload, so each segment binds every electrode to one length; segments repeat until every
    trial is placed; a segment longer than MAX_SEGMENT_S hands its later trials to the next one. Event i
    of the input list carries request=i. No sham trials.

    A segment takes at most MAX_UPLOAD_STIMS electrodes, dealt round-robin across organoids, so the
    organoids rest in parallel. Each electrode gets the length with most trials still waiting, ties at
    random, so lengths are not measured in order (responses drift within a session)."""
    trials = [(int(e), int(n)) for e, n in trials]
    for n in {n for _, n in trials}:
        check_setting(amplitude_uA, phase_us, n, ipi_ms)
    rng = random.Random(seed)
    pending, segs = list(enumerate(trials)), []
    while pending:
        counts = {}
        for _, (e, n) in pending:
            counts.setdefault(e, {}).setdefault(n, 0)
            counts[e][n] += 1
        by_org = {}
        for e in sorted(counts):
            by_org.setdefault(e // 8, []).append(e)
        dealt = [e for i in range(8) for o in sorted(by_org) for e in by_org[o][i:i + 1]][:MAX_UPLOAD_STIMS]
        length = {e: max(counts[e], key=lambda n: (counts[e][n], rng.random())) for e in dealt}
        el = {e: dict(key=i, amplitude_uA=amplitude_uA, phase_us=phase_us, polarity=polarity,
                      n_pulses=n, ipi_ms=float(ipi_ms) if n > 1 else 0.0) for i, (e, n) in enumerate(length.items())}
        now = [(i, e, n) for i, (e, n) in pending if length.get(e) == n]
        pending = [(i, t) for i, t in pending if length.get(t[0]) != t[1]]
        evs = [dict(key=el[e]["key"], kind="stim", stim_electrode=e, level=float(n), pulse_index=0, request=i)
               for i, e, n in now]
        rng.shuffle(evs)
        evs = _spaced_timeline(evs, organoid_isi_s, spacing_s, rng)      # in time order
        later = {ev["request"] for ev in evs if ev["t_s"] > MAX_SEGMENT_S}
        pending += [(i, (e, n)) for i, e, n in now if i in later]
        evs = [ev for ev in evs if ev["request"] not in later]
        sid = f"seg{len(segs):02d}"
        for k, ev in enumerate(evs):
            ev.update(train_id=f"{sid}t{k:03d}", partition="live", block=0)
        segs.append(dict(id=sid, electrodes=el, events=evs, partition="live", block=0,
                         duration_s=round(evs[-1]["t_s"] + 1.5, 1)))
    return dict(step="live", vary="pulse_train", readout="after_train", ipi_ms=ipi_ms, amplitude_uA=amplitude_uA,
                phase_us=phase_us, polarity=polarity, organoid_isi_s=list(organoid_isi_s), segments=segs)
