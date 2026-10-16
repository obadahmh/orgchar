"""The one hardware primitive: deliver trains (stimulating electrode, pulses) and read the evoked spikes on
every other electrode of the stimulated organoid.

Requests are grouped into recordings by train length (a length is part of the upload), at most 6 min
each, organoids resting >= 4 s between trains (protocol.plan_trials). Each recording is aligned to its
triggers and read out as soon as it is saved, and its responses are written to disk at once, so a crash
loses at most the recording in progress. Trains that could not be used (lost in a dropped row,
unmatched) are requested again, up to `retries` times; any still missing stay NaN.

Spike thresholds are fixed per electrode from the pre-train baselines of the first recording (pass them
on to later calls of the same session). No sham trials: a no-current trigger evokes nothing (tested
many times), and the readout subtracts each trial's own pre-train baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol as P
from . import readout as RO
from . import signals as S
from .backends import clear_status, match_triggers, status

CONDITION = dict(ipi_ms=10.0, amplitude_uA=2.0, phase_us=100.0, polarity="negative-first")
# Artifact check: the median across trials of the first ARTIFACT_MS of the counting window, in units of
# what a median of that many trials varies by anyway. An artifact tail is the same on every trial (large
# median, small spread); spikes are not time-locked (fs666 e10, 1 Oct 2026). Pairs over the limit are
# excluded.
ARTIFACT_MS, ARTIFACT_LIMIT, ARTIFACT_TRIALS = 5.0, 3.0, 24


def artifact_score(traces) -> float:
    A = np.stack(traces)
    spread = np.sqrt((A.std(axis=0) ** 2).mean())
    return float(np.sqrt((np.median(A, axis=0) ** 2).mean()) / max(1.2533 * spread / np.sqrt(len(A)), 1e-9))


def organoid_of(e: int) -> range:
    return range(8 * (e // 8), 8 * (e // 8) + 8)


def thresholds_from(X, onsets, spec=RO.SPEC) -> dict[int, float]:
    """-k x robust noise per electrode, from the half-second before each train (at most 60 trains)."""
    nb = int(round(spec["baseline_duration_ms"] * spec["sample_rate_hz"] / 1000))
    on = [o for o in onsets if o > nb][:60]
    return {c: RO.threshold_from_noise(RO.noise_uV(np.concatenate([X[c, o - nb:o - S.samples(20)] for o in on])), spec)
            for c in range(S.N_CH)}


def _record(requests, backend, call_dir: Path, tag: str, codes: bool, thresholds, seed: int, condition, traces):
    """One pass over `requests` [(stim, pulses)]: rows (request, stim, pulses, rec, response, recording)."""
    plan = P.plan_trials(requests, seed=seed, **condition)
    if codes:
        P.add_train_codes(plan)
    call_dir.mkdir(parents=True)
    (call_dir / "plan.json").write_text(json.dumps(plan, indent=1))
    rows = []
    for k, seg in enumerate(plan["segments"]):
        seg_dir = call_dir / seg["id"]
        seg_dir.mkdir()
        n = int(next(iter(seg["electrodes"].values()))["n_pulses"]) if len(seg["electrodes"]) == 1 else None
        print(f"  [{k + 1}/{len(plan['segments'])}] {call_dir.name} {seg['id']}: "
              f"{sum(e['kind'] == 'stim' for e in seg['events'])} trains"
              + (f" of {n} pulses" if n else ""), flush=True)
        raw = backend.run_segment(seg, seg_dir, f"{tag}_{call_dir.name}_{seg['id']}")
        status("checking the recording…", live=True)
        sent_csv = seg_dir / "sent.csv"     # the simulated backend writes none
        sent = pd.read_csv(sent_csv) if sent_csv.exists() and sent_csv.stat().st_size > 1 else None
        trials = match_triggers(seg, raw, sent, min_fraction=0)
        trials.to_csv(seg_dir / "trials.csv", index=False)
        X, _ = S.load_raw(raw)
        stim = trials[~trials.exclude & (trials.kind == "stim")]
        if thresholds is None:
            thresholds = thresholds_from(X, stim.onset.astype(int).tolist())
        request = {e["train_id"]: e["request"] for e in seg["events"] if e["kind"] == "stim"}
        nb = int(round(RO.SPEC["baseline_duration_ms"] * RO.SPEC["sample_rate_hz"] / 1000))
        seg_rows = []
        for t in stim.itertuples():
            n_p, o = int(t.n_pulses), int(t.onset)
            need = RO.post_samples_needed(n_p, condition["ipi_ms"], condition["phase_us"])
            if o < nb or o + need > X.shape[1]:
                continue
            for c in organoid_of(int(t.stim_electrode)):
                if c == t.stim_electrode:
                    continue
                m = RO.measure_trial(X[c, o - nb:o], X[c, o:o + need], n_p, condition["ipi_ms"], condition["phase_us"],
                                     thresholds[c], keep_trace=traces is not None)
                if traces is not None:
                    lst = traces.setdefault((int(t.stim_electrode), c, n_p), [])
                    if len(lst) < ARTIFACT_TRIALS:
                        lst.append(m["trace"][:S.samples(ARTIFACT_MS)])
                seg_rows.append(dict(request=request[t.train_id], stim=int(t.stim_electrode), pulses=n_p, rec=c,
                                     response=m["adjusted"], recording=f"{call_dir.name}/{seg['id']}"))
        pd.DataFrame(seg_rows).to_csv(seg_dir / "responses.csv", index=False)   # on disk before the next one
        rows += seg_rows
        clear_status()
        got = len({r["request"] for r in seg_rows})
        print(f"        {'✓' if got == len(request) else '!'} {got}/{len(request)} trains usable", flush=True)
    return rows, thresholds


def measure(requests, backend, out_dir: Path, tag: str, codes: bool = False, thresholds=None, retries: int = 2,
            seed: int = 0, condition=CONDITION, traces: dict | None = None):
    """Deliver every (stim, pulses) request once; returns (responses, thresholds). `responses` has one
    row per request and recording electrode: request (index into `requests`), stim, pulses, rec,
    response (evoked spikes, minus the trial's own pre-train baseline), recording. `traces`, if given, is
    filled with the start of the counting window per (stim, rec, pulses), for artifact_score."""
    rows, pending, attempt = [], list(range(len(requests))), 0
    while pending and attempt <= retries:
        name = "pass" if attempt == 0 else f"retry{attempt}"
        got, thresholds = _record([requests[i] for i in pending], backend, out_dir / name, tag, codes, thresholds,
                                  seed + attempt, condition, traces)
        for r in got:                      # map back to the caller's request index
            r["request"] = pending[r["request"]]
        rows += got
        done = {r["request"] for r in got}
        pending = [i for i in pending if i not in done]
        attempt += 1
    if pending:
        print(f"  {len(pending)} of {len(requests)} trains still unusable after {retries} retries")
    df = pd.DataFrame(rows, columns=["request", "stim", "pulses", "rec", "response", "recording"])
    df.to_csv(out_dir / "responses.csv", index=False)
    return df, thresholds
