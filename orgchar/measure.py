"""The one hardware primitive: deliver trains (stimulating electrode, pulses) and read the evoked spikes on
every other electrode of the stimulated organoid.

Requests are grouped into recordings by train length (a length is part of the upload), at most 6 min
each, organoids resting >= 4 s between trains (protocol.plan_trials). Each recording is aligned to its
triggers and read out as soon as it is saved, and its responses are written to disk at once, so a crash
loses at most the recording in progress. Trains that could not be used (lost in a dropped row,
unmatched) are requested again, up to `retries` times; any still missing stay NaN.

Spike thresholds are fixed per electrode from the pre-train baselines of the first recording (pass them
on to later calls of the same session). No sham trials: a no-current trigger evokes nothing (tested
many times), and the readout subtracts each trial's own pre-train baseline. A final 'disable all
stimulation' that went unanswered is reported (the next recording starts with one anyway).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol as P
from . import readout as RO
from . import signals as S
from .backends import clear_status, match_triggers, recording, status

CONDITION = dict(ipi_ms=10.0, amplitude_uA=2.0, phase_us=100.0, polarity="negative-first")
# Artifact check: the median across trials of the first ARTIFACT_MS of the counting window, in units of
# what a median of that many trials varies by anyway. An artifact tail is the same on every trial (large
# median, small spread); spikes are not time-locked (fs666 e10, 1 Oct 2026). Pairs over the limit are
# excluded.
ARTIFACT_MS, ARTIFACT_LIMIT, ARTIFACT_TRIALS = 5.0, 3.0, 24
COLUMNS = ["request", "stim", "pulses", "rec", "response", "recording"]


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


def _read(seg, raw, trials, label, thresholds, condition, all_channels, traces):
    """Rows (request, stim, pulses, rec, response, recording) for one recording's usable trains."""
    X, _ = S.load_raw(raw)
    stim = trials[~trials.exclude & (trials.kind == "stim")]
    if thresholds is None:
        thresholds = thresholds_from(X, stim.onset.astype(int).tolist())
    request = {e["train_id"]: e["request"] for e in seg["events"] if e["kind"] == "stim"}
    nb = int(round(RO.SPEC["baseline_duration_ms"] * RO.SPEC["sample_rate_hz"] / 1000))
    rows = []
    for t in stim.itertuples():
        s, n_p, o = int(t.stim_electrode), int(t.n_pulses), int(t.onset)
        need = RO.post_samples_needed(n_p, condition["ipi_ms"], condition["phase_us"])
        if o < nb or o + need > X.shape[1]:
            continue
        own = organoid_of(s)
        for c in range(S.N_CH) if all_channels else own:
            if c == s:
                continue
            keep = traces is not None and c in own
            m = RO.measure_trial(X[c, o - nb:o], X[c, o:o + need], n_p, condition["ipi_ms"], condition["phase_us"],
                                 thresholds[c], keep_trace=keep)
            if keep and len(lst := traces.setdefault((s, c, n_p), [])) < ARTIFACT_TRIALS:
                lst.append(m["trace"][:S.samples(ARTIFACT_MS)])
            rows.append(dict(request=request[t.train_id], stim=s, pulses=n_p, rec=c, response=m["adjusted"],
                             recording=label))
    return rows, thresholds


def _record(requests, ids, backend, call_dir: Path, tag: str, codes: bool, thresholds, seed: int, condition,
            all_channels, traces):
    """One pass over `requests` [(stim, pulses)], whose indices in the caller's list are `ids`.
    Returns (rows, thresholds, problems)."""
    plan = P.plan_trials(requests, seed=seed, **condition)
    for seg in plan["segments"]:
        for e in seg["events"]:
            e["request"] = ids[e["request"]]
    if codes:
        P.add_train_codes(plan)
    call_dir.mkdir(parents=True)
    (call_dir / "plan.json").write_text(json.dumps(plan, indent=1))
    planned = [P.plan_duration_s(dict(segments=[s])) for s in plan["segments"]]
    rows, problems, t0 = [], [], time.monotonic()
    for k, seg in enumerate(plan["segments"]):
        seg_dir = call_dir / seg["id"]
        seg_dir.mkdir()
        lengths = sorted({int(e["n_pulses"]) for e in seg["electrodes"].values()})
        print(f"  [{k + 1}/{len(plan['segments'])}] {call_dir.name} {seg['id']}: "
              f"{sum(e['kind'] == 'stim' for e in seg['events'])} trains of {', '.join(map(str, lengths))} pulses "
              f"on e{', e'.join(map(str, sorted(seg['electrodes'])))}", flush=True)
        raw = backend.run_segment(seg, seg_dir, f"{tag}_{call_dir.name}_{seg['id']}")
        close = seg_dir / "close_status.json"          # hardware only
        if close.exists() and not json.loads(close.read_text())["disable_ok"]:
            problems.append(f"final 'disable all stimulation' unanswered after {call_dir.name}/{seg['id']}")
        status("checking the recording…", live=True)
        sent_csv = seg_dir / "sent.csv"                # the simulated backend writes none
        sent = pd.read_csv(sent_csv) if sent_csv.exists() and sent_csv.stat().st_size > 1 else None
        trials = match_triggers(seg, raw, sent, min_fraction=0)
        trials.to_csv(seg_dir / "trials.csv", index=False)
        got, thresholds = _read(seg, raw, trials, f"{call_dir.name}/{seg['id']}", thresholds, condition, all_channels,
                                traces)
        pd.DataFrame(got, columns=COLUMNS).to_csv(seg_dir / "responses.csv", index=False)  # on disk before the next
        rows += got
        clear_status()
        n_got, n = len({r["request"] for r in got}), sum(e["kind"] == "stim" for e in seg["events"])
        done = time.monotonic() - t0
        left = sum(planned[k + 1:]) * done / sum(planned[:k + 1])     # the plan, scaled by the pace so far
        print(f"        {'✓' if n_got == n else '!'} {n_got}/{n} trains usable"
              + (f"; about {left / 60:.0f} min left in this pass (until ~{datetime.now() + timedelta(seconds=left):%H:%M})"
                 if k + 1 < len(planned) else ""), flush=True)
    return rows, thresholds, problems


def measure(requests, backend, out_dir: Path, tag: str, codes: bool = False, thresholds=None, retries: int = 2,
            seed: int = 0, condition=CONDITION, all_channels: bool = False, traces: dict | None = None):
    """Deliver every (stim, pulses) request once; returns (responses, thresholds, problems). `responses`
    has one row per usable request and recording electrode: request (index into `requests`), stim,
    pulses, rec, response (evoked spikes, minus the trial's own pre-train baseline), recording.
    Recording electrodes: the rest of the stimulated organoid, or every other electrode if
    `all_channels` (the screen's cross-organoid controls). `traces`, if given, is filled with the start
    of the counting window per (stim, rec, pulses) on the stimulated organoid, for artifact_score.
    `problems`: recordings whose final disable went unanswered."""
    rows, problems, pending = [], [], list(range(len(requests)))
    for attempt in range(retries + 1):
        if not pending:
            break
        got, thresholds, prob = _record([requests[i] for i in pending], pending, backend,
                                        out_dir / ("pass" if attempt == 0 else f"retry{attempt}"), tag, codes,
                                        thresholds, seed + attempt, condition, all_channels, traces)
        rows += got
        problems += prob
        done = {r["request"] for r in got}
        pending = [i for i in pending if i not in done]
    if pending:
        print(f"  {len(pending)} of {len(requests)} trains still unusable after {retries} retries")
    for x in problems:
        print(f"  WARNING: {x}")
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(out_dir / "responses.csv", index=False)
    return df, thresholds, problems


def reread(out_dir: Path, all_channels: bool = False, traces: dict | None = None):
    """measure's responses again, from the recordings it saved (e.g. after a change to the readout).
    Returns (responses, thresholds)."""
    rows, thresholds = [], None
    for call_dir in [out_dir / "pass"] + sorted(out_dir.glob("retry*")):
        plan = json.loads((call_dir / "plan.json").read_text())
        condition = {k: plan[k] for k in CONDITION}
        for seg in plan["segments"]:
            seg_dir = call_dir / seg["id"]
            raw = recording(seg_dir)
            if raw is None or not (seg_dir / "trials.csv").exists():
                continue
            got, thresholds = _read(seg, raw, pd.read_csv(seg_dir / "trials.csv"), f"{call_dir.name}/{seg['id']}",
                                    thresholds, condition, all_channels, traces)
            rows += got
    return pd.DataFrame(rows, columns=COLUMNS), thresholds
