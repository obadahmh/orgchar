"""The screen's analysis: which stim -> rec pairs respond to the train length, and how (response
types: short = falling, band = peaked, long = rising with the number of pulses). solve then measures
the best pairs' curves in full.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from . import readout as RO
from . import signals as S

ROLES = ("short", "band", "long")
# Artifact check: the median across trials of the first ARTIFACT_CHECK_MS of the response window,
# in units of what a median of that many trials varies by anyway (1.25 x across-trial SD /
# sqrt(trials), from the same samples). An artifact tail is identical on every trial: large median,
# small spread. Spikes and bursts are not time-locked: they raise the spread as much as the median
# (fs666 e10, 1 Oct 2026: 30-50 µV bursts were flagged when the quiet noise was the yardstick).
ARTIFACT_CHECK_MS = 5.0
ARTIFACT_LIMIT = 3.0


# ------------------------------------------------------------------------------ measurement

def measure(segments, plan, spec=RO.SPEC, all_channels=False):
    """Every trial on every electrode of its organoid (shams: every electrode), with the readout.
    all_channels: stimulation trials on every electrode, so other-organoid pairs can serve as
    negative controls. Thresholds come from the pre-train baselines."""
    segs = {s["id"]: s for s in plan["segments"]}
    nb = int(round(spec["baseline_duration_ms"] * spec["sample_rate_hz"] / 1000))
    # 1) noise -> frozen thresholds, from the pre-train baselines
    noise = {c: [] for c in range(S.N_CH)}
    for raw, trials in segments:
        X, _ = S.load_raw(raw)
        on = trials[~trials.exclude].onset.values
        on = on[on > nb]
        for c in range(S.N_CH):
            v = np.concatenate([X[c, o - nb:o - S.samples(20)] for o in on[:60]]) if len(on) else X[c, :S.FS * 10]
            noise[c].append(RO.noise_uV(v, spec))
    thr = {c: RO.threshold_from_noise(np.median(v), spec) for c, v in noise.items() if v}
    # 2) every trial
    rows, traces = [], {}
    for si, (raw, trials) in enumerate(segments):
        seg = segs[trials.segment.iloc[0]]
        X, _ = S.load_raw(raw)
        n, ipi = int(seg["n_pulses"]), float(plan["ipi_ms"])
        need = RO.post_samples_needed(n, ipi, plan["phase_us"], spec)
        for t in trials[~trials.exclude & trials.kind.isin(["stim", "sham"])].itertuples():
            o = int(t.onset)
            if o < nb or o + need > X.shape[1]:
                continue
            stim = int(t.stim_electrode)
            recs = (range(S.N_CH) if t.kind == "sham" else [c for c in range(S.N_CH) if c != stim] if all_channels
                    else [c for c in range(8 * (stim // 8), 8 * (stim // 8) + 8) if c != stim])
            for c in recs:
                m = RO.measure_trial(X[c, o - nb:o], X[c, o:o + need], n, ipi, plan["phase_us"], thr[c], spec,
                                     keep_trace=t.kind == "stim")
                rows.append(dict(segment=seg["id"], seg_index=si, kind=t.kind, stim=stim, rec=c, level=n, onset=o, train_id=t.train_id,
                                 response=m["response_events"], baseline=m["baseline_events"], adjusted=m["adjusted"]))
                if t.kind == "stim" and c // 8 == stim // 8:   # window start, to check for time-locked artifact
                    lst = traces.setdefault((stim, c, n), [])
                    if len(lst) < 24:
                        lst.append(m["trace"][:S.samples(ARTIFACT_CHECK_MS)])
    D = pd.DataFrame(rows)
    seg_order = {s["id"]: i for i, s in enumerate(plan["segments"])}
    D["order"] = D.segment.map(seg_order) * 1e9 + D.onset
    # artifact = what is common to every trial (median); spikes are not time-locked and drop out
    art = {}
    for k, v in traces.items():
        if len(v) >= 3:
            A = np.stack(v)
            spread = np.sqrt((A.std(axis=0) ** 2).mean())
            art[k] = float(np.sqrt((np.median(A, axis=0) ** 2).mean()) / max(1.2533 * spread / np.sqrt(len(v)), 1e-9))
    return D, thr, art


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


# ------------------------------------------------------------------------------ screen: find candidates

SCOUT_MAX_P = 0.01       # one-sided Mann-Whitney between the two train lengths that define the shape
SCOUT_MIN_EVENTS = 1.5   # and at least this many events apart (a curve the network can use)
SHAPE_SYMBOL = dict(short="↘", band="∧", long="↗")


def _compare(a, b):
    """a vs b, single trials. Rank-based, because evoked bursts make responses heavy-tailed (fs666
    e10: 46 events apart with SD 37 failed a t-test; every rank comparison separated them).
    Returns mean difference (events), P(a single a-trial > a single b-trial) and its p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return dict(diff=0.0, win=0.5, p=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = stats.mannwhitneyu(a, b, alternative="greater")
    return dict(diff=float(a.mean() - b.mean()), win=float(u.statistic / (len(a) * len(b))), p=float(u.pvalue))


def shape_scores(g, levels):
    """For one pair's trials (column y, by level): how clearly it falls, rises and peaks.
    falling: shortest vs longest train; rising: the reverse; peaked: best middle length vs the
    higher of the two ends. 'win' = how often a single trial of the first beats one of the second."""
    by = {n: g[g.level == n].y.values for n in levels}
    lo, hi = by[levels[0]], by[levels[-1]]
    out = dict(short=_compare(lo, hi), long=_compare(hi, lo))
    mids = levels[1:-1]
    if mids:
        peak = max(mids, key=lambda n: by[n].mean())
        out["band"] = _compare(by[peak], lo if lo.mean() >= hi.mean() else hi) | dict(peak=int(peak))
    return out


def drop_chance(same, cross):
    """A pair from another organoid cannot respond, so on its recording electrode it shows what chance
    reaches there; bursting electrodes reach a lot (fs666 1 Oct 2026: e19→e14 and e26→e14 passed,
    and beat e13→e14, the only "peaked" candidate, which the curve did not confirm). A candidate is
    dropped (removed from its pair's shapes, in place) if a pair from another organoid into the same
    electrode did at least as well, in any shape: p as small and as many events apart.
    Heuristic: the null is the ~24 pairs into each electrode from the other screened organoids
    (none if only one organoid was screened). Returns the dropped candidates."""
    null = {}
    for p in cross:
        for v in p["scores"].values():
            if v["p"] < SCOUT_MAX_P:
                null.setdefault(p["rec"], []).append((v["p"], v["diff"], p["stim"]))
    dropped = []
    for p in same:
        for role in list(p["shapes"]):
            v = p["scores"][role]
            beaten_by = next((s for q, d, s in null.get(p["rec"], []) if q <= v["p"] and d >= v["diff"]), None)
            if beaten_by is not None:
                p["shapes"].remove(role)
                dropped.append(dict(stim=p["stim"], rec=p["rec"], role=role, beaten_by=beaten_by))
    return dropped


def scout(segments, plan):
    """The after-train screen: for every electrode pair, the response to trains of each length,
    and which pairs fall, peak or rise with it."""
    D, thr, art = measure(segments, plan, all_channels=True)
    levels = list(plan["levels"])
    sham = D[D.kind == "sham"]
    control = sham.groupby("rec").adjusted.mean().to_dict()
    stim = D[D.kind == "stim"].copy()
    stim["y"] = stim.adjusted - stim.rec.map(control)
    pairs = []
    for (e, c), g in stim.groupby(["stim", "rec"]):
        if g.level.nunique() < len(levels) or g.groupby("level").size().min() < 3:
            continue
        sc = shape_scores(g, levels)
        same = e // 8 == c // 8
        a = max((art.get((e, c, n), 0.0) for n in levels), default=0.0)
        means = g.groupby("level").y.mean().reindex(levels)
        row = dict(stim=int(e), rec=int(c), same_organoid=same, levels=[int(n) for n in levels],
                   control=float(control.get(c, 0.0)), means=means.round(3).tolist(),
                   sd=g.groupby("level").y.std().reindex(levels).round(3).tolist(), artifact=a,
                   artifact_ok=a < ARTIFACT_LIMIT, scores=sc)
        row["shapes"] = [k for k, v in sc.items()
                         if v["p"] < SCOUT_MAX_P and v["diff"] >= SCOUT_MIN_EVENTS and row["artifact_ok"]]
        pairs.append(row)
    same = [p for p in pairs if p["same_organoid"]]
    cross = [p for p in pairs if not p["same_organoid"]]
    chance = drop_chance(same, cross)
    # the best pair per shape: the one whose single trials separate most often, then by size;
    # one electrode may serve two shapes (on different recording electrodes)
    best = {}
    for role in ROLES:
        cands = sorted([p for p in same if role in p["shapes"]],
                       key=lambda p: (-p["scores"][role]["win"], -p["scores"][role]["diff"]))
        if cands:
            sc = cands[0]["scores"][role]
            best[role] = dict(stim=cands[0]["stim"], rec=cands[0]["rec"], win=sc["win"], diff=sc["diff"],
                              p=sc["p"], n_candidates=len(cands), others=[[p["stim"], p["rec"]] for p in cands[1:6]])
    # gross-leak alarm: far more other-organoid pairs pass than p < 0.01 alone would let through
    false_pos = sum(bool(p["shapes"]) for p in cross)
    ceiling = round(len(cross) * len(ROLES) * SCOUT_MAX_P, 1)
    verdict = "CANDIDATES_FOUND" if best else "NO_CANDIDATE"
    if cross and false_pos > max(3, 3 * ceiling):
        verdict = "CONTROLS_FAILED"
    return dict(step="screen", mode="trains", verdict=verdict, levels=levels, best=best,
                chance=chance, controls=dict(cross_organoid_pairs=len(cross), cross_organoid_passing=false_pos,
                                             passing_into=sorted({p["rec"] for p in cross if p["shapes"]}),
                                             nominal_ceiling=ceiling, sham_trials=int(sham.train_id.nunique())),
                pairs=[_clean(p) for p in same], thresholds_uV=_clean(thr),
                n_trials=int(stim.train_id.nunique()), _D=D, _cross=cross)
