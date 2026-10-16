"""The screen: every electrode of the live organoids gets trains of 2, 10 and 20 pulses, read out on every
other electrode; which stim -> rec pairs respond to the train length, and how (response types: short =
falling, band = peaked, long = rising with the number of pulses). solve then measures the clearest pair's
curve in full.

Pairs across organoids cannot respond: they are the negative controls (drop_chance, CONTROLS_FAILED).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy import stats

from .measure import ARTIFACT_LIMIT, CONDITION, artifact_score, measure, reread

ROLES = ("short", "band", "long")
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


def run(fsname, stims, counts, reps, backend, step_dir: Path, codes: bool, condition=CONDITION, seed=0) -> dict:
    """Deliver `reps` trains of each length in `counts` on each electrode in `stims`, and analyse them."""
    requests = [(e, n) for e in stims for n in counts for _ in range(reps)]
    traces = {}
    responses, thr, problems = measure(requests, backend, step_dir, f"orgchar_{fsname}_{step_dir.name}", codes,
                                       seed=seed, condition=condition, all_channels=True, traces=traces)
    return scout(responses, traces, list(counts), len(requests), thr, problems)


def reanalyse(step_dir: Path, plan: dict) -> dict:
    """The screen again from its saved recordings (e.g. after a change to the readout or the analysis)."""
    traces = {}
    responses, thr = reread(step_dir, all_channels=True, traces=traces)
    return scout(responses, traces, plan["levels"], len(plan["stims"]) * len(plan["levels"]) * plan["reps"], thr)


def scout(responses, traces, levels, n_requests, thresholds, warnings_=()) -> dict:
    """For every electrode pair, the response to trains of each length, and which pairs look falling,
    peaked or rising."""
    art = {k: artifact_score(v) for k, v in traces.items() if len(v) >= 3}
    d = responses.rename(columns={"pulses": "level", "response": "y"})
    pairs = []
    for (e, c), g in d.groupby(["stim", "rec"]):
        if g.level.nunique() < len(levels) or g.groupby("level").size().min() < 3:
            continue
        sc = shape_scores(g, levels)
        a = max((art.get((e, c, n), 0.0) for n in levels), default=0.0)
        row = dict(stim=int(e), rec=int(c), same_organoid=e // 8 == c // 8, levels=[int(n) for n in levels],
                   means=g.groupby("level").y.mean().reindex(levels).round(3).tolist(),
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
    return dict(step="screen", mode="trains", verdict=verdict, levels=list(levels), best=best, chance=chance,
                controls=dict(cross_organoid_pairs=len(cross), cross_organoid_passing=false_pos,
                              passing_into=sorted({p["rec"] for p in cross if p["shapes"]}), nominal_ceiling=ceiling),
                pairs=same, thresholds_uV=thresholds, n_trials=int(responses.request.nunique()), n_requests=n_requests,
                warnings=list(warnings_), _responses=responses, _cross=cross)
