"""Characterisation for FuncSolver: which stim -> rec pairs give FuncSolver's activation shapes,
how well they predict single new trials, and the calibration file FuncSolver loads.

FuncSolver (NeuralControlProgram/FuncSolver) uses one organoid pair per role it gets (any subset):
    short = falling curve   L / (1 + exp( k (n - x0))) + b
    band  = peaked curve    A exp(-(n - mu)^2 / (2 sigma^2)) + b
    long  = rising curve    L / (1 + exp(-k (n - x0))) + b
where n = pulses in a hardware train and the output is readout.measure_trial minus the sham mean.
FuncSolver uses ONE stimulation per evaluation, so single-trial scatter is what it gets.

Rules (as in FuncSolver's own design): fit trials alone choose pairs and fit curves. Test trials,
recorded later in the session, only measure the error of those frozen fits.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import readout as RO
from . import signals as S

ROLES = ("short", "band", "long")
ROLE_TEXT = dict(short="falling", band="peaked", long="rising")
# Artifact check: the median across trials of the first ARTIFACT_CHECK_MS of the response window,
# in units of what a median of that many trials varies by anyway (1.25 x across-trial SD /
# sqrt(trials), from the same samples). An artifact tail is identical on every trial: large median,
# small spread. Spikes and bursts are not time-locked: they raise the spread as much as the median
# (fs666 e10, 1 Oct 2026: 30-50 µV bursts were flagged when the quiet noise was the yardstick).
ARTIFACT_CHECK_MS = 5.0
ARTIFACT_LIMIT = 3.0
MIN_EFFECT_P = 0.01         # fit trials: responses must differ across pulse counts (Kruskal-Wallis)
GOOD_R2, FAIR_R2 = 0.5, 0.2  # test-trial single-trial R² for the plain-language grade
# Only these go into calibration.json: a Poor curve predicts new trials worse than R² 0.2 (fs674,
# 12 Oct 2026: a falling curve at R² -1.18 had been handed to FuncSolver)
USABLE_GRADES = ("good", "fair")
# Sessions before 6 Oct 2026 named the partitions "development" and "confirmation".
PARTITION = {"development": "fit", "confirmation": "test"}


# ------------------------------------------------------------------------------ curve models

def model(role, n, p):
    n = np.asarray(n, dtype=float)
    if role == "band":
        A, mu, sig, b = p
        return A * np.exp(-((n - mu) ** 2) / (2 * sig ** 2)) + b
    L, k, x0, b = p
    z = np.clip((k if role == "short" else -k) * (n - x0), -80, 80)
    return L / (1 + np.exp(z)) + b


def fit_role(role, n, y, levels):
    """Least-squares fit to single trials, starting from several sensible guesses."""
    n, y = np.asarray(n, float), np.asarray(y, float)
    lo, hi = min(levels), max(levels)
    m = pd.Series(y).groupby(n).mean()
    if role == "band":
        starts = [[m.max() - m.min(), m.idxmax(), max((hi - lo) / 4, 1.0), m.min()]]
        bounds = ([0, lo - (hi - lo), 0.5, -np.inf], [np.inf, hi + (hi - lo), 4 * (hi - lo), np.inf])
    else:
        starts = [[m.max() - m.min(), s, x0, m.min()] for s in (0.3, 1.0) for x0 in (lo + (hi - lo) * f for f in (0.3, 0.5, 0.7))]
        bounds = ([0, 0.01, lo - (hi - lo), -np.inf], [np.inf, 10.0, hi + (hi - lo), np.inf])
    best = None
    for p0 in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = optimize.curve_fit(lambda x, *q: model(role, x, q), n, y, p0=p0, bounds=bounds, maxfev=20000)
            sse = float(((y - model(role, n, p)) ** 2).sum())
            if best is None or sse < best[1]:
                best = (p, sse)
        except Exception:  # noqa: BLE001 - a start that does not converge is skipped
            continue
    return None if best is None else [float(v) for v in best[0]]


def r2(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def shape_of(levels, means):
    """FuncSolver's shape rule (curve_fitting.classify_response): falling/rising need a clear
    correlation with n, peaked needs its maximum away from both ends."""
    x, y = np.asarray(levels, float), np.asarray(means, float)
    shapes = []
    if np.std(y) < 0.5:
        return shapes
    c = stats.pearsonr(x, y)[0]
    if c < -0.3:
        shapes.append("short")
    if c > 0.3:
        shapes.append("long")
    if 1 < int(np.argmax(y)) < len(y) - 2:
        shapes.append("band")
    return shapes


# ------------------------------------------------------------------------------ measurement

def _plan_index(plan):
    return {s["id"]: s for s in plan["segments"]}


def measure(segments, plan, spec=RO.SPEC, all_channels=False):
    """Every trial on every electrode of its organoid (shams: every electrode), with the readout.
    all_channels: stimulation trials on every electrode, so other-organoid pairs can serve as
    negative controls. Thresholds come from fit and first-reference segments only
    (pre-train baselines)."""
    segs = _plan_index(plan)
    nb = int(round(spec["baseline_duration_ms"] * spec["sample_rate_hz"] / 1000))
    # 1) noise -> frozen thresholds, from pre-train baselines of fit + ref0 segments
    noise = {c: [] for c in range(S.N_CH)}
    for raw, trials in segments:
        seg = segs[trials.segment.iloc[0]]
        part = PARTITION.get(seg["partition"], seg["partition"])
        if part == "test" or (part == "reference" and seg["block"] > 0):
            continue
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
                rows.append(dict(segment=seg["id"], seg_index=si, partition=PARTITION.get(seg["partition"], seg["partition"]),
                                 block=seg["block"],
                                 kind=t.kind, stim=stim, rec=c, level=n, onset=o, train_id=t.train_id,
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


# ------------------------------------------------------------------------------ per-pair analysis

def _cv_r2(role, d, levels):
    """Fit-trials-only score: fit on alternate trials, predict the others, both ways."""
    d = d.sort_values("order").reset_index(drop=True)
    half = d.groupby("level").cumcount() % 2
    preds = np.full(len(d), np.nan)
    for h in (0, 1):
        tr, te = d[half != h], d[half == h]
        p = fit_role(role, tr.level, tr.y, levels)
        if p is None:
            return float("nan")
        preds[te.index] = model(role, te.level, p)
    return r2(d.y, preds)


def analyse_pairs(D, plan, art):
    levels = list(plan["levels"])
    ref_n = plan["reference_pulses"]
    sham = D[D.kind == "sham"]
    # sham correction: one value per electrode from all fit-pass shams (no current flows, so
    # it cannot depend on the pulse count; pooling ~40 trials is far steadier than ~4 per count)
    control = sham[sham.partition == "fit"].groupby("rec").adjusted.mean().to_dict()
    control_sd = sham[sham.partition == "fit"].groupby("rec").adjusted.std().to_dict()
    stim = D[D.kind == "stim"].copy()
    stim["y"] = stim.adjusted - stim.rec.map(control)
    out = []
    for (e, c), g in stim.groupby(["stim", "rec"]):
        dev = g[g.partition == "fit"]
        conf = g[g.partition == "test"]
        ref = g[g.partition == "reference"]
        if dev.level.nunique() < len(levels) or len(dev) < 3 * len(levels):
            continue
        dm = dev.groupby("level").y.agg(["mean", "std", "count"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kw_p = float(stats.kruskal(*[v.y.values for _, v in dev.groupby("level")]).pvalue)
        except ValueError:
            kw_p = 1.0
        kw_p = kw_p if np.isfinite(kw_p) else 1.0
        shapes = shape_of(dm.index.values, dm["mean"].values)
        fits = {}
        for role in ROLES:
            p = fit_role(role, dev.level, dev.y, levels)
            if p is None:
                continue
            f = dict(params=p, r2_means=r2(dm["mean"].values, model(role, dm.index.values, p)),
                     cv_r2=_cv_r2(role, dev, levels))
            if len(conf):   # frozen fit vs new single trials
                pred = model(role, conf.level, p)
                cm = conf.groupby("level").y.mean()
                f.update(conf_r2=r2(conf.y, pred), conf_rmse=float(np.sqrt(((conf.y - pred) ** 2).mean())),
                         conf_means=cm.round(3).tolist(), conf_levels=[int(v) for v in cm.index],
                         conf_means_rmse=float(np.sqrt(((cm.values - model(role, cm.index.values, p)) ** 2).mean())))
            if len(ref):
                expected = float(model(role, ref_n, p))
                rb = ref.groupby("block").y.mean()
                f.update(reference_expected=expected, reference_blocks=rb.round(3).tolist(),
                         drift=float(rb.iloc[-1] - rb.iloc[0]) if len(rb) > 1 else None,
                         max_reference_deviation=float((rb - expected).abs().max()))
            fits[role] = f
        cand = [r for r in shapes if r in fits]
        best = max(cand, key=lambda r: fits[r]["r2_means"]) if cand else None
        art_max = max((art.get((e, c, n), 0.0) for n in levels), default=0.0)
        row = dict(stim=int(e), rec=int(c), shape=best, shapes=cand, kw_p=kw_p,
                   artifact=art_max, artifact_ok=art_max < ARTIFACT_LIMIT,
                   control=float(control.get(c, 0.0)), control_sd=float(control_sd.get(c, 0.0)),
                   levels=[int(v) for v in dm.index], dev_means=dm["mean"].round(3).tolist(),
                   dev_sd=dm["std"].round(3).tolist(), dev_range=float(np.ptp(dm["mean"].values)),
                   trial_sd=float(dm["std"].mean()), fits=fits, n_dev=int(len(dev)), n_conf=int(len(conf)),
                   ref_raw=ref.groupby("block").y.mean().round(3).tolist() if len(ref) else [])
        out.append(row)
    return out, control


def grade(p, role=None):
    """Plain-language verdict for one pair as a FuncSolver activation of `role` (default: its best
    shape), judged on test trials only."""
    role = role or p.get("shape")
    if not role:
        return "none", "no clear falling, peaked or rising curve"
    if not p["artifact_ok"]:
        return "poor", "stimulation artifact reaches the response window"
    if p["kw_p"] >= MIN_EFFECT_P:
        return "poor", "responses do not differ reliably across pulse counts"
    f = p["fits"][role]
    c = f.get("conf_r2")
    if c is None or not np.isfinite(c):
        return "unknown", "no test trials"
    drift = f.get("drift")
    drift_bad = drift is not None and p["dev_range"] > 0 and abs(drift) > 0.5 * p["dev_range"]
    if c >= GOOD_R2 and not drift_bad:
        return "good", "predicts new single trials well"
    if c >= FAIR_R2:
        return "fair", "follows the curve on new trials, but single trials scatter" + (", and drifts" if drift_bad else "")
    return "poor", "does not predict new single trials" + (", and drifts" if drift_bad else "")


def select_roles(pairs):
    """Development data only: per role, the eligible pair with the best cross-validated
    single-trial fit; each role on its own stimulating electrode (as FuncSolver requires).
    Scarce roles are filled first."""
    ok = [p for p in pairs if p.get("shape") and p["artifact_ok"] and p["kw_p"] < MIN_EFFECT_P]
    by_role = {r: sorted([p for p in ok if r in p["shapes"] and r in p["fits"]],
                         key=lambda p: -np.nan_to_num(p["fits"][r]["cv_r2"], nan=-9)) for r in ROLES}
    chosen, used = {}, set()
    for r in sorted(ROLES, key=lambda r: len(by_role[r])):
        for p in by_role[r]:
            if p["stim"] not in used and np.nan_to_num(p["fits"][r]["cv_r2"], nan=-9) > 0:
                chosen[r] = p
                used.add(p["stim"])
                break
    return chosen


# ------------------------------------------------------------------------------ calibration file

def digest(value):
    """Identical to NeuralControlProgram/experiment_control/core.digest (FuncSolver checks it)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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


def calibration(plan, fsname, session_id, thr, chosen, pairs, code_version, simulated, step_dir,
                reference_deviation_events=None):
    """FuncSolver's calibration.json (schema 1, nearest_validated_abs_v1), loadable by
    FuncSolver/calibration.apply_calibration. The live gate (assessment.json) is not written here."""
    counts = list(plan["levels"])
    spec = dict(RO.SPEC, threshold_uV={str(c): round(float(v), 3) for c, v in thr.items()}, response_scale=1.0,
                saturation_uV=6000.0, saturation_source="FinalSpark raw HDF5, clipped at about ±6.3 mV on fs647/fs666 artifacts")
    sel = {}
    for role, p in chosen.items():
        f = p["fits"][role]
        sel[role] = dict(stim_electrode=p["stim"], rec_electrode=p["rec"], type=role, params=f["params"],
                         r2=f["r2_means"], development_cv_r2=f["cv_r2"],
                         control_mean_by_pulses={str(n): p["control"] for n in counts},
                         reference_mean=float(model(role, plan["reference_pulses"], f["params"])),
                         confirmation_r2=f.get("conf_r2"), confirmation_rmse_events=f.get("conf_rmse"),
                         grade=grade(p, role)[0])
    if reference_deviation_events is None:   # default: half the smallest selected curve's range
        ranges = [p["dev_range"] for p in chosen.values()]
        reference_deviation_events = round(0.5 * min(ranges), 2) if ranges else 1.0
    cfg = dict(mea_id=fsname, session_id=session_id, purpose="orgchar_characterisation",
               condition=dict(amplitude_uA=plan["amplitude_uA"], phase_width_us=plan["phase_us"],
                              polarity=plan["polarity"], ipi_ms=plan["ipi_ms"], isi_s=plan["organoid_isi_s"][0]),
               measurement=spec, pulse_counts=counts, pulse_scale=1.0, inference_repetitions=1,
               upload_wait_s=10.0, reference_pulses=plan["reference_pulses"], reference_repetitions=3,
               reference_every_s=300.0, reference_every_samples=10,
               acceptance=dict(max_reference_deviation_events=reference_deviation_events,
                               justification="orgchar default: half the range of the smallest selected curve"),
               stimulation_electrodes=sorted({p["stim"] for p in pairs}))
    cfg = _clean(cfg)
    out = dict(schema_version=1, encoding="nearest_validated_abs_v1", config=cfg, config_sha256=digest(cfg),
               selected=_clean(sel), measured_pulse_counts=counts, simulated=bool(simulated),
               source=dict(tool="orgchar", step=str(step_dir), code_version=code_version),
               missing_roles=[r for r in ROLES if r not in chosen])
    return _clean(out)


def characterise(segments, plan):
    D, thr, art = measure(segments, plan)
    pairs, control = analyse_pairs(D, plan, art)
    for p in pairs:
        p["grade"], p["grade_reason"] = grade(p)
    chosen = select_roles(pairs)
    role_grade = {r: grade(p, r) for r, p in chosen.items()}
    usable = {r: p for r, p in chosen.items() if role_grade[r][0] in USABLE_GRADES}
    verdict = ("ALL_ROLES" if len(chosen) == 3 else "SOME_ROLES" if chosen else "NO_ROLES")
    return dict(step="curve", readout="after_train", verdict=verdict, levels=plan["levels"],
                roles={r: dict(stim=p["stim"], rec=p["rec"], grade=role_grade[r][0], reason=role_grade[r][1],
                               **{k: v for k, v in p["fits"][r].items() if k != "conf_means"}) for r, p in chosen.items()},
                pairs=[_clean({k: v for k, v in p.items()}) for p in pairs], thresholds_uV=_clean(thr),
                n_trials=int(D[D.kind == "stim"].train_id.nunique()), _D=D, _chosen=chosen, _usable=usable)


# ------------------------------------------------------------------------------ screen: find candidates

SCOUT_MAX_P = 0.01       # one-sided Mann-Whitney between the two train lengths that define the shape
SCOUT_MIN_EVENTS = 1.5   # and at least this many events apart (a curve FuncSolver can use)
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
    and which pairs look like FuncSolver's falling, peaked and rising functions."""
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
    curve_stims = list(dict.fromkeys(v["stim"] for v in best.values()))
    # gross-leak alarm: far more other-organoid pairs pass than p < 0.01 alone would let through
    false_pos = sum(bool(p["shapes"]) for p in cross)
    ceiling = round(len(cross) * len(ROLES) * SCOUT_MAX_P, 1)
    verdict = "CANDIDATES_FOUND" if best else "NO_CANDIDATE"
    if cross and false_pos > max(3, 3 * ceiling):
        verdict = "CONTROLS_FAILED"
    return dict(step="screen", mode="trains", verdict=verdict, levels=levels, best=best, curve_stims=curve_stims,
                chance=chance, controls=dict(cross_organoid_pairs=len(cross), cross_organoid_passing=false_pos,
                                             passing_into=sorted({p["rec"] for p in cross if p["shapes"]}),
                                             nominal_ceiling=ceiling, sham_trials=int(sham.train_id.nunique())),
                pairs=[_clean(p) for p in same], thresholds_uV=_clean(thr),
                n_trials=int(stim.train_id.nunique()), _D=D, _cross=cross)
