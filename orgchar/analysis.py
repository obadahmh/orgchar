"""Analyses for the three weekly steps. Every function takes raw HDF5 + a trial table and returns a
plain dict (JSON-serialisable) plus figures; no hardware access here.

Trial table columns (one row per trigger sent; see protocol.py):
    segment, onset (sample of the recorded trigger edge in that segment's raw file), key,
    kind ('stim' | 'sham'), stim_electrode (-1 for sham), amplitude_uA, phase_us, polarity,
    n_pulses, ipi_ms, level (swept value; NaN outside curves), train_id, pulse_index
A multi-pulse train is either one row with n_pulses > 1 (hardware train) or one row per pulse
sharing a train_id (software-timed train, used for pulse-count curves).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import signals as S

# Minimum evidence a channel must show in the census to count as containing firing neurons.
ACTIVE_MIN_RATE_HZ = 0.1
ACTIVE_MIN_SPIKES = 20
ACTIVE_MIN_SNR = 5.5


# ============================================================================== census

def census(raw_path, k: float = 5.0) -> dict:
    """Spontaneous activity, no stimulation. Answers: which channels/organoids contain firing neurons?"""
    X, T = S.load_raw(raw_path)
    edges = S.trigger_edges(T)
    quiet = np.ones(X.shape[1], bool)
    for s, _ in edges:  # a census has no stimulation, but ignore anything near a stray trigger
        quiet[max(0, s - S.samples(50)):s + S.FS] = False
    quiet[:S.samples(100)] = quiet[-S.samples(100):] = False
    p = S.spikes_pipeline(X, None, k=k, quiet_mask=quiet)
    Y, valid, spk, noise = p["Y"], p["valid"] & quiet[None, :], p["spikes"], p["noise"]
    dur = valid[0].sum() / S.FS
    noise_rate = S.expected_noise_rate(Y, valid, noise, k)
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
                common_mode_events_removed=p["n_common_mode"], channels=chans, blocks=blocks,
                live_blocks=[b["block"] for b in blocks if b["verdict"] == "LIVE"],
                _spikes=spk, _dur=dur)


# ============================================================================== helpers

def _pulse_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Expand hardware trains into one row per pulse with its sample onset and the gap to the next
    pulse of the same train (which bounds that pulse's analysis window)."""
    trials = trials.copy()
    if "train_id" not in trials or trials["train_id"].isna().any():
        trials["train_id"] = trials.get("train_id", pd.Series(index=trials.index, dtype=object))
        trials["train_id"] = trials["train_id"].fillna(pd.Series([f"row{i}" for i in trials.index], index=trials.index))
    rows = []
    for _, t in trials.iterrows():
        n = int(t.get("n_pulses", 1) or 1)
        ipi = S.samples(float(t.get("ipi_ms", 0) or 0))
        for i in range(n):
            r = t.to_dict()
            r["pulse_onset"] = int(t["onset"]) + i * ipi
            r["pulse_index"] = int(t.get("pulse_index", 0) or 0) + i
            rows.append(r)
    P = pd.DataFrame(rows).sort_values("pulse_onset").reset_index(drop=True)
    # exclude=True: pulse was delivered (so it is blanked and kept out of the spontaneous baseline)
    # but is not counted as a trial (unmatched trigger, or next to a gap in the recording)
    P["exclude"] = P["exclude"].fillna(False).astype(bool) if "exclude" in P else False
    nxt = P.groupby("train_id")["pulse_onset"].shift(-1)
    P["gap"] = nxt - P["pulse_onset"]
    return P


def _artifact_groups(P: pd.DataFrame, window: int) -> list[S.ArtifactGroup]:
    groups = []
    stim = P[P.kind == "stim"]
    cols = ["stim_electrode", "amplitude_uA", "phase_us", "polarity"]
    for key, g in stim.groupby(cols, dropna=False):
        # the first pulse of a train charges the interface differently from later pulses
        for first, gg in g.groupby(g.pulse_index == 0):
            length = int(min(window, np.nanmin(gg.gap.fillna(window).values)))
            groups.append(S.ArtifactGroup(key=tuple(key) + (bool(first),), stim_electrode=int(key[0]),
                                          onsets=gg.pulse_onset.values.astype(np.int64), length=length))
    return groups


def _quiet_mask(n: int, onsets: np.ndarray, before_ms=50.0, after_ms=500.0) -> np.ndarray:
    q = np.ones(n, bool)
    for s in onsets:
        q[max(0, s - S.samples(before_ms)):s + S.samples(after_ms)] = False
    q[:S.samples(100)] = q[-S.samples(100):] = False
    return q


class _NullSampler:
    """Draws spontaneous-activity windows from quiet periods of the same recording and channel.
    Used as the null for 'did stimulation add spikes?'. Resampling real quiet windows keeps the
    burstiness of spontaneous firing (a Poisson null would not)."""

    def __init__(self, spikes_c: np.ndarray, quiet: np.ndarray, rng: np.random.Generator):
        self.cum = np.concatenate([[0], np.cumsum(np.bincount(spikes_c, minlength=len(quiet)))])
        self.starts = np.flatnonzero(quiet)
        self.rng = rng

    def sums(self, durations: np.ndarray, n_draws: int) -> np.ndarray:
        d = durations.astype(np.int64)
        st = self.rng.choice(self.starts, size=(n_draws, len(d)))
        end = np.minimum(st + d[None, :], len(self.cum) - 1)
        return (self.cum[end] - self.cum[st]).sum(axis=1)


def _responses(spk_c: np.ndarray, valid_c: np.ndarray, onsets: np.ndarray, ends: np.ndarray):
    """Spikes and valid (post-blank) duration in [onset, end) for each pulse."""
    k = np.searchsorted(spk_c, ends) - np.searchsorted(spk_c, onsets)
    first_valid = np.array([o + (np.argmax(valid_c[o:e]) if valid_c[o:e].any() else e - o)
                            for o, e in zip(onsets, ends)])
    return k.astype(int), (ends - first_valid).clip(min=0), first_valid


def _pvalue(obs: int, null: np.ndarray) -> float:
    return float((1 + np.count_nonzero(null >= obs)) / (1 + len(null)))


# ============================================================================== screen

TRACE_PRE_MS = 2.0
EDGE_MS = 1.0


def _peak_fraction(x) -> float | None:
    """Largest fraction of latencies inside any 1-ms window."""
    if len(x) < 5:
        return None
    s = np.sort(np.asarray(x))
    return float((np.searchsorted(s, s + 1.0, "right") - np.arange(len(s))).max() / len(s))


def _robust_spread(x) -> float | None:
    """Interquartile range / 1.349 (= SD for a normal distribution), robust to stray spikes."""
    if len(x) < 5:
        return None
    q1, q3 = np.percentile(x, [25, 75])
    return float((q3 - q1) / 1.349)


def _trace_channels(e: int) -> list[int]:
    """Channels whose raw traces are kept for the report: the stimulated organoid's other electrodes,
    plus the 'mirror' electrodes (31 - e, ±1) in another organoid that receive wiring cross-talk."""
    same = [c for c in range(S.block_of(e) * 8, S.block_of(e) * 8 + 8) if c != e]
    mirror = [c for c in (30 - e, 31 - e, 32 - e) if 0 <= c < S.N_CH and S.block_of(c) != S.block_of(e)]
    return same + mirror


def _collect_traces(traces: dict, P: pd.DataFrame, X, p: dict, W: int, k: float):
    """Per (stim, rec): every pulse's snippet at three stages (raw, artifact removed, filtered),
    the detected spikes and the threshold, so the report can show exactly what was counted."""
    pre = S.samples(TRACE_PRE_MS)
    Xc, Y, valid, spk, noise = p["X_clean"], p["Y"], p["valid"], p["spikes"], p["noise"]
    stim = P[P.kind == "stim"]
    for e, g in stim.groupby("stim_electrode"):
        on = g.pulse_onset.values.astype(np.int64)
        on = on[(on - pre >= 0) & (on + W < X.shape[1])]
        if len(on) == 0:
            continue
        idx = on[:, None] + np.arange(-pre, W)[None, :]
        for c in _trace_channels(int(e)):
            sp = spk[c]
            hits = [sp[(sp >= o) & (sp < o + W)] for o in on]
            traces[(int(e), c)] = dict(raw=X[c][idx], clean=Xc[c][idx], filt=Y[c][idx], valid=valid[c][idx],
                                       spikes=[np.c_[S.ms(h - o), Y[c][h]] for o, h in zip(on, hits)],
                                       thr=float(-k * noise[c]), pre_ms=TRACE_PRE_MS)


def screen(segments: list[tuple[str, pd.DataFrame]], window_ms: float = 30.0, k: float = 5.0,
           n_draws: int = 20000, alpha: float = 0.05, min_effect: float = 0.1, min_spikes: int = 5,
           seed: int = 0) -> dict:
    """Which (stim electrode -> recording electrode) pairs respond?

    For each pulse: spikes in [end of artifact, window_ms) on every channel except the stimulated one.
    Null: same-length windows drawn from quiet periods of the same channel (resampled 20k times).
    Built-in negative controls, analysed identically:
      * sham triggers (no current) -> any channel;
      * pairs across organoids (stim in block A, record in block B): no synaptic path exists, so
        'responses' there measure artifact leakage through the pipeline.
    """
    rng = np.random.default_rng(seed)
    W = S.samples(window_ms)
    acc, blanks, wave_spont, traces = {}, [], {}, {}
    for raw_path, trials in segments:
        X, T = S.load_raw(raw_path)
        P = _pulse_table(trials)
        groups = _artifact_groups(P, W)
        quiet = _quiet_mask(X.shape[1], P.pulse_onset.values)
        p = S.spikes_pipeline(X, groups, k=k, quiet_mask=quiet)
        Y, valid, spk = p["Y"], p["valid"], p["spikes"]
        seg_blanks = []
        for g in groups:
            for c in range(S.N_CH):
                seg_blanks.append(dict(stim=g.stim_electrode, rec=c, key=str(g.key), blank_ms=float(p["blank_ms"][g.key][c]),
                                       template=bool(p["template_used"][g.key])))
        blanks += seg_blanks
        # sham pulses get the typical blank of real pulses so the comparison is like for like
        sham_blank = S.samples(float(np.median([b["blank_ms"] for b in seg_blanks if b["rec"] != b["stim"]] or [3.0])))
        samplers = [_NullSampler(spk[c], quiet, rng) for c in range(S.N_CH)]
        for c in range(S.N_CH):
            wave_spont.setdefault(c, []).append(S.waveforms(Y, c, spk[c][quiet[spk[c]]], max_n=300))
        _collect_traces(traces, P, X, p, W, k)
        P["end"] = P.pulse_onset + np.minimum(W, P.gap.fillna(W).astype(int))
        for (kind, e), g in P[~P.exclude].groupby(["kind", "stim_electrode"]):
            on, end = g.pulse_onset.values.astype(np.int64), g.end.values.astype(np.int64)
            for c in range(S.N_CH):
                if kind == "stim" and c == e:
                    continue  # never read the stimulated electrode
                v = valid[c]
                if kind == "sham":
                    v = v.copy()
                    for o in on:
                        v[o:o + sham_blank] = False
                kc, dur, fv = _responses(spk[c], v, on, end)
                a = acc.setdefault((kind, int(e), c), dict(obs=0, null=np.zeros(n_draws), n=0, blank=[], lat=[], waves=[],
                                                           edge=0, valid_ms=0.0))
                a["obs"] += int(kc.sum())
                a["null"] = a["null"] + samplers[c].sums(dur, n_draws)   # independent draws per segment add up
                a["n"] += len(on)
                a["blank"] += S.ms(fv - on).tolist()
                sp = spk[c]
                ev = [sp[(sp >= f) & (sp < t)] for f, t in zip(fv, end)]
                a["lat"] += [float(x) for o, s_ in zip(on, ev) for x in S.ms(s_ - o)]
                a["edge"] += int(sum(np.count_nonzero(s_ - f < S.samples(EDGE_MS)) for f, s_ in zip(fv, ev)))
                a["valid_ms"] += float(S.ms(dur).sum())
                if kind == "stim" and len(ev):
                    a["waves"].append(S.waveforms(Y, c, np.concatenate(ev), max_n=300))
    rows = []
    for (kind, e, c), a in acc.items():
        rows.append(dict(kind=kind, stim=e, rec=c, stim_block=S.block_of(e) if e >= 0 else -1, rec_block=S.block_of(c),
                         n_pulses=a["n"], spikes=a["obs"], expected=float(a["null"].mean()), null_sd=float(a["null"].std()),
                         mean_blank_ms=float(np.mean(a["blank"])) if a["blank"] else None, p=_pvalue(a["obs"], a["null"]),
                         median_latency_ms=float(np.median(a["lat"])) if a["lat"] else None,
                         latency_spread_ms=_robust_spread(a["lat"]), peak_1ms_fraction=_peak_fraction(a["lat"]),
                         edge_spikes=a["edge"],
                         edge_expected_fraction=a["n"] * EDGE_MS / a["valid_ms"] if a["valid_ms"] else None))
    R = pd.DataFrame(rows)
    R["role"] = np.where(R.kind == "sham", "sham_control",
                         np.where(R.stim_block == R.rec_block, "candidate", "cross_organoid_control"))
    R["delta_per_pulse"] = (R.spikes - R.expected) / R.n_pulses
    R["ratio"] = R.spikes / R.expected.clip(lower=1e-9)
    R["z"] = (R.spikes - R.expected) / R.null_sd.clip(lower=1e-9)
    cand = R.role == "candidate"
    m = int(cand.sum())
    R["p_bonf"] = np.minimum(1.0, R.p * max(m, 1))
    # waveform check: evoked spikes should look like this channel's spontaneous spikes
    corr = []
    for _, r in R.iterrows():
        cc = None
        if r.kind == "stim" and r.spikes - r.expected >= 5:
            ev = np.concatenate(acc[("stim", r.stim, r.rec)]["waves"] or [np.zeros((0, 75))])
            sp = np.concatenate(wave_spont.get(r.rec, [np.zeros((0, 75))]))
            if len(ev) >= 5 and len(sp) >= 10:
                cc = float(np.corrcoef(ev.mean(0), sp.mean(0))[0, 1])
        corr.append(cc)
    R["waveform_corr"] = corr
    # Two signatures of leftover artifact, flagged (not excluded) because a very reliable direct
    # response can look the same; the raw-trace figure is where to decide.
    #  time_locked: most evoked spikes fall in one 1-ms latency window (artifact recurs at a fixed time);
    #  blank_edge:  spikes pile up in the first ms after the blank ends (tail not fully removed).
    excess = (R.kind == "stim") & (R.spikes - R.expected >= min_spikes)
    R["time_locked"] = excess & (R.peak_1ms_fraction * R.spikes >= 5) & (R.peak_1ms_fraction >= 0.4)
    R["blank_edge"] = excess & (R.edge_spikes >= 4) & (R.edge_spikes / R.spikes.clip(lower=1) >= 3 * R.edge_expected_fraction)
    R["responsive"] = (cand & (R.p_bonf < alpha) & (R.delta_per_pulse >= min_effect)
                       & (R.spikes - R.expected >= min_spikes)
                       & (R.waveform_corr.isna() | (R.waveform_corr >= 0.8)))
    # negative-control false-positive rates at the same per-test threshold
    per_test = alpha / max(m, 1)
    ctl = {}
    for role in ["cross_organoid_control", "sham_control"]:
        sub = R[R.role == role]
        ctl[role] = dict(tests=int(len(sub)), significant_uncorrected_p01=int((sub.p < 0.01).sum()),
                         expected_at_p01=round(0.01 * len(sub), 1),
                         significant_bonferroni=int(((sub.p < per_test) & (sub.delta_per_pulse >= min_effect)
                                                             & (sub.spikes - sub.expected >= min_spikes)).sum()))
    leak = ctl["cross_organoid_control"]["significant_bonferroni"] > 0 or ctl["sham_control"]["significant_bonferroni"] > 0
    B = pd.DataFrame(blanks)
    same = B[(B.stim // 8 == B.rec // 8) & (B.stim != B.rec) & B.template]
    return dict(step="screen", window_ms=window_ms, threshold_k=k, alpha=alpha, min_effect_spikes_per_pulse=min_effect,
                candidate_tests=m, controls=ctl,
                verdict=("PIPELINE_LEAK" if leak else ("RESPONSES_FOUND" if R.responsive.any() else "NO_RESPONSE")),
                artifact_blank_ms=dict(same_organoid_median=float(same.blank_ms.median()) if len(same) else None,
                                       same_organoid_p90=float(same.blank_ms.quantile(.9)) if len(same) else None),
                pairs=R.sort_values(["responsive", "z"], ascending=[False, False]).to_dict("records"),
                _latencies={k_: v["lat"] for k_, v in acc.items()}, _blanks=B, _traces=traces)


# ============================================================================== curve

def _sigmoid(x, base, top, x50, slope):
    return base + (top - base) / (1.0 + np.exp(-(x - x50) / max(slope, 1e-6)))


def curve(segments: list[tuple[str, pd.DataFrame]], pairs: list[tuple[int, int]], unit: str = "train",
          window_ms: float = 30.0, k: float = 5.0, seed: int = 0) -> dict:
    """Input-output (transfer) curve per (stim, rec) pair.

    level  = the swept input (amplitude µA or pulse count), taken from the trial table;
    output = evoked spikes per trial: spikes in the post-artifact windows of all pulses of the
             trial ('train') or per pulse ('pulse'), minus the spontaneous expectation for the same
             valid duration (so output 0 = no effect). Sham trials give the zero-input point.
    Reliability = Spearman correlation between the curves from odd and even trials.
    """
    rng = np.random.default_rng(seed)
    W = S.samples(window_ms)
    recs = []
    for raw_path, trials in segments:
        X, T = S.load_raw(raw_path)
        P = _pulse_table(trials)
        quiet = _quiet_mask(X.shape[1], P.pulse_onset.values)
        p = S.spikes_pipeline(X, _artifact_groups(P, W), k=k, quiet_mask=quiet)
        spk, valid = p["spikes"], p["valid"]
        P["end"] = P.pulse_onset + np.minimum(W, P.gap.fillna(W).astype(int))
        P["trial"] = P.train_id.astype(str) + "@" + str(raw_path)
        for e, c in pairs:
            sel = P[~P.exclude & ((P.kind == "sham") | (P.stim_electrode == e))]
            if sel.empty:
                continue
            rate = len(spk[c][quiet[spk[c]]]) / max(quiet.sum() / S.FS, 1e-9)
            v = valid[c].copy()
            for o in sel[sel.kind == "sham"].pulse_onset.values:
                v[o:o + S.samples(3.0)] = False
            kc, dur, _ = _responses(spk[c], v, sel.pulse_onset.values.astype(np.int64), sel.end.values.astype(np.int64))
            sel = sel.assign(spikes=kc, expected=rate * dur / S.FS)
            for trial, g in sel.groupby("trial"):
                lvl = 0.0 if (g.kind == "sham").all() else float(g.level.iloc[0])
                recs.append(dict(stim=e, rec=c, trial=trial, level=lvl, sham=bool((g.kind == "sham").all()),
                                 raw=float(g.spikes.sum()), expected=float(g.expected.sum()), n_pulses=len(g),
                                 time=int(g.pulse_onset.iloc[0]), segment=str(raw_path)))
    D = pd.DataFrame(recs)
    if D.empty:
        return dict(step="curve", pairs=[], verdict="NO_DATA")
    D["evoked"] = D.raw - D.expected
    if unit == "pulse":
        D["evoked"] = D.evoked / D.n_pulses
    out = []
    for (e, c), g in D.groupby(["stim", "rec"]):
        g = g.sort_values(["segment", "time"]).reset_index(drop=True)
        g["half"] = g.index % 2
        tab = g.groupby("level").evoked.agg(["mean", "sem", "std", "count"]).reset_index()
        halves = g.groupby(["half", "level"]).evoked.mean().unstack(0)
        rel = float(stats.spearmanr(halves[0], halves[1]).correlation) if halves.shape[0] >= 3 and halves.notna().all().all() else None
        fit = None
        x, y = tab["level"].values, tab["mean"].values
        if len(x) >= 4 and np.ptp(y) > 0:
            try:
                p0 = [y.min(), y.max(), np.median(x), max(np.ptp(x) / 4, 1e-3)]
                par, _ = optimize.curve_fit(_sigmoid, g.level.values, g.evoked.values, p0=p0, maxfev=20000)
                pred = _sigmoid(x, *par)
                ss = ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
                fit = dict(model="sigmoid", base=par[0], top=par[1], x50=par[2], slope=par[3], r2_of_means=float(1 - ss))
            except Exception:
                fit = None
        sham = g[g.sham].evoked
        top_level = x.max()
        top = g[g.level == top_level].evoked
        effect_p = float(stats.mannwhitneyu(top, sham, alternative="greater").pvalue) if len(sham) >= 5 and len(top) >= 5 else None
        dynamic = float(y.max() - (y[x == 0][0] if (x == 0).any() else y.min()))
        mean_sd = float(tab["std"].mean())
        usable = bool(effect_p is not None and effect_p < 0.01 and rel is not None and rel >= 0.6 and dynamic > mean_sd)
        out.append(dict(stim=int(e), rec=int(c), unit=unit, levels=tab.to_dict("records"), split_half_spearman=rel,
                        fit=fit, top_vs_sham_p=effect_p, dynamic_range=dynamic, mean_trial_sd=mean_sd,
                        snr=float(dynamic / mean_sd) if mean_sd > 0 else None, usable=usable))
    return dict(step="curve", unit=unit, window_ms=window_ms, pairs=out,
                verdict="USABLE_CURVES" if any(o["usable"] for o in out) else "NO_USABLE_CURVE", _trials=D)
