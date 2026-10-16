"""solve: measure the pair's curve now, check it, train the network on it, then run the network on the
tissue. One session, in this order, each step gated:

1. Sweep: trains of 2, 4, ..., 20 pulses (8 each) on each pair's stimulating electrode. Gate: the
   response must depend on the pulse count (Kruskal-Wallis p < 0.01) and the stimulation artifact must
   be gone by the counting window (below 3x its noise floor), else the pair is dropped. No pair
   left: stop, before the costly part. Whether the curve is good enough is the next gate's question.
2. Train the network on the measured curves (no stimulation). Gate: its expected R² on the tissue
   must beat a plain decay a exp(-kt) + b (0.80 at the default times), else stop: below that the run
   could not show the oscillation. Expected = with the noise of the evaluation's trials and of the
   sweep's own means, each count's error shared by every time point that uses it, as in step 3.
3. Evaluate: the network asks for a pulse count per unit and time point; each distinct (electrode,
   count) is measured 6 times and reused wherever it is asked for (same R² as one train per request on
   the 15 Oct live trials, 20x fewer pulses). y(t) is computed from those responses.
4. Drift: the evaluation's responses at counts the sweep also measured, against the sweep.

Responses change within hours (fs675, 15 Oct 2026: the 2-pulse response fell from 48 to 27 between the
morning curve and the live run), so the curve is measured in the same session, minutes before use.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from . import net as N
from .measure import ARTIFACT_LIMIT, artifact_score, measure

SWEEP_COUNTS = tuple(range(2, 21, 2))
SWEEP_REPS, EVAL_REPS, WIDTH, EVAL_POINTS = 8, 6, 12, 26
GATE_P = 0.01


def dependence(values_by_count: dict[int, np.ndarray]) -> tuple[float, float]:
    """Does the response depend on the pulse count? Kruskal-Wallis p, and omega squared: the share of
    single-trial variance the count explains, unbiased for the number of trials."""
    groups = [v for v in values_by_count.values() if len(v)]
    if len(groups) < 2:
        return 1.0, float("nan")
    p = float(stats.kruskal(*groups).pvalue) if np.ptp(np.concatenate(groups)) > 0 else 1.0
    allv = np.concatenate(groups)
    ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ms_w = ss_w / max(len(allv) - len(groups), 1)
    return p, float((ss_b - (len(groups) - 1) * ms_w) / (ss_b + ss_w + ms_w))


def curves_from(responses: pd.DataFrame, pairs, counts):
    """Measured curve of each (stim, rec) pair, with its dependence on the pulse count (p, omega²)."""
    out = []
    for s, r in pairs:
        d = responses[(responses.stim == s) & (responses.rec == r)]
        by = {n: d[d.pulses == n].response.dropna().to_numpy() for n in counts}
        c = N.Curve(int(s), int(r), tuple(int(n) for n in counts), tuple(float(by[n].mean()) for n in counts),
                    tuple(float(by[n].std(ddof=1)) for n in counts))
        out.append((c, *dependence(by)))
    return out


def _save(step_dir: Path, result: dict):
    (step_dir / "result.json").write_text(json.dumps(result, indent=1, default=float))


def run(fsname, pairs, backend, step_dir: Path, codes: bool, ode=N.Oscillator(), sweep_reps=SWEEP_REPS,
        eval_reps=EVAL_REPS, width=WIDTH, points=EVAL_POINTS, starts=8, steps=N.STEPS, seed=0) -> dict:
    tag = f"orgchar_{fsname}_{step_dir.name}"
    stims = sorted({s for s, _ in pairs})
    res = dict(step="solve", fsname=fsname, pairs=[list(p) for p in pairs], sweep_counts=list(SWEEP_COUNTS),
               sweep_reps=sweep_reps, eval_reps=eval_reps, width=width, points=points, ode=vars(ode))

    # 1. sweep
    print(f"\n1/3 Measuring the curve: {sweep_reps} trains at {', '.join(map(str, SWEEP_COUNTS))} pulses on "
          f"e{', e'.join(map(str, stims))}")
    sweep_req = [(s, n) for s in stims for n in SWEEP_COUNTS for _ in range(sweep_reps)]
    traces = {}
    sweep, thr, problems = measure(sweep_req, backend, step_dir / "sweep", tag, codes, seed=seed, traces=traces)
    art = {(s, r): max((artifact_score(v) for (s2, r2, _), v in traces.items() if (s2, r2) == (s, r) and len(v) >= 3),
                       default=0.0) for s, r in pairs}
    measured = curves_from(sweep, pairs, SWEEP_COUNTS)
    res.update(sweep_trains=len(sweep_req), sweep_pulses=sum(n for _, n in sweep_req), thresholds_uV=thr, warnings=problems,
               curves=[dict(stim=c.stim, rec=c.rec, counts=c.counts, mean=c.mean, sd=c.sd, p=pv, explained=w,
                            artifact=art[(c.stim, c.rec)]) for c, pv, w in measured])
    keep = [c for c, pv, _ in measured if pv < GATE_P and art[(c.stim, c.rec)] < ARTIFACT_LIMIT]
    for c, pv, w in measured:
        a = art[(c.stim, c.rec)]
        why = ("" if pv < GATE_P and a < ARTIFACT_LIMIT else "  (dropped: " + ("no dependence on pulse count" if pv >= GATE_P
               else f"artifact {a:.1f}x its noise floor reaches the counting window") + ")")
        print(f"  e{c.stim}→e{c.rec}: range {np.ptp(c.mean):.1f} spikes, pulse count explains {w:.0%} of single-trial "
              f"variance (p {pv:.1g}), artifact {a:.1f}x" + why)
    if not keep:
        res["verdict"] = ("STOPPED after the sweep: no pair responds cleanly to the pulse count (see the curves). "
                          "No further stimulation.")
        _save(step_dir, res)
        return res

    # 2. train
    # a response in the run differs from the table by the run's noise (eval_reps trains) plus the table's
    # own error (sweep_reps trains): train for, and predict with, both (sim, 16 Oct 2026: R² 0.40 against an
    # expected 0.73 when the table was taken as exact)
    reps_eff = 1 / (1 / eval_reps + 1 / sweep_reps)
    print(f"\n2/3 Training on the measured curve{'s' * (len(keep) > 1)} ({width} units, {starts} starts; no stimulation)")
    net, losses = N.train(keep, ode, width, reps_eff, starts, steps)
    torch.save(net.state_dict(), step_dir / "model.pt")
    t = N.held_out_times(ode, points)
    exp, decay = N.expected_r2(net, ode, t, reps_eff), N.decay_r2(ode, t)
    res.update(expected=exp, decay_r2=decay, train_loss={str(k): v for k, v in losses.items()},
               kept=[[c.stim, c.rec] for c in keep])
    print(f"  expected R² {exp['noisy_median']:.2f} (10-90 %: {exp['noisy_p10']:.2f} to {exp['noisy_p90']:.2f}) with "
          f"the noise of {eval_reps} trials and the curve's own uncertainty; {exp['curves']:.2f} on the curve itself; "
          f"a plain decay, no oscillation, scores {decay:.2f}")
    if exp["noisy_median"] <= decay:
        res["verdict"] = (f"STOPPED after training: expected R² {exp['noisy_median']:.2f} does not beat a plain decay "
                          f"({decay:.2f}), so the run could not show the oscillation. No further stimulation.")
        _save(step_dir, res)
        return res

    # 3. evaluate: each distinct (electrode, pulses) measured eval_reps times, reused wherever asked for
    with torch.no_grad():
        p = net.pulses(t).numpy().astype(int)
    needed = sorted({(u.stim, int(n)) for j, u in enumerate(net.curves) for n in p[:, j]})
    eval_req = [(s, n) for s, n in needed for _ in range(eval_reps)]
    print(f"\n3/3 Running the network on the tissue: {len(needed)} pulse counts x {eval_reps} trains")
    ev, _, prob = measure(eval_req, backend, step_dir / "evaluate", tag, codes, thresholds=thr, seed=seed + 100)
    res["warnings"] += prob
    means = ev.groupby(["stim", "rec", "pulses"]).response.mean()
    h = np.array([[means.get((u.stim, u.rec, int(p[i, j])), np.nan) for j, u in enumerate(net.curves)]
                  for i in range(len(p))])
    exact = ode.exact(t.numpy().ravel())
    res.update(eval_trains=len(eval_req), eval_pulses=sum(n for _, n in eval_req), t=t.numpy().ravel().tolist(),
               exact=exact.tolist(), pulses=p.tolist(),
               eval_means=[dict(stim=int(s), rec=int(r), pulses=int(n), mean=float(v)) for (s, r, n), v in means.items()
                           if (s, r) in {(c.stim, c.rec) for c in keep}])
    if np.isnan(h).any():
        res["verdict"] = "FAILED: some pulse counts got no usable train; see evaluate/"
        _save(step_dir, res)
        return res
    with torch.no_grad():
        y = net.output(torch.from_numpy(h)).numpy().ravel()
        model = net(t).numpy().ravel()
    score = N.r2(y, exact)

    # 4. drift: the evaluation against the sweep, at the counts both measured
    drift = []
    for c in keep:
        for n, m in zip(c.counts, c.mean):
            if (c.stim, c.rec, n) in means.index:
                drift.append(abs(means[(c.stim, c.rec, n)] - m) / max(np.ptp(c.mean), 1e-9))
    res.update(y=y.tolist(), y_model=model.tolist(), r2=score, drift_max=float(max(drift)) if drift else None,
               verdict=f"R² {score:.2f} on the tissue (expected {exp['noisy_median']:.2f}; a plain decay scores "
                       f"{decay:.2f}, so this {'shows' if score > decay else 'does not show'} the oscillation)"
                       + (f"; largest drift since the sweep {max(drift):.0%} of the curve's range" if drift else ""))
    _save(step_dir, res)
    return res


def when(step_dir: Path) -> str:
    return datetime.strptime(step_dir.name[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).strftime("%d %b %Y %H:%M UTC")
