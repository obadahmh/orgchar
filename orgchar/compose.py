"""compose: two organoids, each with its own network trained on its own, then connected in series.

A block is u -> train lengths -> one organoid's responses -> y: net.Net on one pair, with a signal u in
[0, 1] in place of time, trained by least squares to a function from [0, 1] to [0, 1]. Organoid A hosts
square (u²) and saturate (an S-curve), organoid B hosts invert (1 - u). Every block takes and gives a
signal in [0, 1] (a block's input is clipped to it), so any block can follow any other without
retraining, as blocks with a standard signal range do in a control loop.

In series the blocks should give the composition f_B(f_A(u)). Its error is about e_B + f_B' e_A, so
every function's slope stays at or below about 2 (saturate's peaks at 2.07). The order of linear blocks never matters; for nonlinear ones
it does, except for special pairs. saturate is symmetric about (0.5, 0.5), so it commutes with invert;
square does not: invert then square is (1 - u)², square then invert 1 - u². The run measures both orders
of both pairs. Offline, on the measured curves of 18 Oct-2 Nov 2026: every block and order expected at
R² 0.94-0.98, and the two orders of the non-commuting pair 0.36 apart against 0.08-0.10 for the
commuting one. Replayed on the later responses of the 30 Oct and 2 Nov chain runs, after organoid A had
changed: A's blocks 0.64-0.89 and the series 0.58-0.89, below the bar, but the order gaps still 0.30
against 0.06-0.09.

Steps: sweep B, sweep A, train each block on its own organoid's curve (gate: every block's expected R²
at least GATE_R2), then three stages, each organoid's train lengths set by the outputs measured before:
A (A's blocks on u), B (invert on u and on A's outputs), A (A's blocks on B's outputs). B is swept first
so the waits balance: each measurement starts within about 33 min of its organoid's sweep, about as long
as in the chain runs, so the design relies on steady pairs, not on a shorter wait. Responses are medians
(solve.center_spread). result.json is written after every step.

Pass: every series R² at least GATE_R2, each commuting pair's order gap at most its predicted 95th
percentile, and each other pair's gap above that. A system working as modelled passes about 90 % of the
time (1000 draws on the simulated session's blocks, 3 Nov 2026).
"""
from __future__ import annotations

from functools import partial, reduce
from pathlib import Path

import numpy as np
import torch

from . import net as N
from .solve import EVAL_REPS, SWEEP_COUNTS, SWEEP_REPS, _save, drift, measure_at, sweep

FUNCTIONS = {"square": lambda u: u ** 2, "saturate": lambda u: 0.5 + 0.5 * np.tanh(4 * (u - 0.5)) / np.tanh(2),
             "invert": lambda u: 1 - u}
BLOCKS = {"A": ("square", "saturate"), "B": ("invert",)}
HOME = {b: o for o, names in BLOCKS.items() for b in names}
PAIRS = [(a, b) for a in BLOCKS["A"] for b in BLOCKS["B"]]
PATHS = [(b,) for b in HOME] + [p for a, b in PAIRS for p in ((a, b), (b, a))]
STAGES = ("A", "B", "A")
# 6 units, 4000 steps: blocks at expected R² 0.94-0.99 on the measured curves, ~30 s each on np14 (3 Nov 2026)
WIDTH, STEPS, POINTS, GATE_R2, TRAIN_MIN = 6, 4000, 26, 0.9, 2


class Missing(Exception):
    pass


def name(path) -> str:
    return ">".join(path)


def inputs(points: int = POINTS) -> np.ndarray:
    """The evaluation inputs: midpoints of `points` evenly spaced values in [0, 1], off the training grid."""
    g = np.linspace(0, 1, points)
    return (g[:-1] + g[1:]) / 2


def target(path, u):
    return reduce(lambda x, b: FUNCTIONS[b](x), path, u)


def pulses(net: N.Net, x) -> np.ndarray:
    """Train lengths the block asks for at inputs x [input, unit]."""
    with torch.no_grad():
        return net.pulses(torch.from_numpy(np.clip(x, 0, 1)).reshape(-1, 1)).numpy().round().astype(int)


def output(net: N.Net, lengths, table: dict) -> np.ndarray:
    """The block's y from the organoid's responses `table` {pulses: spikes} at `lengths`."""
    with torch.no_grad():
        return net.output(torch.from_numpy(np.vectorize(table.__getitem__)(lengths).astype(float))).numpy().ravel()


def cascade(nets: dict, u, stage) -> dict:
    """Every path's output at inputs u. stage(organoid, lengths) -> {pulses: response} is called for A, B,
    then A, each time with every length that organoid's next blocks ask for, given the outputs so far."""
    x, done = {p: u for p in PATHS}, {p: 0 for p in PATHS}
    for org in STAGES:
        todo = [p for p in PATHS if done[p] < len(p) and HOME[p[done[p]]] == org]
        asked = {p: pulses(nets[p[done[p]]], x[p]) for p in todo}
        table = stage(org, sorted({int(n) for a in asked.values() for n in a.ravel()}))
        for p in todo:
            x[p] = output(nets[p[done[p]]], asked[p], table)
            done[p] += 1
    assert all(done[p] == len(p) for p in PATHS)
    return x


def gap(y1, y2) -> float:
    """Root-mean-square difference between two outputs (or targets)."""
    return float(np.sqrt(np.mean((np.asarray(y1) - np.asarray(y2)) ** 2, -1)))


def spread(x) -> dict:
    return dict(median=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                p95=float(np.percentile(x, 95)))


def expected(nets: dict, curves: dict, u, sweep_reps, eval_reps, draws=300, seed=0) -> dict:
    """Each path's R², and each pair's gap between its two orders, as on the tissue: the organoid's true
    curve is the sweep's plus the sweep's own error, and every stage measures it afresh (eval_reps trains)."""
    rng = np.random.default_rng(seed)
    ys = {p: [] for p in PATHS}
    for _ in range(draws):
        true = {o: np.array(c.mean) + rng.normal(0, np.array(c.sd) / sweep_reps ** 0.5) for o, c in curves.items()}
        stage = lambda o, _: dict(zip(curves[o].counts, true[o] + rng.normal(0, np.array(curves[o].sd) / eval_reps ** 0.5)))
        for p, y in cascade(nets, u, stage).items():
            ys[p].append(y)
    return dict(r2={name(p): spread([N.r2(y, target(p, u)) for y in ys[p]]) for p in PATHS},
                gap={name(ab): spread([gap(y1, y2) for y1, y2 in zip(ys[ab], ys[ab[::-1]])]) for ab in PAIRS})


def shuffle_null(net: N.Net, u, table: dict, r2_run: float, goal, draws=2000, seed=0) -> dict:
    """Control: the stage's responses shuffled across its pulse counts, through the same block. Tissue whose
    response did not depend on the count would score like this."""
    keys, rng, lengths = sorted(table), np.random.default_rng(seed), pulses(net, u)
    scores = np.array([N.r2(output(net, lengths, dict(zip(keys, rng.permutation([table[k] for k in keys])))), goal)
                       for _ in range(draws)])
    return dict(median=float(np.median(scores)), p95=float(np.percentile(scores, 95)),
                share_above_run=float((scores >= r2_run).mean()))


def commutes(a, b) -> bool:
    u = np.linspace(0, 1, 1001)
    return gap(target((a, b), u), target((b, a), u)) < 1e-9


def verdict(res: dict) -> str:
    r2, gaps = res["r2"], res["gaps"]
    series = [p for p in PATHS if len(p) > 1]
    bar = max(gaps[name(ab)]["expected"]["p95"] for ab in PAIRS if commutes(*ab))
    fits = all(r2[name(p)] >= GATE_R2 for p in series)
    orders = all((gaps[name(ab)]["measured"] <= bar) == commutes(*ab) for ab in PAIRS)
    alone = ", ".join(f"{b} R² {r2[b]:.2f}" for b in HOME)
    chained = ", ".join(f"{' → '.join(p)} {r2[name(p)]:.2f}" for p in series)
    order = "; ".join(f"{a} and {b} {'commute' if commutes(a, b) else 'do not commute'}: the two orders "
                      f"{gaps[name((a, b))]['measured']:.2f} apart (targets {gaps[name((a, b))]['targets']:.2f} apart)"
                      for a, b in PAIRS)
    return (f"{'PASS' if fits and orders else 'FAIL'}. Alone: {alone}. In series: {chained} (bar {GATE_R2}). "
            f"Order, against {bar:.2f} (the commuting pair's predicted 95th percentile): {order}.")


def run(fs_a, pair_a, fs_b, pair_b, backends: dict, step_dir: Path, codes: bool, sweep_reps=SWEEP_REPS,
        eval_reps=EVAL_REPS, width=WIDTH, points=POINTS, starts=8, steps=STEPS, seed=0) -> dict:
    """backends: {"A": organoid A's MEA, "B": organoid B's MEA}."""
    fs, pair = dict(A=fs_a, B=fs_b), dict(A=tuple(pair_a), B=tuple(pair_b))
    res = dict(step="compose", fs_a=fs_a, pair_a=list(pair_a), fs_b=fs_b, pair_b=list(pair_b), sweep_reps=sweep_reps,
               eval_reps=eval_reps, width=width, points=points, blocks=BLOCKS, response="median")

    def stop(verdict_):
        res["verdict"] = verdict_
        _save(step_dir, res)
        return res

    # 1-2. sweep B, then A
    curves, thr = {}, {}
    for k, org in enumerate("BA"):
        print(f"\n{k + 1}/6 Measuring organoid {org}'s curve ({fs[org]}, e{pair[org][0]}→e{pair[org][1]}): {sweep_reps} "
              f"trains at {', '.join(map(str, SWEEP_COUNTS))} pulses")
        info, keep = sweep([pair[org]], backends[org], step_dir / f"sweep_{org.lower()}",
                           f"orgchar_{fs[org]}_{step_dir.name}_sweep_{org.lower()}", codes, sweep_reps, seed, robust=True)
        res[f"sweep_{org.lower()}"] = info
        _save(step_dir, res)
        if not keep:
            return stop(f"STOPPED after organoid {org}'s sweep: its pair does not respond cleanly to the pulse count. "
                        "No further stimulation.")
        curves[org], thr[org] = keep[0], info["thresholds_uV"]

    # 3. train each block on its own organoid's curve only (no stimulation)
    print(f"\n3/6 Training {len(HOME)} blocks ({', '.join(f'{b} on {o}' for b, o in HOME.items())}; {width} units, "
          f"{starts} starts; no stimulation; about {TRAIN_MIN} min)", flush=True)
    reps_eff = 1 / (1 / eval_reps + 1 / sweep_reps)       # the run's noise plus the sweep's own (see solve)
    nets = {b: N.fit(partial(N.Net, [curves[o]], 1.0, width), N.map_problem(FUNCTIONS[b]), reps_eff, starts, steps)[0]
            for b, o in HOME.items()}
    torch.save({b: n.state_dict() for b, n in nets.items()}, step_dir / "model.pt")
    u = inputs(points)
    exp = expected(nets, curves, u, sweep_reps, eval_reps)
    res.update(u=u.tolist(), targets={name(p): target(p, u).tolist() for p in PATHS}, expected=exp)
    for p in PATHS:
        e = exp["r2"][name(p)]
        print(f"  {name(p).replace('>', ' → '):20s} expected R² {e['median']:.2f} (10-90 %: {e['p10']:.2f} to {e['p90']:.2f})")
    weak = [b for b in HOME if exp["r2"][b]["median"] < GATE_R2]
    if weak:
        return stop(f"STOPPED after training: expected R² of {', '.join(weak)} below {GATE_R2}. No further stimulation.")

    # 4-6. organoid A, B, A, each at the lengths the outputs measured so far ask for
    stages = res["stages"] = []

    def stage(org, lengths):
        k = len(stages)
        print(f"\n{k + 4}/6 Organoid {org} ({fs[org]}): {len(lengths)} pulse counts x {eval_reps} trains"
              + (", set by the outputs measured before" if k else ""))
        means, trains, prob = measure_at(fs[org], pair[org], lengths, backends[org], step_dir, f"stage{k + 1}_{org.lower()}",
                                         codes, thr[org], eval_reps, seed + 100 * (k + 1))
        stages.append(dict(organoid=org, lengths=lengths, responses=means, trains=trains, warnings=prob,
                           drift=drift(curves[org], means) if means else None))
        _save(step_dir, res)
        if set(lengths) - set(means):
            raise Missing(f"stage{k + 1}_{org.lower()}")
        print(f"  largest change against its sweep: {stages[-1]['drift']:.0%} of the curve's range")
        return means

    try:
        y = cascade(nets, u, stage)
    except Missing as e:
        return stop(f"FAILED: some pulse counts got no usable train; see {e}/")

    r2 = {name(p): N.r2(y[p], target(p, u)) for p in PATHS}
    first = {"A": stages[0]["responses"], "B": stages[1]["responses"]}       # where each block ran on u
    res.update(y={name(p): v.tolist() for p, v in y.items()}, r2=r2,
               gaps={name(ab): dict(measured=gap(y[ab], y[ab[::-1]]), targets=gap(target(ab, u), target(ab[::-1], u)),
                                    expected=exp["gap"][name(ab)]) for ab in PAIRS},
               shuffle={b: shuffle_null(nets[b], u, first[o], r2[b], target((b,), u)) for b, o in HOME.items()})
    return stop(verdict(res))
