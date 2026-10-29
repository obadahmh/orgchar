"""chain: two organoids in series, each on its own MEA, solving the damped oscillator. Organoid A's
measured responses set organoid B's train lengths; B's responses, weighted, are the answer.

The network: layer 1 is n1 units on A's pair (t -> pulse count -> A's response, as in net.Net); each of
the n2 units of layer 2 takes a weighted sum of all layer-1 responses as its pulse count on B's pair;
y is a weighted sum of B's responses. Each layer-2 unit's summed |weight| is capped at GAIN_CAP pulses
per unit of A's normalised response, so A's trial noise moves B's lengths little (uncapped, offline on
27 Oct 2026: 49-110, and A's noise changed 9-22 % of B's lengths).

Each organoid has its own backend (its MEA's box): sweep A, sweep B, train, evaluate A, evaluate B.
Each step is gated as in solve; result.json is written after every step.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from . import net as N
from .measure import measure
from .solve import EVAL_POINTS, EVAL_REPS, SWEEP_COUNTS, SWEEP_REPS, _save, sweep

N1, N2, GAIN_CAP, MC = 6, 6, 18.0, 4     # MC: noise samples per training step for the output variance
TRAIN_MIN = 10                            # 6 -> 6, 8 starts, 40000 steps on np14: 9.3 min (offline, 27 Oct 2026)


class Chain(torch.nn.Module):
    def __init__(self, A: N.Curve, B: N.Curve, t_scale: float, n1: int = N1, n2: int = N2):
        super().__init__()
        self.A, self.B, self.t_scale, self.integer = A, B, t_scale, True
        self.scale_a, self.scale_b = (max(abs(m) for m in c.mean) or 1.0 for c in (A, B))
        self.register_buffer("counts_a", torch.tensor(A.counts, dtype=torch.float64))
        self.register_buffer("counts_b", torch.tensor(B.counts, dtype=torch.float64))
        self.h1, self.h2, self.out = torch.nn.Linear(1, n1), torch.nn.Linear(n1, n2), torch.nn.Linear(n2, 1)
        for layer, c, fan_in in ((self.h1, A, 1), (self.h2, B, n1)):      # start inside the measured range
            span = (c.counts[-1] - c.counts[0]) / fan_in ** 0.5
            torch.nn.init.uniform_(layer.weight, -span, span)
            torch.nn.init.uniform_(layer.bias, c.counts[0], c.counts[-1])
        self.double()

    def _lengths(self, p, counts):
        p = p.clamp(counts[0], counts[-1])
        return N.snap(p, counts) if self.integer else p

    def p1(self, t):
        """Pulse counts on A [time, unit]."""
        return self._lengths(self.h1(t / self.t_scale), self.counts_a)

    def w2(self):
        """Layer-2 weights as used: each row scaled down to a summed |weight| of at most GAIN_CAP."""
        w = self.h2.weight
        return w * (GAIN_CAP / w.abs().sum(1, keepdim=True).clamp(min=1e-12)).clamp(max=1.0)

    def p2(self, r1):
        """Pulse counts on B [time, unit] from A's responses r1 (spikes)."""
        return self._lengths(r1 / self.scale_a @ self.w2().T + self.h2.bias, self.counts_b)

    def output(self, r2):
        """y from B's responses r2 (spikes) [time, unit]: the part done on the computer."""
        return self.out(r2 / self.scale_b)

    def forward(self, t, noise=False, repeats=1.0):
        p1 = self.p1(t)
        r1 = N.response(self.A, p1)
        if noise:
            r1 = r1 + N.response(replace(self.A, mean=self.A.sd), p1) * torch.randn_like(r1) / repeats ** 0.5
        p2 = self.p2(r1)
        r2 = N.response(self.B, p2)
        if noise:
            r2 = r2 + N.response(replace(self.B, mean=self.B.sd), p2) * torch.randn_like(r2) / repeats ** 0.5
        return self.output(r2)

    def noise_var(self, t, repeats):
        """Variance of y at times t from both organoids' trial noise (sampled: A's noise moves B's lengths)."""
        return torch.stack([self(t, noise=True, repeats=repeats) for _ in range(MC)]).var(0)


def expected_r2(net: Chain, ode, t, repeats, draws=500, seed=0) -> dict:
    """R² as on the tissue: each pulse count of A and of B is measured once (error drawn once per count,
    sd/sqrt(repeats)) and reused; A's error moves B's lengths. Same keys as net.expected_r2."""
    exact = ode.exact(t.numpy())
    A, B = net.A, net.B
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        clean = N.r2(net(t).numpy(), exact)
        ia = np.searchsorted(A.counts, net.p1(t).numpy().round().astype(int))
        scores = []
        for _ in range(draws):
            r1 = (np.array(A.mean) + rng.normal(0, np.array(A.sd) / repeats ** 0.5))[ia]
            ib = np.searchsorted(B.counts, net.p2(torch.from_numpy(r1)).numpy().round().astype(int))
            r2 = (np.array(B.mean) + rng.normal(0, np.array(B.sd) / repeats ** 0.5))[ib]
            scores.append(N.r2(net.output(torch.from_numpy(r2)).numpy(), exact))
    return dict(curves=clean, noisy_median=float(np.median(scores)), noisy_p10=float(np.percentile(scores, 10)),
                noisy_p90=float(np.percentile(scores, 90)))


def shuffle_null(p2, means_b: dict, w_out, b_out, exact, r2_run, draws=2000, seed=0) -> dict:
    """Control: B's evaluation means shuffled across its pulse counts, through the same output weights.
    Organoid B whose response did not depend on the count would score like this."""
    keys = sorted(means_b)
    rng, scores = np.random.default_rng(seed), []
    for _ in range(draws):
        shuffled = dict(zip(keys, rng.permutation([means_b[k] for k in keys])))
        y = b_out + (np.vectorize(shuffled.get)(p2) * w_out).sum(1)
        scores.append(N.r2(y, exact))
    return dict(median=float(np.median(scores)), p95=float(np.percentile(scores, 95)),
                share_above_run=float((np.array(scores) >= r2_run).mean()))


def _evaluate(fs, pair, lengths, backend, step_dir, part, codes, thresholds, reps, seed):
    """Measure `pair` at each of `lengths` (reps trains each); {pulses: mean response}, and the train count."""
    req = [(pair[0], int(n)) for n in sorted(set(lengths)) for _ in range(reps)]
    ev, _, problems = measure(req, backend, step_dir / part, f"orgchar_{fs}_{step_dir.name}_{part}", codes,
                              thresholds=thresholds, seed=seed)
    d = ev[(ev.stim == pair[0]) & (ev.rec == pair[1])].dropna(subset=["response"])
    return {int(n): float(g.response.mean()) for n, g in d.groupby("pulses")}, len(req), problems


def _drift(curve: N.Curve, means: dict) -> float:
    """Largest change between the sweep and the evaluation, as a share of the curve's range."""
    return max(abs(means[n] - m) for n, m in zip(curve.counts, curve.mean) if n in means) / max(np.ptp(curve.mean), 1e-9)


def run(fs_a, pair_a, fs_b, pair_b, backends: dict, step_dir: Path, codes: bool, ode=N.Oscillator(),
        sweep_reps=SWEEP_REPS, eval_reps=EVAL_REPS, n1=N1, n2=N2, points=EVAL_POINTS, starts=8, steps=N.STEPS,
        seed=0) -> dict:
    """backends: {"A": organoid A's MEA, "B": organoid B's MEA}."""
    res = dict(step="chain", fs_a=fs_a, pair_a=list(pair_a), fs_b=fs_b, pair_b=list(pair_b), sweep_reps=sweep_reps,
               eval_reps=eval_reps, n1=n1, n2=n2, gain_cap=GAIN_CAP, points=points, ode=vars(ode))

    def stop(verdict):
        res["verdict"] = verdict
        _save(step_dir, res)
        return res

    # 1-2. sweep A, then B
    curves, thr = {}, {}
    for k, (name, fs, pair) in enumerate((("A", fs_a, pair_a), ("B", fs_b, pair_b))):
        print(f"\n{k + 1}/5 Measuring organoid {name}'s curve ({fs}, e{pair[0]}→e{pair[1]}): {sweep_reps} trains at "
              f"{', '.join(map(str, SWEEP_COUNTS))} pulses")
        info, keep = sweep([tuple(pair)], backends[name], step_dir / f"sweep_{name.lower()}",
                           f"orgchar_{fs}_{step_dir.name}_sweep_{name.lower()}", codes, sweep_reps, seed)
        res[f"sweep_{name.lower()}"] = info
        _save(step_dir, res)
        if not keep:
            return stop(f"STOPPED after organoid {name}'s sweep: its pair does not respond cleanly to the pulse count. "
                        "No further stimulation.")
        curves[name], thr[name] = keep[0], info["thresholds_uV"]

    # 3. train (no stimulation)
    reps_eff = 1 / (1 / eval_reps + 1 / sweep_reps)     # the run's noise plus the sweep's own (see solve)
    print(f"\n3/5 Training the chain ({n1} -> {n2} units, {starts} starts; no stimulation; about {TRAIN_MIN} min)",
          flush=True)
    net, losses = N.fit(partial(Chain, curves["A"], curves["B"], ode.T, n1, n2), ode, reps_eff, starts, steps)
    torch.save(net.state_dict(), step_dir / "model.pt")
    t = N.held_out_times(ode, points)
    exp, decay = expected_r2(net, ode, t, reps_eff), N.decay_r2(ode, t)
    res.update(expected=exp, decay_r2=decay, train_loss={str(k): v for k, v in losses.items()})
    print(f"  expected R² {exp['noisy_median']:.2f} (10-90 %: {exp['noisy_p10']:.2f} to {exp['noisy_p90']:.2f}); "
          f"{exp['curves']:.2f} on the curves themselves; a plain decay scores {decay:.2f}")
    if exp["noisy_median"] <= decay:
        return stop(f"STOPPED after training: expected R² {exp['noisy_median']:.2f}, not above the plain decay's "
                    f"{decay:.2f}. No further stimulation.")

    # 4. evaluate A at the lengths layer 1 asks for
    with torch.no_grad():
        p1 = net.p1(t).numpy().round().astype(int)
    print(f"\n4/5 Organoid A ({fs_a}): {len(set(p1.ravel()))} pulse counts x {eval_reps} trains")
    means_a, n_a, prob_a = _evaluate(fs_a, pair_a, p1.ravel(), backends["A"], step_dir, "evaluate_a", codes, thr["A"],
                                     eval_reps, seed + 100)
    res.update(t=t.numpy().ravel().tolist(), exact=ode.exact(t.numpy().ravel()).tolist(), p1=p1.tolist(),
               eval_a=means_a, eval_a_trains=n_a, warnings=prob_a)
    _save(step_dir, res)
    if any(int(n) not in means_a for n in p1.ravel()):
        return stop("FAILED: some of organoid A's pulse counts got no usable train; see evaluate_a/")

    # 5. evaluate B at the lengths A's measured responses set
    r1 = np.vectorize(means_a.get)(p1).astype(float)
    with torch.no_grad():
        p2 = net.p2(torch.from_numpy(r1)).numpy().round().astype(int)
    print(f"\n5/5 Organoid B ({fs_b}): {len(set(p2.ravel()))} pulse counts x {eval_reps} trains, set by organoid A")
    means_b, n_b, prob_b = _evaluate(fs_b, pair_b, p2.ravel(), backends["B"], step_dir, "evaluate_b", codes, thr["B"],
                                     eval_reps, seed + 200)
    res.update(r1=r1.tolist(), p2=p2.tolist(), eval_b=means_b, eval_b_trains=n_b, warnings=prob_a + prob_b)
    if any(int(n) not in means_b for n in p2.ravel()):
        return stop("FAILED: some of organoid B's pulse counts got no usable train; see evaluate_b/")

    r2_ = np.vectorize(means_b.get)(p2).astype(float)
    exact = np.array(res["exact"])
    with torch.no_grad():
        y = net.output(torch.from_numpy(r2_)).numpy().ravel()
        model = net(t).numpy().ravel()
        w_out, b_out = (net.out.weight.ravel() / net.scale_b).numpy(), float(net.out.bias)
    score = N.r2(y, exact)
    res.update(r2_responses=r2_.tolist(), y=y.tolist(), y_model=model.tolist(), r2=score,
               drift_a=_drift(curves["A"], means_a), drift_b=_drift(curves["B"], means_b),
               shuffle_b=shuffle_null(p2, means_b, w_out, b_out, exact, score))
    return stop(f"R² {score:.2f} on the two organoids (expected {exp['noisy_median']:.2f}). "
                + (f"Above the plain decay's {decay:.2f}: the chain shows the oscillation." if score > decay else
                   f"Not above the plain decay's {decay:.2f}: the chain does not show the oscillation."))
