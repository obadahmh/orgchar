"""The hybrid network: one hidden layer whose activations are organoid responses, trained to solve the
damped oscillator m y'' + c y' + k y = 0, y(0) = y0, y'(0) = v0 on [0, T].

A hidden unit turns t into a pulse count (linear map, clipped to the measured range, rounded to a
whole number: the tissue only takes whole trains). Its output is the evoked response of one
stimulating -> recording pair to that train, divided by the curve's maximum. The output is a weighted
sum, computed on the computer. Units take the available curves in turn, so one curve or several give
the same architecture; a falling curve read with its input reversed is a rising one, and the linear
map can learn that reversal.

The activation is a measured table (mean evoked spikes per pulse count, linear in between), not a
fitted formula: on fs675 (15 Oct 2026) the curve changed shape within hours, from falling to peaked.

Training (weak form, Kharazmi et al. 2019): the residual against overlapping smooth bumps, integrated
by parts so only values of y are needed (with whole pulse counts the network is a step function of t,
and tissue has no derivative). Plus y(0), a penalty on the output weights, and the output's noise
variance from the measured single-trial spread (the integrals alone average noise away). Two phases:
fractional pulse counts with exact gradients, then whole counts at a low learning rate, keeping the
state with the lowest rounded loss (the straight-through gradient alone drove training uphill).
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import torch
from scipy import optimize

IC_WEIGHT, WEIGHT_DECAY, NOISE_WEIGHT = 10.0, 1e-4, 1.0   # noise weight 0.1 collapsed on fs667, 10 cost accuracy
STEPS, FINETUNE_STEPS, FINETUNE_LR = 6000, 1500, 1e-3
HALF_WIDTH, SPACING, POINTS = 0.5, 0.1, 101               # test bumps and training grid


@dataclass(frozen=True)
class Oscillator:
    m: float = 1.0
    c: float = 2.0
    k: float = 10.0
    y0: float = 1.0
    v0: float = 0.0
    T: float = 10.0          # defaults reproduce TUM Fig. 4f (first trough -0.35 at t ~ 1.05)

    def exact(self, t):
        t = np.asarray(t, dtype=float)
        disc = complex(self.c ** 2 - 4 * self.m * self.k)
        r1, r2_ = (-self.c + disc ** 0.5) / (2 * self.m), (-self.c - disc ** 0.5) / (2 * self.m)
        if abs(r1 - r2_) < 1e-12:
            return np.exp(r1.real * t) * (self.y0 + (self.v0 - r1.real * self.y0) * t)
        a = (self.v0 - r2_ * self.y0) / (r1 - r2_)
        return (a * np.exp(r1 * t) + (self.y0 - a) * np.exp(r2_ * t)).real


def r2(pred, true) -> float:
    pred, true = np.asarray(pred, float).ravel(), np.asarray(true, float).ravel()
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())


def decay_r2(ode: Oscillator, t) -> float:
    """R² of the best plain decay a exp(-k t) + b at times t. A network that drops to 0 and misses the
    oscillation scores about this much (0.80 at the 25 default times), so an R² shows the oscillation
    only above it."""
    t = np.asarray(t, float).ravel()
    y = ode.exact(t)
    p, _ = optimize.curve_fit(lambda x, a, k, b: a * np.exp(-k * x) + b, t, y, p0=(ode.y0, 1.0, 0.0), maxfev=20000)
    return r2(p[0] * np.exp(-p[1] * t) + p[2], y)


@dataclass(frozen=True)
class Curve:
    """One stimulating -> recording pair, measured: mean evoked spikes and single-trial spread per
    pulse count (counts increasing)."""
    stim: int
    rec: int
    counts: tuple[int, ...]
    mean: tuple[float, ...]
    sd: tuple[float, ...]


def response(curve: Curve, pulses: torch.Tensor) -> torch.Tensor:
    """The curve at `pulses`: the measured mean at measured counts, linear in between."""
    c = torch.tensor(curve.counts, dtype=torch.float64)
    m = torch.tensor(curve.mean, dtype=torch.float64)
    x = pulses.clamp(c[0], c[-1])
    i = torch.searchsorted(c, x.detach(), right=True).clamp(1, len(c) - 1)
    return m[i - 1] + (m[i] - m[i - 1]) * (x - c[i - 1]) / (c[i] - c[i - 1])


class Net(torch.nn.Module):
    def __init__(self, curves, t_scale: float, width: int = 12):
        super().__init__()
        self.curves = [curves[j % len(curves)] for j in range(width)]     # one per unit
        self.lo = float(max(c.counts[0] for c in curves))
        self.hi = float(min(c.counts[-1] for c in curves))
        grid = torch.arange(self.lo, self.hi + 1, dtype=torch.float64)     # every whole pulse count
        self.register_buffer("scale", torch.stack([response(c, grid).abs().max().clamp(min=1e-9) for c in self.curves]))
        self.register_buffer("sd", torch.tensor(np.array([np.interp(grid.numpy(), c.counts, c.sd) for c in self.curves])))
        self.t_scale, self.integer = t_scale, True
        self.hidden, self.out = torch.nn.Linear(1, width), torch.nn.Linear(width, 1)
        span = self.hi - self.lo                     # start every unit inside the measured range
        torch.nn.init.uniform_(self.hidden.weight, -span, span)
        torch.nn.init.uniform_(self.hidden.bias, self.lo, self.hi)
        self.double()

    def pulses(self, t: torch.Tensor) -> torch.Tensor:
        """Pulse counts the units ask for at times t [time, unit]; whole numbers unless training's first
        phase (straight-through rounding)."""
        p = self.hidden(t / self.t_scale).clamp(self.lo, self.hi)
        return p + (p.round() - p).detach() if self.integer else p

    def sd_at(self, p: torch.Tensor) -> torch.Tensor:
        """Single-trial spread of each unit at pulse counts p: exact at whole counts, linear in between
        (so its gradient can steer units towards quieter counts)."""
        x = p - self.lo
        i = x.detach().floor().clamp(0, self.sd.shape[1] - 2).long()
        lo, hi = self.sd.gather(1, i.T).T, self.sd.gather(1, (i + 1).T).T
        return lo + (hi - lo) * (x - i)

    def output(self, responses: torch.Tensor) -> torch.Tensor:
        """y from the units' responses in evoked spikes [time, unit]: the part done on the computer."""
        return self.out(responses / self.scale)

    def forward(self, t: torch.Tensor, noise: bool = False, repeats: float = 1) -> torch.Tensor:
        p = self.pulses(t)
        h = torch.stack([response(c, p[:, j]) for j, c in enumerate(self.curves)], 1)
        if noise:
            h = h + self.sd_at(p) * torch.randn_like(h) / repeats ** 0.5
        return self.output(h)

    def noise_var(self, t: torch.Tensor, repeats: float) -> torch.Tensor:
        """Variance of y at times t from single-trial noise, each response the mean of `repeats` trials."""
        sd = self.sd_at(self.pulses(t)) / self.scale
        return (self.out.weight ** 2 * sd ** 2).sum(1, keepdim=True) / repeats


def grid(ode: Oscillator, n: int = POINTS) -> torch.Tensor:
    return torch.linspace(0, ode.T, n, dtype=torch.float64).unsqueeze(1)


def held_out_times(ode: Oscillator, n: int) -> torch.Tensor:
    """Midpoints between the n grid times."""
    t = grid(ode, n)
    return (t[:-1] + t[1:]) / 2


def bump_kernels(ode: Oscillator, t: torch.Tensor, half_width: float = HALF_WIDTH, spacing: float = SPACING):
    """Bumps phi = (1 - s^2)^4 (phi, phi', phi'' vanish at their edges, so the trapezoid rule stays
    accurate), centred every `spacing` from 0 to T - half_width: quadrature-weighted kernels
    w (m phi'' - c phi' + k phi) and phi(0), phi'(0), each divided by the bump's scale k int phi."""
    h = half_width
    s = (t - torch.arange(0, ode.T - h + 1e-9, spacing, dtype=t.dtype)) / h      # [time, bump]
    inside = (s.abs() < 1).to(t.dtype)
    u = 1 - s ** 2
    phi, d1 = u ** 4 * inside, -8 * s * u ** 3 / h * inside
    d2 = (-8 * u ** 3 + 48 * s ** 2 * u ** 2) / h ** 2 * inside
    w = torch.full_like(t, float(t[1] - t[0]))
    w[0] = w[-1] = w[0] / 2
    norm = ode.k * (w * phi).sum(0)
    return w * (ode.m * d2 - ode.c * d1 + ode.k * phi) / norm, phi[0] / norm, d1[0] / norm


def weak_loss(y: torch.Tensor, ode: Oscillator, kernel, phi0, dphi0) -> torch.Tensor:
    """Weak-form residual (integrated by parts; v0 enters through the boundary term) plus y(0) = y0."""
    y0 = y[0, 0]
    r = (y * kernel).sum(0) + ode.m * y0 * dphi0 - ode.c * y0 * phi0 - ode.m * ode.v0 * phi0
    return r.pow(2).mean() + IC_WEIGHT * (y0 - ode.y0) ** 2


def _fit(job):
    curves, ode, width, repeats, steps, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net = Net(curves, ode.T, width)
    t = grid(ode)
    tf = bump_kernels(ode, t)

    def loss_fn():
        return (weak_loss(net(t), ode, *tf) + NOISE_WEIGHT * net.noise_var(t, repeats).mean()
                + WEIGHT_DECAY * net.out.weight.pow(2).sum())

    net.integer = False                                   # phase 1: fractional pulses, exact gradients
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
        sched.step()
    net.integer = True                                    # phase 2: whole pulses, keep the best rounded state
    opt = torch.optim.Adam(net.parameters(), FINETUNE_LR)
    best = (float("inf"), None)
    for _ in range(FINETUNE_STEPS + 1):
        opt.zero_grad()
        loss = loss_fn()
        if loss.item() < best[0]:
            best = (loss.item(), {k: v.clone() for k, v in net.state_dict().items()})
        loss.backward()
        opt.step()
    return seed, best[0] if best[0] == best[0] else float("inf"), best[1]


def train(curves, ode: Oscillator, width: int = 12, repeats: float = 6, starts: int = 8, steps: int = STEPS):
    """Train from `starts` initialisations in parallel; keep the one with the lowest loss (the exact
    solution is never used). Returns the network and {start: loss}."""
    jobs = [(list(curves), ode, width, repeats, steps, seed) for seed in range(starts)]
    with Pool(min(starts, torch.multiprocessing.cpu_count())) as pool:
        results = pool.map(_fit, jobs)
    seed, _, state = min(results, key=lambda r: r[1])
    net = Net(list(curves), ode.T, width)
    net.load_state_dict(state)
    return net, {s: v for s, v, _ in results}


def expected_r2(net: Net, ode: Oscillator, t: torch.Tensor, repeats: float, draws: int = 200, seed: int = 0) -> dict:
    """R² against the exact solution: on the measured curves, and as on the tissue, where each pulse
    count of each curve is measured once (the mean of `repeats` trials) and that one value is reused by
    every unit and time point asking for it. So the error is drawn once per (curve, pulse count), not per
    request. Median and 10-90 % range over `draws`."""
    exact = ode.exact(t.numpy())
    with torch.no_grad():
        clean = r2(net(t).numpy(), exact)
        cells = [list(zip(net.curves, row)) for row in net.pulses(t).numpy().astype(int).tolist()]   # [time][unit]
    h0 = np.array([[np.interp(n, c.counts, c.mean) for c, n in row] for row in cells])
    sd = {k: np.interp(k[1], k[0].counts, k[0].sd) / np.sqrt(repeats) for row in cells for k in row}
    rng = np.random.default_rng(seed)
    noisy = []
    for _ in range(draws):
        e = {k: rng.normal(0, s) for k, s in sd.items()}
        h = h0 + np.array([[e[k] for k in row] for row in cells])
        with torch.no_grad():
            noisy.append(r2(net.output(torch.from_numpy(h)).numpy(), exact))
    return dict(curves=clean, noisy_median=float(np.median(noisy)), noisy_p10=float(np.percentile(noisy, 10)),
                noisy_p90=float(np.percentile(noisy, 90)))
