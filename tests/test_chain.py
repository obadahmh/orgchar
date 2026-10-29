"""chain, on synthetic data."""
import json

import numpy as np
import pytest
import torch

from orgchar import chain as CH
from orgchar import net as N
from orgchar import simulate
from orgchar.backends import SimBackend

COUNTS = tuple(range(2, 21, 2))
FALLING = N.Curve(13, 10, COUNTS, tuple(float(40 / (1 + np.exp(0.6 * (n - 9)))) for n in COUNTS), (3.0,) * 10)
RISING = N.Curve(21, 22, COUNTS, tuple(float(35 / (1 + np.exp(-0.5 * (n - 10)))) for n in COUNTS), (2.0,) * 10)


def test_both_layers_ask_for_measured_counts_and_the_gain_is_capped():
    torch.manual_seed(0)
    ode = N.Oscillator()
    net = CH.Chain(FALLING, RISING, ode.T)
    with torch.no_grad():
        net.h2.weight.mul_(100)                                    # far over the cap
        t = N.held_out_times(ode, 26)
        p1 = net.p1(t)
        p2 = net.p2(N.response(FALLING, p1))
        assert set(p1.ravel().tolist()) | set(p2.ravel().tolist()) <= set(map(float, COUNTS))
        assert net.w2().abs().sum(1).max() == pytest.approx(CH.GAIN_CAP)


def test_expected_r2_without_noise_is_the_curve_r2():
    ode = N.Oscillator()
    quiet = lambda c: N.Curve(c.stim, c.rec, c.counts, c.mean, (0.0,) * 10)
    net = CH.Chain(quiet(FALLING), quiet(RISING), ode.T)
    e = CH.expected_r2(net, ode, N.held_out_times(ode, 26), repeats=4, draws=20)
    assert e["noisy_p10"] == pytest.approx(e["curves"]) and e["noisy_p90"] == pytest.approx(e["curves"])


def test_chain_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "decay_r2", lambda *a: -np.inf)            # a short training; check the plumbing
    swaps, tags = [], []
    backend = SimBackend(simulate.demo_organoid())                     # organoids 1 and 2 stand in for two MEAs
    run_segment = backend.run_segment
    backend.run_segment = lambda seg, seg_dir, tag: tags.append(tag) or run_segment(seg, seg_dir, tag)
    res = CH.run("fsA", (13, 10), "fsB", (21, 22), backend, tmp_path, False, swaps.append, sweep_reps=4,
                 eval_reps=2, n1=3, n2=3, points=11, starts=2, steps=300)
    assert swaps == ["fsB", "fsA", "fsB"] and len(set(tags)) == len(tags)
    assert all("_fsA_" in t for t in tags if "_a_" in t) and all("_fsB_" in t for t in tags if "_b_" in t)
    p1, p2 = np.array(res["p1"]), np.array(res["p2"])
    assert p1.shape == p2.shape == (10, 3) and set(p1.ravel()) | set(p2.ravel()) <= set(COUNTS)
    assert res["eval_a_trains"] == 2 * len(set(p1.ravel())) and res["eval_b_trains"] == 2 * len(set(p2.ravel()))
    assert np.isfinite(res["r2"]) and res["drift_a"] is not None and res["drift_b"] is not None
    assert res["shuffle_b"]["median"] <= res["shuffle_b"]["p95"]
    assert json.loads((tmp_path / "result.json").read_text())["verdict"] == res["verdict"]
