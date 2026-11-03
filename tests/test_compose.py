"""compose, on synthetic data."""
import json
import re
from functools import partial

import numpy as np
import torch

from orgchar import compose as CO
from orgchar import net as N
from orgchar import simulate
from orgchar.backends import SimBackend

COUNTS = tuple(range(2, 21, 2))
FALLING = N.Curve(13, 10, COUNTS, tuple(float(40 / (1 + np.exp(0.6 * (n - 9)))) for n in COUNTS), (3.0,) * 10)
RISING = N.Curve(21, 22, COUNTS, tuple(float(35 / (1 + np.exp(-0.5 * (n - 10)))) for n in COUNTS), (2.0,) * 10)


def test_functions_keep_the_signal_range_and_only_saturate_commutes_with_invert():
    u = np.linspace(0, 1, 1001)
    for f in CO.FUNCTIONS.values():
        y = f(u)
        assert y.min() >= -1e-12 and y.max() <= 1 + 1e-12 and np.abs(np.diff(y) / np.diff(u)).max() <= 2.1
    sq, sat, inv = (CO.FUNCTIONS[k] for k in ("square", "saturate", "invert"))
    assert np.allclose(sat(inv(u)), inv(sat(u)))
    assert CO.gap(sq(inv(u)), inv(sq(u))) > 0.3


def test_each_organoid_runs_at_the_lengths_the_outputs_before_it_set():
    torch.manual_seed(0)
    nets = {b: N.Net([FALLING if o == "A" else RISING], 1.0, 4) for b, o in CO.HOME.items()}
    tables = {"A": dict(zip(COUNTS, FALLING.mean)), "B": dict(zip(COUNTS, RISING.mean))}
    calls = []
    u = CO.inputs(11)
    y = CO.cascade(nets, u, lambda org, lengths: calls.append((org, lengths)) or tables[org])
    assert [o for o, _ in calls] == list(CO.STAGES)
    inv_after_sq = CO.pulses(nets["invert"], y[("square",)])          # B's lengths come from A's output
    assert set(inv_after_sq.ravel()) <= set(calls[1][1])
    assert np.allclose(y[("square", "invert")], CO.output(nets["invert"], inv_after_sq, tables["B"]))
    assert np.allclose(y[("invert", "square")],
                       CO.output(nets["square"], CO.pulses(nets["square"], y[("invert",)]), tables["A"]))


def test_a_block_learns_its_function_on_a_measured_curve():
    net, _ = N.fit(partial(N.Net, [FALLING], 1.0, 6), N.map_problem(CO.FUNCTIONS["square"]), repeats=6, starts=2,
                   steps=1500)
    u = CO.inputs()
    assert N.r2(CO.output(net, CO.pulses(net, u), dict(zip(COUNTS, FALLING.mean))), u ** 2) > 0.9


def test_compose_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(CO, "GATE_R2", -np.inf)                        # a short training; check the plumbing
    backends, log = {}, []
    for org in "AB":                                                   # one simulated MEA each
        b = SimBackend(simulate.demo_organoid())
        b.run_segment = lambda seg, seg_dir, tag, run=b.run_segment, org=org: log.append((org, tag)) or run(seg, seg_dir, tag)
        backends[org] = b
    res = CO.run("fsA", (13, 10), "fsB", (21, 22), backends, tmp_path, False, sweep_reps=4, eval_reps=2, points=11,
                 starts=2, steps=300)
    assert all(f"_fs{org}_" in tag for org, tag in log)
    parts = [(org, re.search(r"(sweep_[ab]|stage\d_[ab])", tag).group(1)) for org, tag in log]
    assert list(dict.fromkeys(parts)) == [("B", "sweep_b"), ("A", "sweep_a"), ("A", "stage1_a"), ("B", "stage2_b"),
                                          ("A", "stage3_a")]
    assert [s["organoid"] for s in res["stages"]] == ["A", "B", "A"]
    assert all(set(s["lengths"]) <= set(COUNTS) and s["trains"] == 2 * len(s["lengths"]) for s in res["stages"])
    assert set(res["y"]) == {CO.name(p) for p in CO.PATHS} and all(len(v) == 10 for v in res["y"].values())
    assert all(np.isfinite(v) for v in res["r2"].values())
    assert res["gaps"]["saturate>invert"]["targets"] < 1e-9 < res["gaps"]["square>invert"]["targets"]
    assert json.loads((tmp_path / "result.json").read_text())["verdict"] == res["verdict"]


def test_verdict_needs_fits_and_the_predicted_order_behaviour():
    res = dict(r2={CO.name(p): 0.95 for p in CO.PATHS},
               gaps={"square>invert": dict(measured=0.35, targets=0.37, expected=dict(p95=0.40)),
                     "saturate>invert": dict(measured=0.08, targets=0.0, expected=dict(p95=0.11))})
    assert CO.verdict(res).startswith("PASS")
    res["gaps"]["saturate>invert"]["measured"] = 0.2                  # the commuting pair's orders came apart
    assert CO.verdict(res).startswith("FAIL")
    res["gaps"]["saturate>invert"]["measured"], res["r2"]["invert>square"] = 0.08, 0.85
    assert CO.verdict(res).startswith("FAIL")
