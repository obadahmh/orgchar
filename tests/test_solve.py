"""net, measure and solve, on synthetic data."""
import json

import h5py
import numpy as np
import pytest
import torch

from orgchar import net as N
from orgchar import report as R
from orgchar import signals as S
from orgchar import simulate
from orgchar import solve as SV
from orgchar.backends import SimBackend, match_triggers
from orgchar.measure import ARTIFACT_LIMIT, artifact_score, measure, reread

FALLING = N.Curve(13, 10, tuple(range(2, 21, 2)), tuple(float(40 / (1 + np.exp(0.6 * (n - 9)))) for n in range(2, 21, 2)),
                  (3.0,) * 10)

# ------------------------------------------------------------------------------ net


def test_activation_is_the_measured_table():
    c = N.Curve(0, 1, (2, 4, 6), (10.0, 6.0, 8.0), (1.0, 1.0, 1.0))
    got = N.response(c, torch.tensor([2.0, 3.0, 5.0, 6.0, 1.0, 9.0], dtype=torch.float64))
    assert got.tolist() == [10.0, 8.0, 7.0, 8.0, 10.0, 8.0]          # linear in between, clamped outside


def test_weak_loss_is_zero_only_on_the_solution():
    ode = N.Oscillator()
    t = N.grid(ode)
    k = N.bump_kernels(ode, t)
    exact = N.weak_loss(torch.from_numpy(ode.exact(t.numpy())), ode, *k).item()
    assert exact < 1e-3 and N.weak_loss(torch.exp(-t), ode, *k).item() > 30 * exact


def test_noise_variance_matches_sampling():
    torch.manual_seed(0)
    ode = N.Oscillator()
    net = N.Net([FALLING], ode.T, width=6)
    t = N.held_out_times(ode, 11)
    with torch.no_grad():
        draws = torch.stack([net(t, noise=True, repeats=3) for _ in range(4000)])
        assert torch.allclose(draws.var(0), net.noise_var(t, 3), rtol=0.1)


def test_expected_r2_without_noise_is_the_curve_r2():
    ode = N.Oscillator()
    net = N.Net([N.Curve(13, 10, FALLING.counts, FALLING.mean, (0.0,) * 10)], ode.T, width=6)
    e = N.expected_r2(net, ode, N.held_out_times(ode, 26), repeats=4)
    assert e["noisy_p10"] == pytest.approx(e["curves"]) and e["noisy_p90"] == pytest.approx(e["curves"])


def test_training_follows_the_solution_on_one_falling_curve():
    """It follows the drop (0.74); the oscillation needs more (a plain decay scores 0.80 here)."""
    ode = N.Oscillator()
    net, _ = N.train([FALLING], ode, width=12, repeats=6, starts=4)
    t = N.held_out_times(ode, 26)
    assert N.expected_r2(net, ode, t, 6)["curves"] > 0.6 and 0.75 < N.decay_r2(ode, t) < 0.85


# ------------------------------------------------------------------------------ measure

class _Edit(SimBackend):
    """The simulated platform, with `edit(seg, raw)` applied to each recording."""

    def __init__(self, organoid, edit):
        super().__init__(organoid)
        self.edit, self.calls = edit, 0

    def run_segment(self, seg, seg_dir, tag):
        raw = super().run_segment(seg, seg_dir, tag)
        self.calls += 1
        self.edit(self.calls, seg, raw)
        return raw


def test_trains_lost_in_a_dropped_row_are_requested_again(tmp_path):
    def drop_a_trigger_row(call, seg, raw):
        if call > 1:
            return
        with h5py.File(raw, "r+") as f:
            rows = np.flatnonzero((f["triggers"][...] != 0).any(axis=1))
            keep = [r for r in range(f["triggers"].shape[0]) if r != rows[2]]
            for name in list(f):
                data = f[name][keep]
                del f[name]
                f.create_dataset(name, data=data, chunks=(1, 1280)).attrs["sampling_rate"] = np.uint32(S.FS)
    requests = [(13, 4)] * 8
    df, thr, problems = measure(requests, _Edit(simulate.demo_organoid(), drop_a_trigger_row), tmp_path, "t")
    assert sorted(df.request.unique()) == list(range(8)) and (tmp_path / "retry1").is_dir() and not problems
    assert set(df.rec) == set(range(8, 16)) - {13} and len(thr) == S.N_CH
    again, _ = reread(tmp_path)                                       # the same responses from the saved recordings
    assert again.sort_values(["request", "rec"]).reset_index(drop=True).equals(
        df.sort_values(["request", "rec"]).reset_index(drop=True))


def test_every_electrode_and_an_unanswered_disable(tmp_path):
    def unanswered(call, seg, raw):
        (raw.parent / "close_status.json").write_text(json.dumps(dict(stop_ok=True, disable_ok=False, messages=[])))
    df, _, problems = measure([(13, 4)] * 3, _Edit(simulate.demo_organoid(), unanswered), tmp_path, "t",
                              all_channels=True)
    assert set(df.rec) == set(range(32)) - {13} and problems == ["final 'disable all stimulation' unanswered after pass/seg00"]


def test_artifact_in_the_counting_window_is_caught(tmp_path):
    from orgchar import readout as RO

    def tail_and_bursts(call, seg, raw):
        t = match_triggers(seg, raw, min_fraction=0)
        n = int(next(iter(seg["electrodes"].values()))["n_pulses"])
        end = RO.train_end_ms(n, 10.0, 100.0)
        rng = np.random.default_rng(call)
        with h5py.File(raw, "r+") as f:
            x = f["electrode_22"][...].reshape(-1)           # a tail still there 10-13 ms after the train
            y = f["electrode_20"][...].reshape(-1)           # big bursts at random times (fs666 e10, 1 Oct)
            for o in t[t.kind == "stim"].onset:
                a = int(o + (end + 10) * 30)
                x[a:a + 90] += 60 * np.sin(np.arange(90) / 4.0)
                y[a:a + 3000] += rng.normal(0, 40, 3000)
            f["electrode_22"][...] = x.reshape(f["electrode_22"].shape)
            f["electrode_20"][...] = y.reshape(f["electrode_20"].shape)
    traces = {}
    measure([(21, n) for n in (2, 4, 6) for _ in range(6)], _Edit(simulate.Organoid(seed=2), tail_and_bursts),
            tmp_path, "a", traces=traces)
    assert max(artifact_score(traces[(21, 22, n)]) for n in (2, 4, 6)) > ARTIFACT_LIMIT
    assert max(artifact_score(traces[(21, c, n)]) for c in (16, 17, 18, 19, 20, 23) for n in (2, 4, 6)) < ARTIFACT_LIMIT


# ------------------------------------------------------------------------------ solve

def test_solve_stops_after_the_sweep_when_nothing_responds(tmp_path):
    res = SV.run("fstest", [(21, 22)], SimBackend(simulate.Organoid(seed=7)), tmp_path, False, sweep_reps=5)
    assert res["verdict"].startswith("STOPPED after the sweep") and res["curves"][0]["p"] >= SV.GATE_P
    assert not (tmp_path / "evaluate").exists() and not (tmp_path / "model.pt").exists()


def test_solve_stops_after_training_when_it_cannot_beat_a_plain_decay(tmp_path):
    res = SV.run("fstest", [(13, 10)], SimBackend(simulate.demo_organoid()), tmp_path, False, sweep_reps=4,
                 width=4, starts=2, steps=300)
    assert res["verdict"].startswith("STOPPED after training") and res["expected"]["noisy_median"] <= res["decay_r2"]
    assert (tmp_path / "model.pt").exists() and not (tmp_path / "evaluate").exists()


def test_solve_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "decay_r2", lambda *a: -np.inf)            # a short training; check the plumbing
    res = SV.run("fstest", [(13, 10)], SimBackend(simulate.demo_organoid()), tmp_path, True, sweep_reps=4,
                 eval_reps=2, width=4, points=11, starts=2, steps=300)
    assert res["curves"][0]["p"] < SV.GATE_P and res["curves"][0]["artifact"] < ARTIFACT_LIMIT
    needed = {(13, n) for row in res["pulses"] for n in row}
    assert res["eval_trains"] == 2 * len(needed) and len(res["y"]) == len(res["t"]) == 10
    assert np.isfinite(res["r2"]) and res["drift_max"] is not None
    assert json.loads((tmp_path / "result.json").read_text())["verdict"] == res["verdict"]
    R.solve_page(tmp_path / "report.html", dict(res, code_version="test", simulated=True), "today")
    assert "Measured curve" in (tmp_path / "report.html").read_text()
