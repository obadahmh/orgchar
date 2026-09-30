"""Run with pytest from the repository root (see README for a venv that leaves `ncp` untouched)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orgchar import analysis as A
from orgchar import protocol as P
from orgchar import signals as S
from orgchar import simulate
from orgchar.backends import SimBackend, match_triggers

FS647_SEP8 = Path.home() / "NeuralControlProgram/ResponseSweep/runs/fs647_waveform_20260908T203700Z"


# ------------------------------------------------------------------------------ protocol

def test_limits_reject_unsafe_settings():
    with pytest.raises(ValueError):
        P.check_setting(5.0, 100)          # amplitude
    with pytest.raises(ValueError):
        P.check_setting(3.0, 200)          # 0.6 nC > 0.4 nC
    P.check_setting(2.0, 100)


def test_screen_plan_uses_sham_key_and_one_segment_per_block():
    plan = P.plan_screen(blocks=[1, 2], reps=5, sham_reps=5)
    assert [s["id"] for s in plan["segments"]] == ["block1_00", "block2_00"]
    for s in plan["segments"]:
        keys = {e["key"] for e in s["events"] if e["kind"] == "stim"}
        assert P.SHAM_KEY not in keys and len(keys) == 8
        assert all(e["key"] == P.SHAM_KEY for e in s["events"] if e["kind"] == "sham")
        t = [e["t_s"] for e in s["events"]]
        assert min(np.diff(t)) >= P.LIMITS["min_isi_s"] - 1e-9


def test_pulse_curve_trains_are_contiguous_and_isolated():
    plan = P.plan_curve_pulses([21], levels=(1, 3), ipi_ms=30, reps=4, sham_reps=2)
    ev = pd.DataFrame([e for s in plan["segments"] for e in s["events"]])
    for _, g in ev[ev.kind == "stim"].groupby("train_id"):
        assert len(g) == g.level.iloc[0]
        assert np.allclose(np.diff(g.t_s), 0.03)


# ------------------------------------------------------------------------------ signals

def test_common_mode_events_across_blocks_are_removed():
    spk = [np.array([], dtype=np.int64) for _ in range(32)]
    for c in (3, 12, 20):                  # same instant on three organoids = interference
        spk[c] = np.array([1000, 5000 + 97 * c])
    out, n = S.reject_common_mode(spk)
    assert n == 3 and all(1000 not in out[c] for c in (3, 12, 20))
    assert all(len(out[c]) == 1 for c in (3, 12, 20))


def _sim_screen(tmp_path, org, reps=24):
    plan = P.plan_screen(blocks=[2], reps=reps, sham_reps=reps, seed=3)
    seg = plan["segments"][0]
    raw = SimBackend(org).run_segment(seg, tmp_path, "t")
    return A.screen([(str(raw), match_triggers(seg, raw))], n_draws=5000)


def test_screen_finds_planted_pair_and_controls_stay_clean(tmp_path):
    org = simulate.Organoid(evoked={(21, 23): (1.0, 1.0, 0.2)}, seed=5)
    r = _sim_screen(tmp_path, org)
    R = pd.DataFrame(r["pairs"])
    hits = R[R.responsive]
    assert list(zip(hits.stim, hits.rec)) == [(21, 23)]
    assert r["controls"]["cross_organoid_control"]["significant_bonferroni"] == 0
    assert r["controls"]["sham_control"]["significant_bonferroni"] == 0
    assert r["verdict"] == "RESPONSES_FOUND"


def test_artifacts_alone_never_look_like_responses(tmp_path):
    # strong artifacts, lively spontaneous firing, but no evoked responses at all
    org = simulate.Organoid(evoked={}, seed=7)
    r = _sim_screen(tmp_path, org)
    assert r["verdict"] == "NO_RESPONSE"


def test_census_separates_live_and_silent_blocks(tmp_path):
    org = simulate.Organoid(spont_hz={20: 3.0, 21: 1.0}, evoked={}, seed=2)
    raw = SimBackend(org).run_segment(P.plan_census(60)["segments"][0], tmp_path, "c")
    r = A.census(raw)
    assert r["live_blocks"] == [2]
    assert {c["channel"] for c in r["channels"] if c["active"]} == {20, 21}


# ------------------------------------------------------------------------------ real data regression

@pytest.mark.skipif(not FS647_SEP8.exists(), reason="fs647 8 Sep 2026 recording not on this machine")
def test_fs647_sep8_known_response_and_clean_controls():
    from orgchar import legacy
    r = A.screen(legacy.waveform_sweep_segments(FS647_SEP8, "ipi_20ms"), window_ms=20.0, n_draws=5000)
    R = pd.DataFrame(r["pairs"])
    resp = set(zip(R[R.responsive].stim, R[R.responsive].rec))
    assert {(21, 23), (23, 21), (15, 13)} <= resp
    assert r["controls"]["cross_organoid_control"]["significant_bonferroni"] == 0
    assert all(S.block_of(s) == S.block_of(c) for s, c in resp)


def test_recording_gaps_do_not_break_trigger_matching(tmp_path):
    """The platform can drop whole 1280-sample rows from the raw file (seen on fs666, 30 Sep 2026)."""
    import h5py
    plan = P.plan_screen(blocks=[2], reps=12, sham_reps=12, seed=4)
    seg = plan["segments"][0]
    raw = SimBackend(simulate.Organoid(evoked={(21, 23): (1.0, 1.0, 0.2)}, seed=9)).run_segment(seg, tmp_path, "g")
    with h5py.File(raw, "r+") as f:                      # drop three rows, one of them holding a trigger
        trig_rows = np.flatnonzero((f["triggers"][...] != 0).any(axis=1))
        drop = {int(trig_rows[40]), int(trig_rows[40]) + 300, int(trig_rows[40]) + 600}
        keep = [r for r in range(f["triggers"].shape[0]) if r not in drop]
        for name in list(f):
            data = f[name][keep]
            del f[name]
            f.create_dataset(name, data=data, chunks=(1, 1280)).attrs["sampling_rate"] = np.uint32(S.FS)
    t = match_triggers(seg, raw)
    assert len(t) == len(seg["events"]) - 1                # the trigger inside the lost row is gone
    assert (t.note == "before recording gap").sum() >= 2   # pulses next to gaps are excluded
    assert (t.note == "unmatched").sum() == 0
    kept = t[~t.exclude]
    planned = {e["train_id"]: e["stim_electrode"] for e in seg["events"]}
    assert all(planned[r.train_id] == r.stim_electrode for r in kept.itertuples())
    r = A.screen([(str(raw), t)], n_draws=3000)
    assert (21, 23) in set(zip(*[pd.DataFrame(r["pairs"]).query("responsive")[c] for c in ("stim", "rec")]))
