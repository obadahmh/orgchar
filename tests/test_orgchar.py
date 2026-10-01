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


def test_bunched_train_triggers_are_excluded_not_mismatched(tmp_path):
    """TCP batching delivered a 30-ms train's triggers late and bunched (fs666 curve, 30 Sep 2026).
    Such a train was not delivered as planned: it must be excluded, and must not derail the matching
    of the other trains."""
    plan = P.plan_curve_pulses([21], levels=(1, 5), ipi_ms=30, reps=6, sham_reps=4, seed=2)
    seg = plan["segments"][0]
    trains = pd.DataFrame(seg["events"])
    bad = trains[trains.level == 5].train_id.iloc[1]
    trig, pulses = [], []
    for ev in seg["events"]:
        s = int(ev["t_s"] * S.FS)
        if ev["train_id"] == bad:          # arrives 188 ms late, 3 of 5 triggers within 0.5 ms, 2 merged away
            if ev["pulse_index"] > 2:
                continue
            s = int((trains[trains.train_id == bad].t_s.min() + 0.188) * S.FS) + 6 * ev["pulse_index"]
        trig.append((s, ev["key"]))
        if ev["kind"] == "stim":
            pulses.append((s + 6, 21, 2.0, 100.0))
    raw = tmp_path / "raw.h5"
    simulate.record(simulate.Organoid(seed=3), seg["duration_s"], pulses, trig, raw)
    t = match_triggers(seg, raw)
    late = t[(t.train_id == bad) | (t.note == "unmatched")]
    assert len(late) == 3 and late.exclude.all()          # the 3 recorded bunched edges: blanked, not counted
    good = t[~t.index.isin(late.index)]
    assert not good.exclude.any() and len(good) == len(seg["events"]) - 5
    assert set(good.train_id) == set(trains.train_id) - {bad}


def test_single_late_trigger_is_jitter_not_lost_rows(tmp_path):
    """One trigger recorded ~5 ms late inside a 30-ms train (fs666 curve, 30 Sep 2026) also fits
    "2 lost rows + 3 skipped pulses" (3 x 30 - 2 x 42.67 = 4.7 ms). Taking that explanation shifted
    every later match and threw away most of the recording."""
    plan = P.plan_curve_pulses([21], levels=(1, 10), ipi_ms=30, reps=6, sham_reps=4, seed=2)
    seg = plan["segments"][0]
    trains = pd.DataFrame(seg["events"])
    late = trains[trains.level == 10].train_id.iloc[0]
    trig, pulses = [], []
    for ev in seg["events"]:
        s = int(ev["t_s"] * S.FS)
        if ev["train_id"] == late and ev["pulse_index"] == 2:
            s += int(0.0047 * S.FS)
        trig.append((s, ev["key"]))
        if ev["kind"] == "stim":
            pulses.append((s + 6, 21, 2.0, 100.0))
    raw = tmp_path / "raw.h5"
    simulate.record(simulate.Organoid(seed=3), seg["duration_s"], pulses, trig, raw)
    t = match_triggers(seg, raw)
    assert len(t) == len(seg["events"]) and not t.exclude.any()
    assert (t.train_id == [e["train_id"] for e in seg["events"]]).all()


def test_late_bursts_and_clock_drift_align(tmp_path):
    """fs666 curve seg02, 30 Sep 2026: three pulses of one train recorded 8-13 ms late, on top of
    <1 ms of drift between the PC clock and the recording clock. Every trigger was recorded, so
    the correct answer is the in-order pairing with nothing excluded."""
    plan = P.plan_curve_pulses([21], levels=(1, 10), ipi_ms=30, reps=6, sham_reps=4, seed=2)
    seg = plan["segments"][0]
    trains = pd.DataFrame(seg["events"])
    late = trains[trains.level == 10].train_id.iloc[1]
    delay = {6: 0.013, 7: 0.010, 9: 0.008}
    trig, pulses = [], []
    for ev in seg["events"]:
        t = ev["t_s"] * (1 - 3e-6)                                   # recording clock slightly slow
        if ev["train_id"] == late:
            t += delay.get(ev["pulse_index"], 0.0)
        s = int(t * S.FS)
        trig.append((s, ev["key"]))
        if ev["kind"] == "stim":
            pulses.append((s + 6, 21, 2.0, 100.0))
    raw = tmp_path / "raw.h5"
    simulate.record(simulate.Organoid(seed=3), seg["duration_s"], pulses, trig, raw)
    t = match_triggers(seg, raw)
    assert list(t.train_id) == [e["train_id"] for e in seg["events"]] and not t.exclude.any()


# ------------------------------------------------------------------------------ train codes

def _coded_curve(reps=6):
    plan = P.add_train_codes(P.plan_curve_pulses([21], levels=(1, 4), ipi_ms=30, reps=reps, sham_reps=4, seed=2))
    return plan["segments"][0]


def _record(seg, path, edit=lambda ev, s: (s, ev["key"]), code_pulse=False):
    trig, pulses = [], []
    for ev in seg["events"]:
        s, key = edit(ev, int(ev["t_s"] * S.FS))
        trig.append((s, key))
        if ev["kind"] == "stim" or (code_pulse and ev["kind"] == "code"):
            pulses.append((s + 6, 21, 2.0, 100.0))
    simulate.record(simulate.Organoid(seed=3), seg["duration_s"], pulses, trig, path)
    return path


def test_train_codes_plan():
    seg = _coded_curve()
    codes = [e for e in seg["events"] if e["kind"] == "code"]
    trains = list(dict.fromkeys(e["train_id"] for e in seg["events"] if e["kind"] != "code"))
    assert len(codes) == 2 * len(trains) and all(e["key"] in P.CODE_KEYS for e in codes)
    for n, tid in enumerate(trains):
        k1, k2 = [e["key"] for e in codes if e["train_id"] == tid]
        assert P.decode_keys(k1, k2) == n % P.N_CODES
    assert [e["t_s"] for e in seg["events"]] == sorted(e["t_s"] for e in seg["events"])
    with pytest.raises(ValueError):          # 10 electrodes would bind keys 8 and 9
        P.add_train_codes(P.plan_curve_pulses(list(range(10)), levels=(1,), reps=1, sham_reps=1))


def test_codes_confirm_alignment_despite_lost_row_and_late_pulse(tmp_path):
    import h5py
    seg = _coded_curve()
    late = next(e["train_id"] for e in seg["events"] if e["kind"] == "stim" and e["level"] == 4)
    raw = _record(seg, tmp_path / "raw.h5", lambda ev, s: (s + (int(0.009 * S.FS) if ev["train_id"] == late
                                                                    and ev["pulse_index"] == 2 else 0), ev["key"]))
    with h5py.File(raw, "r+") as f:                       # drop one row in the middle
        keep = [r for r in range(f["triggers"].shape[0]) if r != f["triggers"].shape[0] // 2]
        for name in list(f):
            data = f[name][keep]
            del f[name]
            f.create_dataset(name, data=data, chunks=(1, 1280)).attrs["sampling_rate"] = np.uint32(S.FS)
    t = match_triggers(seg, raw)
    assert (t.note == "code mismatch").sum() == 0 and set(t.kind) <= {"stim", "sham"}
    assert (t.code == "ok").groupby(t.train_id).all().mean() > 0.95
    assert not t[t.train_id == late].exclude.any()


def test_contradicting_code_excludes_train(tmp_path):
    seg = _coded_curve()
    swapped = [e for e in seg["events"] if e["kind"] == "code"][4:6]     # train 2's code, read as train 3's
    k1, k2 = P.train_code_keys(3)
    def edit(ev, s):
        if ev is swapped[0]:
            return s, k1
        if ev is swapped[1]:
            return s, k2
        return s, ev["key"]
    t = match_triggers(seg, _record(seg, tmp_path / "raw.h5", edit))
    bad = t[t.note == "code mismatch"]
    assert set(bad.train_id) == {swapped[0]["train_id"]} and bad.exclude.all()


def test_code_trigger_that_stimulates_stops_the_run(tmp_path):
    seg = _coded_curve(reps=3)
    with pytest.raises(RuntimeError, match="Code triggers"):
        match_triggers(seg, _record(seg, tmp_path / "raw.h5", code_pulse=True))


# ------------------------------------------------------------------------------ FuncSolver curve

NCP = Path.home() / "NeuralControlProgram"


def test_funcsolver_plan_timing_and_structure():
    plan = P.plan_funcsolver([2, 10, 20, 29], reps=4, sham_reps=2, seed=1)
    parts = [s["partition"] for s in plan["segments"]]
    assert parts[0] == parts[11] == parts[-1] == "reference" and parts.count("development") == 10
    for s in plan["segments"]:
        assert all(v["n_pulses"] == s["n_pulses"] for v in s["electrodes"].values())
        t = np.array([e["t_s"] for e in s["events"]])
        assert np.diff(t).min() >= 1.0 - 1e-9                     # trains >= 1 s apart
        for o in range(4):                                         # each organoid >= 4 s
            to = [e["t_s"] for e in s["events"] if e["stim_electrode"] >= 0 and e["stim_electrode"] // 8 == o]
            assert len(to) < 2 or np.diff(to).min() >= 4.0 - 1e-9
    with pytest.raises(ValueError):
        P.plan_funcsolver([2], counts=(2, 30))                     # beyond the 20-pulse limit


def test_readout_counts_events_after_the_train_only():
    from orgchar import readout as RO
    rng = np.random.default_rng(0)
    base, post = rng.normal(0, 5, 15000), rng.normal(0, 5, 9000)
    post[:200] += 3000                                             # artifact during the train: blanked
    end = int(RO.train_end_ms(5, 10, 100) * 30)
    for ms in (30, 80, 150):                                       # three spikes in the window
        post[end + ms * 30:end + ms * 30 + 8] -= 80
    m = RO.measure_trial(base, post, 5, 10, 100, -25.0)
    assert m["response_events"] == 3 and m["baseline_events"] == 0


def _sim_funcsolver(tmp_path, reps=6):
    from orgchar import funcsolver as F
    org = simulate.demo_organoid()
    plan = P.plan_funcsolver([5, 13, 21], reps=reps, sham_reps=3, seed=1)
    segs = []
    for s in plan["segments"]:
        d = tmp_path / s["id"]
        d.mkdir()
        raw = SimBackend(org).run_segment(s, d, "f")
        segs.append((str(raw), match_triggers(s, raw)))
    return plan, segs, F.characterise(segs, plan)


def test_funcsolver_finds_the_three_planted_shapes_and_writes_a_loadable_calibration(tmp_path):
    from orgchar import funcsolver as F
    plan, segs, r = _sim_funcsolver(tmp_path)
    assert {k: (v["stim"], v["rec"]) for k, v in r["roles"].items()} == {"short": (13, 10), "band": (5, 3), "long": (21, 22)}
    assert all(not p["shape"] for p in r["pairs"] if (p["stim"], p["rec"]) not in {(13, 10), (5, 3), (21, 22)})
    plan = dict(plan, fsname="fstest")
    cal = F.calibration(plan, "fstest", "fstest_s1", {int(k): v for k, v in r["thresholds_uV"].items()},
                        r["_chosen"], r["pairs"], "test", True, tmp_path)
    path = tmp_path / "calibration.json"
    path.write_text(__import__("json").dumps(cal, allow_nan=False))
    if (NCP / "FuncSolver" / "calibration.py").exists():          # FuncSolver's own loader accepts it
        import importlib
        import sys
        sys.path[:0] = [str(NCP), str(NCP / "FuncSolver")]
        try:
            C = importlib.import_module("calibration")
            loaded = C.apply_calibration(path)
            assert set(loaded["selected"]) == {"short", "band", "long"}
        finally:
            del sys.path[:2]


def test_funcsolver_artifact_in_window_is_caught(tmp_path):
    import h5py
    from orgchar import funcsolver as F
    from orgchar import readout as RO
    plan = P.plan_funcsolver([21], counts=(2, 4, 6), reps=6, sham_reps=2, seed=3)
    plan["segments"] = [s for s in plan["segments"] if s["partition"] == "development"]
    segs = []
    for s in plan["segments"]:
        d = tmp_path / s["id"]
        d.mkdir()
        raw = SimBackend(simulate.Organoid(seed=2)).run_segment(s, d, "a")
        t = match_triggers(s, raw)
        end = RO.train_end_ms(s["n_pulses"], plan["ipi_ms"], plan["phase_us"])
        with h5py.File(raw, "r+") as f:                            # a slow tail still there 10-13 ms after the train
            x = f["electrode_22"][...].reshape(-1)
            for o in t[t.kind == "stim"].onset:
                a = int(o + (end + 10) * 30)
                x[a:a + 90] += 60 * np.sin(np.arange(90) / 4.0)
            f["electrode_22"][...] = x.reshape(f["electrode_22"].shape)
        segs.append((str(raw), t))
    D, thr, art = F.measure(segs, plan)
    assert max(art[(21, 22, n)] for n in (2, 4, 6)) > F.ARTIFACT_LIMIT
    assert max(art[(21, c, n)] for c in (16, 17, 18, 19, 20, 23) for n in (2, 4, 6)) < F.ARTIFACT_LIMIT
