"""Run with pytest from the repository root (see README for a venv that leaves `ncp` untouched)."""

import numpy as np
import pytest

from orgchar import analysis as A
from orgchar import protocol as P
from orgchar import screen as SC
from orgchar import signals as S
from orgchar import simulate
from orgchar.backends import SimBackend, match_triggers

# ------------------------------------------------------------------------------ protocol

def test_limits_reject_unsafe_settings():
    with pytest.raises(ValueError):
        P.check_setting(5.0, 100)          # amplitude
    with pytest.raises(ValueError):
        P.check_setting(3.0, 200)          # 0.6 nC > 0.4 nC
    P.check_setting(2.0, 100)


# ------------------------------------------------------------------------------ signals

def test_common_mode_events_across_blocks_are_removed():
    spk = [np.array([], dtype=np.int64) for _ in range(32)]
    for c in (3, 12, 20):                  # same instant on three organoids = interference
        spk[c] = np.array([1000, 5000 + 97 * c])
    out, n = S.reject_common_mode(spk)
    assert n == 3 and all(1000 not in out[c] for c in (3, 12, 20))
    assert all(len(out[c]) == 1 for c in (3, 12, 20))


def test_census_separates_live_and_silent_blocks(tmp_path):
    org = simulate.Organoid(spont_hz={20: 3.0, 21: 1.0}, seed=2)
    raw = SimBackend(org).run_segment(P.plan_census(60)["segments"][0], tmp_path, "c")
    r = A.census(raw)
    assert r["live_blocks"] == [2]
    assert {c["channel"] for c in r["channels"] if c["active"]} == {20, 21}


# ------------------------------------------------------------------------------ trigger matching

def _train_segment(reps=12, coded=False):
    """A screen segment on one electrode: one trigger per train (the board times the pulses), trains
    >= 4 s apart, shams interleaved."""
    plan = P.plan_train_screen(electrodes=[21], counts=(2,), reps=reps, sham_reps=4, seed=2)
    if coded:
        plan = P.add_train_codes(plan)
    return plan["segments"][0]


def _record(seg, path, edit=lambda ev, s: (s, ev["key"]), code_pulse=False, clock=1.0):
    trig, pulses = [], []
    for ev in seg["events"]:
        s, key = edit(ev, int(ev["t_s"] * clock * S.FS))
        trig.append((s, key))
        if ev["kind"] == "stim" or (code_pulse and ev["kind"] == "code"):
            pulses.append((s + 6, 21, 2.0, 100.0))
    simulate.record(simulate.Organoid(seed=3), seg["duration_s"], pulses, trig, path)
    return path


def _drop_rows(raw, rows):
    import h5py
    with h5py.File(raw, "r+") as f:
        keep = [r for r in range(f["triggers"].shape[0]) if r not in rows]
        for name in list(f):
            data = f[name][keep]
            del f[name]
            f.create_dataset(name, data=data, chunks=(1, 1280)).attrs["sampling_rate"] = np.uint32(S.FS)


def test_recording_gaps_do_not_break_trigger_matching(tmp_path):
    """The platform can drop whole 1280-sample rows from the raw file (seen on fs666, 30 Sep 2026).
    One train per ~4 s, so all three losses fall within the first 20 triggers: the offset after them
    is then the most common one, which once made the alignment discard every earlier trigger."""
    import h5py
    seg = _train_segment(reps=40)                         # enough trains that losing 4 leaves > 90 %
    raw = _record(seg, tmp_path / "raw.h5")
    with h5py.File(raw, "r") as f:                       # drop three rows, one of them holding a trigger
        trig_rows = np.flatnonzero((f["triggers"][...] != 0).any(axis=1))
    _drop_rows(raw, {int(trig_rows[5]), int(trig_rows[5]) + 300, int(trig_rows[5]) + 600})
    t = match_triggers(seg, raw)
    assert len(t) == len(seg["events"]) - 1                # the trigger inside the lost row is gone
    assert (t.note == "before recording gap").sum() >= 2   # trains just before a gap are excluded
    assert (t.note == "unmatched").sum() == 0
    planned = {e["train_id"]: e["stim_electrode"] for e in seg["events"]}
    assert all(planned[r.train_id] == r.stim_electrode for r in t[~t.exclude].itertuples())


def test_late_triggers_and_clock_drift_align(tmp_path):
    """Triggers recorded 8-13 ms late (fs666 curve, 30 Sep 2026), on top of drift between the PC clock
    and the recording clock. Every trigger was recorded, so the correct answer is the in-order
    pairing with nothing excluded."""
    seg = _train_segment()
    late = {e["train_id"]: d for e, d in zip([e for e in seg["events"] if e["kind"] == "stim"][2:5], (0.013, 0.010, 0.008))}
    raw = _record(seg, tmp_path / "raw.h5", lambda ev, s: (s + int(late.get(ev["train_id"], 0) * S.FS), ev["key"]),
                  clock=1 - 3e-6)
    t = match_triggers(seg, raw)
    assert list(t.train_id) == [e["train_id"] for e in seg["events"]] and not t.exclude.any()


# ------------------------------------------------------------------------------ train codes

def test_train_codes_plan():
    seg = _train_segment(reps=6, coded=True)
    codes = [e for e in seg["events"] if e["kind"] == "code"]
    trains = list(dict.fromkeys(e["train_id"] for e in seg["events"] if e["kind"] != "code"))
    assert len(codes) == 2 * len(trains) and all(e["key"] in P.CODE_KEYS for e in codes)
    for n, tid in enumerate(trains):
        k1, k2 = [e["key"] for e in codes if e["train_id"] == tid]
        assert P.decode_keys(k1, k2) == n % P.N_CODES
    assert [e["t_s"] for e in seg["events"]] == sorted(e["t_s"] for e in seg["events"])
    with pytest.raises(ValueError):          # 10 electrodes would bind keys 8 and 9
        P.add_train_codes(dict(segments=[dict(id="s", electrodes={e: dict(key=e) for e in range(10)}, events=[])]))


def test_codes_confirm_alignment_despite_lost_row_and_late_trigger(tmp_path):
    import h5py
    seg = _train_segment(reps=6, coded=True)
    late = next(e["train_id"] for e in seg["events"] if e["kind"] == "stim")
    raw = _record(seg, tmp_path / "raw.h5", lambda ev, s: (s + (int(0.009 * S.FS) if ev["train_id"] == late
                                                                    and ev["kind"] == "stim" else 0), ev["key"]))
    with h5py.File(raw, "r") as f:
        middle = f["triggers"].shape[0] // 2
    _drop_rows(raw, {middle})
    t = match_triggers(seg, raw)
    assert (t.note == "code mismatch").sum() == 0 and set(t.kind) <= {"stim", "sham"}
    assert (t.code == "ok").groupby(t.train_id).all().mean() >= 0.9   # one code may sit in the lost row
    assert not t[t.train_id == late].exclude.any()


def test_contradicting_code_excludes_train(tmp_path):
    seg = _train_segment(reps=6, coded=True)
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
    seg = _train_segment(reps=3, coded=True)
    with pytest.raises(RuntimeError, match="Code triggers"):
        match_triggers(seg, _record(seg, tmp_path / "raw.h5", code_pulse=True))


def test_plan_trials_places_every_requested_train_once():
    trials = [(8, 4), (10, 12), (8, 4), (8, 6), (10, 12), (8, 20), (21, 2)]
    plan = P.plan_trials(trials, seed=1)
    placed = {}
    for s in plan["segments"]:
        for e in s["events"]:                                      # each train has its electrode's uploaded length
            assert s["electrodes"][e["stim_electrode"]]["n_pulses"] == e["level"]
            placed[e["request"]] = (e["stim_electrode"], int(e["level"]))
    assert placed == dict(enumerate(trials))
    assert len(plan["segments"]) == 3                              # e8 needs three different lengths
    for s in plan["segments"]:
        t = np.array([e["t_s"] for e in s["events"]])
        assert len(t) < 2 or np.diff(t).min() >= 1.0 - 1e-9        # trains >= 1 s apart
        for o in range(4):                                         # each organoid >= 4 s
            to = [e["t_s"] for e in s["events"] if e["stim_electrode"] // 8 == o]
            assert len(to) < 2 or np.diff(to).min() >= 4.0 - 1e-9
    with pytest.raises(ValueError):
        P.plan_trials([(8, 30)])                                   # beyond the 20-pulse limit


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


def test_train_screen_deals_electrodes_two_per_organoid():
    plan = P.plan_train_screen(blocks=[0, 1, 2, 3], reps=2, sham_reps=1, seed=0)
    assert len(plan["segments"]) == 4 * len(P.SCREEN_COUNTS)
    for s in plan["segments"]:
        orgs = [e // 8 for e in s["electrodes"]]
        assert len(orgs) == P.MAX_UPLOAD_STIMS and all(orgs.count(o) == 2 for o in range(4))
        assert P.SHAM_KEY not in {v["key"] for v in s["electrodes"].values()}
        assert all(e["key"] == P.SHAM_KEY for e in s["events"] if e["kind"] == "sham")
    P.add_train_codes(plan)                                        # codes still fit


def _sim_train_screen(tmp_path, org, blocks):
    plan = P.plan_train_screen(blocks=blocks, seed=2)
    segs = []
    for s in plan["segments"]:
        d = tmp_path / s["id"]
        d.mkdir()
        raw = SimBackend(org).run_segment(s, d, "s")
        segs.append((str(raw), match_triggers(s, raw)))
    return SC.scout(segs, plan)


def test_train_screen_finds_the_planted_shapes(tmp_path):
    r = _sim_train_screen(tmp_path, simulate.demo_organoid(), [0, 1, 2])
    assert {k: (v["stim"], v["rec"]) for k, v in r["best"].items()} == {"short": (13, 10), "band": (5, 3), "long": (21, 22)}
    assert r["verdict"] == "CANDIDATES_FOUND"
    assert r["controls"]["cross_organoid_passing"] <= r["controls"]["nominal_ceiling"]


def test_train_screen_artifacts_alone_find_nothing(tmp_path):
    # strong artifacts and lively spontaneous firing, but nothing responds to the trains
    r = _sim_train_screen(tmp_path, simulate.Organoid(seed=7), [2])
    assert r["verdict"] == "NO_CANDIDATE" and not r["best"]


def test_candidate_matched_by_another_organoid_is_dropped():
    """fs666 screen, 1 Oct 2026: e19→e14 and e26→e14 cannot respond, yet passed on the bursting e14
    with smaller p and larger differences than e13→e14, the only peaked candidate (the curve did not
    confirm it). e10→e15 also has other-organoid pairs passing into e15, but far weaker."""
    def pair(stim, rec, role, p, diff):
        return dict(stim=stim, rec=rec, shapes=[role], scores={role: dict(p=p, diff=diff, win=0.95)})
    same = [pair(13, 14, "band", 0.0096, 1.8), pair(10, 15, "short", 0.0011, 82.1)]
    cross = [pair(19, 14, "short", 0.0041, 2.5), pair(26, 14, "long", 0.0051, 2.7), pair(30, 15, "long", 0.0031, 12.4)]
    dropped = SC.drop_chance(same, cross)
    assert dropped == [dict(stim=13, rec=14, role="band", beaten_by=19)]
    assert same[0]["shapes"] == [] and same[1]["shapes"] == ["short"]


def test_live_trials_fit_in_short_recordings():
    """540 trains of one length on one electrode (FuncSolver, fs675, 14 Oct 2026) made one 38-min recording
    that the platform never saved. Every trial must still be placed exactly once."""
    trials = [(18, 20)] * 540 + [(18, 2)] * 40
    plan = P.plan_trials(trials, seed=1)
    assert all(s["duration_s"] <= P.MAX_SEGMENT_S + 2 for s in plan["segments"])
    placed = sorted(e["request"] for s in plan["segments"] for e in s["events"])
    assert placed == list(range(len(trials)))
    assert all(len({v["n_pulses"] for v in s["electrodes"].values()}) == 1 for s in plan["segments"])


def test_short_recording_with_one_late_first_trigger():
    """FuncSolver retry, fs675, 15 Oct 2026: two code triggers and one train; the first code trigger was
    recorded 2.5 ms later than the other two. All three must be matched."""
    from orgchar.backends import _align
    edges = [(88650, 8), (88890, 8), (91570, 0)]
    pairs, _ = _align(edges, [0.9, 0.91024, 0.999522], [8, 8, 0])
    assert pairs == [(0, 0), (1, 1), (2, 2)]
