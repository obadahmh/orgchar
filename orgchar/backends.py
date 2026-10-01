"""Run one plan segment: upload parameters, record, send triggers, fetch the raw file.

HardwareBackend reproduces the call sequence that worked on fs647 (8 Sep 2026):
disable all -> send + upload params -> wait 10 s -> start raw recording (32 ch + triggers)
-> send one-hot trigger bytes -> stop recording -> disable all -> copy HDF5 from the S3 mount.
Clean-up (stop recording, disable stimulation) always runs, even after an error or Ctrl-C.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import signals as S
from . import simulate

RAW_ROOT = Path("/mnt/s3-intanraws")
PRE_S, POST_S, UPLOAD_WAIT_S = 2.0, 2.0, 10.0


def _utc():
    return datetime.now(timezone.utc).isoformat()


ROW_S = 1280 / S.FS   # the platform writes (and sometimes loses) raw data in rows of 1280 samples = 42.67 ms


# Costs for _align, in units of "one ms of delivery delay" x 2 (see _align). Chosen so that every
# alternative to "this pulse was simply late" needs several unlikely events at once.
DELAY_SCALE_S = 0.002   # typical delivery delay beyond the fastest; cost = delay / scale
EARLY_TOL_S = 0.0015    # an edge cannot be recorded earlier than its send time allows (clock noise only)
EARLY_TOL_ROUGH_S = 0.006  # first pass, before clock drift is known (still far below a 27 ms pulse spacing)
ROW_COST, MISSING_EDGE_COST, EXTRA_EDGE_COST = 8.0, 6.0, 12.0
MAX_LOST_ROWS = 16


def _align(edges, times, keys):
    """Pair recorded trigger edges with sent events: a global, minimum-cost alignment.

    What can happen between a send and its recorded edge (all seen on the platform, Sep 2026):
      * delivery delay: the edge is recorded LATER than its send time (0-13 ms seen; up to ~200 ms
        before TCP_NODELAY), never earlier;
      * lost rows: the file silently drops rows of 42.67 ms, so every later edge appears exactly
        k x 42.67 ms EARLIER, and edges inside a dropped row are missing;
      * slow drift between the PC clock and the recording's sample clock (< 1 ms per 6 min).
    So  recorded = send + offset(t) + delay - rows_lost x 42.67 ms,  with delay >= 0.

    Dynamic programming over (edges, events, rows lost so far) finds the explanation of the whole
    recording with the lowest total cost: delay/DELAY_SCALE_S per match, ROW_COST per lost row,
    MISSING_EDGE_COST per event with no edge, EXTRA_EDGE_COST per edge with no event. Order and keys
    must agree. Unlike walking edges one at a time, one odd edge cannot derail everything after it:
    a wrong local choice makes the rest of the recording expensive, so it is not chosen.
    offset(t) is fitted in two passes: a rough one (constant offset, looser early limit), then a
    linear fit to the promptly delivered edges of the whole recording, then the final alignment.
    Returns (pairs [(edge_index, event_index)], gaps [edge_index just before a lost row])."""
    if not len(edges) or not len(times):
        return [], []
    rec = np.array([s / S.FS for s, _ in edges])
    tm = np.asarray(times, float)
    ek, kk = np.array([k for _, k in edges]), np.asarray(keys)
    # starting offset: the fastest common delivery among the first edges (most pulses have ~no delay)
    d0 = np.array([rec[i] - tm[j] for i in range(min(20, len(rec))) for j in range(min(20, len(tm)))
                   if ek[i] == kk[j]])
    if not len(d0):
        return [], []
    mode = d0[np.argmax([(np.abs(d0 - c) < 0.002).sum() for c in d0])]
    off = np.full(len(tm), d0[np.abs(d0 - mode) < 0.002].min())
    pairs, gaps, rows = _align_pass(rec, tm, ek, kk, off, EARLY_TOL_ROUGH_S)
    if len(pairs) >= 10:                    # linear drift, fitted to the promptly delivered edges
        i, j = np.array(pairs).T
        base = rec[i] - tm[j] + rows * ROW_S
        fast = np.ones(len(base), bool)
        for _ in range(3):                  # lower envelope: drop late deliveries, refit
            a, b = np.polyfit(tm[j[fast]], base[fast], 1)
            fast = base - (a * tm[j] + b) < 0.0005
        off = a * tm + b + min(0.0, float((base - (a * tm[j] + b)).min()))
    pairs, gaps, _ = _align_pass(rec, tm, ek, kk, off, EARLY_TOL_S)
    return pairs, gaps


def _align_pass(rec, tm, ek, kk, off, early_tol):
    n, m, nr = len(rec), len(tm), MAX_LOST_ROWS + 1
    R = np.arange(nr)
    inf = np.inf
    D = np.full((n + 1, m + 1, nr), inf)
    D[0, :, 0] = np.arange(m + 1) * MISSING_EDGE_COST
    jcost = np.arange(m + 1)[:, None] * MISSING_EDGE_COST
    for i in range(n):
        # match edge i with event j, possibly after losing more rows (rows can only accumulate)
        d = rec[i] - (tm + off)[:, None] + R[None, :] * ROW_S                  # (m, nr) delay
        c = np.where((ek[i] == kk)[:, None] & (d > -early_tol), np.maximum(d, 0) / DELAY_SCALE_S, inf)
        prev = np.minimum.accumulate(D[i, :m] - R * ROW_COST, axis=1) + R * ROW_COST
        nxt = np.full((m + 1, nr), inf)
        nxt[1:] = prev + c
        nxt = np.minimum(nxt, D[i] + EXTRA_EDGE_COST)                         # edge i unexplained
        # events with no edge: D[j] = min_k<=j D[k] + cost x (j - k)
        D[i + 1] = np.minimum.accumulate(nxt - jcost, axis=0) + jcost
    # trace back
    i, j, r = n, m, int(np.argmin(D[n, m]))
    out = []
    while i > 0:
        v = D[i, j, r]
        if j > 0 and np.isclose(v, D[i, j - 1, r] + MISSING_EDGE_COST):
            j -= 1
            continue
        if np.isclose(v, D[i - 1, j, r] + EXTRA_EDGE_COST):
            i -= 1
            continue
        d = rec[i - 1] - (tm[j - 1] + off[j - 1]) + r * ROW_S
        c = max(d, 0) / DELAY_SCALE_S
        r0 = next(q for q in range(r, -1, -1) if np.isclose(v, D[i - 1, j - 1, q] + (r - q) * ROW_COST + c))
        out.append((i - 1, j - 1, r))
        i, j, r = i - 1, j - 1, r0
    out.reverse()
    pairs = [(a, b) for a, b, _ in out]
    rows = np.array([q for *_, q in out])
    gaps = [out[k - 1][0] for k in range(1, len(out)) if out[k][2] > out[k - 1][2]]
    return pairs, gaps, rows


CODE_ARTIFACT_LIMIT_UV = 50.0   # a code trigger must leave no stimulation artifact (real ones: ~4000 µV)


def _decode_train_codes(edges, code_keys):
    """Read train codes straight from the recording, independently of the alignment: two code
    edges 5-20 ms apart, then the train's first trigger 60-140 ms after the first code edge.
    Returns {edge index of the train's first trigger: code}."""
    from .protocol import CODE_LEAD_S, decode_keys
    out = {}
    for a in range(len(edges) - 2):
        (s1, k1), (s2, k2) = edges[a], edges[a + 1]
        if k1 not in code_keys or k2 not in code_keys or not 0.005 <= (s2 - s1) / S.FS <= 0.020:
            continue
        nxt = next((b for b in range(a + 2, len(edges)) if edges[b][1] not in code_keys), None)
        if nxt is not None and abs((edges[nxt][0] - s1) / S.FS - CODE_LEAD_S) < 0.040:
            out[nxt] = decode_keys(k1, k2)
    return out


def code_artifact_uV(X, onsets) -> float:
    """Largest change of the median raw signal, on any channel, in the 5 ms after these triggers
    (relative to the 1 ms before). Stimulation leaves thousands of µV; nothing leaves ~noise/sqrt(n)."""
    on = [s for s in onsets if S.samples(1) <= s < X.shape[1] - S.samples(5)][:200]
    if not on:
        return float("nan")
    seg = np.stack([X[:, s - S.samples(1):s + S.samples(5)] for s in on])
    med = np.median(seg, axis=0)
    base = med[:, :S.samples(1)].mean(axis=1, keepdims=True)
    return float(np.abs(med[:, S.samples(1):] - base).max())


def match_triggers(seg: dict, raw_path, sent: pd.DataFrame | None = None, min_fraction=0.9) -> pd.DataFrame:
    """One row per recorded stim/sham trigger edge, with its planned event.

    Every recorded stimulation edge is kept (its artifact must be blanked even if it cannot be
    used); rows with exclude=True are blanked but not counted:
      * edges not matched to the plan;
      * the pulse just before a lost row (its response window may straddle the missing data);
      * every pulse of a multi-pulse train that lost any pulse (the train's total would be wrong);
      * in coded segments, every pulse of a train whose recorded code contradicts the alignment.
    Code triggers (coded segments only) are used for the check and then dropped: no current flows.
    Column `code`: 'ok' | 'missing' (code not readable, e.g. in a lost row) | 'mismatch' | ''."""
    from .protocol import CODE_KEYS
    X, T = S.load_raw(raw_path)
    edges = S.trigger_edges(T)
    evs = seg["events"]
    coded = bool(seg.get("train_codes"))
    els = {int(k): v for k, v in seg["electrodes"].items()}
    by_key = {v["key"]: e for e, v in els.items()}
    code_keys = set(CODE_KEYS) - set(by_key) if coded else set()
    # always align by timing, even when the key sequence matches: lost rows must still be found
    times = [e["t_s"] for e in evs]
    if sent is not None and len(sent) == len(evs) and "utc" in sent:
        t = pd.to_datetime(sent["utc"])
        times = list((t - t.iloc[0]).dt.total_seconds() + evs[0]["t_s"])
    pairs, gaps = _align(edges, times, [e["key"] for e in evs]) if evs else ([], [])
    matched = dict(pairs)
    is_code = [k in code_keys for _, k in edges]
    # a lost row is attributed to the last stim/sham pulse before it (codes carry no response)
    gap_edges = set()
    for g in gaps:
        prev = [i for i, _ in pairs if i <= g and not is_code[i]]
        if prev:
            gap_edges.add(prev[-1])
    train_evs = [e for e in evs if e["kind"] != "code"]
    per_train = pd.Series([e["train_id"] for e in train_evs]).value_counts()
    got = [evs[j]["train_id"] for i, j in matched.items() if not is_code[i]]
    seen = pd.Series(got).value_counts() if got else pd.Series(dtype=int)
    gap_trains = {evs[matched[i]]["train_id"] for i in gap_edges}
    broken = {t for t in set(got) if seen.get(t, 0) < per_train[t] and per_train[t] > 1} | \
             {t for t in gap_trains if per_train[t] > 1}
    # train codes: read from the recording, compared with what the alignment says
    code_of = {e["train_id"]: e["code"] for e in evs if e["kind"] == "code"}
    check = {}
    if coded:
        for i, c in _decode_train_codes(edges, code_keys).items():
            if i in matched:
                tid = evs[matched[i]]["train_id"]
                check[tid] = "ok" if code_of.get(tid) == c else "mismatch"
        art = code_artifact_uV(X, [s for (s, _), c in zip(edges, is_code) if c])
        if art > CODE_ARTIFACT_LIMIT_UV:
            raise RuntimeError(f"Code triggers (keys 8-14) left a {art:.0f} µV artifact in {raw_path}: something "
                               "may be stimulating on those keys. Stop and investigate before any further run.")
    mismatched = {t for t, v in check.items() if v == "mismatch"}
    rows = []
    for i, (s, key) in enumerate(edges):
        if is_code[i]:
            continue
        ev = evs[matched[i]] if i in matched else None
        e = ev["stim_electrode"] if ev else by_key.get(key, -1)
        kind = ev["kind"] if ev else ("stim" if key in by_key else "sham")
        el = els.get(e, {}) if kind == "stim" else {}
        tid = ev["train_id"] if ev else None
        note = ("unmatched" if ev is None else "code mismatch" if tid in mismatched
                else "before recording gap" if i in gap_edges else "incomplete train" if tid in broken else "")
        rows.append(dict(segment=seg["id"], onset=s, key=key, kind=kind, stim_electrode=e,
                         amplitude_uA=el.get("amplitude_uA", 0.0), phase_us=el.get("phase_us", 0.0),
                         polarity=el.get("polarity", ""), n_pulses=1, ipi_ms=0.0,
                         level=ev["level"] if ev else float("nan"),
                         train_id=tid if ev else f"unmatched{i}",
                         pulse_index=ev["pulse_index"] if ev else 0, planned_t_s=ev["t_s"] if ev else float("nan"),
                         code=(check.get(tid, "missing") if coded and ev else ""),
                         exclude=bool(note), note=note))
    df = pd.DataFrame(rows)
    usable_trials = df[~df.exclude].train_id.nunique() if len(df) else 0
    planned_trials = len(per_train)
    n_matched = sum(not is_code[i] for i in matched)
    if n_matched != len(train_evs) or gap_edges:
        print(f"  NOTE {seg['id']}: {len(train_evs)} triggers sent, {len(edges) - sum(is_code)} in the recording, "
              f"{n_matched} matched; {len(gaps)} lost row(s) in the platform's recording; "
              f"{usable_trials} of {planned_trials} trials usable.")
    if coded:
        n_ok, n_bad = sum(v == "ok" for v in check.values()), len(mismatched)
        print(f"  CODES {seg['id']}: {n_ok} of {planned_trials} trains confirmed by their recorded code"
              + (f"; {n_bad} CONTRADICT the alignment (excluded)" if n_bad else "")
              + (f"; {planned_trials - n_ok - n_bad} code(s) unreadable" if planned_trials - n_ok - n_bad else "") + ".")
    if usable_trials < min_fraction * planned_trials:
        raise RuntimeError(f"Only {usable_trials}/{planned_trials} trials usable in {raw_path}; segment not analysed. "
                           "Raw file kept.")
    return df


def _remote_stopped(tag: str, wait_s: float = 10.0) -> bool:
    """True if the platform's raw file for this tag exists and did not grow for wait_s."""
    hits = list(RAW_ROOT.glob(f"*_{tag}.h5"))
    if len(hits) != 1:
        return False
    a = hits[0].stat().st_size
    time.sleep(wait_s)
    return hits[0].stat().st_size == a and a > 0


class SimBackend:
    """Writes a synthetic recording of what the plan would produce (see simulate.py)."""

    def __init__(self, organoid: simulate.Organoid | None = None):
        self.organoid = organoid or simulate.Organoid()

    def run_segment(self, seg: dict, seg_dir: Path, tag: str) -> Path:
        pulses, trig = [], []
        for ev in seg["events"]:
            s = int(ev["t_s"] * S.FS)
            trig.append((s, ev["key"]))
            if ev["kind"] == "stim":
                el = seg["electrodes"][ev["stim_electrode"]]
                pulses.append((s + 6, ev["stim_electrode"], el["amplitude_uA"], el["phase_us"]))
        raw = seg_dir / "raw.h5"
        simulate.record(self.organoid, seg["duration_s"], pulses, trig, raw)
        return raw


class HardwareBackend:
    def __init__(self, operator: str = "admin", log=print):
        self.operator = operator
        self.log = log

    @staticmethod
    def _params(seg):
        from neuroplatformv2.utils.enumerations import StimPolarity
        from neuroplatformv2.utils.schemas import StimParam
        out = []
        for e, p in seg["electrodes"].items():
            out.append(StimParam(index=int(e), enable=True, trigger_key=int(p["key"]), trigger_delay=0,
                                 nb_pulse=0, pulse_train_period=int(round(2 * p["phase_us"])),
                                 phase_duration1=p["phase_us"], phase_duration2=p["phase_us"],
                                 phase_amplitude1=p["amplitude_uA"], phase_amplitude2=p["amplitude_uA"],
                                 polarity=StimPolarity.PositiveFirst if p["polarity"] == "positive-first"
                                 else StimPolarity.NegativeFirst))
        return out

    async def _ack(self, label, coro, timeout=60):
        r = await asyncio.wait_for(coro, timeout)
        if not getattr(r, "status", False):
            raise RuntimeError(f"{label} rejected by platform: {getattr(r, 'message', r)}")

    async def _segment(self, seg, tag, sent):
        from neuroplatformv2.core.intan import IntanController
        from neuroplatformv2.core.trigger import TriggerController
        intan, trig, recording = IntanController(), None, False
        try:
            await self._ack("disable all stimulation", intan._disable_all_stim())
            if seg["electrodes"]:
                # fresh StimParam objects: the SDK only transmits fields changed since creation
                await self._ack("send stimulation parameters", intan._send_stimparam(self._params(seg)))
                await self._ack("upload stimulation parameters",
                                intan._upload_stimparam([int(e) for e in seg["electrodes"]]))
                await asyncio.sleep(UPLOAD_WAIT_S)
            await self._ack("start raw recording", intan._start_raw_recording(list(range(S.N_CH)), tag, True), 30)
            recording = True
            self.log(f"  recording {tag} ({seg['duration_s']:.0f} s, {len(seg['events'])} triggers)")
            if seg["events"]:
                trig = TriggerController(self.operator)
                # Send every 16-byte trigger at once. With TCP's default batching (Nagle), triggers
                # 30 ms apart were held back and delivered in bursts (fs666 curve, 30 Sep 2026).
                trig.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            start = time.monotonic() + PRE_S
            await asyncio.sleep(PRE_S)
            for ev in seg["events"]:
                await asyncio.sleep(max(0.0, start + ev["t_s"] - time.monotonic()))
                if not trig.__check_time__():
                    raise RuntimeError("Booking window ended")
                pattern = np.zeros(16, dtype=np.uint8)
                pattern[ev["key"]] = 1
                trig.sock.sendall(pattern.tobytes())
                sent.append(dict(key=ev["key"], train_id=ev["train_id"], pulse_index=ev["pulse_index"], utc=_utc()))
            await asyncio.sleep(max(0.0, start + seg["duration_s"] - time.monotonic()) if not seg["events"] else POST_S)
        finally:
            errors, stop_ok, disable_ok = [], not recording, False
            if trig is not None:
                trig.close()
            if recording:
                for attempt in range(3):
                    try:
                        await self._ack("stop raw recording", intan._stop_raw_recording(), 90)
                        stop_ok = True
                        break
                    except Exception as e:  # noqa: BLE001 - keep trying, then report
                        errors.append(f"stop recording attempt {attempt + 1}: {e!r}")
                        # the platform often stops without acknowledging: check the file itself
                        if await asyncio.to_thread(_remote_stopped, tag):
                            errors.append("no acknowledgement, but the recording file stopped growing: stopped")
                            stop_ok = True
                            break
            try:
                await self._ack("disable all stimulation", intan._disable_all_stim())
                disable_ok = True
            except Exception as e:  # noqa: BLE001
                errors.append(f"final disable: {e!r}")
            await intan._close()
            self.close_status = dict(stop_ok=stop_ok, disable_ok=disable_ok, messages=errors)
            if errors:
                recovered = stop_ok and disable_ok
                self.log(("  Clean-up needed retries but succeeded: " if recovered else "  CLEAN-UP FAILED: ")
                         + " | ".join(errors))

    def run_segment(self, seg: dict, seg_dir: Path, tag: str) -> Path:
        sent: list = []
        self.close_status = dict(stop_ok=False, disable_ok=False, messages=["segment did not reach clean-up"])
        try:
            asyncio.run(self._segment(seg, tag, sent))
        finally:
            pd.DataFrame(sent).to_csv(seg_dir / "sent.csv", index=False)
            (seg_dir / "close_status.json").write_text(json.dumps(self.close_status, indent=1))
        return self._fetch(tag, seg_dir / "raw.h5")

    def _fetch(self, tag, dest, timeout_s=900):
        """Wait for the platform to finish writing /mnt/s3-intanraws/*_<tag>.h5, then copy it."""
        deadline, last, stable = time.monotonic() + timeout_s, None, 0
        while time.monotonic() < deadline:
            hits = list(RAW_ROOT.glob(f"*_{tag}.h5"))
            if len(hits) == 1:
                st = hits[0].stat()
                sig = (st.st_size, st.st_mtime_ns)
                stable = stable + 1 if sig == last and st.st_size > 0 else 0
                last = sig
                if stable >= 2:
                    tmp = dest.with_suffix(".partial")
                    shutil.copyfile(hits[0], tmp)
                    tmp.replace(dest)
                    (dest.parent / "remote_path.txt").write_text(str(hits[0]) + "\n")
                    return dest
            time.sleep(5)
        raise RuntimeError(f"Raw file for tag {tag} did not appear on {RAW_ROOT} within {timeout_s} s")
