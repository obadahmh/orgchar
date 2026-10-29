"""Run one plan segment: upload parameters, record, send triggers, locate the saved raw file.

HardwareBackend reproduces the call sequence that worked on fs647 (8 Sep 2026):
disable all -> send + upload params -> wait 10 s -> start raw recording (32 ch + triggers)
-> send one-hot trigger bytes -> stop recording -> wait for the file on the S3 mount -> disable all.
The raw file is read where the platform saved it (/mnt/s3-intanraws); its path is kept in the
segment's remote_path.txt. Clean-up (stop recording, disable stimulation) always runs, even after
an error or Ctrl-C.
"""
from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import signals as S
from . import simulate

RAW_ROOT = Path("/mnt/s3-intanraws")
PRE_S, POST_S, UPLOAD_WAIT_S = 2.0, 2.0, 10.0
STOP_ACK_S = 20.0
DISABLE_ACK_S = 180.0  # one final disable, never repeated while the platform may still be busy
SAVE_WAIT_S = 600.0


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
    # starting offset: the fastest common delivery among the first edges (most pulses have ~no delay).
    # The most common one is not always right: when a sparse recording loses rows early, the offset
    # after the loss is the most common among the first edges (one train per 4 s, simulated 5 Oct
    # 2026). So the best-supported candidates are each aligned and the cheapest explanation wins.
    d0 = np.array([rec[i] - tm[j] for i in range(min(20, len(rec))) for j in range(min(20, len(tm)))
                   if ek[i] == kk[j]])
    if not len(d0):
        return [], []
    cands = []
    for k in np.argsort([-(np.abs(d0 - c) < 0.002).sum() for c in d0], kind="stable"):
        c = d0[np.abs(d0 - d0[k]) < 0.002].min()
        if all(abs(c - x) >= 0.002 for x in cands):
            cands.append(c)
        if len(cands) == 5:
            break
    # ranked by the strict cost of the final pass: under the loose limit an edge a little early costs
    # nothing, so the latest candidate always won and the final pass then dropped the other edges
    # (a 3-trigger FuncSolver recording, fs675, 15 Oct 2026). Drift is < 1 ms per 6-min recording.
    off = min((np.full(len(tm), c) for c in cands), key=lambda o: _align_pass(rec, tm, ek, kk, o, EARLY_TOL_S)[3])
    pairs, gaps, rows, _ = _align_pass(rec, tm, ek, kk, off, EARLY_TOL_ROUGH_S)
    if len(pairs) >= 10:                    # linear drift, fitted to the promptly delivered edges
        i, j = np.array(pairs).T
        base = rec[i] - tm[j] + rows * ROW_S
        fast = np.ones(len(base), bool)
        for _ in range(3):                  # lower envelope: drop late deliveries, refit
            a, b = np.polyfit(tm[j[fast]], base[fast], 1)
            fast = base - (a * tm[j] + b) < 0.0005
        off = a * tm + b + min(0.0, float((base - (a * tm[j] + b)).min()))
    pairs, gaps, *_ = _align_pass(rec, tm, ek, kk, off, EARLY_TOL_S)
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
    return pairs, gaps, rows, float(D[n, m].min())


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


def recording(seg_dir: Path) -> Path | None:
    """The segment's raw file: raw.h5 in the folder (simulations; hardware runs before 5 Oct 2026 copied
    it), else the file the platform saved on the S3 mount (remote_path.txt)."""
    if (seg_dir / "raw.h5").exists():
        return seg_dir / "raw.h5"
    if not (seg_dir / "remote_path.txt").exists():
        return None
    p = Path((seg_dir / "remote_path.txt").read_text().strip())
    if not p.exists():
        raise FileNotFoundError(f"Recording {p} is not reachable. Is {RAW_ROOT} mounted?")
    return p


def match_triggers(seg: dict, raw_path, sent: pd.DataFrame | None = None, min_fraction=0.9) -> pd.DataFrame:
    """One row per recorded stim/sham trigger edge (one per train: the board times the pulses),
    with its planned event. Rows with exclude=True are kept, so their stimulation stays visible,
    but not counted:
      * edges not matched to the plan;
      * the train just before a lost row (its response window may straddle the missing data);
      * in coded segments, a train whose recorded code contradicts the alignment.
    Code triggers (coded segments only) are used for the check and then dropped: no current flows.
    Column `code`: 'ok' | 'missing' (code not readable, e.g. in a lost row) | 'mismatch' | ''."""
    from .protocol import CODE_KEYS
    T = S.load_triggers(raw_path)
    edges = S.trigger_edges(T)
    evs = seg["events"]
    coded = bool(seg.get("train_codes"))
    els = {int(k): v for k, v in seg["electrodes"].items()}
    by_key = {v["key"]: e for e, v in els.items()}
    code_keys = set(CODE_KEYS) - set(by_key) if coded else set()
    # always align by timing, even when the key sequence matches: lost rows must still be found
    times = [e["t_s"] for e in evs]
    if sent is not None and len(sent) == len(evs) and "utc" in sent:
        t = pd.to_datetime(sent["utc"], format="ISO8601")
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
    # train codes: read from the recording, compared with what the alignment says
    code_of = {e["train_id"]: e["code"] for e in evs if e["kind"] == "code"}
    check = {}
    if coded:
        for i, c in _decode_train_codes(edges, code_keys).items():
            if i in matched:
                tid = evs[matched[i]]["train_id"]
                check[tid] = "ok" if code_of.get(tid) == c else "mismatch"
        X, _ = S.load_raw(raw_path)
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
                else "before recording gap" if i in gap_edges else "")
        rows.append(dict(segment=seg["id"], onset=s, key=key, kind=kind, stim_electrode=e,
                         amplitude_uA=el.get("amplitude_uA", 0.0), phase_us=el.get("phase_us", 0.0),
                         polarity=el.get("polarity", ""), n_pulses=int(el.get("n_pulses", 1)),
                         ipi_ms=float(el.get("ipi_ms", 0.0)),
                         partition=(ev or {}).get("partition", seg.get("partition", "")),
                         block=(ev or {}).get("block", seg.get("block", -1)),
                         level=ev["level"] if ev else float("nan"),
                         train_id=tid if ev else f"unmatched{i}",
                         pulse_index=ev["pulse_index"] if ev else 0, planned_t_s=ev["t_s"] if ev else float("nan"),
                         code=(check.get(tid, "missing") if coded and ev else ""),
                         exclude=bool(note), note=note))
    df = pd.DataFrame(rows)
    usable_trials = df[~df.exclude].train_id.nunique() if len(df) else 0
    planned_trials = len(train_evs)
    # for the progress output (cli): what this recording delivered
    df.attrs["summary"] = dict(planned=planned_trials, usable=int(usable_trials), lost_rows=len(gaps),
                               unmatched=int((df.note == "unmatched").sum()) if len(df) else 0,
                               recorded_s=len(T) / S.FS, codes_ok=sum(v == "ok" for v in check.values()) if coded else None,
                               codes_bad=len(mismatched) if coded else None)
    if usable_trials < min_fraction * planned_trials:
        raise RuntimeError(f"Only {usable_trials}/{planned_trials} trials usable in {raw_path}; segment not analysed.")
    return df


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
                for i in range(int(el.get("n_pulses", 1))):
                    pulses.append((s + 6 + S.samples(i * el.get("ipi_ms", 0.0)), ev["stim_electrode"],
                                   el["amplitude_uA"], el["phase_us"]))
        raw = seg_dir / "raw.h5"
        simulate.record(self.organoid, seg["duration_s"], pulses, trig, raw)
        return raw


def _bar(fraction, width=20):
    n = int(round(max(0.0, min(1.0, fraction)) * width))
    return "▕" + "█" * n + "░" * (width - n) + "▏"


def status(msg, live=False):
    """The current segment's status line: rewritten in place on a terminal. In a log file only the
    phase changes appear (live=True updates, e.g. the recording clock, are skipped there)."""
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K        " + msg)
        sys.stdout.flush()
    elif not live:
        print("        " + msg, flush=True)


def clear_status():
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


class HardwareBackend:
    def __init__(self, operator: str = "admin", log=print, box: str | None = None):
        """box: the Intan/trigger address of the MEA to drive (default: INTAN_SOFTWARE_IP / TRIGGER_IP)."""
        self.operator = operator
        self.log = log
        self.box = box

    @staticmethod
    def _params(seg):
        from neuroplatformv2.utils.enumerations import StimPolarity
        from neuroplatformv2.utils.schemas import StimParam
        out = []
        for e, p in seg["electrodes"].items():
            # a hardware train: nb_pulse pulses, pulse_train_period µs apart (as FuncSolver and the
            # fs647 sweep of 8 Sep 2026 program it); a single pulse: nb_pulse 0
            n = int(p.get("n_pulses", 1))
            out.append(StimParam(index=int(e), enable=True, trigger_key=int(p["key"]), trigger_delay=0,
                                 nb_pulse=0 if n <= 1 else n,
                                 pulse_train_period=int(round(p["ipi_ms"] * 1000)) if n > 1 else int(round(2 * p["phase_us"])),
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
        from neuroplatformv2.core import intan as sdk_intan, trigger as sdk_trigger
        if self.box:
            # The SDK reads its address once, at import; each box serves Intan and triggers on one
            # address (np14: 172.30.2.170, neuroplatform10: 172.30.1.139, 29 Oct 2026).
            sdk_intan.INTAN_SERVICE_IP = sdk_trigger.TRIGGER_IP = self.box
        intan, trig, recording = sdk_intan.IntanController(), None, False
        try:
            status("disabling stimulation…")
            await self._ack("disable all stimulation", intan._disable_all_stim())
            if seg["electrodes"]:
                status(f"uploading stimulation parameters, then {UPLOAD_WAIT_S:.0f} s for the board…")
                # fresh StimParam objects: the SDK only transmits fields changed since creation
                await self._ack("send stimulation parameters", intan._send_stimparam(self._params(seg)))
                await self._ack("upload stimulation parameters",
                                intan._upload_stimparam([int(e) for e in seg["electrodes"]]))
                await asyncio.sleep(UPLOAD_WAIT_S)
            await self._ack("start raw recording", intan._start_raw_recording(list(range(S.N_CH)), tag, True), 30)
            recording = True
            dur, n_tr = seg["duration_s"], sum(e["kind"] != "code" for e in seg["events"])

            def show_rec(k=None):
                t = min(dur, time.monotonic() - start)
                status(f"recording {_bar(t / dur)} {t:.0f}/{dur:.0f} s" + (f", {k}/{n_tr} trains sent" if n_tr else ""),
                       live=True)
            status(f"recording {dur:.0f} s" + (f", {n_tr} trains" if n_tr else ", no stimulation"))
            if seg["events"]:
                trig = sdk_trigger.TriggerController(self.operator)
                # Send every 16-byte trigger at once. With TCP's default batching (Nagle), triggers
                # 30 ms apart were held back and delivered in bursts (fs666 curve, 30 Sep 2026).
                trig.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            start, n_sent = time.monotonic() + PRE_S, 0
            await asyncio.sleep(PRE_S)
            for ev in seg["events"]:
                await asyncio.sleep(max(0.0, start + ev["t_s"] - time.monotonic()))
                if not trig.__check_time__():
                    raise RuntimeError("Booking window ended")
                pattern = np.zeros(16, dtype=np.uint8)
                pattern[ev["key"]] = 1
                trig.sock.sendall(pattern.tobytes())
                sent.append(dict(key=ev["key"], train_id=ev["train_id"], pulse_index=ev["pulse_index"], utc=_utc()))
                if ev["kind"] != "code":
                    n_sent += 1
                    show_rec(n_sent)
            if seg["events"]:
                await asyncio.sleep(POST_S)
            while time.monotonic() < start + dur and not seg["events"]:   # census: show the clock
                show_rec()
                await asyncio.sleep(min(1.0, max(0.0, start + dur - time.monotonic())))
        finally:
            errors, notes, stop_ok, disable_ok = [], [], not recording, False
            if trig is not None:
                trig.close()
            if recording:
                # One request, short wait. On this platform the recording stops at once but the
                # request is often never acknowledged (every segment of the 30 Sep curves); retries
                # cost minutes and never changed the outcome. Completeness is judged from the file:
                # _wait_saved waits until it stops growing, analysis checks its duration. A recording
                # that never stopped would never stop growing, and _wait_saved fails loudly.
                status(f"stopping the recording (up to {STOP_ACK_S:.0f} s; usually unanswered, which is fine)…")
                try:
                    await self._ack("stop raw recording", intan._stop_raw_recording(), STOP_ACK_S)
                    stop_ok = True
                except Exception as e:  # noqa: BLE001
                    stop_ok = None
                    notes.append(f"stop recording not acknowledged ({e!r}); the saved file is checked instead")
            # Wait until the platform has saved the file (it appears, complete, on the S3 mount)
            # before the final disable, then send ONE disable with a long timeout. The platform
            # handles one request at a time: a disable sent while it was still saving went
            # unanswered (fs666 census, 1 Oct 2026), and repeating it then left the board in
            # "Invalid run mode" (fs666 curve, 1 Oct 2026). No stimulation can happen meanwhile:
            # the trigger connection is already closed.
            if recording:
                status("waiting for the platform to save the file…")
                try:
                    self.saved = await asyncio.to_thread(self._wait_saved, tag, SAVE_WAIT_S)
                except Exception as e:  # noqa: BLE001
                    notes.append(f"file not seen saved before the final disable ({e!r})")
            status("final disable…")
            try:
                await self._ack("disable all stimulation", intan._disable_all_stim(), DISABLE_ACK_S)
                disable_ok = True
            except Exception as e:  # noqa: BLE001
                errors.append(f"final disable: {e!r}")
            await intan._close()
            self.close_status = dict(stop_ok=stop_ok, disable_ok=disable_ok, messages=errors + notes)
            if errors:
                clear_status()
                self.log("        CLEAN-UP FAILED: " + " | ".join(errors))

    def run_segment(self, seg: dict, seg_dir: Path, tag: str) -> Path:
        sent: list = []
        self.close_status = dict(stop_ok=False, disable_ok=False, messages=["segment did not reach clean-up"])
        self.saved = None
        try:
            asyncio.run(self._segment(seg, tag, sent))
        finally:
            clear_status()
            pd.DataFrame(sent).to_csv(seg_dir / "sent.csv", index=False)
            (seg_dir / "close_status.json").write_text(json.dumps(self.close_status, indent=1))
        src = self.saved or self._wait_saved(tag)
        (seg_dir / "remote_path.txt").write_text(str(src) + "\n")
        return src

    @staticmethod
    def _wait_saved(tag, timeout_s=900) -> Path:
        """Wait until /mnt/s3-intanraws/*_<tag>.h5 exists and has stopped changing."""
        deadline, last, stable = time.monotonic() + timeout_s, None, 0
        while time.monotonic() < deadline:
            hits = list(RAW_ROOT.glob(f"*_{tag}.h5"))
            if len(hits) > 1:   # fs675, 18 Oct 2026: sweep and evaluation shared tags, and this waited 25 min
                raise RuntimeError(f"{len(hits)} recordings end in _{tag}.h5: " + ", ".join(sorted(h.name for h in hits)))
            if len(hits) == 1:
                st = hits[0].stat()
                sig = (st.st_size, st.st_mtime_ns)
                stable = stable + 1 if sig == last and st.st_size > 0 else 0
                last = sig
                if stable >= 2:
                    return hits[0]
            time.sleep(5)
        raise RuntimeError(f"Raw file for tag {tag} did not appear on {RAW_ROOT} within {timeout_s:.0f} s")
