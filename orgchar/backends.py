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


GAP_JUMP_S = 0.02   # a lost 1280-sample file row shifts later data by 42.7 ms


def _align(edges, evs, tol_s=0.035):
    """Pair recorded trigger edges with planned events, in order, when they are not identical.

    The platform can drop whole 1280-sample rows (42.7 ms) from the raw file: the triggers inside
    a lost row disappear and everything after it arrives 42.7 ms 'early' relative to the plan. So
    the plan-to-recording offset is tracked from match to match instead of assumed constant; a
    jump in that offset marks a recording gap between two consecutive matched triggers.
    Returns (pairs, gaps): pairs = [(edge_index, event_index)], gaps = [edge_index before a gap]."""
    rec = np.array([s / S.FS for s, _ in edges])
    plan = np.array([e["t_s"] for e in evs])
    # starting offset: the most common key-consistent offset among the first few edges and events
    cands = [round(rec[i] - plan[j], 2) for i in range(min(15, len(rec))) for j in range(min(15, len(plan)))
             if edges[i][1] == evs[j]["key"]]
    off = max(set(cands), key=cands.count) if cands else 0.0
    pairs, gaps, j = [], [], 0
    for i, (_, key) in enumerate(edges):
        for jj in range(j, min(j + 6, len(evs))):
            if evs[jj]["key"] != key:
                continue
            o = rec[i] - plan[jj]
            # recordings only lose data, so the offset can only shrink (by whole rows)
            if -3 * 0.0427 - tol_s < o - off < tol_s:
                if pairs and abs(o - off) > GAP_JUMP_S:
                    gaps.append(pairs[-1][0])
                pairs.append((i, jj))
                off, j = o, jj + 1
                break
    return pairs, gaps


def match_triggers(seg: dict, raw_path, min_fraction=0.9) -> pd.DataFrame:
    """One row per recorded trigger edge, with its planned event.

    Every recorded stimulation edge is kept (its artifact must be blanked even if it cannot be
    used); rows with exclude=True are blanked but not counted: edges not matched to the plan, and
    the pulse just before a recording gap (its response window may straddle the missing data)."""
    _, T = S.load_raw(raw_path)
    edges = S.trigger_edges(T)
    evs = seg["events"]
    els = {int(k): v for k, v in seg["electrodes"].items()}
    by_key = {v["key"]: e for e, v in els.items()}
    if [k for _, k in edges] == [e["key"] for e in evs]:
        pairs, gaps = [(i, i) for i in range(len(edges))], []
    else:
        pairs, gaps = _align(edges, evs)
    matched = dict(pairs)
    rows = []
    for i, (s, key) in enumerate(edges):
        ev = evs[matched[i]] if i in matched else None
        e = ev["stim_electrode"] if ev else by_key.get(key, -1)
        kind = ev["kind"] if ev else ("stim" if key in by_key else "sham")
        el = els.get(e, {}) if kind == "stim" else {}
        rows.append(dict(segment=seg["id"], onset=s, key=key, kind=kind, stim_electrode=e,
                         amplitude_uA=el.get("amplitude_uA", 0.0), phase_us=el.get("phase_us", 0.0),
                         polarity=el.get("polarity", ""), n_pulses=1, ipi_ms=0.0,
                         level=ev["level"] if ev else float("nan"),
                         train_id=ev["train_id"] if ev else f"unmatched{i}",
                         pulse_index=ev["pulse_index"] if ev else 0, planned_t_s=ev["t_s"] if ev else float("nan"),
                         exclude=(i not in matched) or (i in gaps),
                         note="unmatched" if i not in matched else ("before recording gap" if i in gaps else "")))
    df = pd.DataFrame(rows)
    usable = int((~df.exclude).sum()) if len(df) else 0
    if len(pairs) != len(evs) or gaps:
        print(f"  NOTE {seg['id']}: {len(evs)} triggers sent, {len(edges)} in the recording, {len(pairs)} matched; "
              f"{len(gaps)} recording gap(s) detected (platform dropped data); {usable} trials usable.")
    if usable < min_fraction * len(evs):
        raise RuntimeError(f"Only {usable}/{len(evs)} trials usable in {raw_path}; segment not analysed. Raw file kept.")
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
                        await asyncio.sleep(5)
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
