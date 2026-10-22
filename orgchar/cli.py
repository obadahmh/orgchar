"""orgchar: weekly characterisation of FinalSpark organoids.

    orgchar census --fsname fs652          # ~8 min, no stimulation: which organoids fire?
    orgchar screen --fsname fs652          # ~30 min: which pairs respond to the train length?
    orgchar solve  --fsname fs652          # ≤ 50 min: measure a pair's curve, train the network on it, run it on the tissue
    orgchar status [--fsname fs652]        # where each organoid stands (default: every organoid run this week)
    orgchar analyze <step dir>             # re-run the analysis of a finished step, offline
    orgchar codetest --fsname fs652        # ~3 min: check that train codes (--codes) are safe and readable

Add --dry-run to see the plan and its duration without touching hardware, --simulate to run the
whole step on a synthetic organoid.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import analysis as A
from . import screen as SC
from . import protocol as P
from . import report as R

HOME = Path(os.environ.get("ORGCHAR_HOME", Path(__file__).resolve().parents[1] / "sessions"))
RAW_ROOT = Path("/mnt/s3-intanraws")   # where the platform saves every recording
LOCKS = [Path.home() / ".orgchar.lock",
         Path.home() / "NeuralControlProgram/ResponseSweep/overnight_hardware.lock"]  # skipped if absent
SHAPE = dict(short="falling", band="peaked", long="rising")
# Results older than this are flagged and warned about: responses change within a day (fs666 e10's
# 10-pulse reference fell from 22 to 8.5 events within one hour, 2 Oct 2026). Each fsname is one
# preparation; FinalSpark replaces all four organoids at once, under a new fsname.
STALE_H = 24.0


def code_version() -> str:
    """Git commit of this orgchar checkout (+ '-dirty' if tracked files are modified), recorded in
    every plan and result so each report says which code produced it."""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    try:
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (rev or "unknown") + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - never block a run on this
        return "unknown"


def _now():
    return datetime.now(timezone.utc)


def _org_dir(fsname, simulate):
    return HOME / (f"SIMULATED_{fsname}" if simulate else fsname)


def _latest(fsname, simulate, step):
    d = _org_dir(fsname, simulate)
    runs = sorted(d.glob(f"*_{step}/result.json")) if d.exists() else []
    return (runs[-1].parent, json.loads(runs[-1].read_text())) if runs else (None, None)


def hardware_lock():
    """Hold every hardware lock (exits if another job holds one); FuncSolver live inference uses it too."""
    handles = []
    for p in LOCKS:
        if p.parent.exists():
            h = p.open("a")
            try:
                fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                sys.exit(f"Another hardware job holds {p}. Wait for it to finish.")
            handles.append(h)
    return handles


def _describe(plan):
    n_stim = sum(e["kind"] == "stim" for s in plan["segments"] for e in s["events"])
    n_sham = sum(e["kind"] == "sham" for s in plan["segments"] for e in s["events"])
    n_code = sum(e["kind"] == "code" for s in plan["segments"] for e in s["events"])
    print(f"Plan: {plan['step']}, {len(plan['segments'])} segment(s), {n_stim} stimulation pulses, "
          f"{n_sham} sham markers, " + (f"{n_code} code triggers (keys 8-14, no current), " if n_code else "")
          + f"about {P.plan_duration_s(plan) / 60:.1f} min.")
    for s in plan["segments"]:
        el = s["electrodes"]
        desc = ", ".join(f"e{e}:{v['amplitude_uA']}µA×{v['phase_us']:g}µs" for e, v in list(el.items())[:8])
        print(f"  {s['id']}: {s['duration_s']:.0f} s recording" + (f"; {desc}{' …' if len(el) > 8 else ''}" if el else ""))


def _run(a, plan: dict, fn):
    """Create the step directory and its plan.json, take the platform (or the simulated organoid) and
    return (step_dir, fn(backend, step_dir)). A failure is kept with the data as error.txt."""
    step_dir = _org_dir(a.fsname, a.simulate) / f"{_now():%Y%m%dT%H%M%S}_{plan['step']}"
    step_dir.mkdir(parents=True)
    plan = dict(plan, fsname=a.fsname, created_utc=_now().isoformat(), code_version=code_version(),
                backend="simulated" if a.simulate else "hardware", limits=P.LIMITS)
    (step_dir / "plan.json").write_text(json.dumps(plan, indent=1))
    if a.simulate:
        from .backends import SimBackend
        from .simulate import demo_organoid
        backend, locks = SimBackend(demo_organoid()), []
    else:
        from .backends import HardwareBackend
        backend, locks = HardwareBackend(a.operator), hardware_lock()
    print(f"\n{plan['step']}, {a.fsname}{' (SIMULATED)' if a.simulate else ''}, started {_local():%H:%M %Z}")
    try:
        return step_dir, fn(backend, step_dir)
    except BaseException:
        # keep the failure with the data, so an aborted step is never a silent empty folder
        (step_dir / "error.txt").write_text(f"{_now().isoformat()}\n{traceback.format_exc()}")
        print(f"\nStep failed; error saved to {step_dir / 'error.txt'}")
        raise
    finally:
        for h in locks:
            h.close()


def _segments(plan, backend, step_dir):
    """Run every segment of a census or codetest plan; returns [(raw, trials)]."""
    from .backends import clear_status, match_triggers, status
    n = len(plan["segments"])
    planned = [P.plan_duration_s(dict(segments=[s])) for s in plan["segments"]]
    t0 = time.monotonic()
    segs = []
    for i, s in enumerate(plan["segments"]):
        seg_dir = step_dir / s["id"]
        seg_dir.mkdir()
        t_seg = time.monotonic()
        print(f"[{i + 1:>{len(str(n))}}/{n}] {s['id']}: {_seg_text(plan, s)} ({_local():%H:%M:%S})", flush=True)
        raw = backend.run_segment(s, seg_dir, f"orgchar_{plan['fsname']}_{step_dir.name}_{s['id']}")
        status("checking the recording…", live=True)
        sent_csv = seg_dir / "sent.csv"
        sent = pd.read_csv(sent_csv) if sent_csv.exists() and sent_csv.stat().st_size > 1 else None
        trials = match_triggers(s, raw, sent)
        trials.to_csv(seg_dir / "trials.csv", index=False)
        segs.append((str(raw), trials))
        clear_status()
        done = time.monotonic() - t0
        left = sum(planned[i + 1:]) * done / sum(planned[:i + 1])   # the plan, scaled by the pace so far
        print(f"        {_seg_result(s, trials)}; took {_mmss(time.monotonic() - t_seg)}"
              + (f"; {i + 1}/{n} done, about {left / 60:.0f} min left (until ~{_local(left):%H:%M})" if i + 1 < n
                 else f"; {'all ' if n > 1 else ''}done in {_mmss(done)}"), flush=True)
    return segs


def _local(after_s=0.0):
    """Wall-clock time for the operator (the machine's own time zone)."""
    return datetime.now().astimezone() + timedelta(seconds=after_s)


def _mmss(sec):
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def _seg_text(plan, s):
    """What a segment does, in a few words."""
    el = sorted(int(e) for e in s["electrodes"])
    if not el:
        return "spontaneous activity, no stimulation"
    if plan["step"] == "codetest":
        return f"train codes on keys 8-14, then 5 control pulses on e{el[0]}"
    return f"single pulses on e{', e'.join(map(str, el))}"


def _seg_result(s, trials):
    """What the recording delivered: trains usable, lost rows, codes; ✓ if everything is there."""
    m = trials.attrs.get("summary", {})
    if not s["events"]:
        return f"✓ {m.get('recorded_s', 0):.0f} of {s['duration_s']:.0f} s recorded"
    parts = [f"{m['usable']}/{m['planned']} trains usable", f"{m['lost_rows']} lost row{'s' * (m['lost_rows'] != 1)}"]
    if m.get("unmatched"):
        parts.append(f"{m['unmatched']} unmatched")
    if m.get("codes_ok") is not None:
        parts.append(f"codes {m['codes_ok']}/{m['planned']}" + (f", {m['codes_bad']} CONTRADICT" if m["codes_bad"] else ""))
    ok = m["usable"] == m["planned"] and not m.get("codes_bad")
    return ("✓ " if ok else "! ") + ", ".join(parts)


def _segments_from_dir(step_dir: Path, plan: dict):
    """Segments with a recording. Trigger matching is redone from the raw file when trials.csv is
    missing (e.g. the step stopped at matching) so salvageable data can still be analysed."""
    from .backends import match_triggers, recording
    segs = []
    for s in plan["segments"]:
        sd = step_dir / s["id"]
        raw = recording(sd)
        if raw is None:
            continue
        t = sd / "trials.csv"
        if t.exists() and t.stat().st_size > 1:
            trials = pd.read_csv(t)
        elif s["events"]:
            sent = pd.read_csv(sd / "sent.csv") if (sd / "sent.csv").exists() and (sd / "sent.csv").stat().st_size > 1 else None
            trials = match_triggers(s, raw, sent)
            trials.to_csv(t, index=False)
        else:
            trials = pd.DataFrame()
        segs.append((str(raw), trials))
    return segs


# ------------------------------------------------------------------------------ analysis + report

def _census(step_dir: Path, plan: dict, segs) -> dict:
    # reading from the S3 mount dominates: ~13 s per recording
    print(f"\nAnalysing the recording (until ~{_local(13):%H:%M})…", flush=True)
    return A.census(segs[0][0])


def _conclude(step_dir: Path, plan: dict, r: dict) -> dict:
    """Verdict, result.json and report.html of a census or screen."""
    if plan["step"] == "census":
        lines = [f"Recorded {r['duration_s']:.0f} s without stimulation; spike threshold -{r['threshold_k']:g} × noise.",
                 f"Live organoids (≥1 electrode with real spikes): blocks {r['live_blocks'] or 'none'}."]
        for b in r["blocks"]:
            act = [c for c in r["channels"] if c["block"] == b["block"] and c["active"]]
            lines.append(f"Block {b['block']} (e{b['block'] * 8}-{b['block'] * 8 + 7}): {b['verdict']}"
                         + (": " + ", ".join(f"e{c['channel']} {c['rate_hz']:.2f} Hz" for c in act) if act else ""))
        verdict = "GO: screen blocks " + ",".join(map(str, r["live_blocks"])) if r["live_blocks"] else \
            "NO-GO: no organoid is firing. Stimulation results would be artifact only. Re-check later or contact FinalSpark."
        problems = _completeness_problems(step_dir, plan, r)
    else:
        lines = [f"Trains of {', '.join(map(str, plan['levels']))} pulses on e{', e'.join(map(str, plan['stims']))}; "
                 f"{r['n_trials']} of {r['n_requests']} trains usable; evoked spikes 10-210 ms after the train."]
        for role in SC.ROLES:
            v = r["best"].get(role)
            lines.append(f"{SHAPE[role]}: " + (f"e{v['stim']}→e{v['rec']} (train length explains {v['explained']:.0%} of the "
                                               f"trial-to-trial variation, {v['diff']:.1f} events apart; "
                                               f"{v['n_candidates']} candidate pair(s))"
                                               if v else "no candidate"))
        for x in r["chance"]:
            lines.append(f"Dropped e{x['stim']}→e{x['rec']} {SHAPE[x['role']]}: e{x['beaten_by']}→e{x['rec']}, from "
                         "another organoid, did as well.")
        c = r["controls"]
        lines.append(f"Controls: {c['cross_organoid_passing']} of {c['cross_organoid_pairs']} other-organoid pairs pass "
                     f"(p < {SC.SCOUT_MAX_P:g} alone would let up to about {c['nominal_ceiling']:g} through).")
        lines += [f"WARNING: {w}" for w in r["warnings"]]
        if r["best"]:
            v = SC.steadiest(r["best"])
            lines.append(f"Next: ./orgchar.sh solve --fsname {plan['fsname']}   (the steadiest pair, e{v['stim']}→e{v['rec']})")
        verdict, problems = r["verdict"], []
    r["incomplete"] = bool(problems)
    if problems:
        verdict = ("INCOMPLETE: " + "; ".join(problems) + ". Results are for inspection only; re-run this step. "
                   f"(Would otherwise have been: {verdict})")
    r["verdict_text"] = verdict
    r["code_version"] = code_version()
    lines.append(f"Acquired with orgchar {plan.get('code_version', 'unknown (before versioning)')}; "
                 f"analysed with {r['code_version']}.")
    (step_dir / "result.json").write_text(R.to_json(r))
    if plan["step"] == "census":
        title = f"{plan['fsname']}: census, {_when(step_dir)}" + (" (SIMULATED)" if plan.get("backend") == "simulated" else "")
        R.write_html(step_dir / "report.html", title, verdict, lines, R.census_figures(r),
                     pd.DataFrame(r["channels"]).drop(columns=["waveform"]))
    else:
        from . import report_screen as RS
        RS.screen_page(step_dir / "report.html", plan, r, _when(step_dir))
    print("\n" + verdict)
    for s in lines:
        print("  " + s)
    print(f"\nReport: {step_dir / 'report.html'}")
    return r


def _stamp(name: str) -> datetime | None:
    try:
        return datetime.strptime(name[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _when(step_dir: Path) -> str:
    t = _stamp(step_dir.name)
    return t.strftime("%d %b %Y %H:%M UTC") if t else step_dir.name


def _close_status(seg_dir: Path) -> dict | None:
    """How the segment's recording was closed (close_status.json, written by the hardware backend)."""
    f = seg_dir / "close_status.json"
    return json.loads(f.read_text()) if f.exists() else None


def _completeness_problems(step_dir: Path, plan: dict, r: dict) -> list[str]:
    """A step whose recording was cut short or hit platform errors must not produce a GO."""
    problems = []
    from .backends import recording
    for s in plan["segments"]:
        st = _close_status(step_dir / s["id"])
        raw = recording(step_dir / s["id"])
        # a segment that uploaded no stimulation (census) has nothing to disable: an unanswered
        # final disable there is not a problem (it starts with an acknowledged disable)
        if st and not st["disable_ok"] and s["electrodes"]:
            problems.append(f"final 'disable all stimulation' failed for segment {s['id']}")
        if raw is None:
            problems.append(f"segment {s['id']} has no recording")
            continue
        # a stop without acknowledgement is fine if the file holds the whole segment
        with h5py.File(raw, "r") as f:
            got_s = f["triggers"].size / 30000.0
        if got_s < 0.9 * s["duration_s"]:
            problems.append(f"segment {s['id']} recorded {got_s:.0f} of {s['duration_s']:.0f} s")
    return problems


# ------------------------------------------------------------------------------ status

def _age_h(step_dir: Path) -> float:
    """Hours since the step was recorded (from its folder name: re-analysing it does not reset this)."""
    return (_now() - _stamp(step_dir.name)).total_seconds() / 3600


def _ago(step_dir: Path) -> str:
    h = _age_h(step_dir)
    return (f"{h:.0f} h ago" if h < 48 else f"{h / 24:.0f} days ago") + (", STALE" if h > STALE_H else "")


def _runs_on_platform() -> dict[str, set[str]]:
    """{fsname: {'<stamp>_<step>', ...}} for every orgchar recording on the S3 mount, whichever
    machine ran it."""
    out = {}
    try:
        names = os.listdir(RAW_ROOT)
    except OSError:
        return out
    for n in names:
        m = re.match(r"\d{8}_\d{6}_orgchar_(.+?)_(\d{8}T\d{6})_(census|screen|curve|solve|codetest)_", n)
        if m:
            out.setdefault(m[1], set()).add(f"{m[2]}_{m[3]}")
    return out


def _status(fs: str, simulate: bool, elsewhere: set[str]) -> list[str]:
    """Where one organoid stands: the latest complete census, screen and solve, and runs whose results
    are elsewhere."""
    d = _org_dir(fs, simulate)
    runs = sorted(p for p in d.glob("*_*") if p.is_dir() and _stamp(p.name)) if d.exists() else []
    res = {p: json.loads((p / "result.json").read_text()) for p in runs if (p / "result.json").exists()}
    done = [p for p in runs if p in res and not res[p].get("incomplete")]
    out = [fs + (" (SIMULATED)" if simulate else "")]

    def head(label, p, extra=""):
        out.append(f"  {label:7s} {_when(p)} ({_ago(p)}){extra}")

    census = [p for p in done if p.name.endswith("_census")]
    if census:
        c = res[census[-1]]
        head("census", census[-1])
        orgs = []
        for b in c["blocks"]:
            act = [x for x in c["channels"] if x["block"] == b["block"] and x["active"]]
            orgs.append(f"e{8 * b['block']}-{8 * b['block'] + 7}: " + (
                f"{len(act)} active, {sum(x['rate_hz'] for x in act):.1f} Hz" if act else "silent"))
        out.append("          " + "; ".join(orgs))
    screens = [p for p in done if p.name.endswith("_screen") and res[p].get("mode") == "trains"]
    if screens:
        s = res[screens[-1]]
        head("screen", screens[-1])
        parts = []
        for role in SC.ROLES:
            v = s["best"].get(role)
            dropped = ", ".join(f"e{x['stim']}→e{x['rec']}" for x in s.get("chance", []) if x["role"] == role)
            n = v["n_candidates"] if v else 0
            parts.append(f"{SHAPE[role]} " + (f"e{v['stim']}→e{v['rec']} ({n} pair{'s' * (n != 1)})" if v else "none")
                         + (f" ({dropped} dropped as chance)" if dropped else ""))
        out.append("          " + "; ".join(parts)
                   + ("" if "chance" in s else f"  [analysed before the chance check: ./orgchar.sh analyze "
                                               f"{os.path.relpath(screens[-1], Path(__file__).resolve().parents[1])}]"))
    solves = [p for p in runs if p.name.endswith("_solve") and p in res]
    if solves:
        head("solve", solves[-1], f": {res[solves[-1]].get('verdict', '?')}")
    if not (census or screens or solves):
        out.append("  no complete census, screen or solve on this machine")
    if runs:
        last = runs[-1]
        v = res[last].get("verdict_text") or res[last].get("verdict", "?") if last in res else (
            "stopped before analysis, see error.txt" if (last / "error.txt").exists() else "no result")
        out.append(f"  last run: {last.name}: {v[:110]}{'…' if len(v) > 110 else ''}")
    other = sorted(elsewhere - {p.name for p in runs})
    if other:
        out.append(f"  {len(other)} run(s) recorded by another machine (results are there), last {other[-1]}")
    return out


def cmd_status(a):
    remote = {} if a.simulate else _runs_on_platform()
    if a.fsname:
        names = [a.fsname]
    else:
        week = (_now() - timedelta(days=7)).strftime("%Y%m%dT%H%M%S")
        prefix = "SIMULATED_" if a.simulate else ""
        local = {d.name[len(prefix):] for d in HOME.glob(f"{prefix}*") if d.is_dir()
                 and (a.simulate or not d.name.startswith("SIMULATED_"))
                 and any(p.name >= week for p in d.glob("*_*") if p.is_dir())}
        names = sorted(local | {fs for fs, runs in remote.items() if max(runs) >= week})
        if not names:
            print("No organoid was run in the last 7 days. Use --fsname for an older one.")
    for fs in names:
        print("\n".join(_status(fs, a.simulate, remote.get(fs, set()))) + "\n")


# ------------------------------------------------------------------------------ commands

def cmd_census(a):
    plan = P.plan_census(a.duration)
    _describe(plan)
    if a.dry_run or not _confirm(a):
        return
    step_dir, segs = _run(a, plan, lambda backend, step_dir: _segments(dict(plan, fsname=a.fsname), backend, step_dir))
    plan = json.loads((step_dir / "plan.json").read_text())
    _conclude(step_dir, plan, _census(step_dir, plan, segs))


def cmd_screen(a):
    blocks = [int(x) for x in a.blocks.split(",")] if a.blocks else None
    electrodes = [int(x) for x in a.electrodes.split(",")] if a.electrodes else None
    if not blocks and not electrodes:
        d, c = _latest(a.fsname, a.simulate, "census")
        if c is None:
            sys.exit("No census for this organoid yet. Run `orgchar census` first, or pass --blocks.")
        age_h = _age_h(d)
        if c.get("incomplete"):
            sys.exit(f"Latest census ({d.name}) is INCOMPLETE ({c.get('verdict_text', '')[:120]}...). "
                     "Re-run the census, or pass --blocks explicitly.")
        if not c["live_blocks"]:
            sys.exit(f"Latest census ({d.name}) found no firing organoid. Stimulating now would only measure "
                     "artifact. Re-run census later, or pass --blocks explicitly to override.")
        blocks = c["live_blocks"]
        print(f"Using live blocks {blocks} from census {d.name} ({age_h:.1f} h old).")
        if age_h > STALE_H:
            print(f"  Warning: census is more than {STALE_H:.0f} h old; consider re-running it.")
    if a.reps < 5:
        sys.exit("--reps must be at least 5: with 4 trials against 4 the rank test cannot reach p < 0.01.")
    stims = sorted(electrodes) if electrodes else [e for b in sorted(blocks) for e in range(8 * b, 8 * b + 8)]
    counts = sorted(int(x) for x in a.levels.split(",")) if a.levels else list(P.SCREEN_COUNTS)
    cond = dict(ipi_ms=10.0, amplitude_uA=a.amplitude, phase_us=a.phase, polarity=a.polarity)
    requests = [(e, n) for e in stims for n in counts for _ in range(a.reps)]
    est = P.plan_trials(requests, **cond)
    print(f"Plan: screen, {len(requests)} trains ({sum(n for _, n in requests)} pulses) in about {len(est['segments'])} "
          f"recordings, about {P.plan_duration_s(est) / 60:.0f} min.\n  stimulating e{', e'.join(map(str, stims))}; "
          f"{a.reps} trains of {', '.join(map(str, counts))} pulses each, {cond['ipi_ms']:g} ms apart, "
          f"{a.amplitude:g} µA × {a.phase:g} µs; each organoid ≥ 4 s between trains")
    if a.dry_run or not _confirm(a):
        return
    codes = _use_codes(a)
    step_dir, r = _run(a, dict(step="screen", mode="trains", levels=counts, stims=stims, reps=a.reps, codes=codes, **cond),
                       lambda backend, step_dir: SC.run(a.fsname, stims, counts, a.reps, backend, step_dir, codes, cond,
                                                        seed=int(time.time())))
    _conclude(step_dir, json.loads((step_dir / "plan.json").read_text()), r)


def _use_codes(a) -> bool:
    """Train codes are on by default once codetest has passed on this machine's platform setup
    (any organoid: it tests the trigger system, not the tissue). --codes / --no-codes override."""
    if a.codes is not None:
        return a.codes
    if a.simulate:
        return True
    passed = [r for r in sorted(HOME.glob("*/*_codetest/result.json"))
              if not r.parts[-3].startswith("SIMULATED_") and json.loads(r.read_text()).get("pass")]
    if passed:
        print(f"Train codes on (codetest passed: {passed[-1].parent.parent.name}/{passed[-1].parent.name}).")
        return True
    print("Train codes off: codetest has not passed on this machine yet (./orgchar.sh codetest --fsname ...).")
    return False


def cmd_codetest(a):
    if a.electrodes:
        stims = [int(x) for x in a.electrodes.split(",")]
    else:
        d, s = _latest(a.fsname, a.simulate, "screen")
        stims = list(dict.fromkeys(v["stim"] for v in (s or {}).get("best", {}).values()))
        if not stims:   # no screen yet (e.g. a new organoid): one electrode per organoid
            stims = [2, 10, 18, 26]
            print("No screen of this organoid yet; using e2, e10, e18, e26 (one per organoid). The codetest checks "
                  "the trigger system, not the tissue, so the choice does not matter.")
        print(f"Uploading the parameters of e{', e'.join(map(str, stims))} (as a run would), triggering none of them "
              "except 5 positive-control pulses on the first at the end.")
    plan = P.plan_codetest(stims, amplitude_uA=a.amplitude, phase_us=a.phase, seed=int(time.time()))
    _describe(plan)
    if a.dry_run or not _confirm(a):
        return
    step_dir, segs = _run(a, plan, lambda backend, step_dir: _segments(dict(plan, fsname=a.fsname), backend, step_dir))
    r = _codetest_result(plan, segs)
    r["verdict_text"] = ("PASS: train codes are safe and readable; screen and solve now use them by default on this machine"
                         if r["pass"] else "FAIL: do not use --codes. " + "; ".join(r["failures"]))
    (step_dir / "result.json").write_text(R.to_json(r))
    print("\n" + r["verdict_text"])
    for line in r["lines"]:
        print("  " + line)


def _codetest_result(plan, segs):
    """Judge the codetest recording: every code trigger recorded on its key, codes readable, no
    artifact after code triggers, and a clear artifact after the positive-control pulses."""
    from . import signals as S
    from .backends import code_artifact_uV
    seg, (raw, trials) = plan["segments"][0], segs[0]
    X, T = S.load_raw(raw)
    edges = S.trigger_edges(T)
    sent_codes = [e for e in seg["events"] if e["kind"] == "code"]
    rec_codes = [(s, k) for s, k in edges if k in P.CODE_KEYS]
    per_key_sent = pd.Series([e["key"] for e in sent_codes]).value_counts().sort_index()
    per_key_rec = pd.Series([k for _, k in rec_codes], dtype=int).value_counts().reindex(per_key_sent.index, fill_value=0)
    gaps_ms = [(b[0] - a[0]) / S.FS * 1000 for a, b in zip(rec_codes, rec_codes[1:]) if (b[0] - a[0]) / S.FS < 0.05]
    sham = trials[trials.kind == "sham"]
    n_ok = int((sham.code == "ok").sum())
    n_bad = int((sham.code == "mismatch").sum())
    art_code = code_artifact_uV(X, [s for s, _ in rec_codes])
    art_pos = code_artifact_uV(X, trials[trials.kind == "stim"].onset.tolist())
    failures = []
    if per_key_rec.sum() < len(sent_codes) - 4:      # a lost row can take a code or two
        failures.append(f"only {per_key_rec.sum()} of {len(sent_codes)} code triggers recorded")
    if n_bad:
        failures.append(f"{n_bad} codes read back wrong")
    if n_ok < 0.9 * len(sham):
        failures.append(f"only {n_ok} of {len(sham)} codes readable")
    if not art_code < 50:
        failures.append(f"{art_code:.0f} µV after code triggers (must be < 50)")
    if not art_pos > 500:
        failures.append(f"positive control artifact only {art_pos:.0f} µV: the artifact check could not be validated")
    lines = [f"code triggers recorded: {per_key_rec.sum()} of {len(sent_codes)} "
             f"(per key {', '.join(f'{k}: {per_key_rec[k]}/{per_key_sent[k]}' for k in per_key_sent.index)})",
             f"codes read back: {n_ok} correct, {n_bad} wrong, {len(sham) - n_ok - n_bad} unreadable, of {len(sham)}",
             "recorded spacing of the two code triggers: " + (f"{np.median(gaps_ms):.1f} ms median "
                                                              f"({min(gaps_ms):.1f}-{max(gaps_ms):.1f}); sent 10 ms" if gaps_ms else "n/a"),
             f"largest signal change after code triggers: {art_code:.1f} µV (noise level; limit 50)",
             f"after the {int((trials.kind == 'stim').sum())} positive-control pulses: {art_pos:.0f} µV"]
    return dict(step="codetest", **{"pass": not failures}, failures=failures, lines=lines,
                code_artifact_uV=art_code, positive_control_uV=art_pos, codes_ok=n_ok, codes_wrong=n_bad,
                codes_total=len(sham), code_edges_recorded=int(per_key_rec.sum()), code_edges_sent=len(sent_codes))


def cmd_analyze(a):
    step_dir = Path(a.step_dir).resolve()
    plan = json.loads((step_dir / "plan.json").read_text())
    if plan["step"] == "census":
        r = _census(step_dir, plan, _segments_from_dir(step_dir, plan))
    elif plan["step"] == "screen" and (step_dir / "pass").is_dir():
        r = SC.reanalyse(step_dir, plan)
    elif plan["step"] == "solve" and (step_dir / "result.json").exists():
        from . import solve as SV
        res = SV.refresh(step_dir)
        (step_dir / "result.json").write_text(json.dumps(res, indent=1, default=float))
        R.solve_page(step_dir / "report.html", res, _when(step_dir))
        R.runs_page(step_dir.parent.parent)
        print(f"{res['verdict']}\nReport: {step_dir / 'report.html'}\nAll runs: {step_dir.parent.parent / 'runs.html'}")
        return
    else:
        sys.exit(f"{step_dir.name}: only a census, a finished solve or a screen recorded since 16 Oct 2026 can be re-analysed "
                 "(older screens had sham trials; the curve step was replaced by solve). Its report.html and "
                 "result.json stay as they were.")
    _conclude(step_dir, plan, r)


def cmd_disable(a):
    """Send only 'disable all stimulation' (the platform uploads internally). Safe; also a quick test
    of whether the board currently accepts uploads."""
    import asyncio

    async def run():
        from neuroplatformv2.core.intan import IntanController
        intan = IntanController()
        try:
            r = await asyncio.wait_for(intan._disable_all_stim(), 60)
            print(f"disable all stimulation: status={r.status} message={r.message!r}")
        finally:
            await intan._close()
    locks = hardware_lock()
    try:
        asyncio.run(run())
    finally:
        for h in locks:
            h.close()


def cmd_solve(a):
    from . import solve as SV
    if a.pairs:
        pairs = [tuple(int(v) for v in p.split(":")) for p in a.pairs.split(",")]
    else:
        d, s = _latest(a.fsname, a.simulate, "screen")
        if s is None or s.get("mode") != "trains" or s.get("incomplete"):
            sys.exit("No complete screen for this organoid. Run `orgchar screen` first, or pass --pairs stim:rec.")
        if not s["best"]:
            sys.exit(f"The latest screen ({d.name}) found no candidate pair. Pass --pairs stim:rec to try one anyway.")
        # one pair: one response type is enough, and each pair adds ~80 sweep trains and its own pulse counts
        # (sim, 16 Oct 2026: three pairs, 558 trains, expected R² 0.59; one pair, ~200 trains, 0.68)
        v = SC.steadiest(s["best"])
        pairs = [(v["stim"], v["rec"])]
        print(f"Pair from screen {d.name}: e{v['stim']}→e{v['rec']}, the steadiest response ("
              + (f"train length explains {v['explained']:.0%} of the trial-to-trial variation, " if "explained" in v else "")
              + f"{v['diff']:.1f} spikes apart). --pairs to choose others.")
        if _age_h(d) > STALE_H:
            print(f"  Warning: the screen is {_age_h(d):.0f} h old; responses change within a day.")
    stims = sorted({x for x, _ in pairs})
    sweep = P.plan_trials([(x, n) for x in stims for n in SV.SWEEP_COUNTS for _ in range(a.sweep_reps)])
    evaluate = P.plan_trials([(x, n) for x in stims for n in range(2, 21) for _ in range(a.reps)])   # at most
    from . import net as N
    mins = (P.plan_duration_s(sweep) + P.plan_duration_s(evaluate)) / 60 + N.train_minutes()
    print(f"Plan: sweep {a.sweep_reps} trains at {len(SV.SWEEP_COUNTS)} pulse counts on e{', e'.join(map(str, stims))}, "
          f"train ({a.width} units), then at most {len(range(2, 21))} pulse counts x {a.reps} trains: "
          f"at most {mins:.0f} min. Each step stops the run if its check fails.")
    if a.dry_run or not _confirm(a):
        return
    codes = _use_codes(a)
    step_dir, res = _run(a, dict(step="solve", pairs=pairs, sweep_reps=a.sweep_reps, reps=a.reps, width=a.width,
                                 points=a.points, codes=codes),
                         lambda backend, step_dir: SV.run(a.fsname, pairs, backend, step_dir, codes,
                                                          sweep_reps=a.sweep_reps, eval_reps=a.reps, width=a.width,
                                                          points=a.points))
    res.update(code_version=code_version(), simulated=bool(a.simulate))
    (step_dir / "result.json").write_text(json.dumps(res, indent=1, default=float))
    R.solve_page(step_dir / "report.html", res, _when(step_dir))
    R.runs_page(step_dir.parent.parent)
    print(f"\n{res['verdict']}" + (f"\nLargest change since the sweep: {res['drift_max']:.0%} of the curve's range."
                                   if res.get("drift_max") is not None else "") + f"\nReport: {step_dir / 'report.html'}\nAll runs: {step_dir.parent.parent / 'runs.html'}")


def _confirm(a):
    if a.simulate or a.yes:
        return True
    ans = input(f"Run this on the REAL platform for {a.fsname}? Type the fsname to confirm: ").strip()
    if ans != a.fsname:
        print("Not confirmed; nothing sent.")
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orgchar", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--fsname", required=True)
        p.add_argument("--simulate", action="store_true", help="synthetic organoid, no hardware")
        p.add_argument("--dry-run", action="store_true", help="print the plan only")
        p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
        p.add_argument("--operator", default="admin", help="booking e-mail for the trigger controller")

    def stimulus(p):
        p.add_argument("--amplitude", type=float, default=2.0)
        p.add_argument("--phase", type=float, default=100.0)
        p.add_argument("--polarity", default="negative-first", choices=["negative-first", "positive-first"])
        p.add_argument("--codes", action=argparse.BooleanOptionalAction, default=None,
                       help="train codes (keys 8-14) so the recording identifies each train "
                            "(default: on once codetest has passed on this machine)")

    p = sub.add_parser("census", help="spontaneous activity, no stimulation")
    common(p)
    p.add_argument("--duration", type=float, default=300.0)
    p.set_defaults(func=cmd_census)

    p = sub.add_parser("screen", help="~30 min: which pairs respond to the train length (2, 10, 20 pulses)")
    common(p)
    stimulus(p)
    p.add_argument("--levels", help=f"train lengths, default {','.join(map(str, P.SCREEN_COUNTS))}")
    p.add_argument("--blocks", help="e.g. 1,2 (default: live blocks from latest census)")
    p.add_argument("--electrodes", help="subset, e.g. 21,15 (quick re-check before a session)")
    p.add_argument("--reps", type=int, default=6, help="trials per electrode and length (at least 5: with 4 "
                                                       "against 4 the rank test cannot reach p < 0.01)")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("solve", help="≤ 50 min: measure the pair's curve now, train the network on it, run it on the tissue")
    common(p)
    p.add_argument("--pairs", help="stim:rec,... (default: the latest screen's steadiest pair)")
    p.add_argument("--sweep-reps", type=int, default=8, help="trains per pulse count when measuring the curve")
    p.add_argument("--reps", type=int, default=6, help="trains per pulse count when running the network")
    p.add_argument("--width", type=int, default=12, help="hidden units")
    p.add_argument("--points", type=int, default=26, help="time points (the network is evaluated between them)")
    p.add_argument("--codes", action=argparse.BooleanOptionalAction, default=None,
                   help="train codes (default: on once codetest has passed on this machine)")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("codetest", help="~3 min: check that train codes are safe and readable")
    common(p)
    p.add_argument("--electrodes", help="electrodes whose parameters to upload (default: the latest screen's best electrodes)")
    p.add_argument("--amplitude", type=float, default=2.0)
    p.add_argument("--phase", type=float, default=100.0)
    p.set_defaults(func=cmd_codetest)

    p = sub.add_parser("analyze", help="re-run analysis + report of a step directory")
    p.add_argument("step_dir")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("disable", help="disable all stimulation now (also tests that uploads work)")
    p.set_defaults(func=cmd_disable)

    p = sub.add_parser("status", help="where each organoid stands (default: every organoid run in the last 7 days)")
    p.add_argument("--fsname")
    p.add_argument("--simulate", action="store_true", help="the simulated organoids instead")
    p.set_defaults(func=cmd_status)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
