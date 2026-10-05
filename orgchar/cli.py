"""orgchar: weekly characterisation of FinalSpark organoids.

    orgchar census --fsname fs652          # ~8 min, no stimulation: which organoids fire?
    orgchar screen --fsname fs652          # ~40 min: which electrodes give FuncSolver's shapes after trains?
    orgchar curve  --fsname fs652          # 40-65 min: FuncSolver curves (2-20 pulse trains) -> calibration.json
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
from . import funcsolver as F
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


def _lock():
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
    if plan.get("vary") == "pulse_train":
        n_pulses = sum(int(s["n_pulses"]) for s in plan["segments"] for e in s["events"] if e["kind"] == "stim")
        what = "after-train screen" if plan["step"] == "screen" else "FuncSolver curve"
        print(f"Plan: {what}, {len(plan['segments'])} segments, {n_stim} trains "
              f"({n_pulses} pulses), {n_sham} sham trials, "
              + (f"{n_code} code triggers (keys 8-14, no current), " if n_code else "")
              + f"about {P.plan_duration_s(plan) / 60:.0f} min.")
        print(f"  stimulating e{', e'.join(str(e) for e in plan['stims'])}; trains of {', '.join(map(str, plan['levels']))} "
              f"pulses, {plan['ipi_ms']:g} ms apart, {plan['amplitude_uA']} µA × {plan['phase_us']:g} µs; "
              f"each organoid ≥ {plan['organoid_isi_s'][0]:g} s between trains")
        if plan["step"] == "curve":
            print(f"  order: reference, development ({len(plan['levels'])} segments), reference, "
                  f"confirmation ({len(plan['levels'])} segments), reference")
        return
    print(f"Plan: {plan['step']}, {len(plan['segments'])} segment(s), {n_stim} stimulation pulses, "
          f"{n_sham} sham markers, " + (f"{n_code} code triggers (keys 8-14, no current), " if n_code else "")
          + f"about {P.plan_duration_s(plan) / 60:.1f} min.")
    for s in plan["segments"]:
        el = s["electrodes"]
        desc = ", ".join(f"e{e}:{v['amplitude_uA']}µA×{v['phase_us']:g}µs" for e, v in list(el.items())[:8])
        print(f"  {s['id']}: {s['duration_s']:.0f} s recording" + (f"; {desc}{' …' if len(el) > 8 else ''}" if el else ""))


def _execute(args, plan):
    """Create the step directory, run every segment, return (step_dir, segments)."""
    from .backends import match_triggers
    stamp = _now().strftime("%Y%m%dT%H%M%S")
    step_dir = _org_dir(args.fsname, args.simulate) / f"{stamp}_{plan['step']}"
    step_dir.mkdir(parents=True)
    plan = dict(plan, fsname=args.fsname, created_utc=_now().isoformat(), code_version=code_version(),
                backend="simulated" if args.simulate else "hardware", limits=P.LIMITS)
    (step_dir / "plan.json").write_text(json.dumps(plan, indent=1))
    if args.simulate:
        from .backends import SimBackend
        from .simulate import demo_organoid
        backend = SimBackend(demo_organoid())
        locks = []
    else:
        from .backends import HardwareBackend
        backend = HardwareBackend(args.operator)
        locks = _lock()
    try:
        segs = []
        for s in plan["segments"]:
            seg_dir = step_dir / s["id"]
            seg_dir.mkdir()
            tag = f"orgchar_{args.fsname}_{stamp}_{plan['step']}_{s['id']}"
            print(f"[{_now():%H:%M:%S}] segment {s['id']}")
            raw = backend.run_segment(s, seg_dir, tag)
            sent_csv = seg_dir / "sent.csv"
            sent = pd.read_csv(sent_csv) if sent_csv.exists() and sent_csv.stat().st_size > 1 else None
            trials = match_triggers(s, raw, sent)
            trials.to_csv(seg_dir / "trials.csv", index=False)
            segs.append((str(raw), trials))
    except BaseException:
        # keep the failure with the data, so an aborted step is never a silent empty folder
        (step_dir / "error.txt").write_text(f"{_now().isoformat()}\n{traceback.format_exc()}")
        print(f"\nStep failed; error saved to {step_dir / 'error.txt'}")
        raise
    finally:
        for h in locks:
            h.close()
    return step_dir, segs


def _raw(seg_dir: Path) -> Path | None:
    """The segment's recording: raw.h5 in the folder (simulations; hardware runs before 5 Oct 2026
    copied it), else the file the platform saved on the S3 mount (remote_path.txt)."""
    if (seg_dir / "raw.h5").exists():
        return seg_dir / "raw.h5"
    if not (seg_dir / "remote_path.txt").exists():
        return None
    p = Path((seg_dir / "remote_path.txt").read_text().strip())
    if not p.exists():
        sys.exit(f"Recording {p} is not reachable. Is {RAW_ROOT} mounted?")
    return p


def _segments_from_dir(step_dir: Path, plan: dict):
    """Segments with a recording. Trigger matching is redone from the raw file when trials.csv is
    missing (e.g. the step stopped at matching) so salvageable data can still be analysed."""
    from .backends import match_triggers
    segs = []
    for s in plan["segments"]:
        sd = step_dir / s["id"]
        raw = _raw(sd)
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

def analyse(step_dir: Path, plan: dict, segs) -> dict:
    step = plan["step"]
    if step == "census":
        r = A.census(segs[0][0])
        lines = [f"Recorded {r['duration_s']:.0f} s without stimulation; spike threshold -{r['threshold_k']:g} × noise.",
                 f"Live organoids (≥1 electrode with real spikes): blocks {r['live_blocks'] or 'none'}."]
        for b in r["blocks"]:
            act = [c for c in r["channels"] if c["block"] == b["block"] and c["active"]]
            lines.append(f"Block {b['block']} (e{b['block'] * 8}-{b['block'] * 8 + 7}): {b['verdict']}"
                         + (": " + ", ".join(f"e{c['channel']} {c['rate_hz']:.2f} Hz" for c in act) if act else ""))
        verdict = "GO: screen blocks " + ",".join(map(str, r["live_blocks"])) if r["live_blocks"] else \
            "NO-GO: no organoid is firing. Stimulation results would be artifact only. Re-check later or contact FinalSpark."
    elif step == "screen" and plan.get("mode") == "trains":
        r = F.scout(segs, plan)
        lines = [f"Trains of {', '.join(map(str, plan['levels']))} pulses on e{', e'.join(map(str, plan['stims']))}; "
                 "FuncSolver's readout (events 10-210 ms after the train)."]
        for role in F.ROLES:
            v = r["best"].get(role)
            lines.append(f"{SHAPE[role]}: " + (f"e{v['stim']}→e{v['rec']} (a single trial wins {v['win']:.0%} of the time, "
                                               f"{v['diff']:.1f} events apart; {v['n_candidates']} candidate pair(s))"
                                               if v else "no candidate"))
        for x in r["chance"]:
            lines.append(f"Dropped e{x['stim']}→e{x['rec']} {SHAPE[x['role']]}: e{x['beaten_by']}→e{x['rec']}, from "
                         "another organoid, did as well.")
        c = r["controls"]
        lines.append(f"Controls: {c['cross_organoid_passing']} of {c['cross_organoid_pairs']} other-organoid pairs pass "
                     f"(p < {F.SCOUT_MAX_P:g} alone would let up to about {c['nominal_ceiling']:g} through).")
        if r["curve_stims"]:
            lines.append(f"Next: ./orgchar.sh curve --fsname {plan['fsname']}   (stimulates "
                         f"e{', e'.join(map(str, r['curve_stims']))})")
        verdict = r["verdict"]
    elif step == "curve" and plan.get("vary") == "pulse_train":
        r = F.characterise(segs, plan)   # "_" keys stay out of result.json (report.to_json drops them)
        lines = [f"Input: hardware trains of {', '.join(map(str, plan['levels']))} pulses, {plan['ipi_ms']:g} ms apart; "
                 f"output: events 10-210 ms after the train, minus the 500 ms before it and the sham mean."]
        for role in F.ROLES:
            v = r["roles"].get(role)
            lines.append(f"{role} ({F.ROLE_TEXT[role]}): " + (
                f"e{v['stim']}→e{v['rec']}, {v['grade'].upper()} ({v['reason']}); new single trials: R² "
                f"{v.get('conf_r2', float('nan')):.2f}, error ±{v.get('conf_rmse', float('nan')):.1f} events"
                if v else "MISSING: no pair gives this curve"))
        verdict = {"ALL_ROLES": "ALL THREE FUNCTIONS FOUND", "SOME_ROLES": "SOME FUNCTIONS MISSING",
                   "NO_ROLES": "NO USABLE FUNCTION"}[r["verdict"]]
        cal = F.calibration(plan, plan["fsname"], f"{plan['fsname']}_{step_dir.name}",
                            {int(k): v for k, v in r["thresholds_uV"].items()}, r["_chosen"], r["pairs"],
                            code_version(), plan.get("backend") == "simulated", step_dir)
        (step_dir / "calibration.json").write_text(json.dumps(cal, indent=1, allow_nan=False))
        r["calibration"] = str(step_dir / "calibration.json")
    else:
        sys.exit(f"{step_dir.name}: {step} mode '{plan.get('mode') or plan.get('vary')}' was removed on 5 Oct 2026; "
                 "its report.html and result.json stay as they were.")
    problems = _completeness_problems(step_dir, plan, r)
    r["incomplete"] = bool(problems)
    if problems:
        verdict = ("INCOMPLETE: " + "; ".join(problems) + ". Results are for inspection only; re-run this step. "
                   f"(Would otherwise have been: {verdict})")
    r["verdict_text"] = verdict
    r["code_version"] = code_version()
    lines.append(f"Acquired with orgchar {plan.get('code_version', 'unknown (before versioning)')}; "
                 f"analysed with {r['code_version']}.")
    (step_dir / "result.json").write_text(R.to_json(r))
    if step == "census":
        title = f"{plan['fsname']} · census · {_when(step_dir)}" + (" · SIMULATED" if plan.get("backend") == "simulated" else "")
        R.write_html(step_dir / "report.html", title, verdict, lines, R.census_figures(r),
                     pd.DataFrame(r["channels"]).drop(columns=["waveform"]))
    else:
        from . import report_funcsolver as RF
        if step == "screen":
            RF.screen_page(step_dir / "report.html", plan, r, problems, _when(step_dir))
        else:
            RF.page(step_dir / "report.html", plan, r, problems, _when(step_dir))
            if not problems and plan.get("backend") != "simulated" and r["verdict"] == "ALL_ROLES":
                latest = step_dir.parent / "calibration_latest.json"
                latest.write_text((step_dir / "calibration.json").read_text())
                lines.append(f"FuncSolver calibration: {latest}")
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
    for s in plan["segments"]:
        st = _close_status(step_dir / s["id"])
        raw = _raw(step_dir / s["id"])
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
        m = re.match(r"\d{8}_\d{6}_orgchar_(.+?)_(\d{8}T\d{6})_(census|screen|curve|codetest)_", n)
        if m:
            out.setdefault(m[1], set()).add(f"{m[2]}_{m[3]}")
    return out


def _status(fs: str, simulate: bool, elsewhere: set[str]) -> list[str]:
    """Where one organoid stands: the latest complete census, screen and curve, the best pair found
    so far for each FuncSolver function, the calibration, and runs whose results are elsewhere."""
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
        out.append("          " + " · ".join(orgs))
    screens = [p for p in done if p.name.endswith("_screen") and res[p].get("mode") == "trains"]
    if screens:
        s = res[screens[-1]]
        head("screen", screens[-1])
        parts = []
        for role in F.ROLES:
            v = s["best"].get(role)
            dropped = ", ".join(f"e{x['stim']}→e{x['rec']}" for x in s.get("chance", []) if x["role"] == role)
            n = v["n_candidates"] if v else 0
            parts.append(f"{SHAPE[role]} " + (f"e{v['stim']}→e{v['rec']} ({n} pair{'s' * (n != 1)})" if v else "none")
                         + (f" ({dropped} dropped as chance)" if dropped else ""))
        out.append("          " + " · ".join(parts)
                   + ("" if "chance" in s else f"  [analysed before the chance check: ./orgchar.sh analyze "
                                               f"{os.path.relpath(screens[-1], Path(__file__).resolve().parents[1])}]"))
    curves = [p for p in done if p.name.endswith("_curve") and res[p].get("readout") == "after_train"]
    if curves:
        head("curve", curves[-1], f": {res[curves[-1]]['verdict_text']}")
        tried = sorted({e for p in curves for e in json.loads((p / "plan.json").read_text())["stims"]})
        for role in F.ROLES:
            found = [p for p in curves if role in res[p]["roles"]]
            if found:
                v = res[found[-1]]["roles"][role]
                r2 = "–" if v.get("conf_r2") is None else f"{v['conf_r2']:.2f}"
                out.append(f"          {SHAPE[role]:8s} e{v['stim']}→e{v['rec']} {v['grade'].capitalize()}, R² {r2} on new "
                           "trials" + ("" if found[-1] == curves[-1] else f" (curve of {_when(found[-1])}, {_ago(found[-1])})"))
            else:
                out.append(f"          {SHAPE[role]:8s} missing (stimulated so far: e{', e'.join(map(str, tried))})")
    if not (census or screens or curves):
        out.append("  no complete census, screen or curve on this machine")
    cal = d / "calibration_latest.json"
    if cal.exists():
        src = json.loads(cal.read_text())["source"]["step"]
        out.append(f"  FuncSolver: {cal} (from {Path(src).name}, {_ago(Path(src))})")
    elif curves:
        out.append("  FuncSolver: no calibration_latest.json" + (" (simulated runs never write one)" if simulate else
                                                               " yet (written when one complete curve finds all three functions)"))
    if runs:
        last = runs[-1]
        v = res[last].get("verdict_text", "?") if last in res else (
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
    step_dir, segs = _execute(a, plan)
    analyse(step_dir, json.loads((step_dir / "plan.json").read_text()), segs)


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
    plan = P.plan_train_screen(blocks=blocks, electrodes=electrodes,
                               counts=[int(x) for x in a.levels.split(",")] if a.levels else P.SCREEN_COUNTS,
                               amplitude_uA=a.amplitude, phase_us=a.phase, polarity=a.polarity, reps=a.reps,
                               seed=int(time.time()))
    if _use_codes(a):
        plan = P.add_train_codes(plan)
    _describe(plan)
    if a.dry_run or not _confirm(a):
        return
    step_dir, segs = _execute(a, plan)
    analyse(step_dir, json.loads((step_dir / "plan.json").read_text()), segs)


def cmd_curve(a):
    if a.stims:
        stims = [int(x) for x in a.stims.split(",")]
    else:
        d, s = _latest(a.fsname, a.simulate, "screen")
        if s is None or s.get("mode") != "trains":
            sys.exit("No after-train screen for this organoid yet. Run `orgchar screen` first, or pass --stims.")
        if s.get("incomplete"):
            sys.exit(f"Latest screen ({d.name}) is INCOMPLETE. Re-run it, or pass --stims.")
        stims = s.get("curve_stims") or []
        if not stims:
            sys.exit(f"Latest screen ({d.name}) found no electrode giving a falling, peaked or rising "
                     "response after trains. Nothing for FuncSolver; pass --stims to try anyway.")
        print(f"Stimulating e{', e'.join(map(str, stims))}, the screen's best candidates ("
              + ", ".join(f"{SHAPE[k]}: e{v['stim']}→e{v['rec']}" for k, v in s["best"].items())
              + f"; {d.name}). Every other electrode of each organoid is measured.")
        if _age_h(d) > STALE_H:
            print(f"  Warning: this screen is {_age_h(d):.0f} h old and responses change within a day; "
                  "consider re-running it.")
    counts = [int(x) for x in (a.levels or ",".join(map(str, P.FUNCSOLVER_COUNTS))).split(",")]
    plan = P.plan_funcsolver(stims, counts=counts, ipi_ms=a.ipi, amplitude_uA=a.amplitude, phase_us=a.phase,
                             polarity=a.polarity, reps=a.reps, seed=int(time.time()))
    if _use_codes(a):
        plan = P.add_train_codes(plan)
    _describe(plan)
    if a.dry_run or not _confirm(a):
        return
    step_dir, segs = _execute(a, plan)
    analyse(step_dir, json.loads((step_dir / "plan.json").read_text()), segs)


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
        stims = (s or {}).get("curve_stims") or []
        if not stims:   # no screen yet (e.g. a new organoid): one electrode per organoid
            stims = [2, 10, 18, 26]
            print("No screen of this organoid yet; using e2, e10, e18, e26 (one per organoid). The codetest checks "
                  "the trigger system, not the tissue, so the choice does not matter.")
        print(f"Uploading the parameters of e{', e'.join(map(str, stims))} (as a curve would), triggering none of them "
              "except 5 positive-control pulses on the first at the end.")
    plan = P.plan_codetest(stims, amplitude_uA=a.amplitude, phase_us=a.phase, seed=int(time.time()))
    _describe(plan)
    if a.dry_run or not _confirm(a):
        return
    step_dir, segs = _execute(a, plan)
    r = _codetest_result(plan, segs)
    r["verdict_text"] = ("PASS: train codes are safe and readable; screen and curve now use them by default on this machine"
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
    analyse(step_dir, plan, _segments_from_dir(step_dir, plan))


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
    locks = _lock()
    try:
        asyncio.run(run())
    finally:
        for h in locks:
            h.close()


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

    p = sub.add_parser("screen", help="which electrodes give FuncSolver's shapes after trains of 2, 6, 12, 20 pulses")
    common(p)
    stimulus(p)
    p.add_argument("--levels", help="train lengths, default 2,6,12,20")
    p.add_argument("--blocks", help="e.g. 1,2 (default: live blocks from latest census)")
    p.add_argument("--electrodes", help="subset, e.g. 21,15 (quick re-check before a session)")
    p.add_argument("--reps", type=int, default=6, help="trials per electrode and length")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("curve", help="FuncSolver calibration: hardware trains of 2-20 pulses -> calibration.json")
    common(p)
    stimulus(p)
    p.add_argument("--stims", help="stimulating electrodes, e.g. 0,10,30 (default: from latest screen)")
    p.add_argument("--levels", help="pulse counts, default 2,4,...,20")
    p.add_argument("--ipi", type=float, default=10.0, help="ms between the pulses of a train")
    p.add_argument("--reps", type=int, default=10, help="trials per pulse count and pass")
    p.set_defaults(func=cmd_curve)

    p = sub.add_parser("codetest", help="~3 min: check that train codes are safe and readable")
    common(p)
    p.add_argument("--electrodes", help="electrodes whose parameters to upload (default: curve stims from latest screen)")
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
