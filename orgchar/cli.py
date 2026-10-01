"""orgchar: weekly characterisation of FinalSpark organoids.

    orgchar census --fsname fs652          # 5 min, no stimulation: which organoids fire?
    orgchar screen --fsname fs652          # 15-30 min: which electrodes give FuncSolver's shapes after trains?
    orgchar curve  --fsname fs652          # ~30 min: FuncSolver curves (2-20 pulse trains) -> calibration.json
    orgchar analyze <step dir>             # re-run the analysis of a finished step, offline
    orgchar status --fsname fs652          # what has been done for this organoid
    orgchar codetest --fsname fs652        # 1 min: check that train codes (--codes) are safe and readable

Add --dry-run to see the plan and its duration without touching hardware, --simulate to run the
whole step on a synthetic organoid.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import analysis as A
from . import funcsolver as F
from . import protocol as P
from . import report as R

HOME = Path(os.environ.get("ORGCHAR_HOME", Path(__file__).resolve().parents[1] / "sessions"))
LOCKS = [Path.home() / ".orgchar.lock",
         Path.home() / "NeuralControlProgram/ResponseSweep/overnight_hardware.lock"]  # skipped if absent


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
            from .backends import match_triggers
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


def _segments_from_dir(step_dir: Path, plan: dict):
    """Segments with a recording. Trigger matching is redone from the raw file when trials.csv is
    missing (e.g. the step stopped at matching) so salvageable data can still be analysed."""
    from .backends import match_triggers
    segs = []
    for s in plan["segments"]:
        sd = step_dir / s["id"]
        if not (sd / "raw.h5").exists():
            continue
        t = sd / "trials.csv"
        if t.exists() and t.stat().st_size > 1:
            trials = pd.read_csv(t)
        elif s["events"]:
            sent = pd.read_csv(sd / "sent.csv") if (sd / "sent.csv").exists() and (sd / "sent.csv").stat().st_size > 1 else None
            trials = match_triggers(s, sd / "raw.h5", sent)
            trials.to_csv(t, index=False)
        else:
            trials = pd.DataFrame()
        segs.append((str(sd / "raw.h5"), trials))
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
        figs = R.census_figures(r)
        table = pd.DataFrame(r["channels"]).drop(columns=["waveform"])
    elif step == "screen" and plan.get("mode") == "trains":
        r = F.scout(segs, plan)
        names = dict(short="falling", band="peaked", long="rising")
        lines = [f"Trains of {', '.join(map(str, plan['levels']))} pulses on e{', e'.join(map(str, plan['stims']))}; "
                 "FuncSolver's readout (events 10-210 ms after the train)."]
        for role in F.ROLES:
            v = r["best"].get(role)
            lines.append(f"{names[role]}: " + (f"e{v['stim']}→e{v['rec']} (a single trial wins {v['win']:.0%} of the time, "
                                               f"{v['diff']:.1f} events apart; {v['n_candidates']} candidate pair(s))"
                                               if v else "no candidate"))
        c = r["controls"]
        lines.append(f"Controls: {c['cross_organoid_passing']} of {c['cross_organoid_pairs']} other-organoid pairs pass "
                     f"(about {c['expected_by_chance']:g} expected by chance).")
        if r["curve_stims"]:
            lines.append(f"Next: ./orgchar.sh curve --fsname {plan['fsname']}   (stimulates "
                         f"e{', e'.join(map(str, r['curve_stims']))})")
        verdict = r["verdict"]
        figs, table = [], None
    elif step == "screen":
        r = A.screen(segs, window_ms=plan.get("window_ms", 30.0))
        earlier_dir, earlier = _earlier_screen(step_dir)
        Rt = A.classify(r["pairs"], earlier["pairs"] if earlier else None)
        rec = A.recommend(Rt)
        r["pairs"] = Rt.to_dict("records")
        r["recommended_pairs"] = rec
        r["earlier_screen"] = earlier_dir.name if earlier_dir else None
        resp = Rt[Rt.responsive]
        c = r["controls"]
        n = {k: int((Rt.status == k).sum()) for k in ("Use", "Weak", "Check")}
        lines = [f"{len(resp)} responsive pairs: {n['Use']} Use, {n['Weak']} Weak, {n['Check']} Check.",
                 f"Controls: {c['cross_organoid_control']['significant_bonferroni']} of "
                 f"{c['cross_organoid_control']['tests']} other-organoid pairs and "
                 f"{c['sham_control']['significant_bonferroni']} of {c['sham_control']['tests']} sham pairs respond "
                 "(must be 0)."]
        if rec:
            lines.append("Next: ./orgchar.sh curve --fsname " + plan["fsname"]
                         + "   (pairs " + ", ".join(f"e{a}→e{b}" for a, b in rec) + ")")
        verdict = r["verdict"]
        figs, table = [], None
    elif step == "curve" and plan.get("vary") == "pulse_train":
        r = F.characterise(segs, plan)   # "_" keys stay out of result.json (report.to_json drops them)
        chosen = r["_chosen"]
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
        figs, table = [], None
        cal = F.calibration(plan, plan["fsname"], f"{plan['fsname']}_{step_dir.name}", {int(k): v for k, v in r["thresholds_uV"].items()}, chosen, r["pairs"],
                            code_version(), plan.get("backend") == "simulated", step_dir)
        (step_dir / "calibration.json").write_text(json.dumps(cal, indent=1, allow_nan=False))
        r["calibration"] = str(step_dir / "calibration.json")
    elif step == "curve":
        pairs = [tuple(p) for p in plan["pairs"]]
        unit = "pulse" if plan["vary"] == "amplitude" else "train"
        r = A.curve(segs, pairs, unit=unit)
        r["vary"] = plan["vary"]
        lines = [f"Input: {plan['vary']} at levels {plan['levels']}; output: evoked spikes per "
                 f"{'pulse' if unit == 'pulse' else 'trial'} minus spontaneous expectation."]
        fmt = lambda v, f: "n/a" if v is None else format(v, f)  # noqa: E731
        for p in r["pairs"]:
            lines.append(f"e{p['stim']}→e{p['rec']}: {'USABLE' if p['usable'] else 'not usable'}; "
                         f"dynamic range {p['dynamic_range']:.2f}, trial SD {p['mean_trial_sd']:.2f}, "
                         f"split-half ρ {fmt(p['split_half_spearman'], '.2f')}, "
                         f"top level vs sham p={fmt(p['top_vs_sham_p'], '.2g')}")
        verdict = r["verdict"]
        figs = R.curve_figures(r)
        table = pd.DataFrame([dict(stim=p["stim"], rec=p["rec"], level=l["level"], mean=l["mean"], sem=l["sem"],
                                   n=l["count"]) for p in r["pairs"] for l in p["levels"]])
        _write_card(step_dir, plan, r)
    else:
        raise ValueError(step)
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
    if step == "screen" and plan.get("mode") == "trains":
        from . import report_funcsolver as RF
        RF.screen_page(step_dir / "report.html", plan, r, problems, _when(step_dir))
    elif step == "screen":
        R.screen_page(step_dir / "report.html", r, Rt, _screen_context(step_dir, plan, segs, r, problems))
    elif step == "curve" and plan.get("vary") == "pulse_train":
        from . import report_funcsolver as RF
        RF.page(step_dir / "report.html", plan, r, problems, _when(step_dir))
        if not problems and plan.get("backend") != "simulated" and r["verdict"] == "ALL_ROLES":
            latest = step_dir.parent / "calibration_latest.json"
            latest.write_text((step_dir / "calibration.json").read_text())
            lines.append(f"FuncSolver calibration: {latest}")
    else:
        title = f"{plan['fsname']} · {step} · {_when(step_dir)}" + (" · SIMULATED" if plan.get("backend") == "simulated" else "")
        R.write_html(step_dir / "report.html", title, verdict, lines, figs, table)
    print("\n" + verdict)
    for s in lines:
        print("  " + s)
    print(f"\nReport: {step_dir / 'report.html'}")
    return r


def _when(step_dir: Path) -> str:
    try:
        return datetime.strptime(step_dir.name[:15], "%Y%m%dT%H%M%S").strftime("%d %b %Y %H:%M UTC")
    except ValueError:
        return step_dir.name


def _earlier_screen(step_dir: Path):
    """Most recent screen of the same organoid before this one (complete or not: its analysed
    segments are still valid evidence for reproducibility)."""
    earlier = sorted(p for p in step_dir.parent.glob("*_screen") if p.name < step_dir.name and (p / "result.json").exists()
                     and json.loads((p / "result.json").read_text()).get("mode") != "trains")
    if not earlier:
        return None, None
    return earlier[-1], json.loads((earlier[-1] / "result.json").read_text())


def _screen_context(step_dir: Path, plan: dict, segs, r: dict, problems: list[str]) -> dict:
    notes = pd.concat([t.get("note", pd.Series(dtype=str)) for _, t in segs if len(t)]) if segs else pd.Series(dtype=str)
    excl = pd.concat([t.get("exclude", pd.Series(dtype=bool)) for _, t in segs if len(t)]) if segs else pd.Series(dtype=bool)
    stims = sorted({int(e) for s in plan["segments"] for e in s["electrodes"]})
    blocks = sorted({e // 8 for e in stims})
    return dict(fsname=plan["fsname"], when=_when(step_dir), code=r.get("code_version", "unknown"),
                simulated=plan.get("backend") == "simulated", problems=problems,
                n_segments=len(plan["segments"]),
                stimulus=(f"{plan.get('amplitude_uA')} µA × {plan.get('phase_us'):g} µs, {plan.get('polarity')}, "
                          f"{plan.get('reps')} pulses per electrode, organoid{'s' if len(blocks) > 1 else ''} "
                          + ", ".join(map(str, blocks))),
                gaps=int((notes == "before recording gap").sum()), excluded=int(excl.fillna(False).astype(bool).sum()),
                recommended=[tuple(p) for p in r.get("recommended_pairs", [])],
                earlier=r.get("earlier_screen"))


def _close_status(seg_dir: Path) -> dict | None:
    """How the segment's recording was closed. Retries that ended in success are not a problem.
    Segments recorded before close_status.json existed only have cleanup_errors.txt (3 failed stop
    attempts = stop failed; a 'final disable' line = disable failed)."""
    f = seg_dir / "close_status.json"
    if f.exists():
        return json.loads(f.read_text())
    old = seg_dir / "cleanup_errors.txt"
    if old.exists():
        t = old.read_text()
        return dict(stop_ok="stop recording attempt 3" not in t, disable_ok="final disable" not in t)
    return None


def _completeness_problems(step_dir: Path, plan: dict, r: dict) -> list[str]:
    """A step whose recording was cut short or hit platform errors must not produce a GO."""
    problems = []
    for s in plan["segments"]:
        st = _close_status(step_dir / s["id"])
        raw = step_dir / s["id"] / "raw.h5"
        if st and not st["disable_ok"]:
            problems.append(f"final 'disable all stimulation' failed for segment {s['id']}")
        if not raw.exists():
            problems.append(f"segment {s['id']} has no recording")
            continue
        # a stop without acknowledgement is fine if the file holds the whole segment
        with h5py.File(raw, "r") as f:
            got_s = f["triggers"].size / 30000.0
        if got_s < 0.9 * s["duration_s"]:
            problems.append(f"segment {s['id']} recorded {got_s:.0f} of {s['duration_s']:.0f} s")
    return problems


def _write_card(step_dir, plan, r):
    """The dated operating card FuncSolver should read: only this week's usable curves."""
    card = dict(fsname=plan["fsname"], created_utc=_now().isoformat(), source=str(step_dir),
                backend=plan.get("backend"), vary=plan["vary"], levels=plan["levels"],
                stimulation=dict(amplitude_uA=plan.get("amplitude_uA"), phase_us=plan.get("phase_us"),
                                 polarity=plan.get("polarity"), ipi_ms=plan.get("ipi_ms")),
                readout=dict(window_ms=r["window_ms"], threshold="-5 x robust noise, 300-3000 Hz",
                             artifact="per-electrode median template subtraction + adaptive blank",
                             output="evoked spikes per " + r["unit"] + " (spontaneous expectation subtracted)"),
                usable_pairs=[{k: v for k, v in p.items()} for p in r["pairs"] if p["usable"]],
                all_pairs=r["pairs"],
                valid_until_note="Valid for this preparation and week only. Re-run screen on these electrodes "
                                 "before each FuncSolver session; re-run the full workflow after any medium change.")
    txt = R.to_json(card)
    (step_dir / "card.json").write_text(txt)
    (step_dir.parent / "card_latest.json").write_text(txt)


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
        age_h = (time.time() - (d / "result.json").stat().st_mtime) / 3600
        if c.get("incomplete"):
            sys.exit(f"Latest census ({d.name}) is INCOMPLETE ({c.get('verdict_text', '')[:120]}...). "
                     "Re-run the census, or pass --blocks explicitly.")
        if not c["live_blocks"]:
            sys.exit(f"Latest census ({d.name}) found no firing organoid. Stimulating now would only measure "
                     "artifact. Re-run census later, or pass --blocks explicitly to override.")
        blocks = c["live_blocks"]
        print(f"Using live blocks {blocks} from census {d.name} ({age_h:.1f} h old).")
        if age_h > 24:
            print("  Warning: census is more than 24 h old; consider re-running it.")
    if a.mode == "trains":
        plan = P.plan_train_screen(blocks=blocks, electrodes=electrodes,
                                   counts=[int(x) for x in a.levels.split(",")] if a.levels else P.SCREEN_COUNTS,
                                   amplitude_uA=a.amplitude, phase_us=a.phase, polarity=a.polarity,
                                   reps=a.reps if a.reps is not None else 6, seed=int(time.time()))
    else:
        reps = a.reps if a.reps is not None else 30
        plan = P.plan_screen(blocks=blocks, electrodes=electrodes, amplitude_uA=a.amplitude, phase_us=a.phase,
                             polarity=a.polarity, reps=reps, sham_reps=reps, seed=int(time.time()))
        plan["window_ms"] = a.window
    if _use_codes(a):
        plan = P.add_train_codes(plan)
    _describe(plan)
    if a.dry_run or not _confirm(a):
        return
    step_dir, segs = _execute(a, plan)
    analyse(step_dir, json.loads((step_dir / "plan.json").read_text()), segs)


def _screen_pairs(a):
    """Recommended pairs of the latest single-pulse screen (used by --mode pulses/amplitude)."""
    org = _org_dir(a.fsname, a.simulate)
    runs = [r for r in sorted(org.glob("*_screen/result.json"), reverse=True) if json.loads(r.read_text()).get("mode") != "trains"]
    d, s = (runs[0].parent, json.loads(runs[0].read_text())) if runs else (None, None)
    if s is None:
        sys.exit("No screen for this organoid yet. Run `orgchar screen` first, or pass --stims / --pairs.")
    if s.get("incomplete"):
        sys.exit(f"Latest screen ({d.name}) is INCOMPLETE. Re-run it, or pass --stims / --pairs explicitly.")
    rec = s.get("recommended_pairs")
    if rec is None:  # screens analysed before recommendations existed
        rec = A.recommend(A.classify(s["pairs"]))
    if not rec:
        sys.exit(f"Latest screen ({d.name}) has no pair with status Use. Pass --stims / --pairs explicitly.")
    return d, rec


def cmd_curve(a):
    if a.mode == "funcsolver":
        if a.stims:
            stims = [int(x) for x in a.stims.split(",")]
        else:
            d, s = _latest(a.fsname, a.simulate, "screen")
            if s is not None and s.get("mode") == "trains":
                if s.get("incomplete"):
                    sys.exit(f"Latest screen ({d.name}) is INCOMPLETE. Re-run it, or pass --stims.")
                stims = s.get("curve_stims") or []
                if not stims:
                    sys.exit(f"Latest screen ({d.name}) found no electrode giving a falling, peaked or rising "
                             "response after trains. Nothing for FuncSolver; pass --stims to try anyway.")
                names = dict(short="falling", band="peaked", long="rising")
                print(f"Stimulating e{', e'.join(map(str, stims))}, the screen's best candidates ("
                      + ", ".join(f"{names[k]}: e{v['stim']}→e{v['rec']}" for k, v in s["best"].items())
                      + f"; {d.name}). Every other electrode of each organoid is measured.")
            else:   # a single-pulse screen: its recommended stimulating electrodes
                d, rec = _screen_pairs(a)
                stims = list(dict.fromkeys(e for e, _ in rec))[:a.max_stims]
                print(f"Stimulating e{', e'.join(map(str, stims))} (single-pulse screen {d.name}; the after-train "
                      "screen chooses better: ./orgchar.sh screen). Every other electrode of each organoid is measured.")
        counts = [int(x) for x in (a.levels or ",".join(map(str, P.FUNCSOLVER_COUNTS))).split(",")]
        plan = P.plan_funcsolver(stims, counts=counts, ipi_ms=a.ipi if a.ipi is not None else 10.0,
                                 amplitude_uA=a.amplitude, phase_us=a.phase, polarity=a.polarity,
                                 reps=a.reps if a.reps is not None else 10, seed=int(time.time()))
    else:
        if a.pairs:
            pairs = [tuple(int(v) for v in p.split(":")) for p in a.pairs.split(",")]
        else:
            d, rec = _screen_pairs(a)
            keep = list(dict.fromkeys(e for e, _ in rec))[:a.max_stims]
            pairs = [tuple(p) for p in rec if p[0] in keep]
            print(f"Using the recommended pairs from screen {d.name}: " + ", ".join(f"e{e}→e{c}" for e, c in pairs))
        stims = sorted({e for e, _ in pairs})
        reps = a.reps if a.reps is not None else 20
        if a.mode == "pulses":
            levels = [int(x) for x in (a.levels or "1,2,4,6,8,10").split(",")]
            plan = P.plan_curve_pulses(stims, levels=levels, ipi_ms=a.ipi if a.ipi is not None else 30.0,
                                       amplitude_uA=a.amplitude, phase_us=a.phase, polarity=a.polarity, reps=reps,
                                       sham_reps=reps, seed=int(time.time()))
        else:
            levels = [float(x) for x in (a.levels or "0.5,1,1.5,2,3").split(",")]
            plan = P.plan_curve_amplitude(stims, levels=levels, phase_us=a.phase, polarity=a.polarity,
                                          reps_per_pass=max(10, -(-reps // 3)), passes=3, seed=int(time.time()))
        plan["pairs"] = pairs
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
        rec = (s or {}).get("recommended_pairs") or []
        stims = list(dict.fromkeys(e for e, _ in rec))[:4]
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


def cmd_status(a):
    for sim in (False, True):
        d = _org_dir(a.fsname, sim)
        if not d.exists():
            continue
        print(("SIMULATED " if sim else "") + str(d))
        for r in sorted(d.glob("*/result.json")):
            v = json.loads(r.read_text()).get("verdict_text", "?")
            print(f"  {r.parent.name:32s} {v}")
        if (d / "card_latest.json").exists():
            c = json.loads((d / "card_latest.json").read_text())
            print(f"  card_latest.json: {len(c['usable_pairs'])} usable pair(s), from {Path(c['source']).name}")


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

    p = sub.add_parser("census", help="spontaneous activity, no stimulation")
    common(p)
    p.add_argument("--duration", type=float, default=300.0)
    p.set_defaults(func=cmd_census)

    p = sub.add_parser("screen", help="which electrodes give FuncSolver's shapes after trains (default), or "
                                      "which pairs respond to single pulses (--mode single)")
    common(p)
    p.add_argument("--mode", default="trains", choices=["trains", "single"],
                   help="trains (default): trains of 2,6,12,20 pulses on every electrode, FuncSolver's readout; "
                        "single: the earlier single-pulse screen")
    p.add_argument("--levels", help="trains mode: train lengths, default 2,6,12,20")
    p.add_argument("--blocks", help="e.g. 1,2 (default: live blocks from latest census)")
    p.add_argument("--electrodes", help="subset, e.g. 21,15 (quick re-check before a session)")
    p.add_argument("--amplitude", type=float, default=2.0)
    p.add_argument("--phase", type=float, default=100.0)
    p.add_argument("--polarity", default="negative-first", choices=["negative-first", "positive-first"])
    p.add_argument("--reps", type=int, default=None, help="trials per electrode and length (trains: 6; single: 30)")
    p.add_argument("--window", type=float, default=30.0, help="single mode: ms after each pulse in which spikes count")
    p.add_argument("--codes", action=argparse.BooleanOptionalAction, default=None,
                   help="train codes (keys 8-14) so the recording identifies each train "
                        "(default: on once codetest has passed on this machine)")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("curve", help="input-output curves; default: FuncSolver calibration (hardware trains 2-20 pulses)")
    common(p)
    p.add_argument("--mode", default="funcsolver", choices=["funcsolver", "pulses", "amplitude"],
                   help="funcsolver (default): hardware trains, after-train readout, calibration.json for FuncSolver; "
                        "pulses/amplitude: the earlier during-train curves")
    p.add_argument("--stims", help="funcsolver mode: stimulating electrodes, e.g. 2,10,20,29 (default: from latest screen)")
    p.add_argument("--pairs", help="pulses/amplitude modes: stim:rec,... (default: from latest screen)")
    p.add_argument("--max-stims", type=int, default=4, help="at most this many stimulating electrodes")
    p.add_argument("--levels", help="funcsolver: 2,4,...,20  pulses: 1,2,4,6,8,10  amplitude: 0.5,1,1.5,2,3")
    p.add_argument("--ipi", type=float, default=None, help="ms between pulses of a train (funcsolver 10, pulses 30)")
    p.add_argument("--amplitude", type=float, default=2.0)
    p.add_argument("--phase", type=float, default=100.0)
    p.add_argument("--polarity", default="negative-first", choices=["negative-first", "positive-first"])
    p.add_argument("--reps", type=int, default=None, help="trials per level (funcsolver: per pass, default 10; else 20)")
    p.add_argument("--codes", action=argparse.BooleanOptionalAction, default=None,
                   help="train codes (keys 8-14) so the recording identifies each train "
                        "(default: on once codetest has passed on this machine)")
    p.set_defaults(func=cmd_curve)

    p = sub.add_parser("codetest", help="1 min: check that train codes are safe and readable")
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

    p = sub.add_parser("status", help="list steps and verdicts for an organoid")
    p.add_argument("--fsname", required=True)
    p.set_defaults(func=cmd_status)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
