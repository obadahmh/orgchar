"""orgchar: weekly characterisation of FinalSpark organoids.

    orgchar census --fsname fs652          # 5 min, no stimulation: which organoids fire?
    orgchar screen --fsname fs652          # ~6 min per live organoid: which electrode pairs respond?
    orgchar curve  --fsname fs652          # ~10-20 min: input-output curves -> card.json
    orgchar analyze <step dir>             # re-run the analysis of a finished step, offline
    orgchar status --fsname fs652          # what has been done for this organoid

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

import pandas as pd

from . import analysis as A
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
    print(f"Plan: {plan['step']}, {len(plan['segments'])} segment(s), {n_stim} stimulation pulses, "
          f"{n_sham} sham markers, about {P.plan_duration_s(plan) / 60:.1f} min.")
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
        backend = SimBackend()
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
            trials = match_triggers(s, raw)
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
            trials = match_triggers(s, sd / "raw.h5")
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
    elif step == "screen":
        r = A.screen(segs, window_ms=plan.get("window_ms", 30.0))
        Rt = pd.DataFrame(r["pairs"])
        resp = Rt[Rt.responsive]
        c = r["controls"]
        lines = [f"{r['candidate_tests']} stim→record pairs tested inside organoids; {len(resp)} responsive "
                 f"(Bonferroni p<{r['alpha']}, ≥{r['min_effect_spikes_per_pulse']} extra spikes/pulse, spike-shaped).",
                 f"Negative controls: {c['cross_organoid_control']['significant_bonferroni']} of "
                 f"{c['cross_organoid_control']['tests']} cross-organoid pairs and "
                 f"{c['sham_control']['significant_bonferroni']} of {c['sham_control']['tests']} sham pairs significant "
                 "(must be 0; otherwise artifact is leaking into the readout).",
                 f"Artifact blank after template subtraction, same organoid: median "
                 f"{r['artifact_blank_ms']['same_organoid_median'] or float('nan'):.1f} ms, p90 {r['artifact_blank_ms']['same_organoid_p90'] or float('nan'):.1f} ms."]
        for _, x in resp.head(12).iterrows():
            lines.append(f"e{int(x.stim)} → e{int(x.rec)}: +{x.delta_per_pulse:.2f} spikes/pulse "
                         f"({int(x.spikes)} vs {x.expected:.1f} expected), latency ~{x.median_latency_ms:.0f} ms, "
                         f"waveform r={x.waveform_corr if pd.isna(x.waveform_corr) else round(x.waveform_corr, 2)}"
                         + (f"  ⚠ TIME-LOCKED ({x.peak_1ms_fraction:.0%} of spikes in one 1-ms window)"
                            if x.time_locked else "")
                         + (f"  ⚠ BLANK-EDGE ({int(x.edge_spikes)} spikes in the 1st ms after the blank)"
                            if x.blank_edge else "")
                         + ("  → could be artifact; check the raw-trace figure" if x.time_locked or x.blank_edge else ""))
        verdict = r["verdict"]
        figs = R.screen_figures(r)
        cols = ["role", "stim", "rec", "n_pulses", "spikes", "expected", "delta_per_pulse", "z", "p", "p_bonf",
                "waveform_corr", "median_latency_ms", "latency_spread_ms", "peak_1ms_fraction", "time_locked",
                "edge_spikes", "blank_edge", "mean_blank_ms", "responsive"]
        table = Rt[Rt.role == "candidate"][cols].head(60)
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
    title = f"{plan['fsname']} — {step} — {step_dir.name[:15]}" + ("  [SIMULATED]" if plan.get("backend") == "simulated" else "")
    R.write_html(step_dir / "report.html", title, verdict, lines, figs, table)
    print("\n" + verdict)
    for s in lines:
        print("  " + s)
    print(f"\nReport: {step_dir / 'report.html'}")
    return r


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
        if st and not st["stop_ok"]:
            problems.append(f"recording of segment {s['id']} was never stopped cleanly")
        if st and not st["disable_ok"]:
            problems.append(f"final 'disable all stimulation' failed for segment {s['id']}")
        if not (step_dir / s["id"] / "raw.h5").exists():
            problems.append(f"segment {s['id']} has no recording")
    if plan["step"] == "census":
        planned = plan["segments"][0]["duration_s"]
        if r["duration_s"] < 0.9 * planned:
            problems.append(f"recording ended after {r['duration_s']:.0f} of {planned:.0f} s")
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
    analyse(step_dir, dict(plan, fsname=a.fsname, backend="simulated" if a.simulate else "hardware"), segs)


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
    plan = P.plan_screen(blocks=blocks, electrodes=electrodes, amplitude_uA=a.amplitude, phase_us=a.phase,
                         polarity=a.polarity, reps=a.reps, sham_reps=a.reps, seed=int(time.time()))
    plan["window_ms"] = a.window
    _describe(plan)
    if a.dry_run or not _confirm(a):
        return
    step_dir, segs = _execute(a, plan)
    analyse(step_dir, dict(plan, fsname=a.fsname, backend="simulated" if a.simulate else "hardware"), segs)


def cmd_curve(a):
    if a.pairs:
        pairs = [tuple(int(v) for v in p.split(":")) for p in a.pairs.split(",")]
    else:
        d, s = _latest(a.fsname, a.simulate, "screen")
        if s is None:
            sys.exit("No screen for this organoid yet. Run `orgchar screen` first, or pass --pairs stim:rec,...")
        if s.get("incomplete"):
            sys.exit(f"Latest screen ({d.name}) is INCOMPLETE. Re-run it, or pass --pairs explicitly.")
        resp = [p for p in s["pairs"] if p["responsive"]]
        flagged = [p for p in resp if p.get("time_locked") or p.get("blank_edge")]
        if flagged:
            print("Skipping pairs flagged as possible artifact (pass them with --pairs if the traces convince you): "
                  + ", ".join(f"e{p['stim']}→e{p['rec']}" for p in flagged))
        resp = [p for p in resp if p not in flagged]
        if not resp:
            sys.exit(f"Latest screen ({d.name}) found no responsive pair; nothing to build a curve on.")
        best = {}
        for p in sorted(resp, key=lambda p: -p["z"]):
            best.setdefault(p["stim"], []).append(p["rec"])
        stims = list(best)[:a.max_stims]
        pairs = [(e, c) for e in stims for c in best[e]]
        print(f"Using responsive pairs from screen {d.name}: " + ", ".join(f"e{e}→e{c}" for e, c in pairs))
    stims = sorted({e for e, _ in pairs})
    if a.vary == "pulses":
        levels = [int(x) for x in (a.levels or "1,2,4,6,8,10").split(",")]
        plan = P.plan_curve_pulses(stims, levels=levels, ipi_ms=a.ipi, amplitude_uA=a.amplitude, phase_us=a.phase,
                                   polarity=a.polarity, reps=a.reps, sham_reps=a.reps, seed=int(time.time()))
    else:
        levels = [float(x) for x in (a.levels or "0.5,1,1.5,2,3").split(",")]
        plan = P.plan_curve_amplitude(stims, levels=levels, phase_us=a.phase, polarity=a.polarity,
                                      reps_per_pass=max(10, -(-a.reps // 3)), passes=3, seed=int(time.time()))
    plan["pairs"] = pairs
    _describe(plan)
    if a.dry_run or not _confirm(a):
        return
    step_dir, segs = _execute(a, plan)
    analyse(step_dir, dict(plan, fsname=a.fsname, backend="simulated" if a.simulate else "hardware"), segs)


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

    p = sub.add_parser("screen", help="which electrode pairs respond to single pulses")
    common(p)
    p.add_argument("--blocks", help="e.g. 1,2 (default: live blocks from latest census)")
    p.add_argument("--electrodes", help="subset, e.g. 21,15 (quick re-check before a session)")
    p.add_argument("--amplitude", type=float, default=2.0)
    p.add_argument("--phase", type=float, default=100.0)
    p.add_argument("--polarity", default="negative-first", choices=["negative-first", "positive-first"])
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--window", type=float, default=30.0, help="ms after each pulse in which spikes count")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("curve", help="input-output curve for responsive pairs")
    common(p)
    p.add_argument("--pairs", help="stim:rec,... (default: responsive pairs from latest screen)")
    p.add_argument("--max-stims", type=int, default=3)
    p.add_argument("--vary", default="pulses", choices=["pulses", "amplitude"])
    p.add_argument("--levels", help="pulses: 1,2,4,6,8,10  amplitude: 0.5,1,1.5,2,3")
    p.add_argument("--ipi", type=float, default=30.0, help="ms between pulses of a train (pulses mode)")
    p.add_argument("--amplitude", type=float, default=2.0)
    p.add_argument("--phase", type=float, default=100.0)
    p.add_argument("--polarity", default="negative-first", choices=["negative-first", "positive-first"])
    p.add_argument("--reps", type=int, default=20)
    p.set_defaults(func=cmd_curve)

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
