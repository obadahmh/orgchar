# Handover: orgchar (state as of 16 Oct 2026)

How the tool works (method, verdicts, platform quirks) is in `README.md`. This file only holds
what the README doesn't: machines, the current state, open questions, documents and working agreements.

## Goal
Use FinalSpark organoid pairs (stim→rec) as the activation functions of a hybrid network that solves an ODE (TUM's Fig. 4f), with the organoid in the loop. `orgchar solve` does this in one session: measure the curve, train, run the network on the tissue. It replaces the `curve` step and NeuralControlProgram's FuncSolver (16 Oct 2026). The project is heading into a proposed Phase 2 (see Documents).

## Machines
- **np14** runs fs666 (MEA 6); **neuroplatform10** runs fs667 (MEA 7). FinalSpark replaces all four organoids at once, under a new fsXXX, when the user asks.
- Keep both machines on the same commit. GitHub: `obadahmh/orgchar` (private). `sessions/` is local to each machine and stays out of git; raw recordings of both machines are on the shared S3 mount.
- fs667 sessions were copied from neuroplatform10 to np14 (`sessions/fs667/`).
- Tools on np14 without sudo: `~/.local/bin/tectonic` (LaTeX), `~/.local/bin/soffice` (LibreOffice AppImage, for pptx rendering).

## Current state
- **No long-pass anywhere** (curve runs 1–14 Oct on fs666, fs667, fs674, fs675). Not needed: one response type is enough in principle (sufficiency note) and long-pass added nothing in simulation.
- **Responses change within hours.** fs675 (15 Oct): the 2-pulse response fell from 48 to 27 between the 07:20 curve and the live run (pathway adaptation: spontaneous firing stayed steady). A curve from the morning gave live R² −1.5; retrained on the run's own trials, 0.78 (split-half). Hence solve measures its curve minutes before use.
- **Correction:** the fs667 1 Oct curve stimulated e10, **e1**, e30, not e0. An earlier S3 rebuild assumed e0 and produced a fake long-pass on the stimulated electrode. Always take the stims from `plan.json`.
- **Data gaps:** the raw files drop 42.7 ms rows, 96 in 209 recordings (~17 per hour), on both machines. The cause is on FinalSpark's side; the aligner handles it.

## Branch `solve` (16 Oct 2026, not merged)
- New: `net.py` (the network, weak-form training, expected R²), `measure.py` (the one hardware primitive), `solve.py`, `orgchar solve`, its report.
- Removed: the `curve` step, `calibration.json` / `calibration_latest.json`, the curve report. `funcsolver.py` → `screen.py`, `report_funcsolver.py` → `report_screen.py`.
- 26 tests pass; the simulated chain census → screen → solve passes.
- The screen runs on `measure` too (no shams; every other electrode read, for the cross-organoid controls). `plan_trials` deals electrodes across organoids and gives each its lengths in random order.
- Next: merge into `main` once the user agrees; both machines then `git pull`.

## Open questions / next steps
1. **The network does not oscillate yet.** A plain decay a × exp(−kt) + b scores R² 0.80 at the 25 evaluation times, and every network trained on one curve scores 0.72–0.82 there (16 Oct: the simulated sweep and three synthetic shapes, steep and shallow falling, peaked): it follows the first drop and misses the rebound. The 0.92 offline (14 Oct) used two fitted curves (fs675 short + band). solve reports this reference with every run. Since 16 Oct solve's gate 2 requires the expected R² to beat it, so every run currently stops before the evaluation. Candidates: two pairs, a wider layer, more sweep counts.
2. **Drift within a run.** The sweep and the evaluation are ~1 h apart at most; the drift check reports it (simulated: 9 % of the curve's range).
3. **Detector difference with TUM.** TUM used FinalSpark's 6 × SD spike detection and a 0–300 ms window; orgchar uses −5σ (MAD) and 10–210 ms. Not compared yet.
4. **Earlier history** (May–Aug, NCP repo): the live hybrid runs on fs599/618/619/622/626 all failed (R² < 0). Those curves came from the stimulated electrode via the vendor spike database, so they were artefact-dominated.

## Documents (in `~/Documents/FinalSpark_summary/`)
- **Slide deck (live, 19 slides):** https://claude.ai/artifact/3n4rp5qmmQj6m9Q9P1fqcD. The user also edits it directly, so always read the live version before publishing. Build files and figure scripts are in the earlier session's scratchpad; the local PDF/HTML copies are here.
- **MBZUAI PowerPoint:** `Organoid_filter_responses_MBZUAI.pptx`, built on `~/Documents/templates/template.pptx`.
- **PI status report:** `Organoid_filter_responses_status_2026-10-05.pdf/.html` (LaTeX).
- **Phase 2 proposal** for the chair and dean: `proposal/proposal.pdf`, plus a one-page `proposal/executive_summary.pdf` (ASD-STE100 style, ~80%). Both have red placeholders for the department, school, names, TUM collaborators and budget, which the user will supply. Decision thresholds D1–D4 are proposals.
- **Naming in documents:** MEA 1–7 = fs635, fs636, fs647, fs660, fs662, fs666, fs667. The TUM draft paper is called "TUM", not "draft".

## Working agreements
- The user runs all hardware steps. A coding assistant only analyses and changes code, unless explicitly asked.
- Don't call the SDK's unwrapped start/stop/debuginfo endpoints.
- Before handing over a code change, run the full chain in simulation from an empty home:
  ```bash
  ORGCHAR_HOME=<tmp> ./orgchar.sh <step> --simulate --yes   # census, screen, solve, then status --simulate
  ```
  Also run `pytest` (side venv; don't install into `ncp`).
- Never infer which electrode was stimulated from artifact size: neighbours clip at ±6 mV too. Use the plan's key mapping.
- Keep raw-data reads minimal (token cost).
- Keep the code lean: the user dislikes bloat and band-aids. Fix root causes, and say plainly when a fix is a heuristic.
- Slides and reports: short phrase titles, no footnotes except references, plain research terminology, legible when projected.
- No AI attribution in commit messages or PR descriptions.
