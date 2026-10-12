# Handover: orgchar (state as of 7 Oct 2026)

How the tool works (method, verdicts, platform quirks) is in `README.md`. This file only holds
what the README doesn't: machines, the current state, open questions, documents and working agreements.

## Goal
Use FinalSpark organoid channels as activation functions in FuncSolver (`~/NeuralControlProgram/FuncSolver`): three stim→rec pairs, **short-pass**, **band-pass** and **long-pass** (TUM's draft paper), each on its own stimulating electrode. The project is now heading into a proposed Phase 2: a live organoid in the loop, with calibration every session (see Documents).

## Machines
- **np14** runs fs666 (MEA 6); **neuroplatform10** runs fs667 (MEA 7). FinalSpark replaces all four organoids at once, under a new fsXXX, when the user asks.
- Keep both machines on the same commit. GitHub: `obadahmh/orgchar` (private). `sessions/` is local to each machine and stays out of git; raw recordings of both machines are on the shared S3 mount.
- fs667 sessions were copied from neuroplatform10 to np14 (`sessions/fs667/`).
- Tools on np14 without sudo: `~/.local/bin/tectonic` (LaTeX), `~/.local/bin/soffice` (LibreOffice AppImage, for pptx rendering).

## Current state (latest runs: 5 Oct, both MEAs; nothing newer on S3)
- **No long-pass anywhere.** Five curve runs on MEA 6 and MEA 7 (1, 2 and 5 Oct), all re-analysed with the current code. Short-pass in every run, band-pass in three, both predictive within a session (held-out R² 0.53–0.88). The only long-pass screen candidate (fs667 e30→e25) failed its curve.
- **Responses change between days.** For example, fs667 e10→e14 was short-pass on 1 Oct and band-pass on 5 Oct; fs666 e10→e15 fell from a peak of 72 to 17 spikes.
- **MEA 6 is fading**: 30 → 15 → 5 active electrodes (30 Sep, 1 Oct, 5 Oct). MEA 7 had 24, 27 and 19.
- **Correction:** the fs667 1 Oct curve stimulated e10, **e1**, e30, not e0. An earlier S3 rebuild assumed e0 and produced a fake long-pass on the stimulated electrode. Always take the stims from `plan.json`.
- **Data gaps:** the raw files drop 42.7 ms rows, 96 in 209 recordings (~17 per hour), on both machines. The cause is on FinalSpark's side; the aligner handles it.

## Uncommitted code (np14, 9 Oct 2026)
- `protocol.plan_trials`: segments for an arbitrary list of (electrode, pulse count) trains, one length per electrode per segment. FuncSolver's live inference uses it.
- `cli.hardware_lock` (was `_lock`), so FuncSolver's live inference takes the same locks.
- FuncSolver now accepts any subset of the three types: `calibration_latest.json` is written whenever a complete curve finds at least one function; report and README updated; the test uses FuncSolver's new `calibration.load`.
- Tests pass (18) and the simulated chain passes. Commit and push once the user agrees.

## Open questions / next steps
1. **Detector difference with TUM** (most urgent). TUM used FinalSpark's 6 × SD spike detection and a 0–300 ms window; orgchar uses −5σ (MAD) fixed and 10–210 ms. Re-analysing the five curve runs with TUM's settings would show whether the missing long-pass is method or biology. Not done yet.
2. **Long-pass search** with other settings (up to 3 µA, longer pulses, other spacing), on MEA 7 or a new set.
3. **Does the network need all three types?** No, in principle: either type alone is sufficient (`~/Documents/FinalSpark_summary/sufficiency/sufficiency.pdf`). FuncSolver now uses whatever types orgchar finds.
4. **Live FuncSolver** was rewritten on 9 Oct (NeuralControlProgram commit after 88cb31d): it runs through orgchar (`plan_trials`, `HardwareBackend`, `readout.measure_trial`, the calibration's thresholds). Tested on simulated recordings only; not yet run on hardware. Offline it does not yet survive whole pulse counts and trial noise (see FuncSolver's README).
5. **Earlier history** (May–Aug, NCP repo): the live hybrid runs on fs599/618/619/622/626 all failed (R² < 0). Those curves came from the stimulated electrode via the vendor spike database, so they were artefact-dominated.

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
  ORGCHAR_HOME=<tmp> ./orgchar.sh <step> --simulate --yes   # census, screen, curve, then status --simulate
  ```
  Also run `pytest` (side venv; don't install into `ncp`).
- Never infer which electrode was stimulated from artifact size: neighbours clip at ±6 mV too. Use the plan's key mapping.
- Keep raw-data reads minimal (token cost).
- Keep the code lean: the user dislikes bloat and band-aids. Fix root causes, and say plainly when a fix is a heuristic.
- Slides and reports: short phrase titles, no footnotes except references, plain research terminology, legible when projected.
- No AI attribution in commit messages or PR descriptions.
