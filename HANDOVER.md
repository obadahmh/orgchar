# Handover — orgchar (state as of 5 Oct 2026)

How the tool works (method, verdicts, platform quirks) is in `README.md`. This file only holds
what the README doesn't: machines, the current state, open questions and working agreements.

## Goal
Characterise this week's FinalSpark organoids so FuncSolver (`~/NeuralControlProgram/FuncSolver`) can use them as activation functions: three stim→rec pairs, **short** (falling), **band** (peaked) and **long** (rising), each on its own stimulating electrode.

## Machines
- **np14** runs fs666; **neuroplatform10** runs fs667 (week of 30 Sep). FinalSpark replaces all four organoids at once, about weekly, always under a new fsXXX; the user requests it when organoids fatigue.
- Keep both machines on the same commit (`git pull`). GitHub: `obadahmh/orgchar` (private).
- `sessions/` (results) is local to each machine and stays out of git. Raw recordings of both machines are on the shared S3 mount, so `./orgchar.sh status` on one machine lists the other's runs, but not their results.
- `codetest` has passed on np14 (1 Oct). neuroplatform10 ran one on 1 Oct 17:55; check its result there.

## Current state
- **fs666 (np14): falling only.**
  - Organoid 1 (e8–15) is the only lively one (8 electrodes, 180 Hz in total). Organoids 0 and 3 have one electrode each at ~0.2–0.3 Hz.
  - Falling e10→e8 is Good: R² 0.62 on new trials on 2 Oct, and it reproduced on 1 Oct. It is one burst across the organoid; e9, e13 and e15 see it too.
  - Peaked: none (the 1 Oct screen candidate e13→e14 is what the chance check now drops). Rising: none.
- **fs667 (neuroplatform10): falling and peaked, no rising.** Rebuilt on np14 from the S3 recordings of the 1 Oct screen and curve:
  - all four organoids active (38–53 Hz each);
  - falling e10→e13 and peaked e1→e3 both reproduce between development and confirmation, on different stimulating electrodes in different organoids;
  - rising: the screen's only candidate, e30→e25, was flat in the curve.
  - Check this against neuroplatform10's own `report.html`.
- **5 Oct code change** (not yet run on hardware):
  - single-pulse screen and pulses/amplitude curves removed;
  - `status` rewritten;
  - screen chance check per recording electrode;
  - raw files read from S3 instead of copied;
  - alignment starting offset chosen by cost;
  - results over 24 h old marked STALE in `status`; `curve` warns on a screen that old (as `screen` does for the census).
- **FuncSolver integration:**
  - Done: `calibration.json` loads, and offline `--train` works.
  - Not done: live inference, which still uses FuncSolver's own raw readout and a ResponseSweep `assessment.json` gate. Changing that needs the user's go-ahead, because it is in NeuralControlProgram.

## Open questions / next steps
1. **fs667 rising:** `./orgchar.sh screen --fsname fs667 --amplitude 3 --yes` on neuroplatform10. A calibration holds one stimulus, so if a rising pair appears, the curve must also run at 3 µA, with all three electrodes: `curve --amplitude 3 --stims 10,1,<rising stim>`.
2. **Does FuncSolver need all three shapes?** This is a design question for the user. fs666 has one, fs667 two.
3. **Drift:** the fs666 e10 reference fell from 22 to 8.5 events within one hour. Recalibrate right before any FuncSolver session.
4. **Live FuncSolver:** its measurement should call `orgchar.readout.measure_trial` and accept orgchar's checks. Its per-trial recording also treats an unacknowledged stop as an error.
5. The 20 s wait for a stop acknowledgement that never comes may be droppable, saving ~8 min per curve. Test it on hardware before relying on it.

## Working agreements
- The user runs all hardware steps. Claude only analyses and changes code, unless explicitly asked.
- Don't call the SDK's unwrapped start/stop/debuginfo endpoints.
- Before handing over a code change, run the full chain in simulation from an empty home:
  ```bash
  ORGCHAR_HOME=<tmp> ./orgchar.sh <step> --simulate --yes   # census, screen, curve, then status --simulate
  ```
  Also run `pytest`.
- Never infer which electrode was stimulated from artifact size: neighbours clip at ±6 mV too. Use the plan's key mapping.
- Keep the code lean: the user dislikes bloat and band-aids. Fix root causes, and say plainly when a fix is a heuristic.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
