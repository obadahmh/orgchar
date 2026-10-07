# orgchar: weekly organoid characterisation for FuncSolver

Three steps tell you, for this week's FinalSpark organoids:
1. which organoids fire at all;
2. which electrodes give FuncSolver's three response shapes (falling, peaked, rising) after a train;
3. the full curve of those electrodes, checked on new trials, as FuncSolver's `calibration.json`.

```
census   ~8 min, no stimulation       → which organoids are alive?             GO / NO-GO
screen   ~40 min (4 live organoids)   → which electrodes give which shape?     candidates
curve    40–65 min                    → curves checked on new trials           calibration.json
```

One paste runs all three; each step only starts if the previous one found something:

```bash
cd ~/Documents/orgchar && git pull && FS=fs666 && \
./orgchar.sh census --fsname $FS --yes && \
./orgchar.sh screen --fsname $FS --yes && \
./orgchar.sh curve  --fsname $FS --yes
```

Then `./orgchar.sh status` says where every organoid run this week stands.

## Why the method looks like this

Re-analysing the raw HDF5 of earlier work (Aug–Sep 2026) showed four problems:

1. **Often there was nothing to read.** fs635, fs636 and fs647 after 9 Sep fired no more threshold
   crossings than noise alone gives; a two-day calibration ran entirely on silent tissue. Hence the
   census, and the rule: never stimulate a silent organoid.
2. **Artifacts were counted as spikes.** After a pulse, the stimulated organoid's electrodes carry
   hundreds of µV for 3–10 ms (the stimulated electrode saturates at ±6 mV); the vendor spike
   database counts this too. Hence readout windows that start well after the stimulation, an
   artifact check on every pair, and never reading the stimulated electrode.
3. **Cross-talk has a fixed geometry**: up to ~1.7 mV on electrodes near 31 − *e* in another
   organoid. Pairs across organoids cannot respond, so they serve as negative controls.
4. **Responses are bursty and heavy-tailed**, and FuncSolver reads one response per train. Hence
   rank tests and single-trial scores rather than means.

## The method

### Step 1: census (no stimulation)

Record 5 min of spontaneous activity on all 32 electrodes. A channel is **active** if all of the following hold:
- ≥ 20 spikes and ≥ 0.1 Hz;
- more than 10× the rate that Gaussian noise alone would give (Rice's formula, from the channel's own spectrum);
- mean trough ≥ 5.5 × noise.

An organoid (block of 8) with ≥ 1 active electrode is **LIVE**. LIVE covers a wide range: on fs666
(1 Oct) one organoid had 8 active electrodes at 180 Hz in total and another a single one at 0.2 Hz.
`status` shows both numbers per organoid.

### Step 2: screen (which electrodes give which shape after a train)

Every electrode of each live organoid gets hardware trains of 2, 6, 12 and 20 pulses (6 trials each), measured exactly as FuncSolver will (events 10–210 ms after the train; see Step 3). For every stimulating → recording pair:
- **falling** if 2-pulse trials beat 20-pulse trials, **rising** if the reverse, **peaked** if the best middle length beats both ends;
- the comparison is on single trials (rank test, one-sided p < 0.01, at least 1.5 events apart);
- "single trial wins" = how often one trial at the favoured length beats one at the other: the number that decides whether FuncSolver can use it.

**Chance check.** With 6 trials against 6 the smallest possible p is 0.0011, so p < 0.01 alone
lets noise through, mostly on bursting electrodes. Pairs across organoids cannot respond, so for
each recording electrode they show what chance reaches there. A candidate is dropped if a pair from
another organoid into the same electrode did at least as well, in any shape (p as small, as many
events apart). This is a heuristic with ~24 such pairs per electrode, and none when only one
organoid is screened. On fs666 (1 Oct) it drops e13→e14, the only peaked candidate, which e19→e14
and e26→e14 beat and the curve did not confirm; it keeps all nine falling pairs from e9/e10.

The best remaining pair per shape decides which electrodes the curve stimulates (one electrode may serve two shapes). One upload holds at most 8 stimulating electrodes (keys 8–14 carry train codes), so electrodes are dealt two per organoid per recording and all organoids run side by side.

Why not single pulses: on fs666 (1 Oct 2026) a single-pulse screen picked e2, e20, e29 and e10; only e10 responded after trains.

### Step 3: curve (FuncSolver's activation functions)

FuncSolver (NeuralControlProgram) needs three organoid pairs, one per function shape: **short** (falling: more pulses → fewer events), **band** (peaked) and **long** (rising). It sends **one** train per evaluation and reads **one** response, so a curve is only as good as single trials are close to it. Each evaluation gives the three roles different pulse counts at once, so each role needs its own stimulating electrode.

`curve` measures exactly what FuncSolver uses:
- **Input:** a hardware train of 2, 4, …, 20 pulses, 10 ms apart, 2 µA × 100 µs. One trigger; the stimulator repeats the pulse, as FuncSolver delivers it. A train's length is part of the upload, so there is one recording per pulse count.
- **Output** (`orgchar/readout.py`, one function for calibration and live use): 300–3000 Hz band-pass, events = crossings of −5× noise counted **10–210 ms after the train**, minus the 500 ms before it (scaled), minus the mean of no-current sham trials. The raw voltage swings by hundreds of µV on some electrodes; counting on it unfiltered would count those swings.
- **Order:** reference (10 pulses) → **fit** (every count, shuffled) → reference → **test** (reshuffled) → reference. Folders are named `ref0`, `fit_n02` … `fit_n20`, `ref1`, `test_n02` … `test_n20`, `ref2` (sessions before 6 Oct 2026 used `dev_` and `con_`). Each organoid ≥ 4 s between trains, trains ≥ 1 s apart. Every other electrode of each stimulated organoid is measured.
- **Fit trials alone** choose the pairs (FuncSolver's shape rule, then the best cross-validated single-trial fit, each role on its own stimulating electrode) and fit FuncSolver's curve formulas. **Test trials** only score those frozen curves: R² and error on new single trials. The references show drift.
- **Artifact check:** the median across trials of the window's first 5 ms, in units of its own noise floor. Spikes are not time-locked and drop out; an artifact tail does not. Pairs over 3× are excluded. On the fs647 hardware-train recordings of 8 Sep the largest value was 2.6×.

Verdict per function: **Good** (new-trial R² ≥ 0.5, no large drift), **Fair** (≥ 0.2), **Poor**, or **Missing**.

Outputs: `calibration.json` in FuncSolver's format (schema 1, `nearest_validated_abs_v1`; tested with `FuncSolver/calibration.apply_calibration`), copied to `sessions/<fsname>/calibration_latest.json` when one complete run finds all three functions; `report.html` leads with the three functions, then single-trial evidence for each, then every pair measured.

The window starts at the end of the train, so it starts later for longer trains. On fs666 e10, 2–4 pulses set off a burst across the whole organoid lasting ~400 ms; longer trains cut it short and leave the organoid quiet afterwards. Both effects make the curve fall. The four "pairs" e10→e8, e9, e13, e15 are that one burst seen on four electrodes.

## Running it

Everything runs in the FinalSpark `ncp` environment through `./orgchar.sh`:

```bash
cd ~/Documents/orgchar
./orgchar.sh census --fsname fs652 --dry-run    # shows the plan and its duration, touches nothing
./orgchar.sh census --fsname fs652              # asks you to type the fsname to confirm
./orgchar.sh screen --fsname fs652              # after-train screen of the live organoids from the census
./orgchar.sh curve  --fsname fs652              # FuncSolver curves on the screen's candidate electrodes
./orgchar.sh status                             # every organoid run in the last 7 days (or --fsname fs652)
./orgchar.sh analyze sessions/fs652/<step dir>  # re-analyse offline, e.g. after changing a setting
```

- Add `--simulate` to any step to run it on a synthetic organoid with known answers. Results go under `sessions/SIMULATED_<fsname>/` and never mix with real data.
- For long steps, use `tmux new -s orgchar`.
- A lock stops orgchar and the ResponseSweep tools from driving the platform at the same time.

Useful options:
- `screen --blocks 1,2`: override the census choice. `screen --electrodes 21,15`: a subset.
- `curve --stims 0,10,30`: choose the stimulating electrodes; `curve --levels 2,4,8,12,16,20 --reps 12`.
- `--amplitude 3` / `--phase 150`: more current. Stimulation limits are enforced before any upload: ≤ 3 µA, ≤ 200 µs, ≤ 0.4 nC/phase, ≤ 20 pulses. Change them in `orgchar/protocol.py` (`LIMITS`).

Each step directory contains:
- `plan.json` (everything sent);
- one folder per segment with `sent.csv`, `trials.csv` (recorded trigger edge ↔ planned event), `close_status.json` and `remote_path.txt`;
- `result.json`, `report.html` (open in a browser; figures are embedded), and for a curve `calibration.json`.

**Raw recordings stay where the platform saves them**, `/mnt/s3-intanraws` (read at ~60 MB/s);
`remote_path.txt` points to each one. Hardware runs before 5 Oct 2026 copied them into the segment
folder as `raw.h5` (41 GB in the first week); those copies are still used when present.

**`status`** shows, per organoid: the latest complete census (active electrodes and Hz per
organoid), the latest screen (best pair per shape, and what the chance check dropped), the best pair
found so far for each FuncSolver function with its grade and date, the calibration, the last run,
and runs recorded by another machine. A run on neuroplatform10 is visible from np14 through its
recordings on the S3 mount, but its results stay on neuroplatform10. Anything recorded more than
24 h ago is marked STALE, and `screen` / `curve` warn when they build on a census / screen that old:
responses change within a day (fs666 e10's reference fell from 22 to 8.5 events within one hour).

## Reading the verdicts

| Step | Verdict | Do |
|---|---|---|
| census | `GO: screen blocks …` | run screen |
| census | `NO-GO` | do not stimulate; re-check later, contact FinalSpark |
| screen | `CANDIDATES_FOUND` | run curve (it stimulates the candidates) |
| screen | `NO_CANDIDATE` | nothing for FuncSolver with these trains; try `--amplitude 3`, or re-check the census |
| screen | `CONTROLS_FAILED` | far more other-organoid pairs pass than p < 0.01 allows: artifact reaches the readout; do not trust the candidates |
| curve | `ALL THREE FUNCTIONS FOUND` | train FuncSolver on `calibration_latest.json` (command in the report) |
| curve | `SOME FUNCTIONS MISSING` | re-run with other `--stims` for the missing shape, or more current |
| curve | `NO USABLE FUNCTION` | the organoid does not give FuncSolver's shapes with these trains |

Any step can also end `INCOMPLETE: …` (a recording cut short or a failed final disable): inspect, then re-run it.

## Using it for function approximation

- Train FuncSolver offline on the calibration (no hardware):
  `cd ~/NeuralControlProgram && python FuncSolver/main.py --train --fsname <fs> --calibration <path to calibration.json>`.
- **Live inference is not connected yet.** FuncSolver's live path still uses its own readout (`experiment_control/measurement.py`, threshold crossings on the unfiltered voltage) and requires a ResponseSweep `assessment.json`. A calibration from orgchar declares `method: orgchar_bandpass_events_v1`, which FuncSolver's live readout refuses, so it cannot silently measure differently from the calibration. Connecting it means FuncSolver calling `orgchar.readout.measure_trial` and accepting orgchar's checks.
- Responses change within a day. Re-run `curve` at the start of each FuncSolver session; after a medium change, start again from `census`.
- Each fsXXX is one preparation: FinalSpark replaces all four organoids at once, under a new fsXXX, so a new fsname starts from `census` with nothing carried over.
- Do not read the vendor spike database for evoked responses. It counts artifacts and knows nothing about sham or cross-talk.

## Platform quirks (handled)

| What happens | What orgchar does |
|---|---|
| The raw file loses whole rows of 1280 samples (42.7 ms), 1–4 per 6-min recording | Aligns every recorded trigger to its send time and detects each lost row; the train just before a lost row is left out |
| Single triggers are recorded up to ~13 ms after their send time (trigger-path delay; fs666 curves, 30 Sep) | Global minimum-cost alignment of the whole recording (`backends._align`): delays can only be positive, lost rows only shift edges by exactly −42.67 ms, clock drift is fitted; one odd trigger cannot derail the rest. The starting offset is chosen by the same cost, so rows lost early in a sparse recording cannot shift it |
| Triggers were delivered late and bunched by TCP batching (fs666 curve, 30 Sep) | Sends triggers with TCP_NODELAY |
| "Stop recording" is often not acknowledged, although the recording stopped | Sends one stop request and waits 20 s, no retries. Completeness is judged from the recorded duration |
| The platform answers one request at a time and is busy while saving a recording; a request repeated while it is busy left the board in `Invalid run mode` (fs666, 1 Oct) | After stopping, waits until the file appears saved on the S3 mount, then sends ONE final disable (180 s), never repeated. No stimulation is possible meanwhile: the trigger connection is already closed |
| Every upload fails with `Invalid run mode` (even `./orgchar.sh disable`) | Nothing: the board is stuck. Restart the Intan service, then re-run the step |
| Between two recordings the platform needs ~50 s plus half the recording's length (mostly saving the file) | `--dry-run` durations include it (measured on fs666, 2 Oct) |

### Train codes (`--codes`)

Timing alone says *where* each train is, not *which* train it is; the alignment above infers that. With `--codes`, the recording carries the answer itself: ~100 ms before each train, two triggers on keys 8–14 spell the train's number (base 7, 49 codes; trains ≥ 1 s apart, so neighbours are never confused). Keys 8–14 are bound to no electrode, like the sham key 15, so no current flows.

- Codes are read straight from the recording, independently of the alignment, and compared with it. A contradiction excludes that train (`note = code mismatch`) and is reported; a code lost in a dropped row just leaves that train unconfirmed (`code = missing`).
- Every coded run checks that code triggers leave no artifact (< 50 µV; real stimulation leaves thousands). If they ever do, the run stops with an error.
- Before first use on a machine, run `./orgchar.sh codetest --fsname <fs>` (about 3 min). It uploads the curve electrodes' real parameters, sends 60 codes (each followed by a sham marker), then 5 real pulses as a positive control, and prints PASS or FAIL with the evidence.
- Once codetest has passed on a machine, screen and curve use codes by default there (`--no-codes` to switch off). Run it again after FinalSpark changes their setup. Passed on np14, 1 Oct 2026 (130/130 code triggers recorded, 60/60 codes read back, 3.3 µV after codes vs 6259 µV after real pulses).

## What is and is not verified

- **On the platform:** census, screen and curve have run on fs660 (29 Sep), fs666 (30 Sep–2 Oct, np14) and fs667 (30 Sep–1 Oct, neuroplatform10). On fs666 the e10 falling curve reproduced on 1 and 2 Oct (R² 0.62 on new trials on 2 Oct); its events have the shape and size of spontaneous spikes, not artifact.
- **Trigger matching:** all 62 fs666 segments of 1–2 Oct give identical trial tables with the current alignment.
- **Simulation:** the full census → screen → curve chain finds the planted falling, peaked and rising pairs, and an organoid with artifacts but no responses gives `NO_CANDIDATE`.
- **Not verified:** reading raw files from the S3 mount during a hardware run (5 Oct change; the same files read fine offline); FuncSolver live inference (not connected).
- **Sham on key 15 and codes on keys 8–14 assume no electrode is bound to them.** Each segment starts with disable-all and only the planned electrodes (keys 0–7) are uploaded; every coded run checks for an artifact after code triggers.

Tests: about 4 min. pytest is not in `ncp`, so use a side venv that leaves `ncp` untouched:
`python -m venv --system-site-packages /tmp/t && /tmp/t/bin/pip install pytest && /tmp/t/bin/python -m pytest -q` (create the venv with the `ncp` python).
