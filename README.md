# orgchar: organoid characterisation and the hybrid network

Three steps, for this week's FinalSpark organoids:
1. which organoids fire at all;
2. which stimulating → recording pairs respond to the length of a pulse train;
3. the best pair's curve, measured now, the hybrid network trained on it, and the network run on the tissue.

```
census   ~8 min, no stimulation       → which organoids are alive?             GO / NO-GO
screen   ~30 min (4 live organoids)   → which pairs respond to trains?         candidates
solve    ≤ 50 min (one pair)          → network on the tissue                  R² against the true solution
```

One paste runs all three; each step only starts if the previous one found something, and solve
stops itself before its costly part if the curve cannot support the network:

```bash
cd ~/Documents/orgchar && git pull && FS=fs666 && \
./orgchar.sh census --fsname $FS --yes && \
./orgchar.sh screen --fsname $FS --yes && \
./orgchar.sh solve  --fsname $FS --yes
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
4. **Responses are bursty and heavy-tailed.** Hence rank tests and single-trial scores rather than
   means.
5. **Responses change within hours** (fs675, 15 Oct 2026: the 2-pulse response fell from 48 to 27
   spikes between the morning's curve and the live run). Hence solve measures its curve in the same
   session, minutes before the network uses it.

## The method

### Step 1: census (no stimulation)

Record 5 min of spontaneous activity on all 32 electrodes. A channel is **active** if all of the following hold:
- ≥ 20 spikes and ≥ 0.1 Hz;
- more than 10× the rate that Gaussian noise alone would give (Rice's formula, from the channel's own spectrum);
- mean trough ≥ 5.5 × noise.

An organoid (block of 8) with ≥ 1 active electrode is **LIVE**. LIVE covers a wide range: on fs666
(1 Oct) one organoid had 8 active electrodes at 180 Hz in total and another a single one at 0.2 Hz.
`status` shows both numbers per organoid.

### Step 2: screen (which pairs respond to trains)

Every electrode of each live organoid gets hardware trains of 2, 10 and 20 pulses (6 trials each; `--levels`, `--reps` to change; at least 5 trials, since 4 against 4 can never reach p < 0.01), delivered and read out by `measure` as in solve (see Step 3), on every other electrode, so pairs across organoids serve as controls. For every stimulating → recording pair:
- **falling** if 2-pulse trials beat 20-pulse trials, **rising** if the reverse, **peaked** if the best middle length beats both ends;
- the comparison is on single trials (rank test, one-sided p < 0.01, at least 1.5 events apart);
- "single trial wins" = how often one trial at the favoured length beats one at the other.

**Chance check.** With 6 trials against 6 the smallest possible p is 0.0011, so p < 0.01 alone
lets noise through, mostly on bursting electrodes. Pairs across organoids cannot respond, so for
each recording electrode they show what chance reaches there. A candidate is dropped if a pair from
another organoid into the same electrode did at least as well, in any shape (p as small, as many
events apart). This is a heuristic with ~24 such pairs per electrode, and none when only one
organoid is screened. On fs666 (1 Oct) it drops e13→e14, the only peaked candidate, which e19→e14
and e26→e14 beat and a full curve did not confirm; it keeps all nine falling pairs from e9/e10.

solve measures the clearest of these pairs (highest single-trial win, then the largest difference); `--pairs` chooses others. One upload holds at most 8 stimulating electrodes (keys 8–14 carry train codes), so electrodes are dealt two per organoid per recording and all organoids run side by side; each electrode gets its lengths in random order. No sham trials since 16 Oct 2026 (see Step 3).

Why not single pulses: on fs666 (1 Oct 2026) a single-pulse screen picked e2, e20, e29 and e10; only e10 responded after trains.

### Step 3: solve (the hybrid network on the tissue)

**The network.** One hidden layer of 12 units. Unit *j* turns the time *t* into a pulse count
p = w t / T + b, clipped to 2–20 and moved to the nearest count the sweep measured (2, 4, …, 20: a
count in between would be a guess, and on 19 Oct 2026 the guesses were far off); the tissue answers with its pair's evoked spikes after a train
of p pulses; the computer sums the units' responses with weights. The activation is the measured
table (mean evoked spikes per pulse count, linear in between), not a fitted formula. Units take the
available curves in turn: one pair is enough in principle
(`~/Documents/FinalSpark_summary/sufficiency/sufficiency.pdf`), and in simulation a second response
type added nothing.

**The task.** m y'' + c y' + k y = 0, y(0) = 1, y'(0) = 0 on [0, 10], with m, c, k = 1, 2, 10, read
off TUM's Fig. 4f.

Four steps in one session, each gated:
1. **Sweep:** 8 trains at 2, 4, …, 20 pulses on each pair's stimulating electrode. Gate: the response
   depends on the pulse count (Kruskal–Wallis p < 0.01), and the stimulation artifact has faded by
   the counting window (below 3× its noise floor: the median across trials of the window's first 5 ms,
   in units of what a median of that many trials varies by anyway; spikes are not time-locked and
   drop out, an artifact tail does not). A pair that fails is dropped; none left, stop.
2. **Train** (no stimulation): the weak-form residual of the ODE (smooth overlapping test functions,
   integrated by parts, so only values of y are needed), plus y(0) and the output's noise variance
   from the measured trial spread. First with fractional pulse counts, then whole ones; 8 starts,
   lowest loss kept (40,000 steps, about 6 min). Gate: the expected R² must beat a plain decay's (see below). Expected = with
   the noise of the run's 6 trials per count and the sweep's own uncertainty, each count's error
   shared by every time point that uses it (as in step 3). Else stop: no further stimulation.
3. **Evaluate:** at 25 held-out times, each unit asks for a pulse count. Each distinct (electrode,
   count) is measured 6 times and reused wherever it is asked for (on the 15 Oct live trials: the
   same R² as one train per request, with 20× fewer pulses). y(t) is computed from those responses.
4. **Drift:** the run's responses against the sweep, at the counts both measured.

**Reading the R².** A plain decay a × exp(−kt) + b, with no oscillation, scores 0.80 at the 25
times, because the solution is near zero after t ≈ 4. Only an R² above that shows the oscillation;
the report says which side a run is on, and gate 2 stops a run whose expected R² does not clear it.
Training needs 40,000 steps (about 6 min on np14): at 6,000 every network stopped at the decay. With
40,000, one measured curve is enough: in simulation, expected R² 0.97 and 0.92 on the simulated tissue,
trough −0.40 (true −0.35) and rebound +0.10 (true +0.12), with 194 trains in all.

**Measuring** (`orgchar/measure.py`, the one hardware primitive): trains are grouped into recordings
by length (a length is part of the upload), at most 6 min each, each organoid ≥ 4 s between trains;
each recording is read out as soon as it is saved, its responses written at once; trains lost to
dropped rows are requested again (2 retries). Spike thresholds are frozen from the first recording.
No sham trials: a no-current trigger evokes nothing (tested many times), and each trial's own
pre-train baseline is subtracted.

**Readout** (`orgchar/readout.py`): 300–3000 Hz band-pass, crossings of −5× noise counted
**10–210 ms after the train**, minus the 500 ms before it (scaled). The window starts at the end of
the train, so later for longer trains. On fs666 e10, 2–4 pulses set off a burst across the whole
organoid lasting ~400 ms; longer trains cut it short and leave the organoid quiet. Both effects make
the curve fall.

## Running it

Everything runs in the FinalSpark `ncp` environment through `./orgchar.sh`:

```bash
cd ~/Documents/orgchar
./orgchar.sh census --fsname fs652 --dry-run    # shows the plan and its duration, touches nothing
./orgchar.sh census --fsname fs652              # asks you to type the fsname to confirm
./orgchar.sh screen --fsname fs652              # after-train screen of the live organoids from the census
./orgchar.sh solve  --fsname fs652              # the screen's clearest pair: sweep, train, run the network
./orgchar.sh status                             # every organoid run in the last 7 days (or --fsname fs652)
./orgchar.sh analyze sessions/fs652/<step dir>  # re-analyse offline, e.g. after changing a setting
```

- Add `--simulate` to any step to run it on a synthetic organoid with known answers. Results go under `sessions/SIMULATED_<fsname>/` and never mix with real data.
- For long steps, use `tmux new -s orgchar`.
- A lock stops orgchar and the ResponseSweep tools from driving the platform at the same time.

Useful options:
- `screen --blocks 1,2`: override the census choice. `screen --electrodes 21,15`: a subset.
- `solve --pairs 13:10`: choose the pair(s) (stim:rec); `--sweep-reps 8`, `--reps 6` (trains per pulse count in the sweep and the run), `--width 12`, `--points 26`.
- `screen --amplitude 3` / `--phase 150`: more current. Stimulation limits are enforced before any upload: ≤ 3 µA, ≤ 200 µs, ≤ 0.4 nC/phase, ≤ 20 pulses. Change them in `orgchar/protocol.py` (`LIMITS`).

Each step directory contains `plan.json` (the step's settings), `result.json` and `report.html`
(open in a browser; figures are embedded). Recordings sit one folder each, with `sent.csv`,
`trials.csv` (recorded trigger edge ↔ planned event), `close_status.json` and `remote_path.txt`:
- census and codetest: directly in the step directory;
- screen: in `pass/` (and `retry1/`, `retry2/` for trains requested again), each with its own
  `plan.json`, and `responses.csv` per recording and for the whole step;
- solve: the same layout in `sweep/` and `evaluate/`, plus `model.pt`. Its report leads with the result
  and its match score, then every check (each linked to its figure), what the organoid did (its
  response before and during the Run, and how a response is counted on a real voltage trace), how the
  network turns time into an answer (interactive), the session as a timeline, and the details. A box
  under the title defines the few words it uses. `sessions/runs.html` lists every solve run on this
  machine, one row each, linking to its report; solve and `analyze` rewrite it.

`analyze` re-reads a census or a screen (one recorded since 16 Oct 2026) from its recordings, and
rebuilds a finished solve's report.

**Raw recordings stay where the platform saves them**, `/mnt/s3-intanraws` (read at ~60 MB/s);
`remote_path.txt` points to each one. Hardware runs before 5 Oct 2026 copied them into the segment
folder as `raw.h5` (41 GB in the first week); those copies are still used when present.

**`status`** shows, per organoid: the latest complete census (active electrodes and Hz per
organoid), the latest screen (best pair per shape, and what the chance check dropped), the latest
solve, the last run, and runs recorded by another machine. A run on neuroplatform10 is visible from np14 through its
recordings on the S3 mount, but its results stay on neuroplatform10. Anything recorded more than
24 h ago is marked STALE, and `screen` / `solve` warn when they build on a census / screen that old:
responses change within a day (fs666 e10's reference fell from 22 to 8.5 events within one hour).

## Reading the verdicts

| Step | Verdict | Do |
|---|---|---|
| census | `GO: screen blocks …` | run screen |
| census | `NO-GO` | do not stimulate; re-check later, contact FinalSpark |
| screen | `CANDIDATES_FOUND` | run solve (it measures the clearest pair) |
| screen | `NO_CANDIDATE` | nothing responds to these trains; try `--amplitude 3`, or re-check the census |
| screen | `CONTROLS_FAILED` | far more other-organoid pairs pass than p < 0.01 allows: artifact reaches the readout; do not trust the candidates |
| solve | `STOPPED after the sweep` | no pair responds cleanly now; try another screen candidate (`--pairs`) or re-screen |
| solve | `STOPPED after training` | the network on this curve could not show the oscillation; nothing more was stimulated |
| solve | `R² … on the tissue` | compare with the expected R² and the plain decay's (in the verdict) |

Any step can also end `INCOMPLETE: …` (a recording cut short or a failed final disable): inspect, then re-run it.

## Notes

- Each fsXXX is one preparation: FinalSpark replaces all four organoids at once, under a new fsXXX, so a new fsname starts from `census` with nothing carried over.
- Do not read the vendor spike database for evoked responses. It counts artifacts and knows nothing about cross-talk.
- `solve` replaces the `curve` step and FuncSolver's calibration (`~/NeuralControlProgram/FuncSolver`), 16 Oct 2026. Old curve folders keep their reports.

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
- Before first use on a machine, run `./orgchar.sh codetest --fsname <fs>` (about 3 min). It uploads the screen's best electrodes' real parameters, sends 60 codes (each followed by a sham marker), then 5 real pulses as a positive control, and prints PASS or FAIL with the evidence.
- Once codetest has passed on a machine, screen and solve use codes by default there (`--no-codes` to switch off). Run it again after FinalSpark changes their setup. Passed on np14, 1 Oct 2026 (130/130 code triggers recorded, 60/60 codes read back, 3.3 µV after codes vs 6259 µV after real pulses).

## What is and is not verified

- **On the platform:** census and screen have run on fs660, fs666, fs667, fs674 and fs675 (29 Sep–15 Oct). The curve step that solve replaces ran there too; on fs666 the e10 falling curve reproduced on 1 and 2 Oct, and its events have the shape and size of spontaneous spikes, not artifact. solve's pieces ran live through FuncSolver on fs675 (14–15 Oct): trains by plan_trials, the readout, reuse by pulse count.
- **Trigger matching:** all 62 fs666 segments of 1–2 Oct and 110 fs675 FuncSolver segments give identical trial tables with the current alignment.
- **Simulation:** census → screen → solve finds the planted pairs; an organoid with artifacts but no responses gives `NO_CANDIDATE`, and solve on it stops after the sweep.
- **Not verified:** `solve` itself on hardware.
- **Codes on keys 8–14 (and codetest's marker key 15) assume no electrode is bound to them.** Each segment starts with disable-all and only the planned electrodes (keys 0–7) are uploaded; every coded run checks for an artifact after code triggers.

Tests: 28, about 4 min. pytest is not in `ncp`, so use a side venv that leaves `ncp` untouched:
`python -m venv --system-site-packages /tmp/t && /tmp/t/bin/pip install pytest && /tmp/t/bin/python -m pytest -q` (create the venv with the `ncp` python).
