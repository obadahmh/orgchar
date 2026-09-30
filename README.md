# orgchar — weekly organoid characterisation

Three short steps tell you, for this week's FinalSpark organoids:
1. which organoids fire at all;
2. which electrode pairs give a real stimulus-evoked response;
3. the input→output curve of those pairs.

Step 3 produces `card.json`, the file function approximation should read.

```
census   5 min, no stimulation      → which organoids are alive?        GO / NO-GO
screen   ~6 min per live organoid   → which stim→record pairs respond?  responsive pairs
curve    ~10–15 min                 → input-output curves               card.json
```

About 30–40 min of platform time per week in total.

## Why the output could not be read before

These findings come from re-analysing the raw HDF5 already on this machine and on `/mnt/s3-intanraws`.

1. **Most of the time there was nothing to read.**
   - A 5 σ threshold on 300–3000 Hz finds no spikes beyond what pure noise produces in:
     - fs636 (31 Aug–3 Sep);
     - fs635 (2–4 Sep);
     - fs647 after 9 Sep. On 21 Sep: 4 crossings in 31 s over all 32 electrodes, which is what noise alone gives.
   - The two-day calibration (21–23 Sep, 1223 trials) ran entirely on silent tissue.
   - fs647 on 8–9 Sep was the only exception: blocks 1–2 fired at 1–4 Hz per electrode, with clean 30–60 µV spikes.
2. **Artifacts were counted as spikes.**
   - After a pulse, the same-organoid electrodes carry a tail of hundreds of µV for 3–10 ms. The stimulated electrode itself is saturated at ±5–6 mV and takes longer to recover.
   - With 2–3 ms of blanking, the fs647 own-organoid "spike rate" 2–7 ms after a pulse was 40–145 Hz, against a true ~6 Hz.
   - The vendor spike database also counts artifacts. Recording on the stimulated electrode, or in a window that overlaps the train, adds about two "spikes" per pulse. That produced the linear "long-pass" curves in `Replication/`.
3. **Cross-talk has a fixed geometry.**
   - Stimulating electrode *e* puts ~1–6 mV on its own 8-electrode block.
   - It also puts up to ~1.7 mV on electrodes near **31 − *e*** in another organoid, which is wiring/connector coupling.
   - The same adjacent pairs "responded" in every organoid: 1→2, 22↔23, 25↔26.
4. **The real response is per pulse, short-latency.**
   - When the tissue was alive (fs647, 8 Sep, 20 ms between pulses), specific pairs responded 8–20 ms after each pulse:

     | Stim → record | Evoked spikes / pulse |
     |---|---|
     | 21 → 23 | 1.06 (16× spontaneous) |
     | 15 → 13 | 0.95 |
     | 23 → 21 | 0.59 |

   - Nothing changed 10–300 ms *after* the train, which is the window used so far. With 10 ms between pulses, each response is buried under the next pulse's artifact.
5. **Controls could not catch any of this.** Shams had no artifact, or there were none, and n was 3–5 trials per condition.

## The method

### Step 1 — census (no stimulation)

Record 5 min of spontaneous activity on all 32 electrodes. A channel is **active** if all of the following hold:
- ≥ 20 spikes and ≥ 0.1 Hz;
- more than 10× the rate that Gaussian noise alone would give (Rice's formula, from the channel's own spectrum);
- mean trough ≥ 5.5 × noise.

An organoid (block of 8) with ≥ 1 active electrode is **LIVE**.

**Rule: never stimulate a silent organoid for characterisation.** With nothing firing, every "response" is artifact. Re-run the census later in the week, or ask FinalSpark about the preparation.

### Step 2 — screen (which pairs respond)

- Deliver single biphasic pulses on each electrode of each live organoid: 2 µA × 100 µs, negative first, 30 per electrode.
- **Sham markers** go out on trigger key 15, with no electrode bound to it: 30 per organoid.
- Stim and sham trials are randomly interleaved, 1.0–1.5 s apart.

For every pulse and every recording electrode except the stimulated one:
1. **Remove the artifact.** Subtract the median artifact of that electrode and waveform. The median is taken across pulses, so jittered spikes stay in the data.
2. **Blank until the artifact is over.** "Over" means the spread of the residual across pulses, measured *below 300 Hz*, is back to noise level. Spikes carry little energy below 300 Hz, so a real response does not extend the blank. Typical blank: 3–4 ms.
3. **Detect spikes** at −5 × robust noise in the 300–3000 Hz band. Drop events that occur at the same instant in ≥ 2 organoids; that is interference.
4. **Count spikes** from the end of the blank to 30 ms after the pulse.
5. **Test against the null.** Compare the count with 20 000 same-length windows drawn from quiet periods of the same electrode. That keeps spontaneous burstiness in the null.

A pair is **responsive** if all of the following hold:
- Bonferroni p < 0.05;
- ≥ 0.1 extra spikes per pulse and ≥ 5 extra spikes in total;
- the evoked spikes have the same shape as that electrode's spontaneous spikes (r ≥ 0.8).

Each responsive pair gets a status:

| Status | Meaning |
|---|---|
| **✓ Use** | ≥ 0.5 extra spikes per pulse, no artifact warning, not contradicted by an earlier screen of the same organoid |
| **◐ Weak** | responsive but below 0.5 extra spikes per pulse. On fs666 (30 Sep) such pairs did not reproduce |
| **! Check** | an artifact warning (below), or not responsive in the earlier screen; decide from the evidence traces |

The two artifact warnings are not automatic exclusions, because a very reliable direct response can look the same:
- **Locked to one latency**: ≥ 40 % of evoked spikes fall in a single 1 ms window. Artifact recurs at a fixed time; synaptic responses vary by milliseconds.
- **Pile-up where the blank ends**: evoked spikes crowd into the first 1 ms after the blank (≥ 3× an even spread). The artifact tail was not fully removed.

On fs647, 8 Sep, the strong pairs (21→23, 23→21, 8→15) raised no warning. e15→e14 was locked to ~9 ms (a small bump on every pulse); e15→e13 had a 3–4 ms pile-up on top of a genuine 12–17 ms response.

**Built-in negative controls**, analysed exactly like the real pairs:
- **Cross-organoid pairs** (stimulate block A, record block B). No synapse connects them, but they get the mirror cross-talk. They measure whether artifact leaks into the readout.
- **Sham pairs.** They measure whether the trigger path or the analysis window alone produces anything.

If any control comes out significant, the verdict is `PIPELINE_LEAK` and no pair should be trusted.

### Step 3 — curve (input → output)

By default `curve` takes the pairs the screen report recommends: one stimulating electrode per organoid (up to 4), the one with the best **Use** pair (pairs reproduced in an earlier screen first), and its best **Use** pairs (up to 3). Pass `--pairs` to choose differently.
- **Pulse count (default).**
  - 1, 2, 4, 6, 8, 10 pulses, 30 ms apart. That is long enough to read the 8–20 ms response of every pulse.
  - The pulses are timed in software, so all levels and electrodes are randomly interleaved in one recording.
  - 20 trials per level, plus 20 shams as the zero-input level.
- **Amplitude.**
  - 0.5, 1, 1.5, 2, 3 µA.
  - A new amplitude needs a new upload, so the order of levels is reshuffled in each of 3 passes, spreading slow drift over all levels.

**Output** = evoked spikes per trial (post-artifact windows of all pulses), minus the spontaneous expectation for the same duration. 0 means no effect.

A curve is **usable** if all of the following hold:
- the top level beats sham (Mann-Whitney p < 0.01);
- the odd-trial and even-trial curves agree (Spearman ρ ≥ 0.6);
- the dynamic range exceeds the trial-to-trial SD.

Usable curves go to `sessions/<fsname>/card_latest.json`, together with each level's mean, SEM and SD, a sigmoid fit, and the readout settings.

## Running it

Everything runs in the FinalSpark `ncp` environment through `./orgchar.sh`:

```bash
cd ~/Documents/orgchar
./orgchar.sh census --fsname fs652 --dry-run    # shows the plan, touches nothing
./orgchar.sh census --fsname fs652              # asks you to type the fsname to confirm
./orgchar.sh screen --fsname fs652              # uses the live blocks from the census
./orgchar.sh curve  --fsname fs652              # uses the responsive pairs from the screen
./orgchar.sh status --fsname fs652
./orgchar.sh analyze sessions/fs652/<step dir>  # re-analyse offline, e.g. after changing a setting
```

- Add `--simulate` to any step to run it on a synthetic organoid with known answers. Results go under `sessions/SIMULATED_<fsname>/` and never mix with real data.
- For long steps, use `tmux new -s orgchar`.
- A lock stops orgchar and the ResponseSweep tools from driving the platform at the same time.

Useful options:
- `screen --blocks 1,2`: override the census choice.
- `screen --electrodes 21,15 --reps 20`: quick re-check of known pairs before a session.
- `curve --pairs 21:23,15:13`, `curve --vary amplitude --levels 0.5,1,2,3`.
- Stimulation limits are enforced before any upload: ≤ 3 µA, ≤ 200 µs, ≤ 0.4 nC/phase, ≤ 20 pulses. Change them in `orgchar/protocol.py` (`LIMITS`).

Each step directory contains:
- `plan.json` (everything sent);
- one folder per segment with `raw.h5`, `sent.csv` and `trials.csv` (recorded trigger edge ↔ planned event);
- `result.json`;
- `report.html` (open in a browser; figures are embedded).

The **screen report** follows the decisions it supports:
1. **Checks**: recordings complete, controls clean, data lost by the platform, artifact duration.
2. **Next step**: the `curve` command and the recommended pairs; then the pairs to check, with the reason.
3. **Response maps**: one 8 × 8 grid per organoid; coloured cells respond (hover for numbers).
4. **Evidence**, one card per recommended or flagged pair:
   - *Every pulse*: each row is one pulse, 30 ms before to 30 ms after it; blue = detected spikes, shaded = blanked artifact. A real response has more spikes after the pulse than before, scattered over several ms. A column of spikes at one latency, or crowding where the shading ends, is leftover artifact.
   - *Spike timing*: spikes per second around the pulse against the channel's spontaneous rate.
   - *Artifact*: the raw artifact (median and 5–95 % band across pulses) and where the blank ends.
5. **Controls**: every pair's effect, same-organoid vs other-organoid vs sham. Controls must sit at zero.

Figures are vector graphics and stay sharp when zoomed.

## Reading the verdicts

| Step | Verdict | Do |
|---|---|---|
| census | `GO: screen blocks …` | run screen |
| census | `NO-GO` | do not stimulate; re-check later, contact FinalSpark |
| screen | `RESPONSES_FOUND` | run curve |
| screen | `NO_RESPONSE` (controls clean) | tissue fires but does not follow this stimulus; try `--amplitude 3` or `--phase 150` once, then stop |
| screen | `PIPELINE_LEAK` | artifact reaches the readout; look at the Controls section and the evidence traces before anything else |
| curve | `USABLE_CURVES` | use `card_latest.json` |
| curve | `NO_USABLE_CURVE` | the pair responds but not reliably enough to compute with |

## Using it for function approximation

- Each usable pair is one "activation function": input (pulse count or amplitude) → expected evoked spikes, plus the trial-to-trial SD at every level.
  - The SD is the noise the downstream model has to tolerate.
  - Train the ANN readout on the card's curves (or on the raw per-trial table in `result.json`), not on assumed shapes.
- For live use, send the inputs you need as a batch (a `curve`-like segment with your chosen levels). Read the outputs with the same pipeline (`orgchar.analysis.curve`), so the measurement matches the one that produced the card.
- Before each session, run `screen --electrodes <stims> --reps 20`. If a pair's spikes per pulse have dropped by more than about half, re-run `curve`.
- After any medium change or new week, start again from `census`.
- Do not read the vendor spike database for evoked responses. It counts artifacts and knows nothing about sham or cross-talk.

## Platform quirks (handled)

| What happens | What orgchar does |
|---|---|
| The raw file loses whole rows of 1280 samples (42.7 ms), 1–4 per 6-min recording | Aligns every recorded trigger to its send time and detects each lost row; the pulse before a lost row is left out; a train missing any pulse is left out whole |
| Triggers sent 30 ms apart were delivered late and bunched by TCP batching (fs666 curve, 30 Sep) | Sends triggers with TCP_NODELAY; a train whose pulses are not recorded at their send times (±4 ms) is left out |
| "Stop recording" is often not acknowledged, although the recording stopped | Checks whether the file stopped growing; completeness is judged from the recorded duration |
| Every upload fails with `Invalid run mode` (even `./orgchar.sh disable`) | Nothing: the board is stuck. Restart the Intan service, then re-run the step |

Every recorded pulse is blanked, even one that is left out of the counts, so its artifact can't reach the spontaneous-rate baseline.

## What is and is not verified

- **Verified on your data:**
  - census verdicts on the fs647, fs635 and fs636 baselines;
  - the screen on fs647 8 Sep: known pairs found, 0 of 768 cross-organoid controls significant (`tests/test_orgchar.py`);
  - the full census → screen → curve workflow on simulated recordings.
- **Not yet run on the live platform.**
  - The hardware backend (`orgchar/backends.py`) reuses the exact call sequence that worked on 8 Sep 2026: disable all → send → upload → wait 10 s → record → one-hot triggers → stop → disable all → copy HDF5 from `/mnt/s3-intanraws`.
  - Run `census` first: it only records, so it tests the path without stimulating.
- **Sham on key 15 assumes no electrode is bound to key 15.** Each segment starts with disable-all, and only the planned electrodes (keys 0–14) are uploaded. A trigger-only check on fs636 (3 Sep) showed no analog artifact from markers alone.
- **The artifact is never read on the stimulated electrode.** Mirror-channel cross-talk is handled by blanking and is checked by the cross-organoid controls.
- **Electrode 19 on fs647 was unstable** (0 Hz, then ~30–100 Hz during stimulation blocks). The waveform check and controls should catch this kind of channel. Still, look at the report tables before trusting a pair that involves it.

Tests: about 2 min; the last one re-checks the fs647 8 Sep result. pytest is not in `ncp`, so use a side venv that leaves `ncp` untouched:
`python -m venv --system-site-packages /tmp/t && /tmp/t/bin/pip install pytest && /tmp/t/bin/python -m pytest -q` (create the venv with the `ncp` python).
