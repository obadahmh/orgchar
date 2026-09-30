"""Adapters that turn earlier NeuralControlProgram run directories into (raw_path, trials) segments,
so old recordings can be re-analysed with the same pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def waveform_sweep_segments(run_dir, condition_id: str) -> list[tuple[str, pd.DataFrame]]:
    """ResponseSweep/runs/fs647_waveform_* : one raw HDF5 per (condition, 8-electrode bank)."""
    run = Path(run_dir)
    out = []
    for acc in sorted(run.glob(f"accepted/pass*_{condition_id}_electrodes*.json")):
        j = json.loads(acc.read_text())
        c = j["condition"]
        raw = run / "raw" / f"{acc.stem}_attempt{j['attempt']}.h5"
        rows = [dict(onset=t["trigger_sample"], key=t["trigger_key"], kind="stim", stim_electrode=t["stim_electrode"],
                     amplitude_uA=c["amplitude_uA"], phase_us=c["phase_width_us"], polarity=c["polarity"],
                     n_pulses=c["pulses"], ipi_ms=c["ipi_ms"], level=float("nan"), train_id=t["trial_id"], pulse_index=0)
                for t in j["raw_validation"]["trials"]]
        out.append((str(raw), pd.DataFrame(rows)))
    return out
