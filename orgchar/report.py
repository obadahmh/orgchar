"""Self-contained HTML pages: shared page parts and the census page (the screen page is in
report_screen.py). Figures are inline SVG, so a page stays sharp when zoomed and opens offline."""
from __future__ import annotations

import base64
import html
import io
import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import h5py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Colours: reference palette of the dataviz method (series blue/orange, neutral inks, fixed status set).
BLUE, ORANGE, GREY, INK, MUTED, SHADE = "#2a78d6", "#eb6834", "#8a8984", "#2b2b29", "#6b6a66", "#e9e8e2"
CHECK_OK, CHECK_WARN, CHECK_BAD = ("✓", "#0ca30c"), ("!", "#d69300"), ("✗", "#d03b3b")
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 9, "axes.titlelocation": "left", "axes.titleweight": "bold",
                     "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})


def _svg(fig) -> str:
    plt.rcParams["svg.hashsalt"] = uuid.uuid4().hex  # unique element ids: several SVGs share one page
    buf = io.StringIO()
    fig.savefig(buf, format="svg", dpi=250, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    return re.sub(r'<svg([^>]*?) width="([\d.]+)pt" height="([\d.]+)pt"',
                  r'<svg\1 style="width:\2pt;max-width:100%;height:auto"', s, count=1)


CSS = """
:root{--ink:#2b2b29;--muted:#6b6a66;--line:#e3e2dc;--bg:#fcfcfb;--card:#fff;--code:#f4f3ef}
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;padding:24px 16px 56px;
     color:var(--ink);background:var(--bg)}
main{max-width:1100px;margin:0 auto}
h1{font-size:21px;margin:0 0 2px}
h2{font-size:16px;margin:36px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:14px;margin:0 0 2px}
.meta,.note,.meta{margin:0 0 18px}
.lede{font-size:18px;font-weight:600;margin:0 0 10px}
.checks{list-style:none;padding:0;margin:0 0 6px}
.checks li{display:flex;gap:8px;align-items:baseline;margin:3px 0}
.ico{font-weight:700;display:inline-block;width:1.1em;text-align:center}
pre.cmd{background:var(--code);border:1px solid var(--line);border-radius:6px;padding:10px 12px;
        font:13px ui-monospace,SFMono-Regular,Menlo,monospace;overflow-x:auto;margin:6px 0}
.wrap{overflow-x:auto}
table{border-collapse:collapse;font-size:13px}
th{text-align:left;font-weight:600;color:var(--muted);border-bottom:1px solid var(--line);padding:5px 8px;white-space:nowrap}
td{padding:5px 8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;vertical-align:top}
td.n,th.n{text-align:right;white-space:nowrap}
.fig svg{display:block}
.maps{display:flex;flex-wrap:wrap;gap:18px 22px;align-items:flex-start}
.maps figure{margin:0}
.maps figcaption{font-weight:600;font-size:13px;margin-bottom:4px}
dl.terms{display:grid;grid-template-columns:minmax(120px,max-content) 1fr;gap:5px 16px;font-size:13px;margin:6px 0 4px}
dl.terms dt{font-weight:600}
dl.terms dd{margin:0;color:var(--muted)}
@media (max-width:600px){dl.terms{grid-template-columns:1fr}dl.terms dd{margin-bottom:6px}}
details{margin:10px 0}
summary{cursor:pointer;color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 18px}
.chip{font-size:12px;color:var(--muted);background:var(--code);border:1px solid var(--line);border-radius:999px;padding:2px 10px}
.verdict{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 14px}
.verdict .lede{margin:0 0 4px}
.verdict p{margin:0 0 6px}
.cap{max-width:72ch;margin:4px 0 14px}
h3{margin:18px 0 4px}
table.checks td:first-child{width:1.2em}
.sim{color:#d03b3b;font-weight:700;margin:2px 0}
dl.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:3px 28px;margin:10px 0 20px;font-size:13px}
dl.facts div{display:flex;gap:10px}
dl.facts dt{color:var(--muted);min-width:82px}
dl.facts dd{margin:0}
.summary{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:4px 20px 14px;margin:0 0 8px}
.summary h2{border:0;margin:12px 0 6px;padding:0}
.summary dl,dl.methods{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:5px 18px;margin:10px 0 14px;max-width:110ch}
.summary dt,dl.methods dt{font-weight:600}
.summary dd,dl.methods dd{margin:0}
@media (max-width:600px){.summary dl,dl.methods{grid-template-columns:1fr}.summary dd,dl.methods dd{margin-bottom:6px}}
svg.scores{width:100%;max-width:820px;display:block;margin:6px 0 2px;font-size:12px;fill:var(--muted)}
svg.scores .lab{fill:var(--ink)}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:12px 32px;align-items:start}
.panel{position:relative;min-width:0}
.panel>svg{display:block;margin:0 auto}
.pl{position:absolute;left:0;top:0;font-weight:700;font-size:15px}
p.legend{max-width:110ch;font-size:13.5px;margin:6px 0 14px}
table.res{margin:4px 0 10px}
.flow{display:flex;flex-direction:column;gap:2px;margin:6px 0 0 24px;font-size:12.5px}
.flow>div{display:grid;grid-template-columns:62px 1fr;gap:10px;border:1px solid var(--line);border-radius:8px;
          padding:6px 10px;background:var(--card)}
.flow>div p{margin:0}
.flow .gate{border-style:dashed}
.flow .off{opacity:.45}
.flow>span{color:var(--muted);text-align:center;line-height:1.1}
.ok{color:#0ca30c}.bad{color:#d03b3b}
ul.plain{padding-left:20px;max-width:100ch}
ul.plain li{margin:4px 0}
#net{width:100%;max-width:760px;display:block;font-size:12px;fill:var(--ink)}
#net .hd{fill:var(--muted);font-size:11px;text-anchor:middle}
#net .c{text-anchor:middle;font-weight:600}
#net .in{stroke:#dddcd5}
#net .unit,#net .node{fill:#fff;stroke:#8a8984}
#net .tissue{fill:#2a78d6;fill-opacity:.06;stroke:#2a78d6}
.ctl{display:flex;align-items:center;gap:10px;margin:2px 0 8px;max-width:760px}
.ctl input{flex:1}
.ctl button{font:inherit;padding:2px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);cursor:pointer}
[id^=sol-],[id^=hit-]{cursor:pointer}
.lit use{fill:#2b2b29!important}
[id^=hit-].lit path{opacity:.06!important}
@media print{.ctl{display:none}.panel,table,.summary{break-inside:avoid}}
"""


def _page(title: str, body: list[str]) -> str:
    return (f"<title>{html.escape(title)}</title>\n<meta charset=utf-8>\n"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>\n<style>{CSS}</style>\n<main>\n"
            + "\n".join(body) + "\n</main>\n")


def _icon(sym_col) -> str:
    sym, col = sym_col
    return f"<span class=ico style='color:{col}'>{sym}</span>"


# ============================================================================== census

def census_figures(r: dict) -> list[tuple[str, str]]:
    ch = pd.DataFrame(r["channels"])
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.bar(ch.channel, ch.rate_hz, color=[BLUE if a else "#c9c8c2" for a in ch.active], width=0.75)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.axhline(0.1, color=INK, lw=0.6, ls=":")
    ax.text(32.0, 0.1, "0.1 Hz", fontsize=7.5, color=MUTED, va="center", ha="left", clip_on=False)
    ax.set_xticks(range(32))
    ax.set_xlim(-0.8, 31.8)
    for b in range(1, 4):
        ax.axvline(b * 8 - 0.5, color=GREY, lw=0.6)
    for b in range(4):
        ax.text(b * 8 + 3.5, 1.02, f"organoid {b}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=MUTED)
    ax.set_xlabel("electrode")
    ax.set_ylabel("spikes / s")
    figs = [("Firing rate per electrode (blue = active)", _svg(fig))]
    act = ch[ch.active]
    if len(act):
        n = len(act)
        cols = min(n, 8)
        rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(1.35 * cols, 1.8 * rows), squeeze=False, sharey=True,
                                gridspec_kw=dict(hspace=0.6))
        t = (np.arange(75) - 30) / 30.0
        for a in axs.flat[n:]:
            a.axis("off")
        for a, (_, row) in zip(axs.flat, act.iterrows()):
            a.plot(t, row.waveform, color=BLUE, lw=1.4)
            a.set_title(f"e{row.channel}: {row.rate_hz:.1f} Hz", fontsize=8, fontweight="normal")
            a.set_xticks([0, 1])
        for a in axs[:, 0]:
            a.set_ylabel("µV")
        figs.append(("Mean spike shape, active electrodes (ms)", _svg(fig)))
    return figs


# ============================================================================== shared


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def to_json(r: dict) -> str:
    return json.dumps(_clean(r), indent=1)


def write_html(path, title: str, verdict: str, summary_lines: list[str], figs, table: pd.DataFrame | None = None):
    body = [f"<h1>{html.escape(title)}</h1>", f"<p class=lede>{html.escape(verdict)}</p><ul class=checks>"]
    body += [f"<li>{html.escape(s)}</li>" for s in summary_lines]
    body.append("</ul>")
    for t, svg in figs:
        body.append(f"<h2>{html.escape(t)}</h2><div class=fig>{svg}</div>")
    if table is not None and len(table):
        body.append("<details><summary>Table</summary><div class=wrap>"
                    + table.to_html(index=False, float_format=lambda x: f"{x:.3g}", na_rep="") + "</div></details>")
    with open(path, "w") as f:
        f.write(_page(title, body))


# ============================================================================== solve

def _n(v, d=2, sign=False) -> str:
    """A number with a true minus sign."""
    return f"{v:{'+' if sign else ''}.{d}f}".replace("-", "−")


def _tips(svg, tips):
    """Hover text (an SVG <title>) for artists drawn with gid=key: {key: text}."""
    for k, text in tips.items():
        svg = svg.replace(f'<g id="{k}">', f'<g id="{k}"><title>{html.escape(text)}</title>', 1)
    return svg


def _keys(res):
    """Per time point, the (pair, pulse count) every unit asked for, as 'stim-rec-n'."""
    kept = res["kept"]
    return [[f"{kept[j % len(kept)][0]}-{kept[j % len(kept)][1]}-{n}" for j, n in enumerate(row)] for row in res["pulses"]]


def _swing(ode, t, y):
    """Indices into t of y's dip (its lowest point before the exact rebound) and rebound (its highest point
    after the exact dip, up to as far beyond the exact rebound)."""
    dense = np.linspace(0, ode.T, 2001)
    ex = ode.exact(dense)
    t1 = dense[ex.argmin()]
    t2 = dense[dense > t1][ex[dense > t1].argmax()]
    t, y = np.asarray(t), np.asarray(y)
    return int(np.argmin(np.where(t < t2, y, np.inf))), int(np.argmax(np.where((t > t1) & (t < 2 * t2 - t1), y, -np.inf)))


def _recordings(step_dir):
    """Every recording of the run: phase, start (from the platform's file name; None in simulations),
    length in s, trains and pulses delivered, trains usable."""
    rows = []
    for part, phase in (("sweep", "Measure"), ("evaluate", "Run")):
        for call_dir in [step_dir / part / "pass"] + sorted((step_dir / part).glob("retry*")):
            if not (call_dir / "plan.json").exists():
                continue
            for seg in json.loads((call_dir / "plan.json").read_text())["segments"]:
                d = call_dir / seg["id"]
                remote = d / "remote_path.txt"
                start = (datetime.strptime(Path(remote.read_text().strip()).name[:15], "%Y%m%d_%H%M%S")
                         if remote.exists() else None)
                usable = pd.read_csv(d / "responses.csv").request.nunique() if (d / "responses.csv").exists() else 0
                stim = [e for e in seg["events"] if e["kind"] == "stim"]
                rows.append(dict(phase=phase, id=f"{part}/{call_dir.name}/{seg['id']}", start=start, seconds=seg["duration_s"],
                                 trains=len(stim), usable=int(usable),
                                 pulses=sum(seg["electrodes"][str(e["stim_electrode"])]["n_pulses"] for e in stim),
                                 file=Path(remote.read_text().strip()).name if remote.exists() else "raw.h5"))
    return rows


def _span(rows, phase):
    """(start, end) of a phase from its recordings, or None."""
    rs = [r for r in rows if r["phase"] == phase]
    if not rs or any(r["start"] is None for r in rs):
        return None
    return min(r["start"] for r in rs), max(r["start"] + timedelta(seconds=r["seconds"]) for r in rs)


def _examples(step_dir, res):
    """Two trains of the first measurement on the first pair, as the readout saw them: at the counts with
    the highest and the lowest mean response, the train nearest its count's median. [] if the recordings
    cannot be read."""
    from . import readout as RO
    from .backends import recording
    from .measure import CONDITION
    c = res["curves"][0]
    s, r = c["stim"], c["rec"]
    counts = {c["counts"][int(np.argmax(c["mean"]))], c["counts"][int(np.argmin(c["mean"]))]}
    thr = {int(k): v for k, v in res["thresholds_uV"].items()}[r]
    nb = int(round(RO.SPEC["baseline_duration_ms"] * RO.FS / 1000))
    found = {n: [] for n in counts}
    try:
        for d in sorted((step_dir / "sweep").glob("*/seg*")):
            tr = pd.read_csv(d / "trials.csv") if (d / "trials.csv").exists() else pd.DataFrame(columns=["kind"])
            tr = tr[(tr.kind == "stim") & ~tr.exclude.astype(bool) & (tr.stim_electrode == s) & tr.n_pulses.isin(counts)]
            if tr.empty or (raw := recording(d)) is None:
                continue
            with h5py.File(raw, "r") as f:
                v = f[f"electrode_{r}"]
                w = v.shape[1]
                cut = lambda a, b: v[a // w:(b - 1) // w + 1].reshape(-1)[a % w:a % w + b - a].astype(float)  # noqa: E731
                for o, n in zip(tr.onset.astype(int), tr.n_pulses.astype(int)):
                    need = RO.post_samples_needed(n, CONDITION["ipi_ms"], CONDITION["phase_us"])
                    if o >= nb and (o + need) // w < v.shape[0]:
                        found[n].append(RO.measure_trial(cut(o - nb, o), cut(o, o + need), n, CONDITION["ipi_ms"],
                                                         CONDITION["phase_us"], thr, keep_trace=True))
    except (FileNotFoundError, OSError):
        return []
    out = []
    for n in sorted(counts, key=lambda n: -c["mean"][c["counts"].index(n)]):
        if found[n]:
            med = np.median([m["adjusted"] for m in found[n]])
            out.append(dict(n=n, thr=thr, **min(found[n], key=lambda m: abs(m["adjusted"] - med))))
    return out


def _fig_task(ode, t):
    dense = np.linspace(0, ode.T, 1001)
    ex = ode.exact(dense)
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.axhline(0, color=SHADE, lw=1)
    ax.plot(dense, ex, color=INK, lw=1.6, ls=(0, (5, 3)))
    ax.plot(t, np.full(len(t), -0.6), "|", color=MUTED, ms=7, mew=1)
    ax.text(ode.T, -0.54, f"the {len(t)} test times", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    i1 = ex.argmin()
    i2 = i1 + ex[i1:].argmax()
    for i, txt, dy in ((i1, f"dip {_n(ex[i1])}", -3), (i2, f"rebound {_n(ex[i2], sign=True)}", 6)):
        ax.plot(dense[i], ex[i], "o", color=INK, ms=4.5, mew=0)
        ax.annotate(txt, (dense[i], ex[i]), xytext=(8, dy), textcoords="offset points", fontsize=8, color=INK)
    ax.set_ylim(-0.66, 1.08)
    ax.set_xlabel("time t")
    ax.set_ylabel("y(t)")
    return _svg(fig)


def _fig_traces(examples):
    """The examples' band-passed voltage: baseline, blanked train, counting window, crossings."""
    from .readout import FS, SPEC
    per_ms = FS / 1000
    fig, axes = plt.subplots(len(examples), 1, figsize=(5.4, 1.75 * len(examples) + 0.5), sharex=True, squeeze=False)
    k = 15                                                     # 0.5 ms bins: min and max keep every spike
    right = max(m["window_samples"][1] for m in examples) / per_ms + 15
    for ax, m in zip(axes[:, 0], examples):
        v = np.r_[m["baseline"], m["post"]]
        t0 = -len(m["baseline"]) / per_ms
        v = v[:len(v) // k * k].reshape(-1, k)
        x = np.repeat(t0 + (np.arange(len(v)) + 0.5) * k / per_ms, 2)
        a, b = (s / per_ms for s in m["window_samples"])
        lim = 3.2 * abs(m["thr"])
        ax.axvspan(t0, 0, color=SHADE, alpha=0.5, lw=0)
        ax.axvspan(a, b, color=BLUE, alpha=0.08, lw=0)
        ax.axvspan(0, a - SPEC["response_window_ms_after_train"][0] + SPEC["blank_after_train_ms"], color=GREY,
                   alpha=0.25, lw=0)                         # the train and its blanking
        ax.plot(x, np.column_stack([v.min(1), v.max(1)]).ravel(), color=MUTED, lw=0.5)
        ax.axhline(m["thr"], color=INK, lw=0.8, ls=(0, (3, 2)))
        base = np.array(m["baseline_samples"]) / per_ms + t0
        win = (np.array(m["event_samples"]) + m["window_samples"][0]) / per_ms
        ax.plot(base, np.full(len(base), 0.86 * lim), "|", color=MUTED, ms=6, mew=0.9)
        ax.plot(win, np.full(len(win), 0.86 * lim), "|", color=BLUE, ms=6, mew=0.9)
        scale = (m["window_samples"][1] - m["window_samples"][0]) / len(m["baseline"])
        ax.set_title(f"{m['n']} pulses: {m['response_events']} in the window − {m['baseline_events']} before × "
                     f"{scale:.1f} = {_n(m['adjusted'], 1)} evoked spikes", fontsize=8, fontweight="normal")
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("µV")
    ax.set_xlim(t0, right)
    ax.set_xlabel("ms from the start of the train")
    return _svg(fig)


def _fig_curve(c, ev, trains, used):
    """Single trains, the means of the first measurement (grey) and of the run (blue), and below, how often
    the network asked for each count."""
    s, r = c["stim"], c["rec"]
    pair = trains[(trains.stim == s) & (trains.rec == r)] if trains is not None else pd.DataFrame(columns=["part", "pulses"])
    asks = {n: sum(v for k, v in used.items() if k.startswith(f"{s}-{r}-") and int(k.split("-")[2]) == n)
            for n in range(min(c["counts"]), max(c["counts"]) + 1)}
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.4, 3.7), sharex=True, gridspec_kw=dict(height_ratios=[3.2, 1]))
    ax.axhline(0, color=SHADE, lw=1)
    jitter = np.random.default_rng(0)
    for part, col, dx in (("sweep", GREY, -0.15), ("evaluate", BLUE, 0.15)):
        d = pair[pair.part == part]
        ax.scatter(d.pulses + dx + jitter.uniform(-0.08, 0.08, len(d)), d.response, s=6, color=col, alpha=0.35, lw=0)
    n_trains = pair.groupby(["part", "pulses"]).size().to_dict()
    ax.plot(c["counts"], c["mean"], color=GREY, lw=2, label="first measurement, mean")
    tips = {}
    for n, m, sd in zip(c["counts"], c["mean"], c["sd"]):
        ax.plot(n, m, "o", color=GREY, ms=5.5, mec="white", mew=1, gid=f"sw-{s}-{r}-{n}")
        tips[f"sw-{s}-{r}-{n}"] = f"{n} pulses, first measurement\nmean {_n(m, 1)}, SD {sd:.1f}, {n_trains.get(('sweep', n), 0)} trains"
    first = dict(zip(c["counts"], c["mean"]))
    for e in ev:
        n = e["pulses"]
        ax.plot(n, e["mean"], "o", color=BLUE, ms=7, mec="white", mew=1.2, gid=f"crv-{s}-{r}-{n}",
                label="run, mean" if e is ev[0] else None)
        tips[f"crv-{s}-{r}-{n}"] = (f"{n} pulses, run\nmean {_n(e['mean'], 1)}, {n_trains.get(('evaluate', n), 0)} trains"
                                    + (f"\nfirst measurement {_n(first[n], 1)}" if n in first else "")
                                    + f"\nasked for {asks.get(n, 0)} times by the network")
    ax.set_ylabel("evoked spikes")
    ax.legend(frameon=False, fontsize=7.5)
    if used:
        bx.bar(list(asks), list(asks.values()), width=0.6, color=BLUE, alpha=0.55, lw=0)
        for n, v in asks.items():
            if v >= 0.15 * sum(asks.values()):
                bx.text(n, v, str(v), ha="center", va="bottom", fontsize=7, color=INK)
        bx.set_ylim(0, max(asks.values()) * 1.35)
        bx.set_ylabel("requests", fontsize=7.5)
    else:
        bx.set_visible(False)
    bx.set_xticks(range(2, 21, 2))
    (bx if used else ax).set_xlabel("pulses in the train")
    return _tips(_svg(fig), tips)


def _fig_solution(res, ode):
    """The run against the exact solution and the best curve that never dips, with its trial-noise band;
    each time point is a hover target (sol-i, and the column hit-i)."""
    from .net import decay_fit
    dense, t = np.linspace(0, ode.T, 1001), res["t"]
    a, k, b = decay_fit(ode, t)
    fit = dense[dense >= t[0]]                                # scored from the first test time on
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    ax.axhline(0, color=SHADE, lw=1)
    half = (t[1] - t[0]) / 2
    for i, ti in enumerate(t):
        ax.axvspan(ti - half, ti + half, color=INK, alpha=0.001, lw=0, gid=f"hit-{i}")   # hover column, shaded when lit
    if "bootstrap" in res:
        ax.fill_between(t, *res["bootstrap"]["y"], color=BLUE, alpha=0.13, lw=0, label="95% of repeat runs (trial noise)")
    ax.plot(dense, ode.exact(dense), color=INK, lw=1.5, ls=(0, (5, 3)), label="exact solution")
    ax.plot(fit, a * np.exp(-k * fit) + b, color=GREY, lw=1.5, label=f"best curve that never dips, R² {_n(res['decay_r2'])}")
    ax.plot(t, res["y"], color=BLUE, lw=2, label=f"organoid run, R² {_n(res['r2'])}")
    (dip, reb), (tdip, treb) = _swing(ode, t, res["y"]), _swing(ode, t, res["exact"])
    tips = {}
    for i in range(len(t)):
        ax.plot(t[i], res["y"][i], "o", color=BLUE, ms=7, mec="white", mew=1.2, gid=f"sol-{i}")
        band = (f"\nrepeat runs {_n(res['bootstrap']['y'][0][i])} to {_n(res['bootstrap']['y'][1][i])}"
                if "bootstrap" in res else "")
        tips[f"sol-{i}"] = tips[f"hit-{i}"] = (f"t = {t[i]:.1f}\norganoid run {_n(res['y'][i], sign=True)}{band}\n"
                                               f"exact {_n(res['exact'][i], sign=True)}")
    for i, j, word, dx, dy in ((dip, tdip, "dip", 30, -16), (reb, treb, "rebound", 10, 10)):
        ax.annotate(f"{word} {_n(res['y'][i], sign=True)} (exact {_n(res['exact'][j], sign=True)})", (t[i], res["y"][i]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=4))
    ax.set_xlim(0, ode.T)
    ax.set_xlabel("time t")
    ax.set_ylabel("y(t)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _tips(_svg(fig), tips)


def _fig_shuffled(res, ode):
    dense, t = np.linspace(0, ode.T, 1001), res["t"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.axhline(0, color=SHADE, lw=1)
    for k, y in enumerate(res["shuffle"]["examples"]):
        ax.plot(t, y, color=GREY, lw=0.8, alpha=0.4, label="shuffled spikes, 30 of 2,000" if k == 0 else None)
    ax.plot(dense, ode.exact(dense), color=INK, lw=1.5, ls=(0, (5, 3)), label="exact solution")
    ax.plot(t, res["y"], color=BLUE, lw=2, label="organoid run")
    ax.set_xlabel("time t")
    ax.set_ylabel("y(t)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left", bbox_to_anchor=(0, 1))
    return _svg(fig)


def _fig_expected(res, ode, computer):
    dense, t = np.linspace(0, ode.T, 1001), res["t"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.axhline(0, color=SHADE, lw=1)
    ax.plot(dense, ode.exact(dense), color=INK, lw=1.5, ls=(0, (5, 3)), label="exact solution")
    ax.plot(t, res["y_model"], color=GREY, lw=2, label=f"on the computer, R² {_n(computer)}")
    ax.plot(t, res["y"], color=BLUE, lw=2, label=f"organoid run, R² {_n(res['r2'])}")
    ax.set_xlabel("time t")
    ax.set_ylabel("y(t)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left", bbox_to_anchor=(0, 1))
    return _svg(fig)


def _scores(res, computer):
    """Every R² of the run on one 0 to 1 axis, one labelled row each; the best curve that never dips is
    the bar (dashed)."""
    X, e, decay = (lambda v: 270 + 300 * min(max(v, 0.0), 1.0)), res.get("expected"), res["decay_r2"]
    rows = []   # (label, low, high, dot, colour, hollow, value text)
    if "y" in res:
        bs = res.get("bootstrap")
        rows.append(("Organoid run", *(bs["r2"] if bs else (None, None)), res["r2"], BLUE, False,
                     f"{_n(res['r2'])}" + (f", repeat runs {_n(bs['r2'][0])} to {_n(bs['r2'][1])}" if bs else "")))
        rows.append(("Same network on the computer", None, None, computer, GREY, False, _n(computer)))
    if e:
        rows.append(("Predicted before the run", e["noisy_p10"], e["noisy_p90"], e["noisy_median"], BLUE, True,
                     f"{_n(e['noisy_median'])}, 80% within {_n(e['noisy_p10'])} to {_n(e['noisy_p90'])}"))
    rows.append(("Best curve that never dips", None, None, decay, GREY, False, _n(decay)))
    if "shuffle" in res:
        rows.append(("Spikes shuffled across pulse counts", 0.0, res["shuffle"]["p95"], None, GREY, False,
                     f"95% below {_n(res['shuffle']['p95'])}"))
    h = 28 * len(rows) + 58
    g = [f"<svg class=scores viewBox='0 0 820 {h}' role=img aria-label='R² of the run and its references'>",
         f"<line x1={X(decay)} x2={X(decay)} y1=4 y2={h - 42} stroke='{GREY}' stroke-dasharray='3 3' />"]
    for i, (label, lo, hi, dot, col, hollow, text) in enumerate(rows):
        y = 18 + 28 * i
        g.append(f"<text x=0 y={y + 4} class=lab>{html.escape(label)}</text>"
                 f"<line x1={X(0)} x2={X(1)} y1={y} y2={y} stroke='#e3e2dc' />")
        if lo is not None:
            g.append(f"<rect x={X(lo)} y={y - 6} width={X(hi) - X(lo)} height=12 rx=4 fill='{col}' fill-opacity=.22 />")
        if dot is not None:
            g.append(f"<circle cx={X(dot)} cy={y} r=5.5 fill='{'#fff' if hollow else col}' stroke='{col if hollow else '#fff'}' "
                     f"stroke-width={2 if hollow else 1.5} />")
        g.append(f"<text x={X(1) + 16} y={y + 4}>{html.escape(text)}</text>")
    y = h - 38
    g.append(f"<line x1={X(0)} x2={X(1)} y1={y} y2={y} stroke='{GREY}' />")
    g += [f"<line x1={X(v)} x2={X(v)} y1={y} y2={y + 4} stroke='{GREY}' /><text x={X(v)} y={y + 16} "
          f"text-anchor=middle>{'0 or less' if v == 0 else f'{v:g}'}</text>" for v in (0, 0.25, 0.5, 0.75, 1)]
    g.append(f"<text x={X(0.5)} y={y + 31} text-anchor=middle>R² against the exact solution (1 is a perfect match)</text></svg>")
    return "".join(g)


def _protocol(res, rows):
    """The session top to bottom: measure, check, train, check, run, with times, trains and each check's
    outcome. A check that failed stops the session there."""
    from .measure import ARTIFACT_LIMIT
    from .solve import GATE_P
    ok1, ok2 = "expected" in res, "eval_trains" in res
    meas, run = _span(rows, "Measure"), _span(rows, "Run")
    hm = lambda a, b: f"{a:%H:%M} to {b:%H:%M}. " if a and b else ""                    # noqa: E731
    mark = lambda ok, reached=True: ("" if not reached else f" <b class=ok>{CHECK_OK[0]}</b>" if ok   # noqa: E731
                                     else f" <b class=bad>{CHECK_BAD[0]} stopped here</b>")
    checks = " ".join(f"e{c['stim']}→e{c['rec']}: p = {c['p']:.1g} (needs &lt; {GATE_P:g}), artifact "
                      f"{c['artifact']:.1f}× noise (needs &lt; {ARTIFACT_LIMIT:g}×)." for c in res["curves"])
    n_counts = len({n for row in res.get("pulses", []) for n in row})
    steps = [("step", "Measure", f"On the organoid. {hm(*meas) if meas else ''}{res['sweep_trains']} trains of "
              f"2 to 20 pulses."),
             ("gate", "Check", f"Spikes depend on the pulse count. {checks}{mark(ok1)}"),
             ("step" if ok1 else "step off", "Train", f"On the computer. {hm(meas[1], run[0]) if meas and run else ''}"
              "No stimulation." if ok1 else "Not run."),
             ("gate" if ok1 else "gate off", "Check", "Predicted R² above that of a curve that never dips"
              + (f": {_n(res['expected']['noisy_median'])} (needs &gt; {_n(res['decay_r2'])})." if ok1 else ".")
              + mark(ok2, ok1)),
             ("step" if ok2 else "step off", "Run", f"On the organoid. {hm(*run) if run else ''}{res['eval_trains']} "
              f"trains: {n_counts} pulse counts × {res['eval_reps']}." if ok2 else "Not run.")]
    return ("<div class=flow>" + "<span>↓</span>".join(f"<div class='{cls}'><b>{name}</b><p>{text}</p></div>"
                                                      for cls, name, text in steps) + "</div>")


def _network(res, stims, recs, start):
    """The network at one moment, as an SVG that NET_JS (at the page's end) fills in from the run's own
    numbers (netdata): time -> units (pulse counts) -> organoid (spikes) -> weighted lines -> y."""
    k, wt = res["width"], res["weights"]
    ys = [52 + 28 * j for j in range(k)]
    mid, h = (ys[0] + ys[-1]) / 2, ys[-1] + 34
    means = {(m["stim"], m["rec"], m["pulses"]): m["mean"] for m in res["eval_means"]}
    keys = _keys(res)
    spikes = [[means[tuple(int(v) for v in key.split("-"))] for key in row] for row in keys]
    contrib = [abs(w * x) for row in spikes for w, x in zip(wt["w_out"], row)]
    data = dict(t=res["t"], exact=res["exact"], N=res["pulses"], s=spikes, w=wt["w_out"], b=wt["b_out"],
                wi=wt["w_in"], bi=wt["b_in"], k=keys, cmax=max(contrib + [1e-9]), start=start)
    g = [f"<svg id=net viewBox='0 0 760 {h}' role=img aria-label='the network at one moment'>",
         "<text class=hd x=60 y=16>input</text><text class=hd x=250 y=16>units</text>"
         f"<text class=hd x=395 y=16>organoid: pulses on {stims}, spikes on {recs}</text>"
         "<text class=hd x=584 y=16>× weight</text><text class=hd x=690 y=16>output</text>",
         f"<rect class=tissue x=290 y={ys[0] - 18} width=210 height={ys[-1] - ys[0] + 36} rx=10 />"]
    for j, y in enumerate(ys):
        g.append(f"<line class=in x1=82 y1={mid} x2=238 y2={y} /><line id=e{j} x1=500 y1={y} x2=668 y2={mid}>"
                 f"<title id=et{j}></title></line><circle class=unit cx=250 cy={y} r=12><title id=ut{j}></title>"
                 f"</circle><text class=c id=n{j} x=250 y={y + 4}></text><text id=s{j} x=304 y={y + 4}></text>")
    g.append(f"<circle class=node cx=60 cy={mid} r=22 /><text class=c x=60 y={mid + 5}>t</text>"
             f"<text class=c id=tv x=60 y={mid + 42}></text><circle class=node cx=690 cy={mid} r=22><title>y = "
             f"{_n(wt['b_out'], sign=True)} + the {k} weighted spike counts</title></circle><text class=c x=690 "
             f"y={mid + 5}>y</text><text class=c id=yv x=690 y={mid + 42}></text><text class='c hd' id=yt x=690 "
             f"y={mid + 58}></text></svg>")
    return ("".join(g) + f"<div class=ctl><button id=play>▶ play</button><input id=slider type=range min=0 "
            f"max={len(res['t']) - 1} aria-label='test time'></div><script type=application/json id=netdata>"
            f"{json.dumps(data)}</script>")


# Fills the network diagram for one moment, and lights that moment in Figure 4 and its pulse counts in Figure 2.
NET_JS = """<script>
const D = JSON.parse(document.getElementById('netdata').textContent), $ = id => document.getElementById(id);
const f = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2), g = v => v.toFixed(1).replace('-', '−');
function show(i) {
  $('slider').value = i;
  $('tv').textContent = 't = ' + D.t[i].toFixed(1);
  let y = D.b;
  D.N[i].forEach((n, j) => {
    const s = D.s[i][j], c = D.w[j] * s, e = $('e' + j);
    y += c;
    $('n' + j).textContent = n;
    $('s' + j).textContent = `${n} pulses → ${g(s)} spikes`;
    e.style.stroke = c < 0 ? '#eb6834' : '#2a78d6';
    e.style.strokeWidth = 1 + 9 * Math.abs(c) / D.cmax;
    $('et' + j).textContent = `${g(s)} spikes × ${D.w[j].toFixed(3).replace('-', '−')} = ${f(c)}`;
    $('ut' + j).textContent = `unit ${j + 1}: pulses = round(${D.wi[j].toFixed(2).replace('-', '−')} t ${f(D.bi[j])}), kept within 2 to 20`;
  });
  $('yv').textContent = 'y = ' + f(y);
  $('yt').textContent = 'exact ' + f(D.exact[i]);
  document.querySelectorAll('.lit').forEach(e => e.classList.remove('lit'));
  ['sol-' + i, 'hit-' + i, ...D.k[i].map(k => 'crv-' + k)].forEach(id => $(id)?.classList.add('lit'));
}
let timer;
$('slider').oninput = e => { clearInterval(timer); show(+e.target.value); };
$('play').onclick = () => { clearInterval(timer); let i = 0; show(i);
  timer = setInterval(() => ++i < D.t.length ? show(i) : clearInterval(timer), 500); };
document.querySelectorAll('[id^=sol-],[id^=hit-]').forEach(e => e.onmouseenter = () => {
  clearInterval(timer); show(+e.id.split('-')[1]); });
show(D.start);
</script>"""


def _figure(panels, legend):
    """Panels [(letter, html)] side by side where they fit, then the legend."""
    return ("<div class=panels>" + "".join(f"<div class=panel>{f'<span class=pl>{a}</span>' if a else ''}{p}</div>"
                                           for a, p in panels) + f"</div><p class=legend>{legend}</p>")


def solve_page(path, res: dict, when: str):
    """The solve run as a short paper: run facts, a summary with every R², five figures (task and protocol,
    the organoid's response, the network, the result, controls), limitations, checks, methods and the
    recordings. Reads the run's folder next to `path`: single trains, recordings, two raw trains."""
    from .measure import ARTIFACT_LIMIT, CONDITION, organoid_of
    from .net import FINETUNE_STEPS, STEPS, Oscillator, held_out_times, r2
    from .readout import SPEC
    from .solve import GATE_P
    ode = Oscillator(**res["ode"])
    step_dir = Path(path).parent
    parts = [pd.read_csv(f).assign(part=f.parent.name) for f in (step_dir / "sweep" / "responses.csv",
                                                                  step_dir / "evaluate" / "responses.csv") if f.exists()]
    trains = pd.concat(parts, ignore_index=True) if parts else None
    rows, examples = _recordings(step_dir), _examples(step_dir, res)
    done, decay, bs = "y" in res, res.get("decay_r2"), res.get("bootstrap")
    used = pd.Series([k for row in _keys(res) for k in row]).value_counts().to_dict() if done else {}
    c0 = res["curves"][0]
    org = organoid_of(c0["stim"])
    stims = ", ".join(sorted({f"e{c['stim']}" for c in res["curves"]}))
    recs = ", ".join(sorted({f"e{c['rec']}" for c in res["curves"]}))
    thr = {int(k): v for k, v in res["thresholds_uV"].items()}
    computer = r2(res["y_model"], res["exact"]) if done else None
    t, n_t = res.get("t", []), res["points"] - 1
    title = "A brain organoid as the hidden layer of a neural network"

    # ---- header
    session = _span(rows, "Measure"), _span(rows, "Run")
    end = max((s[1] for s in session if s), default=None)
    outcome = ("Completed" if done else "Stopped after training" if "expected" in res
               else "Stopped after the first measurement" if not res["verdict"].startswith("FAILED") else res["verdict"])
    facts = [("Preparation", f"{res['fsname']}, organoid on electrodes {org.start} to {org.stop - 1}"),
             ("Electrodes", f"trains on {stims}, spikes read on {recs}"),
             ("Session", f"{when.removesuffix(' UTC')} to {end:%H:%M} UTC" if end else when),
             ("Stimulation", f"{sum(r['trains'] for r in rows)} trains, {sum(r['pulses'] for r in rows):,} pulses "
                             f"({CONDITION['amplitude_uA']:g} µA, {CONDITION['phase_us']:g} µs per phase)"),
             ("Outcome", outcome),
             ("Software", f"orgchar {res.get('code_version', '?')}")]
    body = [f"<h1>{html.escape(title)}</h1>"
            + ("<p class=sim>SIMULATED: synthetic organoid, no tissue.</p>" if res.get("simulated") else "")
            + "<dl class=facts>" + "".join(f"<div><dt>{a}</dt><dd>{html.escape(b)}</dd></div>" for a, b in facts) + "</dl>"]

    # ---- summary
    if done:
        (dip, reb), (tdip, treb) = _swing(ode, t, res["y"]), _swing(ode, t, res["exact"])
        ci = lambda i: (f" (95% of repeat runs {_n(bs['y'][0][i], sign=True)} to {_n(bs['y'][1][i], sign=True)})"   # noqa: E731
                        if bs else "")
        passed = res["r2"] > decay
        lede = (f"With the organoid as its hidden layer, the network {'followed' if passed else 'did not follow'} the "
                f"spring's dip and rebound: R² {_n(res['r2'])} against the exact solution, "
                f"{'above' if passed else 'not above'} the {_n(decay)} of a curve that never dips.")
        sh = res.get("shuffle")
        summary = [("Aim", "Use the evoked responses of living neurons as the hidden layer of a network that solves "
                           "a differential equation."),
                   ("Method", f"Trains of 2 to 20 pulses on {stims} set the spike count on {recs}. This response served "
                              f"as the activation of a {res['width']}-unit network, trained on the equation alone. The "
                              "network then ran with the organoid in the loop."),
                   ("Result", f"R² {_n(res['r2'])}. Dip {_n(res['y'][dip], sign=True)} at t = {t[dip]:.1f}{ci(dip)}, exact "
                              f"{_n(res['exact'][tdip], sign=True)}. Rebound {_n(res['y'][reb], sign=True)} at t = "
                              f"{t[reb]:.1f}{ci(reb)}, exact {_n(res['exact'][treb], sign=True)}.")]
        if sh:
            summary.append(("Controls", f"A curve that never dips scores {_n(decay)}. With the spikes shuffled across "
                                        f"pulse counts, 95% of runs score below {_n(sh['p95'])} and "
                                        + ("none" if not sh["share_above_run"] else f"{sh['share_above_run']:.1%}")
                                        + f" reach {_n(res['r2'])}."))
        if bs:
            summary.append(("Uncertainty", f"Trial noise is large at {res['eval_reps']} trains per count. Repeat runs "
                                           f"would score {_n(bs['r2'][0])} to {_n(bs['r2'][1])}, and "
                                           f"{bs['below_decay']:.0%} would not clear {_n(decay)}."))
        body.append("<section class=summary><h2>Summary</h2>"
                    f"<p class=lede>{html.escape(lede)}</p><dl>"
                    + "".join(f"<dt>{a}</dt><dd>{html.escape(b)}</dd>" for a, b in summary) + "</dl>"
                    + _scores(res, computer) + "</section>")
    else:
        lede = ("Stopped before the run: the trained network was predicted to score R² "
                f"{_n(res['expected']['noisy_median'])}, not above the {_n(decay)} of a curve that never dips. No "
                "further stimulation." if "expected" in res else
                "Stopped after the first measurement: the response failed a check. No further stimulation."
                if not res["verdict"].startswith("FAILED") else res["verdict"])
        body.append(f"<section class=summary><h2>Summary</h2><p class=lede>{html.escape(lede)}</p>"
                    + (_scores(res, computer) if decay is not None else "") + "</section>")

    # ---- figure 1: task and protocol
    test_t = t if done else held_out_times(ode, res["points"]).numpy().ravel().tolist()
    body.append("<h2>Task and protocol</h2>" + _figure(
        [("a", _fig_task(ode, test_t)), ("b", _protocol(res, rows))],
        f"<b>Figure 1.</b> <b>a</b>, The task: a weight on a spring, let go from y = 1 at rest. m y″ + c y′ + k y = 0 "
        f"with m = {ode.m:g}, c = {ode.c:g}, k = {ode.k:g}. The network is scored at the {len(test_t)} test times. It "
        "never sees this solution, only the equation and the starting point. <b>b</b>, The session. Each check stops "
        "it before any further stimulation."))

    # ---- figure 2: the organoid's response
    panels = []
    if examples:
        panels.append(("a", _fig_traces(examples)))
    for c in res["curves"]:
        ev = [e for e in res.get("eval_means", []) if (e["stim"], e["rec"]) == (c["stim"], c["rec"])]
        panels.append(("b" if examples else "", _fig_curve(c, ev, trains, used)))
    a_text = (f"<b>a</b>, Two single trains of the first measurement, at the pulse counts with the highest and the "
              f"lowest mean response, each the train nearest its count's median. Voltage on e{c0['rec']}, band-passed "
              f"{SPEC['band_hz'][0]:g} to {SPEC['band_hz'][1]:g} Hz. The train and {SPEC['blank_after_train_ms']:g} ms "
              f"after it are blanked (dark grey). Ticks: downward crossings of the threshold (dashed, "
              f"{_n(thr[c0['rec']], 1)} µV), counted in the {SPEC['baseline_duration_ms']:g} ms before the train (grey) "
              f"and in the window {SPEC['response_window_ms_after_train'][0]:g} to "
              f"{SPEC['response_window_ms_after_train'][1]:g} ms after it (blue). <b>b</b>, " if examples else "")
    body.append("<h2>The organoid's response</h2>" + _figure(
        panels, f"<b>Figure 2.</b> {a_text}Evoked spikes against train length: the network's activation function. "
                f"Dots: single trains. Grey: the first measurement, {res['sweep_reps']} trains per count. "
                + (f"Blue: the run, {res['eval_reps']} trains per count. " if done else "")
                + "Below zero, the train left fewer spikes than there were before it."
                + (" Bars: how often the units asked for each count, over all test times. Black: the counts in use at "
                   "the test time shown in Figure 3." if done else "")))

    # ---- figure 3: the network
    if done:
        body.append("<h2>The network</h2>" + _figure(
            [("", _network(res, stims, recs, dip))],
            "<b>Figure 3.</b> The trained network at one test time, with the organoid's answers from the run. Unit j "
            "asks for N<sub>j</sub> = round(a<sub>j</sub> t + b<sub>j</sub>) pulses, kept within 2 to 20. The organoid "
            "answers with spikes. The computer forms y = c + Σ<sub>j</sub> v<sub>j</sub> × spikes<sub>j</sub>. Line "
            "width: each unit's share of y. Blue adds, orange subtracts. Move the slider, press play, or hover "
            "Figure 4. Hover a unit or a line for its numbers."))

    # ---- figure 4: the result
    if done:
        cell = lambda i: f"{_n(bs['y'][0][i], sign=True)} to {_n(bs['y'][1][i], sign=True)}" if bs else ""  # noqa: E731
        table = ("<div class=wrap><table class=res><tr><th></th><th class=n>Organoid run</th><th class=n>95% of repeat "
                 "runs</th><th class=n>Exact</th></tr>"
                 f"<tr><td>R²</td><td class=n>{_n(res['r2'])}</td><td class=n>"
                 + (f"{_n(bs['r2'][0])} to {_n(bs['r2'][1])}" if bs else "") + "</td><td class=n>1</td></tr>"
                 + "".join(f"<tr><td>{w}</td><td class=n>{_n(res['y'][i], sign=True)} at t = {t[i]:.1f}</td>"
                           f"<td class=n>{cell(i)}</td><td class=n>{_n(res['exact'][j], sign=True)} at t = {t[j]:.1f}</td></tr>"
                           for w, i, j in (("Dip", dip, tdip), ("Rebound", reb, treb))) + "</table></div>")
        body.append("<h2>Network output on the organoid</h2>" + _figure(
            [("", _fig_solution(res, ode))],
            f"<b>Figure 4.</b> Network output with the organoid in the loop, at the {len(t)} test times (blue). "
            + ("Band: 95% of repeat runs, from resampling the run's trains within each pulse count. " if bs else "")
            + "Dashed: the exact solution. Grey: the best curve a e<sup>−kt</sup> + b, which never dips. R² rewards the "
            f"long flat tail, so this curve already scores {_n(decay)}. Only a higher score shows the dip and rebound. "
            "Black: the test time shown in Figure 3. Hover the plot to trace another.") + table)

    # ---- figure 5: controls
    drift, worst = [], None
    for c in res["curves"]:
        by = {e["pulses"]: e["mean"] for e in res.get("eval_means", []) if (e["stim"], e["rec"]) == (c["stim"], c["rec"])}
        span = max(np.ptp(c["mean"]), 1e-9)
        for n, v in zip(c["counts"], c["mean"]):
            if n in by:
                ch = (by[n] - v) / span
                worst = worst if worst and abs(worst[2]) >= abs(ch) else (n, v, ch, by[n])
                warn = f" style='color:{CHECK_WARN[1]}'" if round(abs(ch), 2) >= 0.3 else ""
                drift.append(f"<tr><td class=n>{n}</td><td class=n>{_n(v, 1)}</td><td class=n>{_n(by[n], 1)}</td>"
                             f"<td class=n{warn}>{ch:+.0%}</td></tr>".replace("-", "−").replace("−0%", "0%").replace("+0%", "0%"))
    if done and "shuffle" in res:
        sh = res["shuffle"]
        n, v, ch, now = worst
        body.append("<h2>Controls</h2>" + _figure(
            [("a", _fig_shuffled(res, ode)), ("b", _fig_expected(res, ode, computer))],
            "<b>Figure 5.</b> <b>a</b>, The same network, fed the run's spikes shuffled across pulse counts. If the "
            "spike count did not depend on the pulse count, runs would look like this. Of 2,000 shuffles, 95% score "
            f"below {_n(sh['p95'])} and " + ("none" if not sh["share_above_run"] else f"{sh['share_above_run']:.1%}")
            + f" reach {_n(res['r2'])}. <b>b</b>, The same network on the computer, with the first measurement's curve "
            "as activation (grey), against the organoid run (blue). The difference comes from trial noise and from the "
            f"organoid's change during the session. The largest change: {n} pulses, from {_n(v, 1)} spikes in the first "
            f"measurement to {_n(now, 1)} in the run ({f'{ch:+.0%}'.replace('-', '−')} of the curve's range).")
            + "<details><summary>Change at every pulse count</summary><div class=wrap><table>"
            "<tr><th class=n>Pulses</th><th class=n>First measurement</th><th class=n>Run</th><th class=n>Change, % of "
            f"range</th></tr>{''.join(drift)}</table></div></details>")

    # ---- limitations
    if done:
        lims = ["One run on one electrode pair. Repeats on other days and other pairs are needed before a general claim."]
        if bs:
            lims.append(f"Trial noise is large at {res['eval_reps']} trains per count: {bs['below_decay']:.0%} of repeat "
                        f"runs would not clear {_n(decay)}. More trains per count would narrow this.")
        if worst and round(abs(worst[2]), 2) >= 0.3:
            lims.append(f"The response changed during the session: {worst[0]} pulses went from {_n(worst[1], 1)} to "
                        f"{_n(worst[3], 1)} spikes ({abs(worst[2]):.0%} of the curve's range).")
        lo, hi = min(c0["counts"]), max(c0["counts"])
        ends = sum(v for k, v in used.items() if int(k.split("-")[2]) in (lo, hi))
        if ends > sum(used.values()) / 2:
            lims.append(f"The network leaned on the ends of the range: {lo} and {hi} pulses made up {ends} of the "
                        f"{sum(used.values())} requests.")
        body.append("<h2>Limitations</h2><ul class=plain>" + "".join(f"<li>{html.escape(x)}</li>" for x in lims) + "</ul>")

    # ---- checks
    checks = []

    def check(ok, what, value, need):
        icon = CHECK_OK if ok else CHECK_WARN if ok is None else CHECK_BAD
        checks.append(f"<tr><td>{_icon(icon)}</td><td>{html.escape(what)}</td><td>{html.escape(value)}</td>"
                      f"<td>{html.escape(need)}</td></tr>")
    for c in res["curves"]:
        check(c["p"] < GATE_P, f"Spikes depend on the pulse count, e{c['stim']}→e{c['rec']}", f"p = {c['p']:.1g}",
              f"p < {GATE_P:g}")
        check(c["artifact"] < ARTIFACT_LIMIT, f"No stimulation artifact in the window, e{c['stim']}→e{c['rec']}",
              f"{c['artifact']:.1f}× noise", f"below {ARTIFACT_LIMIT:g}×")
    if "sweep_usable" in res:
        check(res["sweep_usable"] == res["sweep_trains"] or None, "Trains usable, first measurement",
              f"{res['sweep_usable']} of {res['sweep_trains']}", "all")
    if "expected" in res:
        check(res["expected"]["noisy_median"] > decay, "Predicted R²", _n(res["expected"]["noisy_median"]),
              f"above {_n(decay)}")
    if "eval_usable" in res:
        check(res["eval_usable"] == res["eval_trains"] or None, "Trains usable, run",
              f"{res['eval_usable']} of {res['eval_trains']}", "all")
    if done:
        check(res["r2"] > decay, "R² on the organoid", _n(res["r2"]), f"above {_n(decay)}")
        if "shuffle" in res:
            check(res["r2"] > res["shuffle"]["p95"], "R² against shuffled spikes", _n(res["r2"]),
                  f"above {_n(res['shuffle']['p95'])}")
    if res.get("drift_max") is not None:
        check(round(res["drift_max"], 2) < 0.3 or None, "Change during the session", f"{res['drift_max']:.0%} of range",
              "below 30%")
    for w in res.get("warnings", []):
        check(None, w[0].upper() + w[1:], "", "")
    body.append("<h2>Quality checks</h2><div class=wrap><table class=checks><tr><th></th><th>Check</th><th>Result</th>"
                f"<th>Needed</th></tr>{''.join(checks)}</table></div>")

    # ---- methods
    cond, win = CONDITION, SPEC["response_window_ms_after_train"]
    scale = (win[1] - win[0]) / SPEC["baseline_duration_ms"]
    methods = [
        ("Preparation", f"FinalSpark {html.escape(res['fsname'])}: four organoids, 8 electrodes each. This run used the organoid on "
                        f"electrodes {org.start} to {org.stop - 1}: trains on {stims}, spikes read on {recs}. The "
                        "stimulated electrode is never read."),
        ("Stimulation", f"Biphasic pulses, {cond['polarity']}, {cond['amplitude_uA']:g} µA and {cond['phase_us']:g} µs "
                        f"per phase, {cond['ipi_ms']:g} ms apart. Trains of 2 to 20 pulses, at least 4 s apart on an "
                        "organoid, grouped by length into recordings of at most 6 min, in random order."),
        ("Spike count", f"Band-pass {SPEC['band_hz'][0]:g} to {SPEC['band_hz'][1]:g} Hz (Butterworth, order "
                        f"{SPEC['filter_order']}, zero phase), after blanking the train and "
                        f"{SPEC['blank_after_train_ms']:g} ms after it. A spike is a downward crossing of "
                        f"−{SPEC['threshold_noise_multiple']:g} × the electrode's noise (median absolute deviation / "
                        f"0.6745), at most one per {SPEC['refractory_ms']:g} ms. Thresholds are fixed from the first "
                        f"recording's pre-train baselines. Evoked spikes = crossings {win[0]:g} to {win[1]:g} ms after "
                        f"the train, minus those in the {SPEC['baseline_duration_ms']:g} ms before it × {scale:.1f}."),
        ("Response curve", f"{res['sweep_reps']} trains at each of 2, 4, …, 20 pulses. The activation is the mean per "
                           f"count, linear in between. Checks before training: Kruskal–Wallis test across counts, p < "
                           f"{GATE_P:g}, and no artifact in the window (the median across trials of its first 5 ms below "
                           f"{ARTIFACT_LIMIT:g}× what trial-to-trial spread gives)."),
        ("Network", f"{res['width']} hidden units. Unit j asks for N<sub>j</sub>(t) = round(a<sub>j</sub> t + "
                    "b<sub>j</sub>) pulses, kept within 2 to 20, and outputs the organoid's evoked spikes for that train. "
                    "y(t) = c + Σ<sub>j</sub> v<sub>j</sub> × spikes<sub>j</sub>, computed on the computer."),
        ("Training", "Physics-informed, in weak form: the equation's residual against overlapping smooth test functions, "
                     "integrated by parts, with the starting conditions. Added: the output variance expected from the "
                     "measured trial-to-trial spread, and a small penalty on the output weights. The exact solution is "
                     f"never used. {STEPS:,} steps with fractional pulse counts, then {FINETUNE_STEPS:,} with whole counts. "
                     "8 random starts, the lowest loss kept."),
        ("Prediction", f"Before the run, the run was simulated 200 times. Each pulse count's mean was drawn once, with the "
                       f"noise of {res['eval_reps']} run trains and the first measurement's own uncertainty, and shared by "
                       "every unit and time that asks for it. Median and 10 to 90% range."),
        ("Run", f"The network's pulse counts at the {n_t} test times ({test_t[0]:.1f} to {test_t[-1]:.1f}, every "
                f"{test_t[1] - test_t[0]:.1f}). Each distinct count was delivered "
                f"{res['eval_reps']} times, and its mean used wherever that count was asked for."),
        ("Scores", f"R² = 1 − Σ(y − y<sub>exact</sub>)² / Σ(y<sub>exact</sub> − mean)², over the {n_t} test times. "
                   "Curve that never dips: least-squares fit of a e<sup>−kt</sup> + b to the exact solution at the same "
                   "times. Repeat runs: the run's "
                   "trains resampled with replacement within each pulse count, 2,000 times, through the same network. "
                   "The resamples add noise to means that already carry it, so their R² sits below the run's own. "
                   "Shuffle: the run's mean responses permuted across pulse counts, 2,000 times."),
        ("Software", f"orgchar {html.escape(str(res.get('code_version', '?')))}, readout {SPEC['method']}.")]
    body.append("<h2>Methods</h2><dl class=methods>" + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in methods) + "</dl>")

    # ---- recordings and data
    rec_rows = "".join(f"<tr><td>{r['phase']}</td><td>{r['id']}</td><td class=n>"
                       + (f"{r['start']:%H:%M:%S}" if r["start"] else "") + f"</td><td class=n>{r['seconds']:.0f} s</td>"
                       f"<td class=n>{r['trains']}</td><td class=n>{r['usable']}</td><td>{html.escape(r['file'])}</td></tr>"
                       for r in rows)
    data = json.dumps(dict(result=_clean(res), trains=[] if trains is None else trains.to_dict("records")), default=float)
    body.append(f"<h2>Recordings and data</h2><details><summary>{len(rows)} recordings</summary><div class=wrap><table>"
                "<tr><th>Phase</th><th>Recording</th><th class=n>Start (UTC)</th><th class=n>Length</th>"
                f"<th class=n>Trains</th><th class=n>Usable</th><th>File</th></tr>{rec_rows}</table></div></details>"
                f"<p><a download='{html.escape(res['fsname'])}_{step_dir.name}.json' href='data:application/json;base64,"
                f"{base64.b64encode(data.encode()).decode()}'>Download the data</a> (JSON, {len(data) // 1024} KB: the "
                "result and every single train).</p>"
                "<script>addEventListener('beforeprint', () => document.querySelectorAll('details').forEach(d => d.open = true))"
                "</script>" + (NET_JS if done else ""))      # after Figures 2 and 4, which it lights
    with open(path, "w") as f:
        f.write(_page(f"{res['fsname']}: {title}", body))
