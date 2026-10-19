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
p.facts{color:var(--muted);margin:0 0 12px}
details.vocab{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 16px;margin:0 0 6px}
details.vocab dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:3px 16px;margin:8px 0 4px;font-size:13px}
details.vocab dt{font-weight:600}
details.vocab dd{margin:0;color:var(--muted)}
@media (max-width:600px){details.vocab dl{grid-template-columns:1fr}details.vocab dd{margin-bottom:5px}}
section{scroll-margin-top:12px}
p.cap{max-width:90ch;margin:4px 0 14px;color:var(--ink)}
.fig{margin:6px 0 0}
svg.scores,svg.timeline{width:100%;max-width:860px;display:block;margin:8px 0 4px;font-size:12.5px;fill:var(--muted)}
svg.scores .lab{fill:var(--ink)}
svg.timeline .blk{fill:#2a78d6;fill-opacity:.1;stroke:#2a78d6}
svg.timeline .off{fill:#e9e8e2;fill-opacity:1;stroke:#c9c8c2}
svg.timeline .nm{fill:var(--ink);font-weight:600}
svg.timeline .gl{font-size:10.5px}
table.checks td:last-child a{color:var(--muted)}
details.table,details.details{margin:6px 0 14px}
details.details h3{margin:14px 0 4px}
ol.method{padding-left:22px;max-width:100ch}
ol.method li{margin:5px 0}
table.runs td{vertical-align:middle}
svg.mini{width:160px;height:44px;display:block}
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
@media print{.ctl{display:none}section,table{break-inside:avoid}}
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


def _rule(res) -> str:
    """How a unit turns a + b t into a pulse count, as this run did it (runs before 19 Oct 2026 rounded)."""
    c = res.get("counts_used")
    if not c:
        return "round(a<sub>j</sub> t + b<sub>j</sub>) pulses, kept within 2 to 20"
    listed = f"{c[0]}, {c[1]}, …, {c[-1]}" if len(c) > 3 else ", ".join(map(str, c))
    return f"the measured pulse count ({listed}) nearest to a<sub>j</sub> t + b<sub>j</sub>"


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


def _fig_result(res, ode):
    """The organoid-driven output (blue, with its spread) against the true curve (dashed). Each time point is
    a hover target (sol-i, and its column hit-i), linked to the network diagram."""
    dense, t, bs = np.linspace(0, ode.T, 1001), res["t"], res.get("bootstrap")
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    ax.axhline(0, color=SHADE, lw=1)
    half = (t[1] - t[0]) / 2
    for i, ti in enumerate(t):
        ax.axvspan(ti - half, ti + half, color=INK, alpha=0.001, lw=0, gid=f"hit-{i}")   # hover column, shaded when lit
    if bs:
        ax.fill_between(t, *bs["y"], color=BLUE, alpha=0.13, lw=0)
    ax.plot(dense, ode.exact(dense), color=INK, lw=1.5, ls=(0, (5, 3)))
    ax.plot(t, res["y"], color=BLUE, lw=2)
    tips = {}
    for i in range(len(t)):
        ax.plot(t[i], res["y"][i], "o", color=BLUE, ms=7, mec="white", mew=1.2, gid=f"sol-{i}")
        tips[f"sol-{i}"] = tips[f"hit-{i}"] = (
            f"t = {t[i]:.1f}\norganoid-driven output {_n(res['y'][i], sign=True)}"
            + (f"\nspread {_n(bs['y'][0][i], sign=True)} to {_n(bs['y'][1][i], sign=True)}" if bs else "")
            + f"\ntrue {_n(res['exact'][i], sign=True)}")
    k = int(np.argmin(np.abs(dense - 2.6)))
    ax.annotate("true curve", (dense[k], ode.exact(dense[k])), xytext=(12, 22), textcoords="offset points", fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.text(t[-1], (bs["y"][1][-1] if bs else res["y"][-1]) + 0.04, "organoid-driven output" + (", shaded: spread" if bs else ""),
            ha="right", fontsize=8.5, color=INK)
    ax.set_xlim(0, ode.T)
    ax.set_xlabel("time t")
    ax.set_ylabel("y(t)")
    return _tips(_svg(fig), tips)


def _fig_random(res, ode):
    dense, t = np.linspace(0, ode.T, 1001), res["t"]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.axhline(0, color=SHADE, lw=1)
    for k, y in enumerate(res["shuffle"]["examples"]):
        ax.plot(t, y, color=GREY, lw=0.8, alpha=0.4, label="random pairings (30 of 2,000)" if k == 0 else None)
    ax.plot(dense, ode.exact(dense), color=INK, lw=1.5, ls=(0, (5, 3)), label="true curve")
    ax.plot(t, res["y"], color=BLUE, lw=2, label="organoid-driven output")
    ax.set_xlabel("time t")
    ax.set_ylabel("y(t)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left", bbox_to_anchor=(0, 1))
    return _svg(fig)


def _fig_expected(res, ode):
    dense, t = np.linspace(0, ode.T, 1001), res["t"]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.axhline(0, color=SHADE, lw=1)
    ax.plot(dense, ode.exact(dense), color=INK, lw=1.5, ls=(0, (5, 3)), label="true curve")
    ax.plot(t, res["y_model"], color=GREY, lw=2, label="expected, from Measure")
    ax.plot(t, res["y"], color=BLUE, lw=2, label="got, from the Run")
    ax.set_xlabel("time t")
    ax.set_ylabel("y(t)")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left", bbox_to_anchor=(0, 1))
    return _svg(fig)


def _scores(res):
    """Three rows on one 0 to 1 axis: this run (with its spread), the score expected before the Run, and the
    no-dip baseline (also a dashed line through every row)."""
    X, e, decay, bs = (lambda v: 230 + 360 * min(max(v, 0.0), 1.0)), res.get("expected"), res["decay_r2"], res.get("bootstrap")
    rows = []   # (label, low, high, dot, colour, hollow, value text)
    if "y" in res:
        rows.append(("This run", *(bs["r2"] if bs else (None, None)), res["r2"], BLUE, False,
                     _n(res["r2"]) + (f", spread {_n(bs['r2'][0])} to {_n(bs['r2'][1])}" if bs else "")))
    if e:
        rows.append(("Expected before the Run", e["noisy_p10"], e["noisy_p90"], e["noisy_median"], BLUE, True,
                     f"{_n(e['noisy_median'])}, likely {_n(e['noisy_p10'])} to {_n(e['noisy_p90'])}"))
    rows.append(("No-dip baseline", None, None, decay, GREY, False, _n(decay)))
    h = 30 * len(rows) + 62
    g = [f"<svg class=scores viewBox='0 0 860 {h}' role=img aria-label='match scores'>",
         f"<line x1={X(decay)} x2={X(decay)} y1=4 y2={h - 52} stroke='{GREY}' stroke-dasharray='3 3' />"]
    for i, (label, lo, hi, dot, col, hollow, text) in enumerate(rows):
        y = 18 + 30 * i
        g.append(f"<text x=0 y={y + 5} class=lab>{html.escape(label)}</text>"
                 f"<line x1={X(0)} x2={X(1)} y1={y} y2={y} stroke='#e3e2dc' />")
        if lo is not None:
            g.append(f"<rect x={X(lo)} y={y - 7} width={X(hi) - X(lo)} height=14 rx=4 fill='{col}' fill-opacity=.22 />")
        g.append(f"<circle cx={X(dot)} cy={y} r=6.5 fill='{'#fff' if hollow else col}' stroke='{col if hollow else '#fff'}' "
                 f"stroke-width={2 if hollow else 1.5} />")
        g.append(f"<text x={X(1) + 18} y={y + 5}>{html.escape(text)}</text>")
    y = h - 46
    g.append(f"<line x1={X(0)} x2={X(1)} y1={y} y2={y} stroke='{GREY}' /><text x={X(0.5)} y={y + 33} "
             "text-anchor=middle>match score</text>")
    g += [f"<line x1={X(v)} x2={X(v)} y1={y} y2={y + 4} stroke='{GREY}' /><text x={X(v)} y={y + 17} "
          f"text-anchor=middle>{'0 or less' if v == 0 else f'{v:g}'}</text>" for v in (0, 0.25, 0.5, 0.75, 1)]
    return "".join(g) + "</svg>"


def _timeline(res, rows):
    """The session left to right: Measure, Fit, Run as blocks sized by their minutes (equal without times),
    with the check between them that could stop the session."""
    meas, run = _span(rows, "Measure"), _span(rows, "Run")
    ok1, ok2 = "expected" in res, "eval_trains" in res
    mins = ([(b - a).total_seconds() / 60 for a, b in ((meas[0], meas[1]), (meas[1], run[0]), (run[0], run[1]))]
            if meas and run else [1, 1, 1])
    bursts = lambda p: sum(r["trains"] for r in rows if r["phase"] == p)   # noqa: E731
    blocks = [("Measure", True, f"{bursts('Measure')} bursts"), ("Fit", ok1, "no stimulation"), ("Run", ok2, f"{bursts('Run')} bursts")]
    gates = [(ok1, ("response depends", "on burst size")), (ok2 if ok1 else None, ("expected score", "above baseline"))]
    W, gap = 860, 56
    widths = [max(130, (W - 2 * gap) * m / sum(mins)) for m in mins]
    widths = [w * (W - 2 * gap) / sum(widths) for w in widths]
    g, x = [f"<svg class=timeline viewBox='0 0 {W} 104' role=img aria-label='the session'>"], 0.0
    for i, ((name, on, what), w) in enumerate(zip(blocks, widths)):
        took = f", {mins[i]:.0f} min" if meas and run and on else ""
        g.append(f"<rect x={x:.0f} y=8 width={w:.0f} height=46 rx=8 class='{'blk' if on else 'blk off'}' />"
                 f"<text x={x + 12:.0f} y=28 class=nm>{name}{took}</text>"
                 f"<text x={x + 12:.0f} y=46>{what if on else 'not run'}</text>")
        x += w
        if i < 2:
            ok, why = gates[i]
            sym, col = (CHECK_OK if ok else CHECK_BAD) if ok is not None else ("", GREY)
            g.append(f"<circle cx={x + gap / 2:.0f} cy=31 r=11 fill='#fff' stroke='{col}' stroke-width=1.5 />"
                     f"<text x={x + gap / 2:.0f} y=36 text-anchor=middle fill='{col}' font-weight=700>{sym}</text>"
                     f"<text x={x + gap / 2:.0f} y=72 text-anchor=middle class=gl>check: {why[0]}</text>"
                     f"<text x={x + gap / 2:.0f} y=86 text-anchor=middle class=gl>{why[1]}</text>")
            x += gap
    return "".join(g) + "</svg>"


def _changes(res):
    """[(pulses, Measure average, Run average, change as a share of the Measure curve's range)] at the
    sizes both measured, for every pair."""
    out = []
    for c in res["curves"]:
        by = {e["pulses"]: e["mean"] for e in res.get("eval_means", []) if (e["stim"], e["rec"]) == (c["stim"], c["rec"])}
        span = max(np.ptp(c["mean"]), 1e-9)
        out += [(n, v, by[n], (by[n] - v) / span) for n, v in zip(c["counts"], c["mean"]) if n in by]
    return out


def _checks(res):
    """Every check of the run: [(ok, what, value, needed, anchor)], ok None for a warning, the main ones first."""
    from .measure import ARTIFACT_LIMIT
    from .solve import GATE_P
    out, decay, bs, done = [], res.get("decay_r2"), res.get("bootstrap"), "y" in res
    if done:
        out.append((res["r2"] > decay, "Score above the no-dip baseline",
                    _n(res["r2"]) + (f" (spread {_n(bs['r2'][0])} to {_n(bs['r2'][1])})" if bs else ""),
                    f"above {_n(decay)}", "result"))
    if "shuffle" in res:
        out.append((res["r2"] > res["shuffle"]["p95"], "Score above random pairing", _n(res["r2"]),
                    f"above {_n(res['shuffle']['p95'])}, where 95% of random pairings fall", "random"))
    if res.get("drift_max") is not None:
        out.append((round(res["drift_max"], 2) < 0.3 or None, "Organoid unchanged between Measure and Run",
                    f"largest change {res['drift_max']:.0%} of the range", "below 30%", "organoid"))
    for part, n_ok, n in (("Measure", "sweep_usable", "sweep_trains"), ("Run", "eval_usable", "eval_trains")):
        if n_ok in res:
            out.append((res[n_ok] == res[n] or None, f"Clean bursts, {part}", f"{res[n_ok]} of {res[n]}", "all", ""))
    for c in res["curves"]:
        pair = f", e{c['stim']}→e{c['rec']}" if len(res["curves"]) > 1 else ""
        out.append((c["p"] < GATE_P, "Response depends on burst size" + pair, f"p = {c['p']:.1g}", f"p < {GATE_P:g}",
                    "organoid"))
        out.append((c["artifact"] < ARTIFACT_LIMIT, "No leftover stimulation signal in the count" + pair,
                    f"{c['artifact']:.1f}× noise", f"below {ARTIFACT_LIMIT:g}×", "counting"))
    if "expected" in res:
        out.append((res["expected"]["noisy_median"] > decay, "Expected score above the no-dip baseline",
                    _n(res["expected"]["noisy_median"]), f"above {_n(decay)}", "session"))
    out += [(None, w[0].upper() + w[1:], "", "", "") for w in res.get("warnings", [])]
    return out


def _mini(res, ode):
    """A thumbnail of the result: true curve (dashed) and the organoid-driven output, 0 <= t <= 5."""
    X = lambda t: 4 + 152 * t / 5                                          # noqa: E731
    Y = lambda y: 30 - 24 * y                                              # noqa: E731
    dense = np.linspace(0, 5, 101)
    true = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(dense, ode.exact(dense)))
    out = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(res["t"], res["y"]) if t <= 5)
    return (f"<svg class=mini viewBox='0 0 160 44'><line x1=4 x2=156 y1={Y(0)} y2={Y(0)} stroke='#e3e2dc' />"
            f"<polyline points='{true}' fill=none stroke='{INK}' stroke-width=1.2 stroke-dasharray='3 2' />"
            f"<polyline points='{out}' fill=none stroke='{BLUE}' stroke-width=2 /></svg>")


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
                   alpha=0.25, lw=0)                         # the burst, blanked
        ax.plot(x, np.column_stack([v.min(1), v.max(1)]).ravel(), color=MUTED, lw=0.5)
        ax.axhline(m["thr"], color=INK, lw=0.8, ls=(0, (3, 2)))
        base = np.array(m["baseline_samples"]) / per_ms + t0
        win = (np.array(m["event_samples"]) + m["window_samples"][0]) / per_ms
        ax.plot(base, np.full(len(base), 0.86 * lim), "|", color=MUTED, ms=6, mew=0.9)
        ax.plot(win, np.full(len(win), 0.86 * lim), "|", color=BLUE, ms=6, mew=0.9)
        scale = (m["window_samples"][1] - m["window_samples"][0]) / len(m["baseline"])
        ax.set_title(f"{m['n']}-pulse burst: {m['response_events']} spikes after − {m['baseline_events']} before × "
                     f"{scale:.1f} = response {_n(m['adjusted'], 1)}", fontsize=8, fontweight="normal")
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("µV")
    ax.set_xlim(t0, right)
    ax.set_xlabel("ms from the start of the burst")
    return _svg(fig)


def _fig_curve(c, ev, trains, used):
    """Response against burst size: single bursts, the averages of Measure (grey) and of the Run (blue), and
    below, how often the network used each size."""
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
    ax.plot(c["counts"], c["mean"], color=GREY, lw=2, label="Measure, average")
    tips = {}
    for n, m, sd in zip(c["counts"], c["mean"], c["sd"]):
        ax.plot(n, m, "o", color=GREY, ms=5.5, mec="white", mew=1, gid=f"sw-{s}-{r}-{n}")
        tips[f"sw-{s}-{r}-{n}"] = f"{n} pulses, Measure\naverage {_n(m, 1)} ± {sd:.1f}, {n_trains.get(('sweep', n), 0)} bursts"
    first = dict(zip(c["counts"], c["mean"]))
    for e in ev:
        n = e["pulses"]
        ax.plot(n, e["mean"], "o", color=BLUE, ms=7, mec="white", mew=1.2, gid=f"crv-{s}-{r}-{n}",
                label="Run, average" if e is ev[0] else None)
        tips[f"crv-{s}-{r}-{n}"] = (f"{n} pulses, Run\naverage {_n(e['mean'], 1)}, {n_trains.get(('evaluate', n), 0)} bursts"
                                    + (f"\nMeasure {_n(first[n], 1)}" if n in first else "")
                                    + f"\nused {asks.get(n, 0)} times by the network")
    ax.set_ylabel("response (spikes)")
    ax.legend(frameon=False, fontsize=7.5)
    if used:
        bx.bar(list(asks), list(asks.values()), width=0.6, color=BLUE, alpha=0.55, lw=0)
        for n, v in asks.items():
            if v >= 0.15 * sum(asks.values()):
                bx.text(n, v, str(v), ha="center", va="bottom", fontsize=7, color=INK)
        bx.set_ylim(0, max(asks.values()) * 1.35)
        bx.set_ylabel("times used", fontsize=7.5)
    else:
        bx.set_visible(False)
    bx.set_xticks(range(2, 21, 2))
    (bx if used else ax).set_xlabel("burst size (pulses)")
    return _tips(_svg(fig), tips)


def _network(res, stims, recs, start):
    """The network at one moment, as an SVG that NET_JS (at the page's end) fills in from the run's own
    numbers (netdata): time -> units (burst sizes) -> organoid (responses) -> weighted lines -> y."""
    k, wt = res["width"], res["weights"]
    ys = [52 + 28 * j for j in range(k)]
    mid, h = (ys[0] + ys[-1]) / 2, ys[-1] + 34
    means = {(m["stim"], m["rec"], m["pulses"]): m["mean"] for m in res["eval_means"]}
    keys = _keys(res)
    spikes = [[means[tuple(int(v) for v in key.split("-"))] for key in row] for row in keys]
    contrib = [abs(w * x) for row in spikes for w, x in zip(wt["w_out"], row)]
    data = dict(t=res["t"], exact=res["exact"], N=res["pulses"], s=spikes, w=wt["w_out"], b=wt["b_out"],
                wi=wt["w_in"], bi=wt["b_in"], k=keys, cmax=max(contrib + [1e-9]), start=start,
                rule=re.sub("<[^>]+>", "", _rule(res)).replace("aj t + bj", "{a} t {b}"))
    g = [f"<svg id=net viewBox='0 0 760 {h}' role=img aria-label='the network at one moment'>",
         "<text class=hd x=60 y=16>input</text><text class=hd x=250 y=16>units</text>"
         f"<text class=hd x=395 y=16>organoid: bursts on {stims}, responses on {recs}</text>"
         "<text class=hd x=584 y=16>× weight</text><text class=hd x=690 y=16>output</text>",
         f"<rect class=tissue x=290 y={ys[0] - 18} width=210 height={ys[-1] - ys[0] + 36} rx=10 />"]
    for j, y in enumerate(ys):
        g.append(f"<line class=in x1=82 y1={mid} x2=238 y2={y} /><line id=e{j} x1=500 y1={y} x2=668 y2={mid}>"
                 f"<title id=et{j}></title></line><circle class=unit cx=250 cy={y} r=12><title id=ut{j}></title>"
                 f"</circle><text class=c id=n{j} x=250 y={y + 4}></text><text id=s{j} x=304 y={y + 4}></text>")
    g.append(f"<circle class=node cx=60 cy={mid} r=22 /><text class=c x=60 y={mid + 5}>t</text>"
             f"<text class=c id=tv x=60 y={mid + 42}></text><circle class=node cx=690 cy={mid} r=22><title>y = "
             f"{_n(wt['b_out'], sign=True)} + the {k} weighted responses</title></circle><text class=c x=690 "
             f"y={mid + 5}>y</text><text class=c id=yv x=690 y={mid + 42}></text><text class='c hd' id=yt x=690 "
             f"y={mid + 58}></text></svg>")
    return ("".join(g) + f"<div class=ctl><button id=play>▶ play</button><input id=slider type=range min=0 "
            f"max={len(res['t']) - 1} aria-label='test time'></div><script type=application/json id=netdata>"
            f"{json.dumps(data)}</script>")


# Fills the network diagram for one moment, and lights that moment in the result and its burst sizes in the
# response curve.
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
    $('s' + j).textContent = `${n} pulses → response ${g(s)}`;
    e.style.stroke = c < 0 ? '#eb6834' : '#2a78d6';
    e.style.strokeWidth = 1 + 9 * Math.abs(c) / D.cmax;
    $('et' + j).textContent = `response ${g(s)} × weight ${D.w[j].toFixed(3).replace('-', '−')} = ${f(c)}`;
    $('ut' + j).textContent = `unit ${j + 1}: ` + D.rule.replace('{a}', D.wi[j].toFixed(2).replace('-', '−')).replace('{b}', f(D.bi[j]));
  });
  $('yv').textContent = 'y = ' + f(y);
  $('yt').textContent = 'true ' + f(D.exact[i]);
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


VOCAB = [("Task", "produce the motion of a weight on a spring, let go from y = 1: it dips below zero, rebounds a little "
                   "and settles (the dashed true curve)"),
         ("Burst", "a group of 2 to 20 electrical pulses, 10 ms apart, on one electrode"),
         ("Response", "spikes in the 200 ms after a burst, minus the spikes normally there; below zero, the burst "
                      "silenced the organoid"),
         ("Unit", "one of the network's hidden neurons: it picks a burst size, and the organoid's response is its output"),
         ("Measure, Fit, Run", "the three steps: measure the organoid's response to each burst size; fit the network "
                               "on the computer; run it, with the organoid giving every response"),
         ("Match score", "how closely the output follows the true curve (R²); 1 is a perfect match"),
         ("No-dip baseline", "the best score of a curve that never dips below zero"),
         ("Spread", "where the score could have landed from burst-to-burst noise alone"),
         ("Random pairing", "the same responses reassigned to burst sizes at random")]


def _outcome(res) -> str:
    from .measure import ARTIFACT_LIMIT
    from .solve import GATE_P
    if "y" in res:
        ok = res["r2"] > res["decay_r2"]
        return (f"The organoid-driven network {'followed' if ok else 'did not follow'} the spring's dip and rebound: "
                f"match score {_n(res['r2'])}, {'above' if ok else 'below'} the no-dip baseline of {_n(res['decay_r2'])}.")
    if "expected" in res:
        return (f"Stopped before the Run: the expected score, {_n(res['expected']['noisy_median'])}, was not above the "
                f"no-dip baseline of {_n(res['decay_r2'])}. No more stimulation.")
    if res["verdict"].startswith("FAILED"):
        return res["verdict"]
    weak = any(c["p"] >= GATE_P for c in res["curves"])
    return ("Stopped after Measure: the response " + ("did not depend clearly on burst size" if weak else
            f"carried leftover stimulation signal (over {ARTIFACT_LIMIT:g}× noise)") + ". No more stimulation.")


def _section(anchor, heading, *parts):
    return f"<section id={anchor}><h2>{heading}</h2>" + "".join(parts) + "</section>"


def _cap(text):
    return f"<p class=cap>{text}</p>"


def solve_page(path, res: dict, when: str):
    """One solve run: the result and its score, the checks, what the organoid did, how the network works, the
    session, and the details. Reads the run's folder next to `path`: single bursts, recordings, two raw bursts."""
    from .measure import ARTIFACT_LIMIT, CONDITION, organoid_of
    from .net import FINETUNE_STEPS, STEPS, Oscillator
    from .readout import SPEC
    from .solve import GATE_P
    ode = Oscillator(**res["ode"])
    step_dir = Path(path).parent
    parts = [pd.read_csv(f).assign(part=f.parent.name) for f in (step_dir / "sweep" / "responses.csv",
                                                                  step_dir / "evaluate" / "responses.csv") if f.exists()]
    trains = pd.concat(parts, ignore_index=True) if parts else None
    rows, examples = _recordings(step_dir), _examples(step_dir, res)
    done = "y" in res
    used = pd.Series([k for row in _keys(res) for k in row]).value_counts().to_dict() if done else {}
    c0 = res["curves"][0]
    org = organoid_of(c0["stim"])
    stims = ", ".join(sorted({f"e{c['stim']}" for c in res["curves"]}))
    recs = ", ".join(sorted({f"e{c['rec']}" for c in res["curves"]}))
    meas, run = _span(rows, "Measure"), _span(rows, "Run")
    end = (run or meas or (None, None))[1]
    title = f"{res['fsname']}, {stims}→{recs}: organoid-driven network"

    # ---- header
    facts = [f"{res['fsname']}, organoid on electrodes {org.start} to {org.stop - 1}",
             f"Bursts on {stims}, responses read on {recs}",
             f"{when.removesuffix(' UTC')}" + (f" to {end:%H:%M} UTC" if end else ""),
             f"{sum(r['trains'] for r in rows)} bursts in all"]
    body = [f"<h1>{html.escape(title)}</h1>"
            + ("<p class=sim>SIMULATED: synthetic organoid, no tissue.</p>" if res.get("simulated") else "")
            + f"<p class=lede>{html.escape(_outcome(res))}</p><p class=facts>{html.escape('. '.join(facts))}.</p>"
            + "<details class=vocab open><summary>Words used here</summary><dl>"
            + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in VOCAB) + "</dl></details>"]

    # ---- result
    if done:
        ok = res["r2"] > res["decay_r2"]
        body.append(_section(
            "result", f"The output {'followed' if ok else 'did not follow'} the dip and rebound",
            f"<div class=fig>{_fig_result(res, ode)}</div>",
            _cap("Blue: the network's output when every hidden unit's value is a response of the organoid, at the "
                 f"{len(res['t'])} times it was scored. Hover a point to see that moment in the network below."),
            _scores(res),
            _cap("A curve that never dips already scores " + _n(res["decay_r2"]) + ", because the true curve is flat "
                 "after t ≈ 4. Only a score above that line shows the dip and rebound.")))
    elif "expected" in res:
        body.append(_section("result", "No Run: the expected score was too low", _scores(res)))

    # ---- checks
    checks = _checks(res)
    n_ok = sum(1 for c in checks if c[0])
    body.append(_section(
        "checks", f"{n_ok} of {len(checks)} checks passed",
        "<div class=wrap><table class=checks><tr><th></th><th>Check</th><th>Result</th><th>Needed</th><th></th></tr>"
        + "".join(f"<tr><td>{_icon(CHECK_OK if ok else CHECK_WARN if ok is None else CHECK_BAD)}</td><td>{html.escape(w)}"
                  f"</td><td>{html.escape(v)}</td><td>{html.escape(need)}</td><td>"
                  + (f"<a href='#{a}'>see</a>" if a else "") + "</td></tr>" for ok, w, v, need, a in checks)
        + "</table></div>",
        f"<h3 id=random>Random pairing</h3><div class=fig>{_fig_random(res, ode)}</div>"
        + _cap(f"If the organoid's responses did not depend on burst size, the output would look like these grey lines. "
               f"95% of 2,000 random pairings score below {_n(res['shuffle']['p95'])}, and "
               + ("none" if not res["shuffle"]["share_above_run"] else f"{res['shuffle']['share_above_run']:.1%}")
               + f" reach this run's {_n(res['r2'])}.") if "shuffle" in res else ""))

    # ---- the organoid
    changes = _changes(res)
    worst = max(changes, key=lambda c: abs(c[3])) if changes else None
    heading = ("The organoid's response to bursts" if not worst else
               "The organoid changed between Measure and Run" if round(abs(worst[3]), 2) >= 0.3 else
               "The organoid held steady between Measure and Run")
    curves = "".join(f"<div class=fig>{_fig_curve(c, [e for e in res.get('eval_means', []) if (e['stim'], e['rec']) == (c['stim'], c['rec'])], trains, used)}</div>"
                     for c in res["curves"])
    body.append(_section(
        "organoid", heading, curves,
        _cap(f"Grey: Measure, {res['sweep_reps']} bursts per size. "
             + (f"Blue: the Run, {res['eval_reps']} per size, where the two split the organoid changed "
                f"(most at {worst[0]} pulses: {_n(worst[1], 1)} then {_n(worst[2], 1)}). Bars: how often the network "
                "used each size." if done and worst else "Faint dots: single bursts.")),
        "<details class=table><summary>Every burst size</summary><div class=wrap><table><tr><th class=n>Pulses</th>"
        "<th class=n>Measure</th><th class=n>Run</th><th class=n>Change, share of range</th></tr>"
        + "".join(f"<tr><td class=n>{n}</td><td class=n>{_n(a, 1)}</td><td class=n>{_n(b, 1)}</td><td class=n>"
                  f"{_n(100 * ch, 0, sign=True)}%</td></tr>" for n, a, b, ch in changes)
        + "</table></div></details>" if changes else "",
        (f"<h3 id=counting>How a response is counted</h3><div class=fig>{_fig_traces(examples)}</div>"
         + _cap(f"Two single bursts from Measure, voltage on e{c0['rec']}. Ticks mark spikes: grey in the 500 ms before "
                "the burst, blue in the 200 ms counted after it (the burst itself is blanked).")) if examples else ""))

    # ---- the network
    if done:
        dip = _swing(ode, res["t"], res["y"])[0]
        body.append(_section(
            "network", "How the network turns time into an answer",
            _network(res, stims, recs, dip),
            _cap("Each unit turns the time t into a burst size. The organoid's response to that burst is multiplied "
                 "by the unit's weight, and the results add up to y: thick lines count most, blue adds, orange subtracts."),
            f"<h3>Expected and got</h3><div class=fig>{_fig_expected(res, ode)}</div>",
            _cap("Grey: the output the network expected, from the responses seen in Measure. Blue: what it got in the "
                 "Run. The gap is the organoid's doing: noise and change.")))

    # ---- the session
    minutes = f"{((end - meas[0]).total_seconds() / 60):.0f} minutes, " if meas and end else ""
    body.append(_section(
        "session", f"The session: {minutes}{sum(r['trains'] for r in rows)} bursts", _timeline(res, rows),
        _cap("Each check could stop the session before more stimulation.")))

    # ---- details
    cond, win = CONDITION, SPEC["response_window_ms_after_train"]
    steps = [
        f"Bursts: {cond['amplitude_uA']:g} µA, {cond['phase_us']:g} µs per phase ({cond['polarity']}), pulses "
        f"{cond['ipi_ms']:g} ms apart, at least 4 s between bursts on an organoid, sizes in random order. The "
        "stimulated electrode is never read.",
        f"Spikes: voltage filtered to {SPEC['band_hz'][0]:g} to {SPEC['band_hz'][1]:g} Hz; a spike is a dip below "
        f"{SPEC['threshold_noise_multiple']:g}× the electrode's typical noise, at most one per {SPEC['refractory_ms']:g} "
        f"ms. Response: spikes {win[0]:g} to {win[1]:g} ms after the burst, minus {(win[1] - win[0]) / SPEC['baseline_duration_ms']:.1f}"
        f"× the spikes in the {SPEC['baseline_duration_ms']:g} ms before it.",
        f"Measure: {res['sweep_reps']} bursts at each size 2, 4, …, 20. Checks: the response depends on burst size "
        f"(Kruskal–Wallis, p < {GATE_P:g}) and carries no leftover stimulation signal (below {ARTIFACT_LIMIT:g}× noise).",
        f"Fit: {res['width']} units; unit j asks for {_rule(res)}. y = c + Σ weight<sub>j</sub> × response<sub>j</sub>. "
        "The weights are fitted so that y obeys the spring equation and starts at 1, at rest (the physics-informed "
        "approach), allowing for burst-to-burst noise. The true curve is never used. "
        f"{STEPS + FINETUNE_STEPS:,} steps, best of 8 starts.",
        "Expected score: the Run simulated 200 times with the noise seen in Measure. The Run goes ahead only if the "
        "middle of these is above the no-dip baseline.",
        f"Run: every burst size the network asks for, delivered {res['eval_reps']} times; the average is used wherever "
        "that size is asked for.",
        f"Scores: match score R² at the {res['points'] - 1} test times. No-dip baseline: the best a e<sup>−kt</sup> + b. "
        "Spread: the Run's bursts resampled 2,000 times. Random pairing: the Run's responses reassigned to burst sizes, "
        "2,000 times.",
        f"Software: orgchar {html.escape(str(res.get('code_version', '?')))}, readout {SPEC['method']}."]
    data = json.dumps(dict(result=_clean(res), trains=[] if trains is None else trains.to_dict("records")), default=float)
    body.append(
        "<details class=details><summary>Details: method, recordings, data</summary><h3>Method</h3><ol class=method>"
        + "".join(f"<li>{x}</li>" for x in steps) + "</ol><h3>Recordings</h3><div class=wrap><table>"
        "<tr><th>Step</th><th>Recording</th><th class=n>Start (UTC)</th><th class=n>Length</th><th class=n>Bursts</th>"
        "<th class=n>Clean</th><th>File</th></tr>"
        + "".join(f"<tr><td>{r['phase']}</td><td>{r['id']}</td><td class=n>"
                  + (f"{r['start']:%H:%M:%S}" if r["start"] else "") + f"</td><td class=n>{r['seconds']:.0f} s</td>"
                  f"<td class=n>{r['trains']}</td><td class=n>{r['usable']}</td><td>{html.escape(r['file'])}</td></tr>"
                  for r in rows)
        + f"</table></div><p><a download='{html.escape(res['fsname'])}_{step_dir.name}.json' href='data:application/json;"
        f"base64,{base64.b64encode(data.encode()).decode()}'>Download the data</a> (JSON, {len(data) // 1024} KB: the "
        "result and every single burst).</p></details>"
        "<script>addEventListener('beforeprint', () => document.querySelectorAll('details').forEach(d => d.open = true))"
        "</script>" + (NET_JS if done else ""))      # last: it lights the result and the response curve
    with open(path, "w") as f:
        f.write(_page(title, body))


def runs_page(root):
    """root/runs.html: every solve run recorded on this machine (simulations left out), newest first, one row
    each, linking to its report."""
    from .net import Oscillator
    from .solve import when
    root, rows, n_done, n_ok, n_stop = Path(root), [], 0, 0, 0
    for f in sorted(root.glob("*/*_solve/result.json"), key=lambda f: f.parent.name, reverse=True):
        if f.parts[-3].startswith("SIMULATED_"):
            continue
        res, d = json.loads(f.read_text()), f.parent
        done = "y" in res
        n_done += done
        n_ok += done and res["r2"] > res["decay_r2"]
        n_stop += not done
        checks = _checks(res)
        bs = res.get("bootstrap")
        pairs = ", ".join(f"e{a}→e{b}" for a, b in res["pairs"])
        rows.append(f"<tr><td><a href='{d.parent.name}/{d.name}/report.html'>{when(d)}</a></td><td>{res['fsname']}, "
                    f"{pairs}</td><td>{_mini(res, Oscillator(**res['ode'])) if done else ''}</td><td class=n>"
                    + (_n(res["r2"]) if done else "") + "</td><td class=n>"
                    + (f"{_n(bs['r2'][0])} to {_n(bs['r2'][1])}" if bs else "") + "</td><td>"
                    + "".join(f"<span title='{html.escape(c[1])}: {html.escape(c[2])}'>"
                              f"{_icon(CHECK_OK if c[0] else CHECK_WARN if c[0] is None else CHECK_BAD)}</span>" for c in checks)
                    + f"</td><td>{html.escape(_outcome(res))}</td></tr>")
    body = ["<h1>Organoid-driven network: all runs</h1>",
            f"<p class=lede>{n_done} completed run{'s' * (n_done != 1)}, {n_ok} above the no-dip baseline."
            + (f" {n_stop} more stopped by a check before the Run." if n_stop else "") + "</p>",
            "<details class=vocab><summary>Words used here</summary><dl>"
            + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in VOCAB) + "</dl></details>",
            "<div class=wrap><table class=runs><tr><th>Run</th><th>Organoid</th><th>Output (dashed: true)</th>"
            "<th class=n>Score</th><th class=n>Spread</th><th>Checks</th><th>Outcome</th></tr>" + "".join(rows)
            + "</table></div>"]
    (root / "runs.html").write_text(_page("Organoid-driven network: all runs", body))
