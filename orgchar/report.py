"""Self-contained HTML pages: shared page parts and the census page (the screen page is in
report_screen.py). Figures are inline SVG, so a page stays sharp when zoomed and opens offline."""
from __future__ import annotations

import base64
import html
import io
import json
import re
import uuid
from pathlib import Path

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
.summary{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px 14px;margin:0 0 8px}
.summary ul{margin:0 0 14px;padding-left:18px}
.summary li{margin:2px 0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat{border-top:1px solid var(--line);padding-top:8px}
.stat .v{font-size:26px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums}
.stat .k{font-size:13px}
.stat .s{font-size:12px;color:var(--muted)}
.cap{max-width:72ch;margin:4px 0 14px}
h3{margin:18px 0 4px}
table.checks td:first-child{width:1.2em}
ol.method{padding-left:20px;max-width:80ch}
ol.method li{margin:4px 0}
svg.scale{width:100%;max-width:640px;display:block;margin:6px 0 2px;font-size:11px;fill:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:13px;color:var(--muted)}
.legend i{display:inline-block;width:11px;height:11px;margin-right:6px;vertical-align:-1px}
[data-k]{cursor:pointer}
[data-k]:hover use,.lit use,.lit path{fill:#eb6834!important}
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

def _stat(value, label, sub=""):
    return (f"<div class=stat><div class=v>{html.escape(value)}</div><div class=k>{html.escape(label)}</div>"
            + (f"<div class=s>{html.escape(sub)}</div>" if sub else "") + "</div>")


def _tips(svg, tips):
    """Hover text (an SVG <title>) and HTML attributes for artists drawn with gid=key: {key: (text, attrs)}."""
    for k, (text, attrs) in tips.items():
        svg = svg.replace(f'<g id="{k}">', f'<g id="{k}"{attrs}><title>{html.escape(text)}</title>', 1)
    return svg


def _keys(res):
    """Per time point, the (pair, pulse count) every unit asked for, as 'stim-rec-n'."""
    kept = res["kept"]
    return [[f"{kept[j % len(kept)][0]}-{kept[j % len(kept)][1]}-{n}" for j, n in enumerate(row)] for row in res["pulses"]]


def _fig_solution(res, ode):
    dense, t = np.linspace(0, ode.T, 1001), res["t"]
    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    ax.axhline(0, color=SHADE, lw=1)
    ax.plot(dense, ode.exact(dense), color=GREY, lw=2, ls=(0, (5, 3)), label="true solution")
    ax.plot(t, res["y_model"], color=MUTED, lw=1.2, label="network on the measured curve")
    ax.plot(t, res["y"], color=BLUE, lw=2.2, label="network on the tissue")
    tips = {}
    for i, keys in enumerate(_keys(res)):
        ax.plot(t[i], res["y"][i], "o", color=BLUE, ms=6, mew=0, gid=f"sol-{i}")
        tips[f"sol-{i}"] = (f"t = {t[i]:.1f}\ntrue {res['exact'][i]:+.2f}\ntissue {res['y'][i]:+.2f}\n"
                            f"network on the measured curve {res['y_model'][i]:+.2f}\n"
                            f"pulses per unit: {' '.join(k.split('-')[-1] for k in keys)}",
                            f' data-k="{" ".join(sorted(set(keys)))}"')
    ax.set_xlabel("time t")
    ax.set_ylabel("y(t)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _tips(_svg(fig), tips)


def _fig_curve(c, ev, trains, used):
    """Single trains (faint) and means of the sweep (black) and the run (blue) for one pair."""
    s, r = c["stim"], c["rec"]
    pair = trains[(trains.stim == s) & (trains.rec == r)] if trains is not None else pd.DataFrame(columns=["part", "pulses"])
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.axhline(0, color=SHADE, lw=1)
    jitter = np.random.default_rng(0)
    for part, col, dx in (("sweep", GREY, -0.15), ("evaluate", BLUE, 0.15)):
        d = pair[pair.part == part]
        ax.scatter(d.pulses + dx + jitter.uniform(-0.08, 0.08, len(d)), d.response, s=6, color=col, alpha=0.35, lw=0)
    ax.plot(c["counts"], c["mean"], color=INK, lw=2, label="sweep: mean")
    n_trains = pair.groupby(["part", "pulses"]).size().to_dict()
    tips = {}
    for n, m, sd in zip(c["counts"], c["mean"], c["sd"]):
        ax.plot(n, m, "o", color=INK, ms=4.5, mew=0, gid=f"sw-{s}-{r}-{n}")
        tips[f"sw-{s}-{r}-{n}"] = (f"{n} pulses, sweep\nmean {m:.1f}, SD {sd:.1f}, {n_trains.get(('sweep', n), 0)} trains", "")
    sweep = dict(zip(c["counts"], c["mean"]))
    for e in ev:
        n = e["pulses"]
        ax.plot(n, e["mean"], "o", color=BLUE, ms=6, mew=0, gid=f"crv-{s}-{r}-{n}", label="run: mean" if e is ev[0] else None)
        tips[f"crv-{s}-{r}-{n}"] = (f"{n} pulses, run\nmean {e['mean']:.1f}, {n_trains.get(('evaluate', n), 0)} trains"
                                    + (f"\nsweep {sweep[n]:.1f}" if n in sweep else "")
                                    + f"\nasked for {used.get(f'{s}-{r}-{n}', 0)} times by the network", "")
    ax.set_xticks(range(2, 21, 2))
    ax.set_xlabel("pulses in the train")
    ax.set_ylabel("evoked spikes")
    ax.legend(frameon=False, fontsize=7.5)
    return _tips(_svg(fig), tips)


def _fig_use(pulses):
    n, k = np.unique(np.array(pulses), return_counts=True)
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    for b, x in zip(ax.bar(n, k, color=BLUE, width=0.7), n):
        b.set_gid(f"use-n{x}")
    ax.set_xticks(range(2, 21, 2))
    ax.set_xlabel("pulses in the train")
    ax.set_ylabel("requests")
    return _tips(_svg(fig), {f"use-n{x}": (f"{x} pulses: {y} requests", "") for x, y in zip(n, k)})


def _fig_null(shuffle, r2, decay):
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    ax.bar(np.linspace(-2.95, 0.95, 40), shuffle["hist"], width=0.09, color=SHADE, edgecolor=GREY, lw=0.4)
    ax.axvline(decay, color=GREY, lw=1.2, ls=(0, (4, 3)), label=f"plain decay {decay:.2f}")
    ax.axvline(r2, color=BLUE, lw=2, label=f"this run {r2:.2f}")
    ax.set_xlim(-3.05, 1.05)
    ax.set_xlabel("R² with the responses shuffled across pulse counts (−3 includes everything below)")
    ax.set_ylabel("shuffles")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    return _svg(fig)


def _scale(res):
    """The run's R² on one 0 to 1 axis, with the expected range, the plain decay and the shuffled control."""
    e, X = res["expected"], lambda v: 20 + 560 * min(max(v, 0.0), 1.0)
    out = ["<svg class=scale viewBox='0 0 600 50' role=img aria-label='R² of this run against its references'>",
           f"<line x1={X(0)} x2={X(1)} y1=22 y2=22 stroke='#c9c8c2'/>"]
    out += [f"<text x={X(v)} y=46 text-anchor=middle>{v:g}</text><line x1={X(v)} x2={X(v)} y1=19 y2=25 stroke='#c9c8c2'/>"
            for v in (0, 0.25, 0.5, 0.75, 1)]
    out += [f"<rect x={X(e['noisy_p10'])} y=14 width={X(e['noisy_p90']) - X(e['noisy_p10'])} height=16 rx=3 "
            f"fill='{BLUE}' fill-opacity=.18><title>expected R², 10–90%: {e['noisy_p10']:.2f} to {e['noisy_p90']:.2f}</title></rect>",
            f"<line x1={X(res['decay_r2'])} x2={X(res['decay_r2'])} y1=6 y2=38 stroke='{GREY}' stroke-width=2 stroke-dasharray='4 3'>"
            f"<title>plain decay: {res['decay_r2']:.2f}</title></line>"]
    if "shuffle" in res:
        p95 = res["shuffle"]["p95"]
        out.append(f"<line x1={X(p95)} x2={X(p95)} y1=6 y2=38 stroke='{ORANGE}' stroke-width=2>"
                   f"<title>shuffled control, 95% of shuffles below {p95:.2f}</title></line>")
    out.append(f"<circle cx={X(res['r2'])} cy=22 r=7 fill='{BLUE}' stroke='#fff' stroke-width=2>"
               f"<title>this run: {res['r2']:.2f}</title></circle></svg>")
    legend = [(f"background:{BLUE};border-radius:50%", f"this run {res['r2']:.2f}"),
              (f"background:{BLUE};opacity:.25", f"expected {e['noisy_p10']:.2f} to {e['noisy_p90']:.2f}"),
              (f"background:{GREY};width:3px", f"plain decay {res['decay_r2']:.2f}")]
    if "shuffle" in res:
        legend.append((f"background:{ORANGE};width:3px", f"shuffled, 95% below {res['shuffle']['p95']:.2f}"))
    return "".join(out) + "<div class=legend>" + "".join(f"<span><i style='{st}'></i>{html.escape(t)}</span>"
                                                         for st, t in legend) + "</div>"


# Hovering a time point lights up the pulse counts it used, on the curve and in the bar chart.
LINK_JS = """<script>
for (const g of document.querySelectorAll('[data-k]'))
  for (const [ev, on] of [['mouseenter', true], ['mouseleave', false]])
    g.addEventListener(ev, () => g.dataset.k.split(' ').forEach(k => ['crv-' + k, 'use-n' + k.split('-').pop()]
      .forEach(id => document.getElementById(id)?.classList.toggle('lit', on))));
</script>"""


def solve_page(path, res: dict, when: str):
    """The solve run as a short research report: summary, solution, the tissue's curve and its stability,
    how the network used the tissue, checks, method."""
    from .measure import ARTIFACT_LIMIT, CONDITION
    from .net import Oscillator
    from .solve import GATE_P
    ode = Oscillator(**res["ode"])
    step_dir = Path(path).parent           # the run's own responses, for the single trains and the data link
    parts = [pd.read_csv(f).assign(part=f.parent.name) for f in (step_dir / "sweep" / "responses.csv",
                                                                  step_dir / "evaluate" / "responses.csv") if f.exists()]
    trains = pd.concat(parts, ignore_index=True) if parts else None
    used = pd.Series([k for row in _keys(res) for k in row]).value_counts().to_dict() if "y" in res else {}
    pairs = ", ".join(f"e{x['stim']} → e{x['rec']}" for x in res["curves"])
    decay = res.get("decay_r2")
    done = "y" in res
    title = f"{res['fsname']}: damped oscillator on organoid tissue"
    body = [f"<h1>{html.escape(title)}</h1>",
            f"<p class=meta>{html.escape(when)}{' (SIMULATED)' if res.get('simulated') else ''}. Pair {pairs}. "
            f"orgchar {html.escape(str(res.get('code_version', '?')))}.</p>"]

    # ---- summary
    if done:
        above = res["r2"] > decay
        head = (f"The tissue output follows the true solution, R² {res['r2']:.2f}." if above else
                f"The tissue output does not show the oscillation, R² {res['r2']:.2f}.")
        points = [f"A plain decay with no oscillation scores {decay:.2f}. The run is "
                  + ("above it." if above else "not above it.")]
        t, y, ex = np.array(res["t"]), np.array(res["y"]), np.array(res["exact"])
        dip, true_dip = int(np.argmin(np.where(t < 2, y, np.inf))), int(np.argmin(np.where(t < 2, ex, np.inf)))
        reb = int(np.argmax(np.where((t > t[dip]) & (t < 3.5), y, -np.inf)))
        true_reb = int(np.argmax(np.where((t > t[true_dip]) & (t < 3.5), ex, -np.inf)))
        points.append(f"First dip: {y[dip]:+.2f} at t = {t[dip]:.1f}. True: {ex[true_dip]:+.2f} at t = {t[true_dip]:.1f}.")
        points.append(f"Rebound: {y[reb]:+.2f} at t = {t[reb]:.1f}. True: {ex[true_reb]:+.2f} at t = {t[true_reb]:.1f}.")
        if "shuffle" in res:
            points.append(f"Control: with the responses shuffled across pulse counts, 95% of 2,000 shuffles score "
                          f"below {res['shuffle']['p95']:.2f}.")
        points.append(f"Stimulation: {res['sweep_trains'] + res['eval_trains']} trains, "
                      f"{res['sweep_pulses'] + res['eval_pulses']:,} pulses.")
    else:
        if "expected" not in res:
            head, points = "Stopped after the sweep.", ["No pair passed the sweep's checks (see Checks).",
                                                        "The tissue was not stimulated further."]
        elif "eval_means" not in res:
            head, points = "Stopped after training.", [
                f"Expected R² {res['expected']['noisy_median']:.2f}. A plain decay scores {decay:.2f}.",
                "Below that bar, the run could not show the oscillation.", "The tissue was not stimulated further."]
        else:
            head, points = "The run failed.", ["Some pulse counts had no usable train, even after retries.",
                                               "The recordings are in the evaluate folder."]
        stats = [_stat(f"{res['sweep_trains']}", "trains", f"{res['sweep_pulses']:,} pulses")]
        if "expected" in res:
            stats[:0] = [_stat(f"{res['expected']['noisy_median']:.2f}", "expected R²"),
                         _stat(f"{decay:.2f}", "plain decay", "the bar for an oscillation")]
    body.append(f"<section class=summary><p class=lede>{html.escape(head)}</p><ul>"
                + "".join(f"<li>{html.escape(x)}</li>" for x in points) + "</ul>"
                + (_scale(res) if done else f"<div class=stats>{''.join(stats)}</div>") + "</section>")

    # ---- solution
    if done:
        body.append("<h2>Solution</h2>" + f"<div class=fig>{_fig_solution(res, ode)}</div>"
                    f"<p class=cap>The task: m y″ + c y′ + k y = 0 with m = {ode.m:g}, c = {ode.c:g}, k = {ode.k:g}, "
                    f"y(0) = {ode.y0:g}, y′(0) = {ode.v0:g}, for t from 0 to {ode.T:g}. The network is read at "
                    f"{len(res['t'])} times between its training points. Blue uses the tissue's responses during "
                    "the run. Grey uses the curve measured just before. Hover a blue point for the pulse counts "
                    "the units asked for at that time. They light up on the curve and in the bar chart below.</p>")
        if "shuffle" in res:
            body.append("<h3>Shuffled-count control</h3>"
                        f"<div class=fig>{_fig_null(res['shuffle'], res['r2'], decay)}</div><p class=cap>The run's "
                        "responses, shuffled across pulse counts and fed through the same network, 2,000 times. Tissue "
                        "whose response ignored the count would score like this. "
                        + (f"{res['shuffle']['share_above_run']:.1%} of shuffles" if res['shuffle']['share_above_run']
                           else "None of the shuffles") + " reach this run's R².</p>")

    # ---- the tissue
    body.append("<h2>Tissue response</h2>")
    for x in res["curves"]:
        ev = [e for e in res.get("eval_means", []) if (e["stim"], e["rec"]) == (x["stim"], x["rec"])]
        m = np.array(x["mean"])
        cap = [f"Stimulating e{x['stim']}, counting spikes on e{x['rec']}. Faint dots are single trains."]
        if np.isfinite(m).all():
            cap.append(f"Most spikes after {x['counts'][int(np.argmax(m))]} pulses ({m.max():.1f}). "
                       f"Fewest after {x['counts'][int(np.argmin(m))]} ({m.min():.1f}).")
            cap.append(f"The pulse count explains {x['explained']:.0%} of the trial-to-trial variance.")
        body.append(f"<div class=fig>{_fig_curve(x, ev, trains, used)}</div><p class=cap>{html.escape(' '.join(cap))}</p>")
        if ev:
            by = {e["pulses"]: e["mean"] for e in ev}
            rng_ = max(np.ptp(m), 1e-9)
            rows = "".join(
                f"<tr><td class=n>{n}</td><td class=n>{v:.1f}</td><td class=n>{by[n]:.1f}</td>"
                f"<td class=n{' style=color:' + CHECK_WARN[1] if round(abs(by[n] - v) / rng_, 2) >= 0.3 else ''}>"
                f"{(by[n] - v) / rng_:+.0%}</td></tr>" for n, v in zip(x["counts"], m) if n in by)
            body.append("<h3>Stability during the run</h3><p class=cap>The same pulse counts, measured in the sweep "
                        "and again during the run. Change is in units of the curve's range.</p>"
                        "<div class=wrap><table><tr><th class=n>Pulses</th><th class=n>Sweep</th><th class=n>Run</th>"
                        f"<th class=n>Change</th></tr>{rows}</table></div>")

    # ---- how the network used the tissue
    if done:
        p = np.array(res["pulses"])
        n, k = np.unique(p, return_counts=True)
        top = sorted(zip(k, n), reverse=True)[:2]
        body.append("<h2>How the network used the tissue</h2>" + f"<div class=fig>{_fig_use(p)}</div>"
                    f"<p class=cap>{res['width']} units, each asking for a pulse count at each of {len(res['t'])} "
                    f"times: {p.size} requests. Each distinct count was measured {res['eval_reps']} times and "
                    f"reused. {top[0][0]} requests were for {top[0][1]} pulses"
                    + (f" and {top[1][0]} for {top[1][1]}." if len(top) > 1 else ".") + "</p>")

    # ---- checks
    rows = []

    def check(ok, what, value, need):
        icon = CHECK_OK if ok else CHECK_WARN if ok is None else CHECK_BAD
        rows.append(f"<tr><td>{_icon(icon)}</td><td>{html.escape(what)}</td><td>{html.escape(value)}</td>"
                    f"<td>{html.escape(need)}</td></tr>")
    for x in res["curves"]:
        check(x["p"] < GATE_P, f"Response depends on pulse count (e{x['stim']} → e{x['rec']})",
              f"p = {x['p']:.1g}", f"p < {GATE_P:g}")
        check(x["artifact"] < ARTIFACT_LIMIT, "Stimulation artifact in the counting window",
              f"{x['artifact']:.1f}× noise floor", f"below {ARTIFACT_LIMIT:g}×")
    if "sweep_usable" in res:
        check(res["sweep_usable"] == res["sweep_trains"] or None, "Sweep trains usable",
              f"{res['sweep_usable']} of {res['sweep_trains']}", "all")
    if "expected" in res:
        check(res["expected"]["noisy_median"] > decay, "Expected R² before using the tissue",
              f"{res['expected']['noisy_median']:.2f}", f"above {decay:.2f}")
    if "eval_usable" in res:
        check(res["eval_usable"] == res["eval_trains"] or None, "Run trains usable",
              f"{res['eval_usable']} of {res['eval_trains']}", "all")
    if done:
        check(res["r2"] > decay, "R² on the tissue", f"{res['r2']:.2f}", f"above {decay:.2f}")
        if "shuffle" in res:
            check(res["r2"] > res["shuffle"]["p95"], "Shuffled-count control",
                  f"95% of shuffles below {res['shuffle']['p95']:.2f}", "below the run")
    if res.get("drift_max") is not None:
        check(round(res["drift_max"], 2) < 0.3 or None, "Largest change since the sweep", f"{res['drift_max']:.0%} of the range",
              "below 30%")
    for w in res.get("warnings", []):
        check(None, w[0].upper() + w[1:], "", "")
    body.append("<h2>Checks</h2><div class=wrap><table class=checks><tr><th></th><th>Check</th><th>Result</th>"
                f"<th>Needed</th></tr>{''.join(rows)}</table></div>")

    # ---- method
    cond = CONDITION
    body.append("<h2>Method</h2><ol class=method>"
                f"<li><b>Sweep.</b> {res['sweep_reps']} trains at each of 2, 4, …, 20 pulses on the stimulating "
                f"electrode, in random order. Pulses are {cond['amplitude_uA']:g} µA, {cond['phase_us']:g} µs per "
                f"phase, {cond['ipi_ms']:g} ms apart. The organoid rests at least 4 s between trains.</li>"
                "<li><b>Readout.</b> Spikes counted 10 to 210 ms after the train, minus the count in the 500 ms "
                "before it. Thresholds are fixed at the start of the run.</li>"
                f"<li><b>Network.</b> {res['width']} hidden units. Each turns the time t into a pulse count from 2 "
                "to 20. The tissue's measured response to that train is the unit's output. The computer adds the "
                "units' outputs with learned weights.</li>"
                "<li><b>Training.</b> The weak form of the equation, with the measured trial noise. 8 starts. "
                "No stimulation.</li>"
                f"<li><b>Run.</b> Each pulse count the network needs is measured {res['eval_reps']} times. The mean "
                "is used wherever that count is asked for.</li></ol>")
    data = json.dumps(dict(result=_clean(res), trains=[] if trains is None else trains.to_dict("records")), default=float)
    body.append(f"<p class=meta><a download='{html.escape(res['fsname'])}_{step_dir.name}.json' "
                f"href='data:application/json;base64,{base64.b64encode(data.encode()).decode()}'>Download the data</a> "
                f"(JSON, {len(data) // 1024} KB): this result and every train's response.</p>" + LINK_JS)
    with open(path, "w") as f:
        f.write(_page(title, body))
