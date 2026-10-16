"""Self-contained HTML pages: shared page parts and the census page (the screen page is in
report_screen.py). Figures are inline SVG, so a page stays sharp when zoomed and opens offline."""
from __future__ import annotations

import html
import io
import json
import re
import uuid

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
.meta,.note,.card p{color:var(--muted);font-size:13px}
.meta{margin:0 0 18px}
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
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px 8px;margin:12px 0}
.card p{margin:0 0 6px}
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
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;margin:0 0 8px}
.tile{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--type);border-radius:10px;
      padding:12px 14px;display:flex;flex-direction:column;gap:4px}
.tile.missing{border-style:dashed;border-left-style:dashed;color:var(--muted)}
.tile .head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.tile .type{font-weight:700}
.tile .pair{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:600}
.tile .big{font-size:26px;font-weight:700;line-height:1.1}
.tile .sub{font-size:12px;color:var(--muted)}
.pill{display:inline-flex;align-items:center;gap:4px;font-size:12px;font-weight:600;border-radius:999px;padding:1px 9px;
      background:color-mix(in srgb,var(--status) 16%,transparent);white-space:nowrap}
.pill .ico{color:var(--status);width:auto}
.stim{display:flex;flex-wrap:wrap;gap:16px 28px;align-items:flex-start;margin:14px 0 22px}
.stim h3{margin:0 0 6px}
table.heat{border-collapse:separate;border-spacing:2px;font-size:12px}
table.heat th{border:0;padding:0 4px;font-weight:500;text-align:center}
table.heat th.row{text-align:left;padding-right:10px;font-weight:600;color:var(--ink)}
table.heat td{border:0;padding:0;width:30px;height:24px;border-radius:3px}
table.heat td.type{width:auto;padding:0 0 0 10px;font-size:12px;color:var(--muted);white-space:nowrap}
.scale{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);margin:4px 0 10px}
.scale .bar{width:160px;height:10px;border-radius:3px}
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

def solve_page(path, res: dict, when: str):
    """The solve run: verdict, the network on the tissue against the true solution, the measured curve
    (with the evaluation's responses on it), and each check."""
    from .measure import ARTIFACT_LIMIT
    from .net import Oscillator
    from .solve import GATE_P
    sim = res.get("simulated")
    chips = [when] + (["SIMULATED"] if sim else []) + [
        "pairs " + ", ".join(f"e{s}→e{r}" for s, r in res["pairs"]),
        f"curve: {res['sweep_trains']} trains, {res['sweep_pulses']} pulses"]
    if "eval_trains" in res:
        chips.append(f"network: {res['eval_trains']} trains, {res['eval_pulses']} pulses")
    chips.append(f"orgchar {res.get('code_version', '?')}")
    body = [f"<h1>{html.escape(res['fsname'])}: solve</h1>",
            "<div class=chips>" + "".join(f"<span class=chip>{html.escape(c)}</span>" for c in chips) + "</div>",
            f"<div class=verdict><p class=lede>{html.escape(res['verdict'])}</p></div>"]
    if "y" in res:
        ode = Oscillator(**res["ode"])
        dense = np.linspace(0, ode.T, 1001)
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        ax.plot(dense, ode.exact(dense), color=GREY, lw=2, ls=(0, (5, 3)), label="true solution")
        ax.plot(res["t"], res["y_model"], color=MUTED, lw=1, label="network on the measured curve")
        ax.plot(res["t"], res["y"], color=BLUE, lw=2.2, marker="o", ms=4, label="network on the tissue")
        ax.text(0.98, 0.9, f"$R^2$ = {res['r2']:.3f} (plain decay: {res['decay_r2']:.2f})", transform=ax.transAxes,
                ha="right")
        ax.text(0.98, 0.72, r"$m\ddot{y} + c\dot{y} + ky = 0$", transform=ax.transAxes, ha="right")
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
        body.append("<h2>Solution</h2>" + f"<div class=fig>{_svg(fig)}</div>")
    fig, axes = plt.subplots(1, len(res["curves"]), figsize=(4.2 * len(res["curves"]), 2.8), squeeze=False)
    for ax, c in zip(axes[0], res["curves"]):
        m, sd = np.array(c["mean"]), np.array(c["sd"])
        ax.fill_between(c["counts"], m - sd, m + sd, color=SHADE, lw=0)
        ax.plot(c["counts"], m, color=INK, lw=2, marker="o", ms=4, label="curve (sweep mean ± SD)")
        ev = [e for e in res.get("eval_means", []) if (e["stim"], e["rec"]) == (c["stim"], c["rec"])]
        if ev:
            ax.plot([e["pulses"] for e in ev], [e["mean"] for e in ev], "o", color=BLUE, ms=4,
                    label="during the run")
        ax.axhline(0, color=GREY, lw=0.6)
        ax.set_title(f"e{c['stim']}→e{c['rec']}", fontsize=9)
        ax.set_xlabel("pulses in the train")
    axes[0][0].set_ylabel("evoked spikes")
    axes[0][0].legend(frameon=False, fontsize=7.5)
    body.append("<h2>Measured curve</h2>" + f"<div class=fig>{_svg(fig)}</div>")
    checks = []
    for c in res["curves"]:
        ok = c["p"] < GATE_P and c["artifact"] < ARTIFACT_LIMIT
        checks.append((CHECK_OK if ok else CHECK_BAD,
                       f"e{c['stim']}→e{c['rec']}: the pulse count explains {c['explained']:.0%} of single-trial variance "
                       f"(p {c['p']:.1g}; needs p < {GATE_P:g}); artifact {c['artifact']:.1f}x its noise floor "
                       f"(limit {ARTIFACT_LIMIT:g}x)"))
    if "expected" in res:
        e = res["expected"]
        checks.append((CHECK_OK if e["noisy_median"] > res["decay_r2"] else CHECK_BAD,
                       f"Expected R² with the noise of {res['eval_reps']} trials: {e['noisy_median']:.2f} "
                       f"(10-90 %: {e['noisy_p10']:.2f} to {e['noisy_p90']:.2f}; must beat a plain decay's {res['decay_r2']:.2f})"))
    if "r2" in res:
        checks.append((CHECK_OK if res["r2"] > res["decay_r2"] else CHECK_WARN,
                       f"Oscillation: R² {res['r2']:.2f} on the tissue; the best plain decay a × exp(-kt) + b, which has "
                       f"no oscillation, scores {res['decay_r2']:.2f} at the same times. Only an R² above it shows the "
                       "oscillation."))
    checks += [(CHECK_WARN, w[0].upper() + w[1:]) for w in res.get("warnings", [])]
    if res.get("drift_max") is not None:
        checks.append((CHECK_OK if res["drift_max"] < 0.3 else CHECK_WARN,
                       f"Largest change of a response between the sweep and the run: {res['drift_max']:.0%} of the "
                       "curve's range (sampling noise alone gives ~10 %)"))
    body.append("<h2>Checks</h2><ul class=checks>"
                + "".join(f"<li>{_icon(i)}<span>{html.escape(t)}</span></li>" for i, t in checks) + "</ul>")
    with open(path, "w") as f:
        f.write(_page(f"{res['fsname']}: solve, {when}", body))
