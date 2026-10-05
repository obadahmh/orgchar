"""Self-contained HTML pages: shared page parts and the census page (the screen and curve pages are
in report_funcsolver.py). Figures are inline SVG, so a page stays sharp when zoomed and opens offline."""
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
details{margin:10px 0}
summary{cursor:pointer;color:var(--muted)}
"""


def _page(title: str, body: list[str]) -> str:
    return (f"<title>{html.escape(title)}</title>\n<meta charset=utf-8>\n<style>{CSS}</style>\n<main>\n"
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
            a.set_title(f"e{row.channel} · {row.rate_hz:.1f} Hz", fontsize=8, fontweight="normal")
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
