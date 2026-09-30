"""Self-contained HTML pages, one per step. Figures are inline SVG (text and axes stay vector; only
dense trace lines are rasterised, at 250 dpi), so the page stays sharp when zoomed and opens offline."""
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
from matplotlib.collections import LineCollection  # noqa: E402

# Colours: reference palette of the dataviz method (series blue/orange, neutral inks, fixed status set).
BLUE, ORANGE, GREY, INK, MUTED, SHADE = "#2a78d6", "#eb6834", "#8a8984", "#2b2b29", "#6b6a66", "#e9e8e2"
STATUS = {"Use": ("✓", "#0ca30c"), "Weak": ("◐", "#d69300"), "Check": ("!", "#e0663a")}
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


def _status_cell(status: str) -> str:
    if status not in STATUS:
        return ""
    return f"{_icon(STATUS[status])}{status}"


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


# ============================================================================== curve

def curve_figures(r: dict) -> list[tuple[str, str]]:
    pairs = r["pairs"]
    if not pairs:
        return []
    D = r["_trials"]
    n = len(pairs)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.9 * cols, 2.5 * rows), squeeze=False)
    for a in axs.flat[n:]:
        a.axis("off")
    for a, p in zip(axs.flat, pairs):
        L = pd.DataFrame(p["levels"])
        g = D[(D.stim == p["stim"]) & (D.rec == p["rec"])].sort_values(["segment", "time"]).reset_index(drop=True)
        g["half"] = g.index % 2
        for h in (0, 1):
            gh = g[g.half == h].groupby("level").evoked.mean()
            a.plot(gh.index, gh.values, color=GREY, lw=0.8, ls=":" if h else "--")
        a.errorbar(L.level, L["mean"], yerr=L["sem"], color=BLUE, marker="o", ms=4, lw=1.8, capsize=2)
        if p["fit"]:
            x = np.linspace(L.level.min(), L.level.max(), 100)
            f = p["fit"]
            a.plot(x, f["base"] + (f["top"] - f["base"]) / (1 + np.exp(-(x - f["x50"]) / max(f["slope"], 1e-6))),
                   color=ORANGE, lw=1)
        a.axhline(0, color=INK, lw=0.5)
        rho = p["split_half_spearman"]
        a.set_title(f"e{p['stim']} → e{p['rec']}  {'✓ usable' if p['usable'] else 'not usable'}"
                    + ("" if rho is None else f"   ρ {rho:.2f}"), fontsize=8.5)
        a.set_xlabel("pulses per train" if r.get("vary") != "amplitude" else "amplitude (µA)")
        a.set_ylabel("evoked spikes / trial" if r["unit"] == "train" else "evoked spikes / pulse")
    return [("Input → output (blue: mean ± SEM · dashed/dotted: odd/even trials · orange: fit)", _svg(fig))]


# ============================================================================== screen

def _pair_figure(d: dict, window_ms: float) -> str:
    """Three panels for one stim → record pair: every pulse, spike timing, artifact."""
    filt, valid = d["filt"], d["valid"]
    n, L = filt.shape
    t = np.arange(L) / 30.0 - d["pre_ms"]
    post = t > 0
    blank_end = np.array([t[np.argmax(v & post)] if (v & post).any() else t[-1] for v in valid])
    fig = plt.figure(figsize=(10.6, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1.3, 1.0], wspace=0.32)
    a0, a1, a2 = (fig.add_subplot(gs[i]) for i in range(3))

    # 1. every pulse, one row each; negative (spike) deflections point down
    scale = 0.45 / (6 * d["noise"])
    y = np.arange(n)[:, None] - np.clip(filt * scale, -0.95, 0.95)
    a0.barh(np.arange(n), blank_end + 0.2, left=-0.2, height=1.0, color=SHADE, zorder=0, linewidth=0)
    a0.add_collection(LineCollection([np.c_[t, y[j]] for j in range(n)], colors=GREY, linewidths=0.45,
                                     rasterized=True, zorder=1))
    for j, s in enumerate(d["spikes"]):
        for lat in s[:, 0]:
            m = (t >= lat - 0.35) & (t <= lat + 0.55)
            a0.plot(t[m], y[j, m], color=BLUE, lw=1.1, zorder=2, solid_capstyle="round")
    a0.axvline(0, color=INK, lw=0.8, zorder=3)
    a0.set_xlim(t[0], t[-1])
    a0.set_ylim(n - 0.4, -0.9)
    ticks = sorted({0, n // 2, n - 1})
    a0.set_yticks(ticks, [str(k + 1) for k in ticks])
    a0.set_ylabel("pulse")
    a0.set_xlabel("ms from pulse")
    a0.set_title("Every pulse")
    a0.text(0.4, -0.85, "pulse", fontsize=7.5, color=INK, va="bottom")

    # 2. spike timing vs the channel's spontaneous rate
    edges = np.arange(t[0], t[-1] + 1e-9, 2.0)
    lat = np.concatenate([s[:, 0] for s in d["spikes"]]) if d["spikes"] else np.zeros(0)
    rate = np.histogram(lat, edges)[0] / (n * 0.002)
    a1.axvspan(-0.2, float(np.median(blank_end)), color=SHADE, zorder=0, lw=0)
    a1.bar(edges[:-1], rate, width=2.0, align="edge", color=BLUE, edgecolor="white", linewidth=0.6, zorder=2)
    a1.axhline(d["spont_hz"], color=INK, lw=0.8, ls="--", zorder=3, label="spontaneous")
    a1.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=1.6, borderaxespad=0.1)
    a1.axvline(0, color=INK, lw=0.8)
    a1.set_xlim(edges[0], edges[-1])
    a1.set_ylim(0, max(rate.max() * 1.3 if len(rate) else 1, d["spont_hz"] * 2.5, 5))
    a1.set_xlabel("ms from pulse")
    a1.set_ylabel("spikes / s")
    a1.set_title("Spike timing")

    # 3. raw artifact: median across pulses and 5-95 % band
    m = t >= 1.0
    lim = float(np.clip(1.3 * np.abs(d["raw_med"][m]).max(), 60, 3000))
    a2.fill_between(t, d["raw_lo"], d["raw_hi"], color="#d6d5cf", lw=0, rasterized=True)
    a2.plot(t, d["raw_med"], color=INK, lw=0.9)
    be = float(np.median(blank_end))
    xmax = min(max(20.0, be + 5.0), t[-1])
    a2.axvline(be, color=INK, lw=0.7, ls=":")
    a2.text(be + 0.4, lim * 0.82, "blank ends", fontsize=7.5, color=MUTED, clip_on=True)
    a2.set_xlim(-1, xmax)
    a2.set_ylim(-lim, lim)
    a2.set_xlabel("ms from pulse")
    a2.set_ylabel("µV")
    a2.set_title("Artifact (raw)")
    return _svg(fig)


_DIV = [(0.0, "#cde2fb"), (0.5, "#9ec5f4"), (1.0, "#5598e7"), (2.0, "#1c5cab"), (4.0, "#0d366b")]


def _div_colour(v: float) -> str:
    v = float(np.clip(v, _DIV[0][0], _DIV[-1][0]))
    for (x0, c0), (x1, c1) in zip(_DIV, _DIV[1:]):
        if v <= x1:
            f = (v - x0) / (x1 - x0)
            a = np.array([int(c0[i:i + 2], 16) for i in (1, 3, 5)])
            b = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)])
            return "#%02x%02x%02x" % tuple(np.round(a + f * (b - a)).astype(int))
    return _DIV[-1][1]


def _dark(hexcol: str) -> bool:
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45


def _response_maps(R: pd.DataFrame, stims: set) -> str:
    """One 8×8 grid per organoid, drawn as plain SVG: crisp, and each cell has a hover tooltip."""
    cand = {(int(x.stim), int(x.rec)): x for x in R[R.role == "candidate"].itertuples()}
    cs, left, top = 26, 32, 22
    figs = []
    for b in range(4):
        el = list(range(b * 8, b * 8 + 8))
        w, h = left + 8 * cs + 2, top + 8 * cs + 2
        s = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' "
             f"aria-label='Response map organoid {b}' style='max-width:100%;height:auto;font:10px system-ui,sans-serif'>"]
        for j, c in enumerate(el):
            s.append(f"<text x='{left + j * cs + cs / 2}' y='{top - 6}' text-anchor='middle' fill='{MUTED}'>e{c}</text>")
        for i, e in enumerate(el):
            s.append(f"<text x='{left - 4}' y='{top + i * cs + cs / 2 + 4}' text-anchor='end' fill='{MUTED}'>e{e}</text>")
            for j, c in enumerate(el):
                x0, y0 = left + j * cs, top + i * cs
                if e == c:
                    s.append(f"<rect x='{x0}' y='{y0}' width='{cs - 2}' height='{cs - 2}' fill='#dcdbd5'>"
                             f"<title>e{e} is the stimulated electrode; never read</title></rect>")
                    continue
                x = cand.get((e, c))
                if e not in stims or x is None:
                    s.append(f"<rect x='{x0}' y='{y0}' width='{cs - 2}' height='{cs - 2}' fill='#f7f6f3'>"
                             f"<title>e{e} not stimulated in this screen</title></rect>")
                    continue
                col = _div_colour(x.delta_per_pulse) if x.responsive else "#f1f0ec"
                st = getattr(x, "status", "") or ""
                tip = (f"e{e} → e{c}: {x.delta_per_pulse:+.2f} extra spikes/pulse ({int(x.spikes)} observed, "
                       f"{x.expected:.1f} expected)" + (f", latency {x.median_latency_ms:.0f} ms" if pd.notna(x.median_latency_ms) else "")
                       + (f". {st}" if st else ". Not responsive"))
                s.append(f"<rect x='{x0}' y='{y0}' width='{cs - 2}' height='{cs - 2}' fill='{col}'>"
                         f"<title>{html.escape(tip)}</title></rect>")
                if x.responsive:
                    txt = f"{x.delta_per_pulse:.1f}" + ("!" if st == "Check" else "")
                    fw = "700" if st == "Use" else "400"
                    s.append(f"<text x='{x0 + (cs - 2) / 2}' y='{y0 + cs / 2 + 3}' text-anchor='middle' font-weight='{fw}' "
                             f"fill='{'#fff' if _dark(col) else INK}' pointer-events='none'>{txt}</text>")
        s.append("</svg>")
        figs.append(f"<figure><figcaption>Organoid {b} · e{b * 8}–e{b * 8 + 7}</figcaption>{''.join(s)}</figure>")
    sw = "".join(f"<span style='display:inline-block;width:26px;height:12px;background:{_div_colour(v)};"
                 f"vertical-align:middle'></span><span style='margin:0 10px 0 4px'>{'≥' if v == 4 else ''}{v:g}</span>"
                 for v in (0.5, 1, 2, 4))
    legend = (f"<p class=note>Rows: stimulated electrode · columns: recorded electrode. Coloured cells respond; "
              f"colour and number = extra spikes per pulse {sw}<br>"
              "<b>bold</b> = Use · regular = Weak · ! = Check traces · dark grey diagonal = stimulated electrode (not read). "
              "Hover any cell for its numbers.</p>")
    return f"<div class=maps>{''.join(figs)}</div>{legend}"


def _controls_figure(R: pd.DataFrame) -> str:
    rows = [("Same organoid", R.role == "candidate"), ("Other organoid", R.role == "cross_organoid_control"),
            ("Sham (no current)", R.role == "sham_control")]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10.6, 2.1))
    for yi, (label, m) in enumerate(rows):
        sub = R[m]
        jit = rng.uniform(-0.22, 0.22, len(sub))
        base = ~sub.responsive if yi == 0 else np.ones(len(sub), bool)
        ax.scatter(sub.delta_per_pulse[base], (yi + jit)[base], s=9, color="#b9b8b2", lw=0, zorder=2)
        if yi == 0:
            for st, (sym, col) in STATUS.items():
                mm = (sub.status == st).values
                ax.scatter(sub.delta_per_pulse[mm], (yi + jit)[mm], s=22, color=col, edgecolor="white", lw=0.8,
                           zorder=3, label=f"{sym} {st}")
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(range(3), [r[0] for r in rows])
    ax.set_ylim(2.6, -0.6)
    ax.set_xlabel("extra spikes per pulse (observed − expected from spontaneous firing)")
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return _svg(fig)


def screen_page(path, r: dict, R: pd.DataFrame, ctx: dict):
    """The screen report, ordered by the decisions it supports: is the run valid, which pairs to
    take to the curve step, and the evidence for each."""
    tr = r.get("_traces", {})
    c = r["controls"]
    cross, sham = c["cross_organoid_control"], c["sham_control"]
    resp = R[R.responsive]
    n_use = int((R.status == "Use").sum())
    leak = r["verdict"] == "PIPELINE_LEAK"
    body = [f"<h1>{html.escape(ctx['fsname'])} · screen</h1>",
            f"<p class=meta>{html.escape(ctx['when'])} · {html.escape(ctx['stimulus'])} · "
            f"orgchar {html.escape(ctx['code'])}{' · SIMULATED' if ctx.get('simulated') else ''}</p>"]
    if ctx["problems"]:
        lede = "Incomplete run: use for inspection only, re-run the screen."
    elif leak:
        lede = "Artifact reaches the readout: do not use these results."
    elif len(resp):
        lede = f"{n_use} pair{'s' if n_use != 1 else ''} ready for the curve step ({len(resp)} responsive in total)."
    else:
        lede = "No pair responds to this stimulus."
    body.append(f"<p class=lede>{html.escape(lede)}</p><ul class=checks>")
    if ctx["problems"]:
        body.append(f"<li>{_icon(CHECK_BAD)}{html.escape('; '.join(ctx['problems']))}</li>")
    else:
        body.append(f"<li>{_icon(CHECK_OK)}All {ctx['n_segments']} recordings complete</li>")
    ok = cross["significant_bonferroni"] == 0 and sham["significant_bonferroni"] == 0
    body.append(f"<li>{_icon(CHECK_OK if ok else CHECK_BAD)}Controls {'clean' if ok else 'NOT clean'}: "
                f"{cross['significant_bonferroni']} of {cross['tests']} other-organoid pairs and "
                f"{sham['significant_bonferroni']} of {sham['tests']} sham pairs respond (must be 0)</li>")
    if ctx["gaps"]:
        body.append(f"<li>{_icon(CHECK_WARN)}{ctx['gaps']} data gap(s) in the platform's recording; "
                    f"{ctx['excluded']} pulse(s) left out</li>")
    else:
        body.append(f"<li>{_icon(CHECK_OK)}No data gaps</li>")
    ab = r["artifact_blank_ms"]
    if ab.get("same_organoid_median") is not None:
        body.append(f"<li>{_icon(CHECK_OK)}Artifact removed after {ab['same_organoid_median']:.1f} ms "
                    f"(90 % of pairs within {ab['same_organoid_p90']:.1f} ms)</li>")
    body.append("</ul>")

    # ---- next step
    rec = ctx["recommended"]
    body.append("<h2>Next step</h2>")
    if rec and not leak and not ctx["problems"]:
        stims = list(dict.fromkeys(e for e, _ in rec))
        body.append(f"<p>Build input–output curves for {len(rec)} pairs, one stimulating electrode per organoid "
                    f"({', '.join(f'e{e}' for e in stims)}), strongest reproducible pairs first. "
                    "<code>curve</code> uses exactly these pairs by default:</p>")
        body.append(f"<pre class=cmd>./orgchar.sh curve --fsname {html.escape(ctx['fsname'])}</pre>")
        body.append(f"<p class=note>Same as <code>--pairs {','.join(f'{e}:{c}' for e, c in rec)}</code>. "
                    "Change the list to include or drop pairs.</p>")
    elif not len(resp):
        body.append("<p>Nothing to build curves on. If the controls are clean, try one stronger screen "
                    "(<code>--phase 200</code>) on the organoid with the most spontaneous activity; if that also "
                    "finds nothing, stop stimulating this preparation.</p>")
    else:
        body.append("<p>Do not continue with this screen; see the checks above.</p>")

    # ---- tables
    def table(rows_df, with_status=True, with_why=True):
        out = []
        for x in rows_df.itertuples():
            ev = x.earlier_responsive
            if ev is None or (isinstance(ev, float) and np.isnan(ev)):
                early = "—"
            else:
                early = f"{_icon(CHECK_OK) if ev else _icon(CHECK_BAD)}{x.earlier_delta:+.2f}"
            out.append("<tr>" + (f"<td>{_status_cell(x.status)}</td>" if with_status else "")
                       + f"<td><b>e{int(x.stim)} → e{int(x.rec)}</b></td><td class=n>{int(x.stim_block)}</td>"
                       f"<td class=n>{x.delta_per_pulse:+.2f}</td><td class=n>{int(x.spikes)} / {x.expected:.1f}</td>"
                       f"<td class=n>{x.median_latency_ms:.0f} ms</td><td class=n>{early}</td>"
                       + (f"<td>{html.escape(x.reason)}</td>" if with_why else "") + "</tr>")
        head = ("<tr>" + ("<th>Status</th>" if with_status else "") + "<th>Pair</th><th class=n>Organoid</th>"
                "<th class=n>Extra spikes / pulse</th><th class=n>Spikes: seen / expected</th><th class=n>Latency</th>"
                "<th class=n>Earlier screen</th>"
                + ("<th>Why</th>" if with_why else "") + "</tr>")
        return f"<div class=wrap><table>{head}{''.join(out)}</table></div>"

    if rec and not leak and not ctx["problems"]:
        byk = R[R.kind == "stim"].set_index(["stim", "rec"])
        recdf = byk.loc[rec].reset_index()
        body.append(table(recdf, with_status=False, with_why=False))
    checks = resp[resp.status == "Check"].sort_values("z", ascending=False)
    if len(checks):
        body.append("<h2>Check before using</h2><p class=note>Responsive, but the timing looks like leftover "
                    "artifact or the pair did not hold up before. Their traces are under Evidence.</p>")
        body.append(table(checks, with_status=False))
    if len(resp):
        order = {"Use": 0, "Weak": 1, "Check": 2}
        T = resp.assign(_o=resp.status.map(order)).sort_values(["stim_block", "_o", "delta_per_pulse"],
                                                               ascending=[True, True, False])
        n = {k: int((resp.status == k).sum()) for k in order}
        body.append(f"<details><summary>All {len(resp)} responsive pairs, by organoid "
                    f"({n['Use']} Use · {n['Weak']} Weak · {n['Check']} Check)</summary>{table(T)}</details>")

    # ---- maps
    stims = {int(s) for s in R[(R.kind == "stim")].stim.unique()}
    body.append("<h2>Response maps</h2>" + _response_maps(R, stims))

    # ---- evidence
    groups = [("Recommended pairs", [p for p in rec if p in tr]),
              ("Pairs to check", [(int(x.stim), int(x.rec)) for x in checks.itertuples() if (int(x.stim), int(x.rec)) in tr])]
    if any(g for _, g in groups):
        body.append("<h2>Evidence</h2>")
        body.append("<p class=note>Left: one row per pulse, 30 ms before to 30 ms after it; blue = detected spikes, "
                    "shaded = blanked artifact. A real response has more spikes after the pulse than before it, "
                    "scattered over several ms. Spikes lined up at one time, or right where the shading ends, suggest "
                    "leftover artifact.</p>")
    byk = {(int(x.stim), int(x.rec)): x for x in R[R.kind == "stim"].itertuples()}
    for gname, ev in groups:
        if ev:
            body.append(f"<h3 style='margin-top:22px'>{gname}</h3>")
        for e, cc in ev:
            x = byk[(e, cc)]
            head = f"e{e} → e{cc} · organoid {int(x.stim_block)}"
            sub = (f"{x.delta_per_pulse:+.2f} extra spikes/pulse ({int(x.spikes)} seen, {x.expected:.1f} expected) · "
                   f"latency {x.median_latency_ms:.0f} ms" + (f" · {x.reason}" if x.reason else ""))
            body.append(f"<div class=card><h3>{_status_cell(x.status)} · {html.escape(head)}</h3>"
                        f"<p>{html.escape(sub)}</p><div class=fig>{_pair_figure(tr[(e, cc)], r['window_ms'])}</div></div>")

    # ---- controls
    body.append("<h2>Controls</h2>")
    body.append("<p class=note>Pairs across organoids (no synapse, but they receive wiring cross-talk) and sham "
                "markers (no current) are analysed like real pairs. They should sit at zero; real responses stand "
                "out to the right.</p>")
    body.append(f"<div class=fig>{_controls_figure(R)}</div>")
    mirrors = [(e, 31 - e) for e in dict.fromkeys(e for e, _ in rec) if (e, 31 - e) in tr]
    if mirrors:
        body.append("<details><summary>Cross-talk controls for the recommended electrodes</summary>")
        byk = {(int(x.stim), int(x.rec)): x for x in R[R.kind == "stim"].itertuples()}
        for e, m in mirrors:
            x = byk.get((e, m))
            sub = "" if x is None else f"{x.delta_per_pulse:+.2f} extra spikes/pulse ({int(x.spikes)} seen, {x.expected:.1f} expected)"
            body.append(f"<div class=card><h3>Control · e{e} → e{m} (other organoid)</h3><p>{html.escape(sub)}</p>"
                        f"<div class=fig>{_pair_figure(tr[(e, m)], r['window_ms'])}</div></div>")
        body.append("</details>")

    # ---- full table
    cols = ["stim", "rec", "stim_block", "n_pulses", "spikes", "expected", "delta_per_pulse", "z", "p_bonf",
            "waveform_corr", "median_latency_ms", "peak_1ms_fraction", "edge_spikes", "mean_blank_ms", "status", "reason"]
    full = R[R.role == "candidate"][[c_ for c_ in cols if c_ in R]].sort_values("z", ascending=False)
    body.append("<details><summary>All same-organoid pairs (numbers)</summary><div class=wrap>"
                + full.to_html(index=False, float_format=lambda v: f"{v:.3g}", na_rep="") + "</div></details>")
    with open(path, "w") as f:
        f.write(_page(f"{ctx['fsname']} screen", body))


# ============================================================================== census / curve pages

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
