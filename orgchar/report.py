"""Figures and a self-contained HTML page per step (figures embedded, opens offline)."""
from __future__ import annotations

import base64
import html
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#2b2b29"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _blocks(ax, horizontal=True, vertical=True):
    for b in range(1, 4):
        if horizontal:
            ax.axhline(b * 8 - 0.5, color=GREY, lw=0.6)
        if vertical:
            ax.axvline(b * 8 - 0.5, color=GREY, lw=0.6)


# ------------------------------------------------------------------------------ census

def census_figures(r: dict) -> list[tuple[str, str]]:
    ch = pd.DataFrame(r["channels"])
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.bar(ch.channel, ch.rate_hz, color=[BLUE if a else GREY for a in ch.active], width=0.8)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.axhline(0.1, color=INK, lw=0.6, ls=":")
    ax.set_xticks(range(32))
    ax.set_xlabel("electrode (blocks of 8 = one organoid each)")
    ax.set_ylabel("spikes / s")
    _blocks(ax, horizontal=False)
    ax.set_title("Spontaneous firing rate (blue = active channel; dotted = 0.1 Hz)", loc="left")
    figs = [("Firing rate per electrode", _png(fig))]
    act = ch[ch.active]
    if len(act):
        n = len(act)
        fig, axs = plt.subplots(1, n, figsize=(1.6 * n + 0.5, 1.9), squeeze=False, sharey=True)
        t = (np.arange(75) - 30) / 30.0
        for a, (_, row) in zip(axs[0], act.iterrows()):
            a.plot(t, row.waveform, color=BLUE, lw=1.5)
            a.set_title(f"e{row.channel}  {row.rate_hz:.2f} Hz", fontsize=8)
            a.set_xlabel("ms")
        axs[0, 0].set_ylabel("µV")
        figs.append(("Mean spike waveform of active electrodes", _png(fig)))
    return figs


# ------------------------------------------------------------------------------ screen

def screen_figures(r: dict) -> list[tuple[str, str]]:
    R = pd.DataFrame(r["pairs"])
    st = R[R.kind == "stim"]
    M = np.full((32, 32), np.nan)
    for _, x in st.iterrows():
        M[int(x.stim), int(x.rec)] = x.delta_per_pulse
    stims = sorted(st.stim.unique())
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(stims) + 1.6))
    vmax = max(0.3, np.nanmax(np.abs(M)) if np.isfinite(M).any() else 0.3)
    im = ax.imshow(M[stims], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    for i, e in enumerate(stims):
        for _, x in st[(st.stim == e) & st.responsive].iterrows():
            ax.text(x.rec, i, "●", ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks(range(len(stims)))
    ax.set_yticklabels([f"e{e}" for e in stims])
    ax.set_xticks(range(32))
    ax.set_xticklabels(range(32), fontsize=6)
    ax.set_xlabel("recording electrode")
    ax.set_ylabel("stimulated electrode")
    _blocks(ax, horizontal=False)
    fig.colorbar(im, ax=ax, label="evoked spikes / pulse (minus spontaneous)", shrink=0.8)
    ax.set_title("Response map  (● = responsive after correction; off-block cells are negative controls)", loc="left")
    figs = [("Response map", _png(fig))]
    resp = R[R.responsive].head(8)
    if len(resp):
        n = len(resp)
        fig, axs = plt.subplots(1, n, figsize=(2.0 * n + 0.5, 2.0), squeeze=False, sharey=True)
        for a, (_, x) in zip(axs[0], resp.iterrows()):
            lat = r["_latencies"].get(("stim", int(x.stim), int(x.rec)), [])
            a.hist(lat, bins=np.arange(0, r["window_ms"] + 1, 1.0), color=BLUE)
            a.set_title(f"e{int(x.stim)}→e{int(x.rec)}\n{x.delta_per_pulse:.2f}/pulse", fontsize=8)
            a.set_xlabel("ms after pulse")
        axs[0, 0].set_ylabel("spikes")
        figs.append(("Latency of evoked spikes (responsive pairs)", _png(fig)))
    tf = trace_figure(r)
    if tf:
        figs.append(tf)
    B = r["_blanks"]
    if len(B):
        Bt = B[B.template].groupby(["stim", "rec"]).blank_ms.median().unstack()
        if len(Bt):
            fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(Bt) + 1.6))
            im = ax.imshow(Bt.values, cmap="Blues", vmin=0, vmax=15, aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(Bt)))
            ax.set_yticklabels([f"e{e}" for e in Bt.index])
            ax.set_xticks(range(len(Bt.columns)))
            ax.set_xticklabels(Bt.columns, fontsize=6)
            ax.set_xlabel("recording electrode")
            _blocks(ax, horizontal=False)
            fig.colorbar(im, ax=ax, label="ms blanked after each pulse", shrink=0.8)
            ax.set_title("Artifact duration after template subtraction (stimulated electrode never read)", loc="left")
            figs.append(("Artifact map", _png(fig)))
    return figs


def _trace_rows(r: dict, max_pairs: int = 6):
    """Responsive pairs (or, if none, the best-scoring candidates), then one cross-organoid
    'mirror' control per stimulating electrode shown."""
    R = pd.DataFrame(r["pairs"])
    tr = r.get("_traces", {})
    cand = R[(R.role == "candidate") & np.array([(int(s), int(c)) in tr for s, c in zip(R.stim, R.rec)])]
    shown = cand[cand.responsive].head(max_pairs)
    if shown.empty:
        shown = cand.sort_values("z", ascending=False).head(3)
    rows = [(int(x.stim), int(x.rec),
             f"e{int(x.stim)}→e{int(x.rec)}  " + ("RESPONSIVE" if x.responsive else "not responsive")
             + f"  {x.delta_per_pulse:+.2f} spikes/pulse" + ("  ⚠ TIME-LOCKED" if x.get("time_locked") else "") + ("  ⚠ BLANK-EDGE" if x.get("blank_edge") else ""))
            for _, x in shown.iterrows()]
    for e in dict.fromkeys(int(s) for s in shown.stim):
        m = 31 - e
        if (e, m) in tr:
            x = R[(R.kind == "stim") & (R.stim == e) & (R.rec == m)]
            d = f"  {x.delta_per_pulse.iloc[0]:+.2f} spikes/pulse" if len(x) else ""
            rows.append((e, m, f"CONTROL e{e}→e{m} (other organoid, wiring cross-talk){d}"))
    return rows


def trace_figure(r: dict):
    """What the pipeline did to every pulse, stage by stage."""
    rows = _trace_rows(r)
    if not rows:
        return None
    tr = r["_traces"]
    fig, axs = plt.subplots(len(rows), 4, figsize=(15, 2.3 * len(rows) + 0.6), squeeze=False)
    for i, (e, c, label) in enumerate(rows):
        d = tr[(e, c)]
        n, L = d["raw"].shape
        t = np.arange(L) / 30.0 - d["pre_ms"]
        # each pulse's own blank = first valid sample after the pulse; shade the median in panels 1-3
        per = np.array([t[np.argmax(v & (t > 0))] if (v & (t > 0)).any() else t[-1] for v in d["valid"]])
        blank_end = float(np.median(per))
        tmpl = np.median(d["raw"], axis=0)
        a = axs[i, 0]
        a.plot(t, d["raw"].T, color=GREY, lw=0.4, alpha=0.35)
        a.plot(t, tmpl, color=ORANGE, lw=1.4)
        lim = min(2500.0, max(150.0, 1.3 * np.percentile(np.abs(tmpl[t >= 1.0]), 99)))
        a.set_ylim(-lim, lim)
        a.set_ylabel("µV", fontsize=8)
        a.text(0.02, 0.96, label, transform=a.transAxes, fontsize=8, va="top", fontweight="bold",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        noise = abs(d["thr"]) / 5.0
        for a, y, ylim in [(axs[i, 1], d["clean"], 10 * noise), (axs[i, 2], d["filt"], None)]:
            a.plot(t, y.T, color=GREY, lw=0.4, alpha=0.35)
            if ylim:
                a.set_ylim(-ylim, ylim)
        a = axs[i, 2]
        amp = np.concatenate([s[:, 1] for s in d["spikes"]]) if d["spikes"] else np.zeros(0)
        lim = max(8 * noise, 1.3 * np.abs(amp).max() if len(amp) else 0)
        a.set_ylim(-lim, lim / 2)
        a.axhline(d["thr"], color=INK, lw=0.8, ls="--")
        for s in d["spikes"]:
            if len(s):
                a.plot(s[:, 0], s[:, 1], "o", color=BLUE, ms=3.5)
        a = axs[i, 3]
        for j, s in enumerate(d["spikes"]):
            if len(s):
                a.plot(s[:, 0], np.full(len(s), j), "|", color=BLUE, ms=6, mew=1.5)
        a.barh(np.arange(n), per + 0.2, left=-0.2, height=0.8, color="#e8e7e1", zorder=0)
        a.set_ylim(-1, n)
        a.set_ylabel("pulse #", fontsize=8)
        for a in axs[i, :3]:
            a.axvspan(-0.2, blank_end, color="#e8e7e1", zorder=0)
        for a in axs[i]:
            a.set_xlim(t[0], t[-1])
    titles = ["1. raw, every pulse (orange = median = artifact template)",
              "2. template subtracted, artifact blanked (shaded = median blank)",
              "3. 300–3000 Hz, threshold (dashed), detected spikes (blue)",
              "4. detected spikes per pulse (grey = that pulse's blank)"]
    for a, tt in zip(axs[0], titles):
        a.set_title(tt, fontsize=8.5, loc="left")
    for a in axs[-1]:
        a.set_xlabel("ms after pulse")
    fig.tight_layout()
    return ("Raw traces: what was counted for each pair (grey = individual pulses; shaded = blanked)", _png(fig))


# ------------------------------------------------------------------------------ curve

def curve_figures(r: dict) -> list[tuple[str, str]]:
    pairs = r["pairs"]
    if not pairs:
        return []
    D = r["_trials"]
    n = len(pairs)
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n + 0.5, 2.8), squeeze=False)
    for a, p in zip(axs[0], pairs):
        L = pd.DataFrame(p["levels"])
        g = D[(D.stim == p["stim"]) & (D.rec == p["rec"])].sort_values(["segment", "time"]).reset_index(drop=True)
        g["half"] = g.index % 2
        for h, col in [(0, GREY), (1, GREY)]:
            gh = g[g.half == h].groupby("level").evoked.mean()
            a.plot(gh.index, gh.values, color=col, lw=0.8, ls=":" if h else "--")
        a.errorbar(L.level, L["mean"], yerr=L["sem"], color=BLUE, marker="o", ms=4, lw=1.8, capsize=2)
        if p["fit"]:
            x = np.linspace(L.level.min(), L.level.max(), 100)
            f = p["fit"]
            a.plot(x, f["base"] + (f["top"] - f["base"]) / (1 + np.exp(-(x - f["x50"]) / max(f["slope"], 1e-6))),
                   color=ORANGE, lw=1)
        a.axhline(0, color=INK, lw=0.5)
        a.set_title(f"e{p['stim']}→e{p['rec']}  {'USABLE' if p['usable'] else 'not usable'}\n"
                    f"split-half ρ={p['split_half_spearman'] if p['split_half_spearman'] is None else round(p['split_half_spearman'], 2)}",
                    fontsize=8)
        a.set_xlabel("pulses per train" if r["unit"] == "train" and r.get("vary") != "amplitude" else "input level")
        a.set_ylabel("evoked spikes / trial" if r["unit"] == "train" else "evoked spikes / pulse")
    return [("Transfer curves (blue = mean ± SEM, dashed/dotted = odd/even trials, orange = sigmoid fit)", _png(fig))]


# ------------------------------------------------------------------------------ page

def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def to_json(r: dict) -> str:
    return json.dumps(_clean(r), indent=1)


def write_html(path, title: str, verdict: str, summary_lines: list[str], figs, table: pd.DataFrame | None = None):
    parts = [f"<title>{html.escape(title)}</title>",
             "<style>body{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;"
             "color:#2b2b29;background:#fcfcfb}h1{font-size:20px}h2{font-size:15px;margin-top:28px}"
             ".v{display:inline-block;padding:3px 10px;border-radius:4px;background:#eef3fb;font-weight:600}"
             "table{border-collapse:collapse;font-size:12px}td,th{padding:3px 8px;border-bottom:1px solid #e3e2dc;"
             "text-align:right}img{max-width:100%}.wrap{overflow-x:auto}</style>",
             f"<h1>{html.escape(title)}</h1><p class=v>{html.escape(verdict)}</p><ul>"]
    parts += [f"<li>{html.escape(s)}</li>" for s in summary_lines]
    parts.append("</ul>")
    for t, b64 in figs:
        parts.append(f"<h2>{html.escape(t)}</h2><img src='data:image/png;base64,{b64}'>")
    if table is not None and len(table):
        parts.append("<h2>Table</h2><div class=wrap>" + table.to_html(index=False, float_format=lambda x: f"{x:.3g}") + "</div>")
    with open(path, "w") as f:
        f.write("\n".join(parts))
