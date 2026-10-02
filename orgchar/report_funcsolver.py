"""Report for the FuncSolver curve step: decision first (which pair for each of FuncSolver's three
functions, and how much to trust it), evidence after. Figures are SVG: sharp at any zoom."""
from __future__ import annotations

import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import funcsolver as F
from .report import BLUE, CHECK_BAD, CHECK_OK, CHECK_WARN, GREY, INK, MUTED, ORANGE, _icon, _page, _svg

GRADE = {"good": ("✓", "#0ca30c", "Good"), "fair": ("◐", "#d69300", "Fair"),
         "poor": ("✗", "#d03b3b", "Poor"), "unknown": ("?", MUTED, "Unknown"), "none": ("–", MUTED, "No curve")}
DEV, CONF = "#a3a29c", BLUE
ROLE_TITLE = dict(short="Short · falling curve", band="Band · peaked curve", long="Long · rising curve")
ROLE_NEED = dict(short="more pulses → fewer events", band="most events at a middle pulse count",
                 long="more pulses → more events")


def _grade(g):
    sym, col, label = GRADE.get(g, GRADE["unknown"])
    return f"{_icon((sym, col))}<b>{label}</b>"


def _fmt(v, f=".2f", na="–"):
    return na if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, f)


def _jitter(n, width, seed):
    return np.random.default_rng(seed).uniform(-width, width, n)


def _role_figure(p, role, D, levels, ref_n):
    f = p["fits"][role]
    stim = D[(D.kind == "stim") & (D.stim == p["stim"]) & (D.rec == p["rec"])].copy()
    stim["y"] = stim.adjusted - p["control"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 2.9), gridspec_kw=dict(width_ratios=[4.2, 1.5], wspace=0.28))
    step = np.median(np.diff(levels)) if len(levels) > 1 else 1.0
    for part, col, dx, lab in (("development", DEV, -0.22, "development trials (fit)"),
                               ("confirmation", CONF, 0.22, "confirmation trials (new)")):
        g = stim[stim.partition == part]
        a.scatter(g.level + dx * step + _jitter(len(g), 0.08 * step, 1), g.y, s=12, color=col, alpha=0.75,
                  linewidths=0, label=lab, zorder=2)
        if part == "confirmation" and len(g):
            m = g.groupby("level").y.mean()
            a.plot(m.index + dx * step, m.values, "o", ms=4.5, mfc="white", mec=CONF, mew=1.4, zorder=4,
                   label="confirmation mean")
    xs = np.linspace(min(levels), max(levels), 200)
    a.plot(xs, F.model(role, xs, f["params"]), color=INK, lw=2, zorder=3, label="curve FuncSolver will use")
    a.axhline(0, color=GREY, lw=0.6, zorder=1)
    a.set_xticks(levels)
    a.set_xlabel("pulses in the train")
    a.set_ylabel("events after the train\n(minus baseline and sham)")
    a.legend(loc="upper left", bbox_to_anchor=(0, -0.26), ncol=4, frameon=False, fontsize=7.5, markerscale=1.6,
             handletextpad=0.3, columnspacing=1.4)
    ref = stim[stim.partition == "reference"]
    if len(ref):
        blocks = sorted(ref.block.unique())
        for i, bk in enumerate(blocks):
            g = ref[ref.block == bk]
            b.scatter(i + _jitter(len(g), 0.12, 2), g.y, s=12, color=ORANGE, alpha=0.75, linewidths=0, zorder=2)
            b.plot([i - 0.25, i + 0.25], [g.y.mean()] * 2, color=ORANGE, lw=2, zorder=3)
        b.axhline(f.get("reference_expected", np.nan), color=INK, lw=1.2, ls="--", zorder=1)
        b.set_xticks(range(len(blocks)))
        b.set_xticklabels(["start", "mid", "end"][:len(blocks)])
        b.set_xlim(-0.6, len(blocks) - 0.4)
        b.set_title(f"{ref_n}-pulse reference\n(dashed: curve)", fontsize=8.5)
    else:
        b.axis("off")
    return _svg(fig)


def _all_pairs_figure(pairs, D, chosen):
    stims = sorted({p["stim"] for p in pairs})
    recs = {s: sorted(p["rec"] for p in pairs if p["stim"] == s) for s in stims}
    ncol = max(len(v) for v in recs.values())
    fig, axes = plt.subplots(len(stims), ncol, figsize=(1.45 * ncol, 1.35 * len(stims)), squeeze=False,
                             gridspec_kw=dict(hspace=0.75, wspace=0.45))
    by = {(p["stim"], p["rec"]): p for p in pairs}
    picked = {(p["stim"], p["rec"]): r for r, p in chosen.items()}
    for i, s in enumerate(stims):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(recs[s]):
                ax.axis("off")
                continue
            c = recs[s][j]
            p = by[(s, c)]
            lv = np.array(p["levels"])
            dm, sd = np.array(p["dev_means"]), np.array(p["dev_sd"])
            ax.fill_between(lv, dm - sd, dm + sd, color=DEV, alpha=0.25, linewidth=0)
            ax.plot(lv, dm, color=DEV, lw=1.4)
            if p.get("shape") and p["fits"][p["shape"]].get("conf_means"):
                f = p["fits"][p["shape"]]
                ax.plot(f.get("conf_levels", lv), f["conf_means"], color=CONF, lw=1.4, marker="o" if len(f["conf_means"]) < len(lv) else None, ms=2.5)
            ax.axhline(0, color=GREY, lw=0.5)
            ax.set_xticks([lv[0], lv[-1]])
            ax.tick_params(labelsize=6.5, length=2)
            role = picked.get((s, c))
            label = f"e{s}→e{c}"
            sub = (f"★ {role}" if role else (F.ROLE_TEXT[p["shape"]] if p.get("shape") else "flat/none"))
            ax.set_title(f"{label}\n{sub}", fontsize=7.5, fontweight="bold" if role else "normal",
                         color=INK if (role or p.get("shape")) else MUTED, loc="left", pad=3)
            if role:
                for sp in ("left", "bottom"):
                    ax.spines[sp].set_color(INK)
                    ax.spines[sp].set_linewidth(1.2)
    return _svg(fig)


def page(path, plan, r, problems, when):
    D = r["_D"]
    chosen = r["_chosen"]
    pairs = r["pairs"]
    levels = list(plan["levels"])
    fs = plan["fsname"]
    sim = plan.get("backend") == "simulated"
    title = f"{fs} · FuncSolver curves · {when}" + (" · SIMULATED" if sim else "")
    body = [f"<h1>{html.escape(fs)} · FuncSolver curves</h1>",
            f"<p class=meta>{html.escape(when)}{' · SIMULATED' if sim else ''} · stimulating "
            f"{', '.join('e' + str(s) for s in plan['stims'])} · trains of {levels[0]}–{levels[-1]} pulses, "
            f"{plan['ipi_ms']:g} ms apart, {plan['amplitude_uA']} µA × {plan['phase_us']:g} µs · "
            f"{r['n_trials']} trains</p>"]

    # ---- the decision
    n = len(chosen)
    lede = {3: "All three FuncSolver functions found.", 0: "No pair gives a usable FuncSolver function."}.get(
        n, f"{n} of FuncSolver's 3 functions found: " + ", ".join(F.ROLE_TEXT[x] for x in F.ROLES if x not in chosen)
        + " missing.")
    body.append(f"<p class=lede>{html.escape(lede)}</p>")
    rows = []
    for role in F.ROLES:
        v = r["roles"].get(role)
        if v:
            rows.append(f"<tr><td><b>{role}</b> <span class=note>({F.ROLE_TEXT[role]})</span></td>"
                        f"<td>e{v['stim']} → e{v['rec']}</td><td>{_grade(v['grade'])}</td>"
                        f"<td class=n>{_fmt(v.get('conf_r2'))}</td><td class=n>±{_fmt(v.get('conf_rmse'), '.1f')}</td>"
                        f"<td class=n>{_fmt(np.ptp(F.model(role, levels, v['params'])), '.1f')}</td>"
                        f"<td>{html.escape(v['reason'])}</td></tr>")
        else:
            rows.append(f"<tr><td><b>{role}</b> <span class=note>({F.ROLE_TEXT[role]})</span></td><td>–</td>"
                        f"<td>{_icon(CHECK_BAD)}<b>Missing</b></td><td></td><td></td><td></td>"
                        f"<td>no pair gives this curve ({ROLE_NEED[role]})</td></tr>")
    body.append("<div class=wrap><table><tr><th>FuncSolver function</th><th>Pair</th><th>Verdict</th>"
                "<th class=n title='Share of the trial-to-trial variation in NEW trials that the curve explains'>"
                "R² on new trials</th><th class=n title='Typical distance of a new single trial from the curve'>"
                "Error per trial</th><th class=n title='Difference between the curve maximum and minimum'>"
                "Curve range</th><th>Why</th></tr>" + "".join(rows) + "</table></div>")
    body.append("<p class=note><b>How to read this.</b> FuncSolver sends one train per evaluation and reads one "
                "response, so what matters is how close a <i>single new</i> trial lands to the curve. "
                "<b>R² on new trials</b>: 1 = every trial on the curve, 0 = the curve is no better than guessing "
                "the average. <b>Error per trial</b> vs <b>curve range</b>: an error of a third of the range means "
                "a single response tells apart only about three input levels. Verdicts: Good ≥ "
                f"{F.GOOD_R2:g}, Fair ≥ {F.FAIR_R2:g}, otherwise Poor; a reference that drifts by more than half "
                "the range also lowers the verdict.</p>")

    # ---- checks
    body.append("<h2>Checks</h2><ul class=checks>")
    if problems:
        body.append(f"<li>{_icon(CHECK_BAD)}<span><b>Incomplete:</b> {html.escape('; '.join(problems))}. "
                    "No calibration_latest.json was written.</span></li>")
    elif n < 3:
        body.append(f"<li>{_icon(CHECK_WARN)}<span>calibration_latest.json not updated: FuncSolver needs all three "
                    "functions (this run's calibration.json is kept in its folder for inspection).</span></li>")
    else:
        body.append(f"<li>{_icon(CHECK_OK)}<span>All {len(plan['segments'])} recordings complete.</span></li>")
    art = max((p["artifact"] for p in pairs), default=0.0)
    bad_art = [p for p in pairs if not p["artifact_ok"]]
    body.append(f"<li>{_icon(CHECK_OK if not bad_art else CHECK_WARN)}<span>Stimulation artifact gone before the "
                f"response window (10 ms after the train): "
                + ("yes on every pair" if not bad_art else f"no on {len(bad_art)} pair(s), excluded: "
                   + ", ".join(f"e{p['stim']}→e{p['rec']}" for p in bad_art))
                + f" (largest time-locked residue {art:.1f}× its noise floor; limit {F.ARTIFACT_LIMIT:g}×).</span></li>")
    sh = D[(D.kind == "sham") & (D.partition == "development")]
    body.append(f"<li>{_icon(CHECK_OK)}<span>Sham correction from {sh.train_id.nunique()} no-current trials per "
                f"electrode (same timing, no pulse).</span></li>")
    thr = pd.Series(r["thresholds_uV"]).astype(float)
    body.append(f"<li>{_icon(CHECK_OK)}<span>Event threshold frozen per electrode at −{plan.get('k', 5)}× its "
                f"noise (300–3000 Hz): {thr.abs().min():.0f}–{thr.abs().max():.0f} µV.</span></li></ul>")

    # ---- next step
    body.append("<h2>Next step</h2>")
    if chosen:
        cal = r.get("calibration", "calibration.json")
        body.append("<p>Train FuncSolver offline on these curves (no hardware):</p>"
                    f"<pre class=cmd>cd ~/NeuralControlProgram\npython FuncSolver/main.py --train --fsname {html.escape(fs)} "
                    f"--calibration {html.escape(cal)}</pre>")
        if n < 3:
            body.append("<p class=note>FuncSolver's network needs all three functions. With one missing, try a "
                        "different stimulating electrode for the missing shape (<code>--stims</code>), or more "
                        "current (<code>--amplitude</code>), and re-run this step.</p>")
    else:
        body.append("<p>Nothing to hand to FuncSolver. Try other stimulating electrodes (<code>--stims</code>) or "
                    "stronger trains, after re-checking the screen.</p>")

    # ---- evidence per role
    body.append("<h2>The chosen pairs</h2><p class=note>Grey: development trials, which chose the pair and fitted "
                "the black curve. Blue: confirmation trials, recorded later and never used for fitting; open "
                "circles are their means. Right: the same reference train at the start, middle and end of the "
                "session (dashed: what the curve predicts), which shows drift.</p>")
    for role in F.ROLES:
        p = chosen.get(role)
        if not p:
            continue
        v = r["roles"][role]
        refs = v.get("reference_blocks") or []
        body.append(f"<div class=card><h3>{ROLE_TITLE[role]}: e{p['stim']} → e{p['rec']} &nbsp; {_grade(v['grade'])}</h3>"
                    f"<p>On new trials the curve explains <b>{_fmt(v.get('conf_r2'), '.0%') if v.get('conf_r2') is not None else '–'}</b> "
                    f"of the trial-to-trial variation; a single trial typically lands within "
                    f"<b>±{_fmt(v.get('conf_rmse'), '.1f')}</b> events of it (curve range "
                    f"{_fmt(np.ptp(F.model(role, levels, v['params'])), '.1f')}). "
                    + (f"Reference ({plan['reference_pulses']} pulses): {' → '.join(_fmt(x, '.1f') for x in refs)} events; "
                       f"curve says {_fmt(v.get('reference_expected'), '.1f')}." if refs else "")
                    + "</p>"
                    f"<div class=fig>{_role_figure(p, role, D, levels, plan['reference_pulses'])}</div>"
                    f"<details><summary>Fitted parameters (as written to calibration.json)</summary><p class=note>"
                    f"{', '.join(f'{x:.4g}' for x in v['params'])} · development R² of means {_fmt(v.get('r2_means'))}, "
                    f"cross-validated single-trial R² {_fmt(v.get('cv_r2'))} · sham mean {p['control']:.2f} events"
                    f"</p></details></div>")

    # ---- all pairs
    body.append("<h2>Every pair measured</h2><p class=note>Each panel is one stimulating → recording electrode. "
                "Grey line and band: development mean ± one trial SD; blue: confirmation mean. Each panel has its "
                "own vertical scale. ★ marks the chosen pairs; the subtitle is the shape FuncSolver's rule assigns "
                "(falling, peaked, rising) or flat.</p>")
    body.append(f"<div class=fig>{_all_pairs_figure(pairs, D, chosen)}</div>")
    trs = []
    for p in sorted(pairs, key=lambda q: (q["stim"], q["rec"])):
        best = p.get("shape")
        f = p["fits"].get(best, {}) if best else {}
        trs.append(f"<tr><td>e{p['stim']}→e{p['rec']}</td><td>{F.ROLE_TEXT.get(best, 'flat/none')}</td>"
                   f"<td>{_grade(p['grade'])}</td><td class=n>{_fmt(p['dev_range'], '.1f')}</td>"
                   f"<td class=n>{_fmt(p['trial_sd'], '.1f')}</td><td class=n>{_fmt(f.get('cv_r2'))}</td>"
                   f"<td class=n>{_fmt(f.get('conf_r2'))}</td><td class=n>{_fmt(f.get('conf_rmse'), '.1f')}</td>"
                   f"<td class=n>{_fmt(f.get('drift'), '+.1f')}</td><td class=n>{p['artifact']:.1f}</td>"
                   f"<td class=n>{p['kw_p']:.1g}</td><td>{html.escape(p['grade_reason'])}</td></tr>")
    body.append("<details><summary>All pairs as a table</summary><div class=wrap><table><tr><th>Pair</th><th>Shape</th>"
                "<th>Verdict</th><th class=n>Range</th><th class=n>Trial SD</th><th class=n>Dev. R² (cross-val.)</th>"
                "<th class=n>New-trial R²</th><th class=n>Error/trial</th><th class=n>Reference drift</th>"
                "<th class=n>Artifact ×floor</th><th class=n>Differs across counts p</th><th>Why</th></tr>"
                + "".join(trs) + "</table></div></details>")

    # ---- method
    body.append("<h2>How this was measured</h2><details><summary>Method</summary><ul>"
                "<li>Each train is one trigger; the stimulator itself repeats the pulse, exactly as FuncSolver "
                "delivers it. One recording per pulse count (a train's length is part of the upload).</li>"
                "<li>Order: reference → development (every count, shuffled) → reference → confirmation (reshuffled) → "
                "reference. Each organoid gets ≥ 4 s between trains; trains on different organoids are ≥ 1 s apart.</li>"
                "<li>Readout (<code>orgchar/readout.py</code>, the same function FuncSolver should call live): "
                "300–3000 Hz band-pass; the train and 4 ms after it are blanked; events = crossings of −5× noise, "
                "≥ 1 ms apart, counted 10–210 ms after the train, minus the 500 ms before it (scaled), minus the "
                "mean of no-current sham trials.</li>"
                "<li>Development trials choose the pairs (FuncSolver's shape rule + cross-validated single-trial "
                "fit) and fit FuncSolver's curve formulas; confirmation trials only score those frozen curves.</li>"
                "</ul></details>")
    with open(path, "w") as fh:
        fh.write(_page(title, body))


# ============================================================================== after-train screen

SHAPE_COLOUR = dict(short=ORANGE, band="#1baf7a", long=BLUE)   # reference palette slots 2, 3, 1
SHAPE_NAME = dict(short="falling", band="peaked", long="rising")


def _shape_maps(pairs, stims):
    """One 8 x 8 grid per organoid: rows = stimulated electrode, columns = recording electrode.
    A coloured cell with its symbol = a candidate shape; hover for the numbers."""
    by = {(p["stim"], p["rec"]): p for p in pairs}
    out = []
    for o in sorted({e // 8 for e in stims}):
        cs, x0, y0 = 26, 34, 30
        es = list(range(8 * o, 8 * o + 8))
        svg = [f"<svg viewBox='0 0 {x0 + 8 * cs + 4} {y0 + 8 * cs + 4}' style='width:{x0 + 8 * cs + 4}px;max-width:100%'>"
               f"<text x='{x0 + 4 * cs}' y='10' font-size='10' text-anchor='middle' fill='{MUTED}'>recording electrode</text>"]
        for j, c in enumerate(es):
            svg.append(f"<text x='{x0 + j * cs + cs / 2}' y='{y0 - 6}' font-size='10' text-anchor='middle' fill='{MUTED}'>{c}</text>")
        for i, e in enumerate(es):
            tested = e in stims
            svg.append(f"<text x='{x0 - 6}' y='{y0 + i * cs + cs / 2 + 4}' font-size='10' text-anchor='end' "
                       f"fill='{INK if tested else '#c9c8c2'}'>e{e}</text>")
            for j, c in enumerate(es):
                x, y = x0 + j * cs, y0 + i * cs
                p = by.get((e, c))
                if e == c or not tested or p is None:
                    svg.append(f"<rect x='{x + 1}' y='{y + 1}' width='{cs - 2}' height='{cs - 2}' rx='3' fill='#f1f0eb'/>")
                    continue
                shp = max(p["shapes"], key=lambda k: p["scores"][k]["win"]) if p["shapes"] else None
                tip = (f"e{e}→e{c}: " + ", ".join(f"{n} p: {m:.1f}" for n, m in zip(p["levels"], p["means"]))
                       + (f" · {SHAPE_NAME[shp]}, single trial wins {p['scores'][shp]['win']:.0%}" if shp else " · no shape"))
                fill = SHAPE_COLOUR[shp] if shp else "#ffffff"
                svg.append(f"<g><title>{html.escape(tip)}</title><rect x='{x + 1}' y='{y + 1}' width='{cs - 2}' "
                           f"height='{cs - 2}' rx='3' fill='{fill}' stroke='#e3e2dc'/>"
                           + (f"<text x='{x + cs / 2}' y='{y + cs / 2 + 5}' font-size='14' text-anchor='middle' "
                              f"fill='white' font-weight='700'>{F.SHAPE_SYMBOL[shp]}</text>" if shp else "") + "</g>")
        svg.append("</svg>")
        out.append(f"<figure><figcaption>Organoid {o + 1} (e{8 * o}–e{8 * o + 7})</figcaption>{''.join(svg)}</figure>")
    legend = " ".join(f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:{SHAPE_COLOUR[k]};"
                      f"vertical-align:-1px'></span> {F.SHAPE_SYMBOL[k]} {SHAPE_NAME[k]}" for k in F.ROLES)
    return f"<p class=note>{legend} &nbsp;·&nbsp; white: tested, no shape &nbsp;·&nbsp; grey: not stimulated</p><div class=maps>{''.join(out)}</div>"


def _candidate_figure(cands, D, levels):
    n = len(cands)
    ncol = min(4, n)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.0 * nrow), squeeze=False,
                             gridspec_kw=dict(hspace=0.85, wspace=0.4))
    for k, (role, p) in enumerate(cands):
        ax = axes[k // ncol, k % ncol]
        g = D[(D.kind == "stim") & (D.stim == p["stim"]) & (D.rec == p["rec"])]
        y = g.adjusted - p.get("control", 0.0)
        xs = np.array([levels.index(v) for v in g.level])
        ax.scatter(xs + _jitter(len(xs), 0.12, k), y, s=10, color=SHAPE_COLOUR[role], alpha=0.7, linewidths=0)
        ax.plot(range(len(levels)), p["means"], color=INK, lw=1.8, marker="o", ms=3.5)
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.tick_params(labelsize=7)
        ax.set_title(f"e{p['stim']}→e{p['rec']}  {F.SHAPE_SYMBOL[role]} {SHAPE_NAME[role]}\n"
                     f"wins {p['scores'][role]['win']:.0%}, {p['scores'][role]['diff']:.1f} events", fontsize=7.5, loc="left")
        if k % ncol == 0:
            ax.set_ylabel("events after train", fontsize=7.5)
        if k // ncol == nrow - 1:
            ax.set_xlabel("pulses in the train", fontsize=7.5)
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    return _svg(fig)


def screen_page(path, plan, r, problems, when):
    D = r["_D"]
    levels = list(plan["levels"])
    fs = plan["fsname"]
    sim = plan.get("backend") == "simulated"
    best, stims = r["best"], plan["stims"]
    body = [f"<h1>{html.escape(fs)} · after-train screen</h1>",
            f"<p class=meta>{html.escape(when)}{' · SIMULATED' if sim else ''} · every electrode of organoid(s) "
            f"{', '.join(str(o + 1) for o in sorted({e // 8 for e in stims}))} · trains of {', '.join(map(str, levels))} "
            f"pulses, {plan['ipi_ms']:g} ms apart, {plan['amplitude_uA']} µA × {plan['phase_us']:g} µs · "
            f"{r['n_trials']} trains</p>"]
    if r["curve_stims"]:
        lede = (f"Next, the curve stimulates e{', e'.join(map(str, r['curve_stims']))}. Candidates found for "
                + ", ".join(SHAPE_NAME[k] for k in F.ROLES if k in best)
                + ("" if len(best) == 3 else "; none for " + ", ".join(SHAPE_NAME[k] for k in F.ROLES if k not in best)) + ".")
    else:
        lede = "No electrode gives a falling, peaked or rising response after trains."
    if r["verdict"] == "CONTROLS_FAILED":
        lede = "Controls failed: too many other-organoid pairs pass. Do not trust the candidates below."
    body.append(f"<p class=lede>{html.escape(lede)}</p>")
    rows = []
    for role in F.ROLES:
        v = best.get(role)
        if v:
            p = next(q for q in r["pairs"] if q["stim"] == v["stim"] and q["rec"] == v["rec"])
            rows.append(f"<tr><td><b>{SHAPE_NAME[role]}</b> {F.SHAPE_SYMBOL[role]} <span class=note>({role})</span></td>"
                        f"<td>e{v['stim']} → e{v['rec']}</td><td class=n>{v['win']:.0%}</td><td class=n>{v['diff']:.1f}</td>"
                        f"<td class=n>{' · '.join(f'{m:.1f}' for m in p['means'])}</td>"
                        f"<td class=n>{v['n_candidates'] - 1}</td></tr>")
        else:
            rows.append(f"<tr><td><b>{SHAPE_NAME[role]}</b> {F.SHAPE_SYMBOL[role]} <span class=note>({role})</span></td>"
                        f"<td>{_icon(CHECK_BAD)}none</td><td></td><td></td><td></td><td></td></tr>")
    body.append("<div class=wrap><table><tr><th>Shape FuncSolver needs</th><th>Best pair</th>"
                "<th class=n title='How often one trial at the favoured length beats one at the other'>Single trial wins</th>"
                "<th class=n>Difference (events)</th>"
                f"<th class=n>Mean events at {' · '.join(map(str, levels))} pulses</th><th class=n>Other candidates</th></tr>"
                + "".join(rows) + "</table></div>")
    body.append("<p class=note><b>How to read this.</b> FuncSolver reads one response per train, so a shape is only "
                "useful if single trials differ. <b>Single trial wins</b>: how often one trial at the length that should "
                "give more events (2 pulses for falling, 20 for rising, the middle for peaked) beats one at the length "
                "that should give fewer. 50% = no difference, 100% = always. A pair counts as a candidate at p &lt; "
                f"{F.SCOUT_MAX_P:g} (one-sided rank test) and ≥ {F.SCOUT_MIN_EVENTS:g} events apart. The curve step then "
                "measures the full curve for the chosen electrodes and checks it on new trials.</p>")
    c = r["controls"]
    body.append("<h2>Checks</h2><ul class=checks>")
    body.append(f"<li>{_icon(CHECK_BAD if problems else CHECK_OK)}<span>"
                + (f"<b>Incomplete:</b> {html.escape('; '.join(problems))}." if problems
                   else f"All {len(plan['segments'])} recordings complete.") + "</span></li>")
    ok = r["verdict"] != "CONTROLS_FAILED"
    body.append(f"<li>{_icon(CHECK_OK if ok else CHECK_BAD)}<span>Negative controls: {c['cross_organoid_passing']} of "
                f"{c['cross_organoid_pairs']} other-organoid pairs pass (about {c['expected_by_chance']:g} expected by "
                "chance; a train cannot reach another organoid).</span></li>")
    bad = [p for p in r["pairs"] if not p["artifact_ok"]]
    body.append(f"<li>{_icon(CHECK_OK if not bad else CHECK_WARN)}<span>Stimulation artifact gone 10 ms after the train"
                + (" on every pair." if not bad else f": not on {len(bad)} pair(s), which were excluded.") + "</span></li>")
    body.append(f"<li>{_icon(CHECK_OK)}<span>Sham correction from {c['sham_trials']} no-current trials.</span></li></ul>")
    body.append("<h2>Next step</h2>")
    if r["curve_stims"]:
        body.append(f"<pre class=cmd>./orgchar.sh curve --fsname {html.escape(fs)}</pre><p class=note>Same as "
                    f"<code>--stims {','.join(map(str, r['curve_stims']))}</code>. About "
                    f"{20 + 10 * len(r['curve_stims'])}–{30 + 15 * len(r['curve_stims'])} min.</p>")
    else:
        body.append("<p>Nothing for FuncSolver with these trains. Options: more current (<code>--amplitude 3</code>), "
                    "or re-check the census: the organoids may be quiet today.</p>")
    body.append("<h2>Every electrode pair</h2>" + _shape_maps(r["pairs"], stims))
    cands = []
    for role in F.ROLES:
        cands += [(role, p) for p in sorted([q for q in r["pairs"] if role in q["shapes"]],
                                            key=lambda q: -q["scores"][role]["win"])[:4]]
    if cands:
        body.append("<h2>The candidates</h2><p class=note>Up to four per shape, best first. Dots: single trials; line: "
                    "mean. Events are counted 10–210 ms after the train, minus the 500 ms before it and the sham mean.</p>"
                    f"<div class=fig>{_candidate_figure(cands, D, levels)}</div>")
    trs = "".join(
        f"<tr><td>e{p['stim']}→e{p['rec']}</td><td>{', '.join(SHAPE_NAME[k] for k in p['shapes']) or '–'}</td>"
        f"<td class=n>{' · '.join(f'{m:.1f}' for m in p['means'])}</td>"
        + "".join(f"<td class=n>{p['scores'][k]['win']:.0%} (p {p['scores'][k]['p']:.1g})</td>" if k in p["scores"] else "<td></td>"
                  for k in F.ROLES)
        + f"<td class=n>{p['artifact']:.1f}</td></tr>" for p in sorted(r["pairs"], key=lambda q: (q["stim"], q["rec"])))
    body.append("<details><summary>All pairs as a table</summary><div class=wrap><table><tr><th>Pair</th><th>Shape</th>"
                f"<th class=n>Mean events at {' · '.join(map(str, levels))}</th><th class=n>Falling: wins</th>"
                "<th class=n>Peaked: wins</th><th class=n>Rising: wins</th><th class=n>Artifact ×</th></tr>"
                + trs + "</table></div></details>")
    with open(path, "w") as fh:
        fh.write(_page(f"{fs} · after-train screen · {when}", body))
