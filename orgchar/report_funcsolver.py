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
                ax.plot(lv, p["fits"][p["shape"]]["conf_means"], color=CONF, lw=1.4)
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
