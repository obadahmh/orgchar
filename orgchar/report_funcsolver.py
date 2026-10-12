"""Reports for the curve and screen steps, written for readers outside the project: the decision
first, then the terms it uses, then the evidence. Electrode panels follow FinalSpark's electrode
map. Figures are SVG, so they stay sharp at any zoom."""
from __future__ import annotations

import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import funcsolver as F
from . import protocol as P
from .report import BLUE, CHECK_BAD, CHECK_OK, CHECK_WARN, GREY, INK, MUTED, ORANGE, _icon, _page, _svg

GRADE = {"good": ("✓", "#0ca30c", "Good"), "fair": ("◐", "#d69300", "Fair"),
         "poor": ("✗", "#d03b3b", "Poor"), "unknown": ("?", MUTED, "Unknown"), "none": ("–", MUTED, "No curve")}
FIT, TEST = "#a3a29c", BLUE
TYPE = dict(short="short-pass", band="band-pass", long="long-pass")
TYPE_DESC = dict(short="fewer evoked spikes after longer trains", band="most evoked spikes at a middle train length",
                 long="more evoked spikes after longer trains")
TYPE_COLOUR = dict(short=ORANGE, band="#1baf7a", long=BLUE)
# FinalSpark's electrode map: each organoid is a cross of eight electrodes over three rows.
# Key: electrode number within the organoid (0-7). Value: (column, row) in the cross.
LAYOUT = {0: (1, 0), 7: (2, 0), 1: (0, 1), 2: (1, 1), 5: (2, 1), 6: (3, 1), 3: (1, 2), 4: (2, 2)}


def _grade(g):
    sym, col, label = GRADE.get(g, GRADE["unknown"])
    return f"{_icon((sym, col))}<b>{label}</b>"


def _fmt(v, f=".2f", na="–"):
    return na if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, f)


def _jitter(n, width, seed):
    return np.random.default_rng(seed).uniform(-width, width, n)


def _pair(s, c):
    return f"e{s} → e{c}"


def _type(role):
    return f"{F.SHAPE_SYMBOL[role]} {TYPE[role]}"


def _planned(plan):
    return sum(1 for s in plan["segments"] for e in s["events"] if e["kind"] == "stim")


def _organoid(e):
    return f"e{8 * (e // 8)}–e{8 * (e // 8) + 7}"


def _terms(items):
    return "<dl class=terms>" + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in items) + "</dl>"


def _p_text(p):
    return "yes (p < 0.001)" if p < 0.001 else f"yes (p = {p:.3f})" if p < F.MIN_EFFECT_P else f"no (p = {p:.2f})"


TERMS_COMMON = [
    ("e8 → e15", "Stimulate electrode 8 and count spikes on electrode 15. Electrode numbers are FinalSpark's. "
                 "Each organoid sits on eight electrodes: e0–e7, e8–e15, e16–e23 and e24–e31."),
    ("Train", "A series of biphasic pulses, 10 ms apart. The stimulator generates the whole train from one trigger."),
    ("Trial", "One train on one stimulating electrode, then counting spikes on the other seven electrodes of the "
              "same organoid. Trials on the same organoid are at least 4 s apart."),
    ("Evoked spikes", "Spikes counted 10 to 210 ms after the end of the train. From this we subtract the spontaneous "
                      "count in the 500 ms before the train (scaled to the same length) and the sham average."),
    ("Sham trial", "A trigger that delivers no current. It shows what the counting gives when nothing is stimulated."),
    ("Response types", f"<b>{TYPE['short']}</b>: {TYPE_DESC['short']}. <b>{TYPE['band']}</b>: {TYPE_DESC['band']}. "
                       f"<b>{TYPE['long']}</b>: {TYPE_DESC['long']}."),
]


# ============================================================================== curve step

def _role_figure(p, role, D, levels, ref_n):
    f = p["fits"][role]
    stim = D[(D.kind == "stim") & (D.stim == p["stim"]) & (D.rec == p["rec"])].copy()
    stim["y"] = stim.adjusted - p["control"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 2.9), gridspec_kw=dict(width_ratios=[4.2, 1.5], wspace=0.28))
    step = np.median(np.diff(levels)) if len(levels) > 1 else 1.0
    for part, col, dx, lab in (("fit", FIT, -0.22, "fit trials"), ("test", TEST, 0.22, "test trials")):
        g = stim[stim.partition == part]
        a.scatter(g.level + dx * step + _jitter(len(g), 0.08 * step, 1), g.y, s=12, color=col, alpha=0.75,
                  linewidths=0, label=lab, zorder=2)
        if part == "test" and len(g):
            m = g.groupby("level").y.mean()
            a.plot(m.index + dx * step, m.values, "o", ms=4.5, mfc="white", mec=TEST, mew=1.4, zorder=4,
                   label="test-trial mean")
    xs = np.linspace(min(levels), max(levels), 200)
    a.plot(xs, F.model(role, xs, f["params"]), color=INK, lw=2, zorder=3, label="curve fitted to the fit trials")
    a.axhline(0, color=GREY, lw=0.6, zorder=1)
    a.set_xticks(levels)
    a.set_xlabel("pulses in the train")
    a.set_ylabel("evoked spikes per trial")
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
        b.set_xticklabels(["start", "middle", "end"][:len(blocks)])
        b.set_xlim(-0.6, len(blocks) - 0.4)
        b.set_title(f"{ref_n}-pulse reference train\n(dashed: what the curve predicts)", fontsize=8.5)
    else:
        b.axis("off")
    return _svg(fig)


def _all_pairs_figure(pairs, chosen):
    """One block per stimulating electrode. The panels sit where the recording electrodes sit on
    FinalSpark's electrode map."""
    stims = sorted({p["stim"] for p in pairs})
    by = {(p["stim"], p["rec"]): p for p in pairs}
    picked = {(p["stim"], p["rec"]): r for r, p in chosen.items()}
    fig = plt.figure(figsize=(9.6, 4.4 * len(stims)))
    subs = fig.subfigures(len(stims), 1, hspace=0.04) if len(stims) > 1 else [fig]
    for sf, s in zip(subs, stims):
        sf.suptitle(f"Stimulating e{s}  (organoid on {_organoid(s)})", x=0.02, y=0.99, ha="left", va="top",
                    fontsize=11, fontweight="bold")
        axes = sf.subplots(3, 4, gridspec_kw=dict(hspace=0.95, wspace=0.42, top=0.84, bottom=0.07,
                                                   left=0.06, right=0.98))
        for ax in axes.flat:
            ax.axis("off")
        for k, (col, row) in LAYOUT.items():
            e, ax = 8 * (s // 8) + k, axes[row, col]
            if e == s:
                ax.text(0.5, 0.5, f"e{e}\nstimulated", ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white", transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.6", fc=INK, ec="none"))
                continue
            p = by.get((s, e))
            if p is None:
                ax.text(0.5, 0.5, f"e{e}\nnot analysed", ha="center", va="center", fontsize=7.5, color=MUTED,
                        transform=ax.transAxes)
                continue
            ax.axis("on")
            lv = np.array(p["levels"])
            dm, sd = np.array(p["dev_means"]), np.array(p["dev_sd"])
            ax.fill_between(lv, dm - sd, dm + sd, color=FIT, alpha=0.25, linewidth=0)
            ax.plot(lv, dm, color=FIT, lw=1.4)
            if p.get("shape") and p["fits"][p["shape"]].get("conf_means"):
                f = p["fits"][p["shape"]]
                ax.plot(f.get("conf_levels", lv), f["conf_means"], color=TEST, lw=1.4)
            ax.axhline(0, color=GREY, lw=0.5)
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, -1.0), max(hi, 3.0))   # flat pairs: no magnified noise
            ax.set_xticks([lv[0], lv[-1]])
            ax.tick_params(labelsize=7.5, length=2)
            role = picked.get((s, e))
            sub = f"★ {TYPE[role]}" if role else (TYPE[p["shape"]] if p.get("shape") else "no clear type")
            ax.set_title(f"e{e}\n{sub}", fontsize=8.5, fontweight="bold" if role else "normal",
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
    planned = _planned(plan)
    title = f"{fs} · response curves · {when}" + (" · SIMULATED" if sim else "")
    body = [f"<h1>{html.escape(fs)} · response curves to pulse trains</h1>",
            f"<p class=meta>{html.escape(when)}{' · SIMULATED' if sim else ''} · stimulating "
            f"{', '.join('e' + str(s) for s in plan['stims'])} · trains of {levels[0]} to {levels[-1]} pulses, "
            f"{plan['ipi_ms']:g} ms apart, {plan['amplitude_uA']:g} µA, {plan['phase_us']:g} µs per phase · "
            f"{r['n_trials']} of {planned} trials usable</p>",
            "<p>This step looks for electrode pairs with the three response types to pulse trains. Our "
            "function-approximation network needs one pair of each type. It sends one train per evaluation and "
            "reads a single response, so the question is how well a fitted curve predicts single new trials.</p>"]

    # ---- the decision
    n = len(chosen)
    missing = [TYPE[x] for x in F.ROLES if x not in chosen]
    lede = {3: "All three response types found.", 0: "No pair gives a usable response type."}.get(
        n, f"{n} of the 3 response types found. Missing: {', '.join(missing)}.")
    body.append(f"<p class=lede>{html.escape(lede)}</p>")
    rows = []
    for role in F.ROLES:
        v = r["roles"].get(role)
        name = f"<b>{_type(role)}</b><br><span class=note>{TYPE_DESC[role]}</span>"
        if v:
            rows.append(f"<tr><td>{name}</td><td>{_pair(v['stim'], v['rec'])}</td><td>{_grade(v['grade'])}</td>"
                        f"<td class=n>{_fmt(v.get('conf_r2'))}</td><td class=n>±{_fmt(v.get('conf_rmse'), '.1f')}</td>"
                        f"<td class=n>{_fmt(np.ptp(F.model(role, levels, v['params'])), '.1f')}</td>"
                        f"<td>{html.escape(v['reason'])}</td></tr>")
        else:
            rows.append(f"<tr><td>{name}</td><td>–</td><td>{_icon(CHECK_BAD)}<b>Missing</b></td><td></td><td></td>"
                        "<td></td><td>no pair gives this curve</td></tr>")
    body.append("<div class=wrap><table><tr><th>Response type</th><th>Pair</th><th>Verdict</th>"
                "<th class=n>R² on test trials</th><th class=n>Error per trial</th><th class=n>Curve range</th>"
                "<th>Why</th></tr>" + "".join(rows) + "</table></div>")
    body.append("<p class=note>Errors and ranges are in evoked spikes. Verdicts: <b>Good</b> means R² on test "
                f"trials of at least {F.GOOD_R2:g}. <b>Fair</b> means at least {F.FAIR_R2:g}. Anything lower is "
                "<b>Poor</b>. A reference train that drifts by more than half the curve range also lowers the "
                "verdict. As a rule of thumb, if the error per trial is a third of the curve range, one response "
                "can tell apart only about three train lengths.</p>")

    # ---- terms
    body.append("<h2>Terms used in this report</h2>" + _terms(TERMS_COMMON + [
        ("Fit trials and test trials", "Every train length is measured twice. The first pass (fit trials) picks "
                                       "the pairs and fits the curves. The second pass (test trials) comes later in "
                                       "the same session, in a new random order. It is never used for fitting. It "
                                       "only checks the fitted curves on new data."),
        ("Reference train", f"The same {plan['reference_pulses']}-pulse train, given at the start, middle and end "
                            "of the session. If the responses drift during the session, these three blocks "
                            "disagree."),
        ("R² on test trials", "The share of the trial-to-trial variation in the test trials that the fitted curve "
                              "explains. 1 means every trial falls on the curve. 0 means the curve does no better "
                              "than the overall average."),
        ("Error per trial", "How far a single test trial typically falls from the curve (root mean square)."),
        ("p", "The probability of seeing differences across train lengths at least this large if train length had "
              f"no effect (Kruskal–Wallis test on the fit trials). A pair needs p below {F.MIN_EFFECT_P:g}."),
    ]))

    # ---- checks
    body.append("<h2>Checks</h2><ul class=checks>")
    if problems:
        body.append(f"<li>{_icon(CHECK_BAD)}<span><b>Incomplete.</b> "
                    + " ".join(html.escape(x[0].upper() + x[1:]) + "." for x in problems) + "</span></li>")
    else:
        body.append(f"<li>{_icon(CHECK_OK)}<span>All {len(plan['segments'])} recordings are complete.</span></li>")
    lost = planned - r["n_trials"]
    body.append(f"<li>{_icon(CHECK_OK if lost <= 0.1 * planned else CHECK_WARN)}<span>{r['n_trials']} of {planned} "
                "planned trials are usable."
                + (f" The other {lost} could not be used. Either their trigger was not found in the recording, or a "
                   "gap in the raw data cut into their counting window. The platform sometimes drops 42.7 ms of "
                   "data without a marker." if lost else "") + "</span></li>")
    art = max((p["artifact"] for p in pairs), default=0.0)
    bad_art = [p for p in pairs if not p["artifact_ok"]]
    body.append(f"<li>{_icon(CHECK_OK if not bad_art else CHECK_WARN)}<span>Stimulation artifact. "
                + ("It has faded on every pair by the start of the counting window (10 ms after the train)."
                   if not bad_art else
                   f"It still reaches the counting window on {len(bad_art)} pair(s), which were excluded: "
                   + ", ".join(_pair(p["stim"], p["rec"]) for p in bad_art) + ".")
                + f" The largest leftover is {art:.1f} times its noise floor. The limit is {F.ARTIFACT_LIMIT:g}."
                "</span></li>")
    sh = D[(D.kind == "sham") & (D.partition == "fit")]
    body.append(f"<li>{_icon(CHECK_OK)}<span>The sham correction uses {sh.train_id.nunique()} sham trials.</span></li>")
    thr = pd.Series(r["thresholds_uV"]).astype(float)
    body.append(f"<li>{_icon(CHECK_OK)}<span>Spike thresholds were fixed per electrode at {plan.get('k', 5)} times "
                f"its noise, in the 300–3000 Hz band. They range from {thr.abs().min():.0f} to "
                f"{thr.abs().max():.0f} µV.</span></li></ul>")

    # ---- evidence per role
    body.append("<h2>The chosen pairs</h2><p class=note>Grey dots are fit trials. They chose the pair and fitted the "
                "black curve. Blue dots are test trials, and open circles are their means. On the right, the "
                "reference train at the start, middle and end of the session shows drift.</p>")
    for role in F.ROLES:
        p = chosen.get(role)
        if not p:
            continue
        v = r["roles"][role]
        refs = v.get("reference_blocks") or []
        body.append(f"<div class=card><h3>{_type(role)}: {_pair(p['stim'], p['rec'])} &nbsp; {_grade(v['grade'])}</h3>"
                    f"<p>On test trials the curve explains <b>{_fmt(v.get('conf_r2'), '.0%')}</b> of the "
                    f"trial-to-trial variation. A single trial typically falls within "
                    f"<b>±{_fmt(v.get('conf_rmse'), '.1f')}</b> spikes of the curve, whose range is "
                    f"{_fmt(np.ptp(F.model(role, levels, v['params'])), '.1f')} spikes. "
                    + (f"The reference train gave {', then '.join(_fmt(x, '.1f') for x in refs)} spikes. The curve "
                       f"predicts {_fmt(v.get('reference_expected'), '.1f')}." if refs else "")
                    + "</p>"
                    f"<div class=fig>{_role_figure(p, role, D, levels, plan['reference_pulses'])}</div>"
                    f"<details><summary>Fitted parameters (as written to calibration.json)</summary><p class=note>"
                    f"Parameters {', '.join(f'{x:.4g}' for x in v['params'])}. R² of the fit-trial means "
                    f"{_fmt(v.get('r2_means'))}. Cross-validated R² within the fit trials {_fmt(v.get('cv_r2'))}. "
                    f"Sham average {p['control']:.2f} spikes.</p></details></div>")

    # ---- all pairs
    body.append("<h2>Every pair measured</h2><p class=note>One block per stimulating electrode. Each panel is a "
                "recording electrode, placed where it sits on FinalSpark's electrode map. The grey line and band "
                "are the fit-trial mean ± one standard deviation. The blue line is the test-trial mean. Each panel "
                "has its own vertical scale. ★ marks the chosen pairs.</p>")
    body.append(f"<div class=fig>{_all_pairs_figure(pairs, chosen)}</div>")
    trs = []
    for p in sorted(pairs, key=lambda q: (q["stim"], q["rec"])):
        best = p.get("shape")
        f = p["fits"].get(best, {}) if best else {}
        trs.append(f"<tr><td>{_pair(p['stim'], p['rec'])}</td><td>{TYPE.get(best, 'no clear type')}</td>"
                   f"<td>{_grade(p['grade'])}</td><td>{_p_text(p['kw_p'])}</td>"
                   f"<td class=n>{_fmt(p['dev_range'], '.1f')}</td><td class=n>{_fmt(p['trial_sd'], '.1f')}</td>"
                   f"<td class=n>{_fmt(f.get('conf_r2'))}</td><td class=n>{_fmt(f.get('conf_rmse'), '.1f')}</td>"
                   f"<td class=n>{_fmt(f.get('drift'), '+.1f')}</td><td class=n>{p['artifact']:.1f}</td>"
                   f"<td>{html.escape(p['grade_reason'])}</td></tr>")
    body.append("<details><summary>All pairs as a table</summary><p class=note>Range: the difference between the "
                "largest and smallest fit-trial mean. Trial SD: the typical scatter of single trials at one train "
                "length. Drift: the last reference block minus the first. Artifact: the leftover at the start of "
                "the counting window, in units of its noise floor.</p><div class=wrap><table><tr><th>Pair</th>"
                "<th>Type</th><th>Verdict</th><th>Changes with train length?</th><th class=n>Range</th>"
                "<th class=n>Trial SD</th><th class=n>R² on test trials</th><th class=n>Error per trial</th>"
                "<th class=n>Drift</th><th class=n>Artifact</th><th>Why</th></tr>"
                + "".join(trs) + "</table></div></details>")

    # ---- method
    body.append("<h2>How this was measured</h2><details><summary>Method</summary><ul>"
                "<li>Each train comes from one trigger. The stimulator repeats the pulse itself, which is also how "
                "the network delivers it. A train's length is part of the uploaded settings, so each train length "
                "has its own recording.</li>"
                "<li>Order: reference train, fit pass (every train length, shuffled), reference train, test pass "
                "(every train length, reshuffled), reference train. Trials on the same organoid are at least 4 s "
                "apart. Trials on different organoids are at least 1 s apart.</li>"
                "<li>Spike counting: 300–3000 Hz band-pass. The train and the 4 ms after it are blanked. A spike is a "
                "crossing of −5 times the noise, at least 1 ms after the previous one.</li>"
                "<li>Fit trials choose the pairs and fit the curves. Test trials only score those frozen curves.</li>"
                "</ul></details>")

    # ---- operator
    body.append("<h2>For the operator</h2>")
    if chosen:
        cal = r.get("calibration", "calibration.json")
        body.append("<p>Train the network offline on these curves (no hardware needed); it uses the types found:</p>"
                    f"<pre class=cmd>cd ~/NeuralControlProgram/FuncSolver\npython main.py train {html.escape(cal)}</pre>")
        if n < 3:
            body.append("<p class=note>To look for the missing type, re-run this step with other stimulating "
                        "electrodes (<code>--stims</code>) or more current (<code>--amplitude</code>).</p>")
    else:
        body.append("<p>Nothing to hand to the network. Try other stimulating electrodes (<code>--stims</code>) or "
                    "stronger trains, after re-checking the screen.</p>")
    with open(path, "w") as fh:
        fh.write(_page(title, body))


# ============================================================================== screen step

def _shape_maps(pairs, stims):
    """FinalSpark's electrode map, with a small copy of the organoid's map at every electrode. The small
    map at electrode e shows what stimulating e did: each recording electrode is coloured by the
    response type it gave. Hover a square for the numbers."""
    by = {(p["stim"], p["rec"]): p for p in pairs}
    cell, pitch = 16, 19
    mini_w, mini_h = 4 * pitch, 3 * pitch
    slot_w, slot_h = mini_w + 16, mini_h + 26
    org_w, org_h = 4 * slot_w, 3 * slot_h
    out = []
    for o in sorted({e // 8 for e in stims}):
        svg = [f"<svg viewBox='0 0 {org_w} {org_h}' style='width:{round(1.4 * org_w)}px;max-width:100%'>"]
        for k, (col, row) in LAYOUT.items():
            e = 8 * o + k
            x0, y0 = col * slot_w + 8, row * slot_h + 18
            tested = e in stims
            svg.append(f"<text x='{x0 + mini_w / 2}' y='{y0 - 6}' font-size='13' text-anchor='middle' "
                       f"font-weight='{700 if tested else 400}' fill='{INK if tested else '#b9b8b2'}'>e{e}</text>")
            for kk, (cc, rr) in LAYOUT.items():
                c = 8 * o + kk
                x, y = x0 + cc * pitch, y0 + rr * pitch
                if not tested:
                    svg.append(f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' rx='2' fill='#f1f0eb'/>")
                    continue
                if c == e:
                    svg.append(f"<g><title>e{e} stimulated</title><rect x='{x}' y='{y}' width='{cell}' height='{cell}' "
                               f"rx='2' fill='{INK}'/></g>")
                    continue
                p = by.get((e, c))
                shp = max(p["shapes"], key=lambda q: p["scores"][q]["win"]) if p and p["shapes"] else None
                tip = (f"{_pair(e, c)}: " + ", ".join(f"{n} pulses {m:.1f}" for n, m in zip(p["levels"], p["means"]))
                       + (f". {TYPE[shp]}, single trial wins {p['scores'][shp]['win']:.0%}" if shp else ". No clear type")
                       if p else f"{_pair(e, c)}: not analysed")
                fill = TYPE_COLOUR[shp] if shp else "#ffffff"
                svg.append(f"<g><title>{html.escape(tip)}</title><rect x='{x}' y='{y}' width='{cell}' height='{cell}' "
                           f"rx='2' fill='{fill}' stroke='{'none' if shp else '#c9c8c2'}'/></g>")
        svg.append("</svg>")
        out.append(f"<figure><figcaption>Organoid on {_organoid(8 * o)}</figcaption>{''.join(svg)}</figure>")
    key = (f"<span style='display:inline-block;width:12px;height:12px;border-radius:2px;background:{INK};"
           "vertical-align:-1px'></span> stimulated electrode &nbsp; "
           + " &nbsp; ".join(f"<span style='display:inline-block;width:12px;height:12px;border-radius:2px;"
                             f"background:{TYPE_COLOUR[k]};vertical-align:-1px'></span> {TYPE[k]}" for k in F.ROLES)
           + " &nbsp; <span style='display:inline-block;width:12px;height:12px;border-radius:2px;background:#fff;"
           "border:1px solid #c9c8c2;vertical-align:-2px'></span> no clear type")
    return f"<p class=note>{key}</p><div class=maps>{''.join(out)}</div>"


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
        ax.scatter(xs + _jitter(len(xs), 0.12, k), y, s=10, color=TYPE_COLOUR[role], alpha=0.7, linewidths=0)
        ax.plot(range(len(levels)), p["means"], color=INK, lw=1.8, marker="o", ms=3.5)
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.tick_params(labelsize=7)
        ax.set_title(f"e{p['stim']} → e{p['rec']}  {TYPE[role]}\n"
                     f"wins {p['scores'][role]['win']:.0%}, {p['scores'][role]['diff']:.1f} spikes apart",
                     fontsize=7.5, loc="left")
        if k % ncol == 0:
            ax.set_ylabel("evoked spikes per trial", fontsize=7.5)
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
    planned = _planned(plan)
    organoids = sorted({e // 8 for e in stims})
    body = [f"<h1>{html.escape(fs)} · screen with pulse trains</h1>",
            f"<p class=meta>{html.escape(when)}{' · SIMULATED' if sim else ''} · every electrode of the "
            f"organoid(s) on {', '.join(_organoid(8 * o) for o in organoids)} · trains of "
            f"{', '.join(map(str, levels))} pulses, {plan['ipi_ms']:g} ms apart, {plan['amplitude_uA']:g} µA, "
            f"{plan['phase_us']:g} µs per phase · {r['n_trials']} of {planned} trials usable</p>",
            "<p>This step is a quick scan. It stimulates every electrode with four train lengths and looks for "
            "pairs that might show one of the three response types. The curve step then measures the most "
            "promising stimulating electrodes in full.</p>"]
    if r["curve_stims"]:
        lede = (f"Next, the curve step stimulates e{', e'.join(map(str, r['curve_stims']))}. Candidates found for "
                + ", ".join(TYPE[k] for k in F.ROLES if k in best) + "."
                + ("" if len(best) == 3 else " None for " + ", ".join(TYPE[k] for k in F.ROLES if k not in best) + "."))
    else:
        lede = "No electrode gives a short-pass, band-pass or long-pass response."
    if r["verdict"] == "CONTROLS_FAILED":
        lede = "Controls failed: too many pairs across organoids pass. Do not trust the candidates below."
    body.append(f"<p class=lede>{html.escape(lede)}</p>")
    rows = []
    for role in F.ROLES:
        v = best.get(role)
        name = f"<b>{_type(role)}</b><br><span class=note>{TYPE_DESC[role]}</span>"
        if v:
            p = next(q for q in r["pairs"] if q["stim"] == v["stim"] and q["rec"] == v["rec"])
            rows.append(f"<tr><td>{name}</td><td>{_pair(v['stim'], v['rec'])}</td><td class=n>{v['win']:.0%}</td>"
                        f"<td class=n>{v['diff']:.1f}</td><td class=n>{' · '.join(f'{m:.1f}' for m in p['means'])}</td>"
                        f"<td class=n>{v['n_candidates'] - 1}</td></tr>")
        else:
            rows.append(f"<tr><td>{name}</td><td>{_icon(CHECK_BAD)}none</td><td></td><td></td><td></td><td></td></tr>")
    body.append("<div class=wrap><table><tr><th>Response type</th><th>Best pair</th><th class=n>Single trial wins</th>"
                "<th class=n>Difference (spikes)</th>"
                f"<th class=n>Mean evoked spikes at {' · '.join(map(str, levels))} pulses</th>"
                "<th class=n>Other candidates</th></tr>" + "".join(rows) + "</table></div>")

    body.append("<h2>Terms used in this report</h2>" + _terms(TERMS_COMMON + [
        ("Single trial wins", "Take one trial at the train length that should give more spikes (2 pulses for "
                              "short-pass, 20 for long-pass, the middle lengths for band-pass) and one at the length "
                              "that should give fewer. This is how often the first has more spikes. 50% means no "
                              "difference. 100% means always."),
        ("Candidate", f"A pair whose single trial wins clearly (one-sided rank test, p below {F.SCOUT_MAX_P:g}) and "
                      f"whose means differ by at least {F.SCOUT_MIN_EVENTS:g} spikes. A pair is dropped if a pair "
                      "from another organoid does as well on the same recording electrode."),
        ("p", "The probability of a difference at least this large if train length had no effect."),
        ("Control pairs", "A stimulating electrode paired with recording electrodes on another organoid. A train "
                          "cannot reach another organoid through tissue, so these pairs should never pass."),
    ]))

    c = r["controls"]
    body.append("<h2>Checks</h2><ul class=checks>")
    body.append(f"<li>{_icon(CHECK_BAD if problems else CHECK_OK)}<span>"
                + ("<b>Incomplete.</b> " + " ".join(html.escape(x[0].upper() + x[1:]) + "." for x in problems)
                   if problems else f"All {len(plan['segments'])} recordings are complete.") + "</span></li>")
    lost = planned - r["n_trials"]
    if lost:
        body.append(f"<li>{_icon(CHECK_OK if lost <= 0.1 * planned else CHECK_WARN)}<span>{lost} of {planned} trials "
                    "could not be used. Either their trigger was not found in the recording, or a gap in the raw "
                    "data cut into their counting window.</span></li>")
    ok = r["verdict"] != "CONTROLS_FAILED"
    into = c.get("passing_into", [])
    body.append(f"<li>{_icon(CHECK_OK if ok else CHECK_BAD)}<span>Control pairs: {c['cross_organoid_passing']} of "
                f"{c['cross_organoid_pairs']} pass. Chance alone would let up to about {c['nominal_ceiling']:g} "
                "through." + (f" The passing ones record on e{', e'.join(map(str, into))}." if into else "")
                + "</span></li>")
    for x in r.get("chance", []):
        body.append(f"<li>{_icon(CHECK_WARN)}<span>Dropped {_pair(x['stim'], x['rec'])} as {TYPE[x['role']]}. "
                    f"The control pair {_pair(x['beaten_by'], x['rec'])}, from another organoid, did at least as "
                    "well, so chance explains it.</span></li>")
    bad = [p for p in r["pairs"] if not p["artifact_ok"]]
    body.append(f"<li>{_icon(CHECK_OK if not bad else CHECK_WARN)}<span>Stimulation artifact. "
                + ("It has faded on every pair 10 ms after the train." if not bad
                   else f"It still reaches the counting window on {len(bad)} pair(s), which were excluded.")
                + "</span></li>")
    body.append(f"<li>{_icon(CHECK_OK)}<span>The sham correction uses {c['sham_trials']} sham trials.</span></li></ul>")

    body.append("<h2>Every electrode pair</h2><p class=note>The large layout is FinalSpark's electrode map. At each "
                "electrode sits a small copy of its organoid's map, showing what stimulating that electrode did. "
                "Each square in the small map is a recording electrode, coloured by the response type it gave. "
                "Hover a square for the numbers.</p>" + _shape_maps(r["pairs"], stims))
    cands = []
    for role in F.ROLES:
        cands += [(role, p) for p in sorted([q for q in r["pairs"] if role in q["shapes"]],
                                            key=lambda q: -q["scores"][role]["win"])[:4]]
    if cands:
        body.append("<h2>The candidates</h2><p class=note>Up to four per response type, best first. Dots are "
                    "single trials. The line joins the means.</p>"
                    f"<div class=fig>{_candidate_figure(cands, D, levels)}</div>")
    trs = "".join(
        f"<tr><td>{_pair(p['stim'], p['rec'])}</td><td>{', '.join(TYPE[k] for k in p['shapes']) or '–'}</td>"
        f"<td class=n>{' · '.join(f'{m:.1f}' for m in p['means'])}</td>"
        + "".join(f"<td class=n>{'<b>' if k in p['shapes'] else ''}{p['scores'][k]['win']:.0%}"
                  f"{'</b>' if k in p['shapes'] else ''}</td>" if k in p["scores"] else "<td></td>" for k in F.ROLES)
        + f"<td class=n>{p['artifact']:.1f}</td></tr>" for p in sorted(r["pairs"], key=lambda q: (q["stim"], q["rec"])))
    body.append("<details><summary>All pairs as a table</summary><p class=note>Single trial wins for each type. "
                "Bold means the pair is a candidate for that type. Artifact: the leftover at the start of the "
                "counting window, in units of its noise floor.</p><div class=wrap><table><tr><th>Pair</th>"
                f"<th>Candidate for</th><th class=n>Mean evoked spikes at {' · '.join(map(str, levels))}</th>"
                "<th class=n>Short-pass wins</th><th class=n>Band-pass wins</th><th class=n>Long-pass wins</th>"
                "<th class=n>Artifact</th></tr>" + trs + "</table></div></details>")

    body.append("<h2>For the operator</h2>")
    if r["curve_stims"]:
        mins = P.plan_duration_s(P.plan_funcsolver(r["curve_stims"])) / 60
        body.append(f"<pre class=cmd>./orgchar.sh curve --fsname {html.escape(fs)}</pre><p class=note>This is the "
                    f"same as <code>--stims {','.join(map(str, r['curve_stims']))}</code>. It takes about "
                    f"{mins:.0f} min.</p>")
    else:
        body.append("<p>Nothing to measure further with these trains. Try more current (<code>--amplitude 3</code>), "
                    "or re-check the census, because the organoids may be quiet today.</p>")
    with open(path, "w") as fh:
        fh.write(_page(f"{fs} · screen with pulse trains · {when}", body))
