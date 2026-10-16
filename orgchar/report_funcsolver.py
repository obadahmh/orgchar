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

TYPE = dict(short="short-pass", band="band-pass", long="long-pass")
TYPE_DESC = dict(short="fewer evoked spikes after longer trains", band="most evoked spikes at a middle train length",
                 long="more evoked spikes after longer trains")
TYPE_COLOUR = dict(short=ORANGE, band="#1baf7a", long=BLUE)
# FinalSpark's electrode map: each organoid is a cross of eight electrodes over three rows.
# Key: electrode number within the organoid (0-7). Value: (column, row) in the cross.
LAYOUT = {0: (1, 0), 7: (2, 0), 1: (0, 1), 2: (1, 1), 5: (2, 1), 6: (3, 1), 3: (1, 2), 4: (2, 2)}


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

# Sequential blue (dataviz reference ramp, steps 100-700) for evoked spikes; near zero recedes to the surface.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]
ZERO = "#f1f0eb"
STATUS = dict(good=("✓", "#0ca30c", "Good"), fair=("◐", "#fab219", "Fair"), poor=("✗", "#d03b3b", "Poor"),
              unknown=("?", MUTED, "Unknown"), none=("–", MUTED, "No curve"))


def _pill(g, extra=""):
    sym, col, label = STATUS.get(g, STATUS["unknown"])
    return f"<span class=pill style='--status:{col}'><span class=ico>{sym}</span>{label}{extra}</span>"


def _heat(v, vmax):
    """(fill, dark) for a mean evoked-spike count on the page's shared scale."""
    if v is None or not np.isfinite(v) or vmax <= 0 or v <= 0.03 * vmax:
        return ZERO, False
    i = int(round(min(1.0, v / vmax) * (len(RAMP) - 1)))
    return RAMP[i], i >= 6


def _map_svg(stim, peak, vmax, picked, cs=34):
    """FinalSpark's cross of eight electrodes for the stimulated electrode's organoid: the stimulated
    electrode in ink, each recording electrode filled by its largest mean response (the table's scale),
    the chosen ones ringed and starred. Hover for numbers."""
    o, r = 8 * (stim // 8), cs * 0.4
    out = [f"<svg viewBox='0 0 {4 * cs} {3 * cs}' width='{4 * cs}' height='{3 * cs}' role=img "
           f"aria-label='Electrode map of e{o}-e{o + 7}, stimulating e{stim}'>"]
    for k, (col, row) in LAYOUT.items():
        e, x, y = o + k, (col + 0.5) * cs, (row + 0.5) * cs
        if e == stim:
            fill, dark, tip, ring = INK, True, f"e{e}: stimulated", ""
        else:
            v = peak.get(e)
            fill, dark = _heat(v, vmax)
            tip = f"e{e}: up to {v:.1f} evoked spikes" if v is not None else f"e{e}: not analysed"
            role = picked.get(e)
            ring = (f"<circle cx='{x}' cy='{y}' r='{r + 3}' fill='none' stroke='{INK}' stroke-width='2'/>"
                    if role else "")
        txt = "#fff" if dark else INK
        out.append(f"<g><title>{tip}</title>{ring}<circle cx='{x}' cy='{y}' r='{r}' fill='{fill}' "
                   f"stroke='{'none' if e == stim else '#d6d5ce'}'/><text x='{x}' y='{y + 4}' font-size='{cs * 0.3:.0f}' "
                   f"text-anchor='middle' fill='{txt}' font-weight='600'>{e}{'★' if picked.get(e) else ''}</text></g>")
    return "".join(out) + "</svg>"


def _spark(role, v, p, levels):
    """The fitted curve (line) and the test-trial means (dots), for a tile."""
    w, h, pad = 240, 64, 6
    xs = np.linspace(min(levels), max(levels), 60)
    ys = F.model(role, xs, v["params"])
    tl, tm = p["fits"][role].get("conf_levels", []), p["fits"][role].get("conf_means", [])
    lo, hi = min(0.0, ys.min(), *tm), max(ys.max(), *tm, 1e-9)
    X = lambda n: pad + (n - min(levels)) / (max(levels) - min(levels)) * (w - 2 * pad)  # noqa: E731
    Y = lambda y: h - pad - (y - lo) / (hi - lo) * (h - 2 * pad)  # noqa: E731
    col = TYPE_COLOUR[role]
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}" for i, (a, b) in enumerate(zip(xs, ys)))
    dots = "".join(f"<circle cx='{X(n):.1f}' cy='{Y(m):.1f}' r='3.5' fill='{col}' stroke='#fff' stroke-width='2'>"
                   f"<title>{n} pulses: {m:.1f} evoked spikes (test-trial mean)</title></circle>" for n, m in zip(tl, tm))
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' preserveAspectRatio='none' role=img "
            f"aria-label='fitted curve and test-trial means'><line x1='{pad}' x2='{w - pad}' y1='{Y(0):.1f}' "
            f"y2='{Y(0):.1f}' stroke='#e1e0d9'/><path d='{path}' fill='none' stroke='{INK}' stroke-width='2'/>{dots}</svg>"
            f"<div class=sub>{levels[0]} to {levels[-1]} pulses; line: fitted curve; dots: test-trial means</div>")


def _tile(role, r, levels, pairs, peak_by_stim, vmax):
    v = r["roles"].get(role)
    col = TYPE_COLOUR[role]
    head = f"<div class=head><span class=type>{_type(role)}</span>"
    if not v:
        return (f"<div class='tile missing' style='--type:var(--line)'>{head}<span class=pill style='--status:{MUTED}'>"
                f"Not found</span></div><div class=sub>{TYPE_DESC[role]}</div><div>No pair gives this curve.</div></div>")
    p = next(q for q in pairs if q["stim"] == v["stim"] and q["rec"] == v["rec"])
    used = role in r["_usable"]
    rng = np.ptp(F.model(role, levels, v["params"]))
    glyph = _map_svg(v["stim"], peak_by_stim[v["stim"]], vmax, {v["rec"]: role}, cs=27)
    return (f"<div class=tile style='--type:{col}'>{head}{_pill(v['grade'])}</div>"
            f"<div class=pair>{glyph}<span>e{v['stim']} → e{v['rec']}</span></div>"
            f"<div><span class=big>{_fmt(v.get('conf_r2'))}</span> <span class=sub>R² on test trials</span></div>"
            f"<div class=sub>±{_fmt(v.get('conf_rmse'), '.1f')} spikes per trial on a {rng:.0f}-spike range"
            + ("" if used else "; <b>left out of the calibration</b>") + "</div>"
            + _spark(role, v, p, levels) + "</div>")


def _role_figure(p, role, D, levels, ref_n):
    f = p["fits"][role]
    col = TYPE_COLOUR[role]
    stim = D[(D.kind == "stim") & (D.stim == p["stim"]) & (D.rec == p["rec"])].copy()
    stim["y"] = stim.adjusted - p["control"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.0), gridspec_kw=dict(width_ratios=[4.2, 1.4], wspace=0.3))
    step = np.median(np.diff(levels)) if len(levels) > 1 else 1.0
    for part, c, dx, lab in (("fit", "#b5b4ad", -0.18, "fit trials"), ("test", col, 0.18, "test trials")):
        g = stim[stim.partition == part]
        a.scatter(g.level + dx * step + _jitter(len(g), 0.07 * step, 1), g.y, s=14, color=c, alpha=0.8,
                  linewidths=0, label=lab, zorder=2)
        if part == "test" and len(g):
            m = g.groupby("level").y.mean()
            a.plot(m.index + dx * step, m.values, "o", ms=6, mfc="white", mec=col, mew=1.8, zorder=4,
                   label="test-trial mean")
    xs = np.linspace(min(levels), max(levels), 200)
    a.plot(xs, F.model(role, xs, f["params"]), color=INK, lw=2, zorder=3, label="fitted curve")
    a.axhline(0, color="#c3c2b7", lw=0.8, zorder=1)
    a.set_xticks(levels)
    a.set_xlabel("pulses in the train")
    a.set_ylabel("evoked spikes per trial")
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, frameon=False, fontsize=8, markerscale=1.3,
             handletextpad=0.3, columnspacing=1.2)
    ref = stim[stim.partition == "reference"]
    if len(ref):
        blocks = sorted(ref.block.unique())
        for i, bk in enumerate(blocks):
            g = ref[ref.block == bk]
            b.scatter(i + _jitter(len(g), 0.1, 2), g.y, s=14, color="#898781", alpha=0.8, linewidths=0, zorder=2)
            b.plot([i - 0.25, i + 0.25], [g.y.mean()] * 2, color=INK, lw=2, zorder=3)
        exp = f.get("reference_expected", np.nan)
        b.axhline(exp, color=col, lw=1.4, zorder=1)
        b.text(len(blocks) - 0.45, exp, f" curve: {exp:.0f}", color=MUTED, fontsize=7.5, va="bottom", ha="right")
        b.set_xticks(range(len(blocks)))
        b.set_xticklabels(["start", "middle", "end"][:len(blocks)])
        b.set_xlim(-0.6, len(blocks) - 0.4)
        b.set_title(f"Drift check: {ref_n}-pulse train", fontsize=9, loc="left")
    else:
        b.axis("off")
    return _svg(fig)


def _pair_maps(pairs, chosen, levels):
    """Every pair, one block per stimulating electrode: the organoid's electrode map coloured by response
    size, beside a table of mean evoked spikes per train length (rows = recording electrodes, largest
    first). One colour scale for the whole page, so a 2-spike wobble never looks like a 50-spike response."""
    vmax = max([max(p["dev_means"]) for p in pairs] + [1.0])
    picked = {(p["stim"], p["rec"]): r for r, p in chosen.items()}
    out = [f"<div class=scale><span>evoked spikes, mean of fit trials:</span><span>0</span><span class=bar style="
           f"'background:linear-gradient(90deg,{ZERO},{','.join(RAMP)})'></span><span>{vmax:.0f}</span>"
           "<span>(★ chosen pair; hover a cell for its value)</span></div>"]
    for s in sorted({p["stim"] for p in pairs}):
        rows = sorted([p for p in pairs if p["stim"] == s], key=lambda p: -max(p["dev_means"]))
        peak = {p["rec"]: max(p["dev_means"]) for p in rows}
        tr = []
        for p in rows:
            role = picked.get((s, p["rec"]))
            cells = "".join(
                f"<td style='background:{_heat(m, vmax)[0]}' title='e{s} → e{p['rec']}, {n} pulses: {m:.1f} "
                f"evoked spikes (SD {sd:.1f})'></td>" for n, m, sd in zip(p["levels"], p["dev_means"], p["dev_sd"]))
            kind = (f"★ {_type(role)}" if role else _type(p["shape"]) if p.get("shape") else "no clear type")
            tr.append(f"<tr><th class=row>e{p['rec']}</th>{cells}<td class=type"
                      f"{' style=\"color:var(--ink);font-weight:600\"' if role else ''}>{kind}</td></tr>")
        out.append(f"<div class=stim><div><h3>Stimulating e{s}</h3>"
                   f"{_map_svg(s, peak, vmax, {rec: r for (st, rec), r in picked.items() if st == s})}"
                   f"<div class=sub>{_organoid(s)}, FinalSpark layout</div></div>"
                   f"<div class=wrap><table class=heat><tr><th></th>"
                   + "".join(f"<th>{n}</th>" for n in levels) + "<th></th></tr>" + "".join(tr)
                   + f"<tr><th></th><th colspan={len(levels)} class=sub>pulses in the train →</th></tr></table></div></div>")
    return "".join(out), vmax


def page(path, plan, r, problems, when):
    D, chosen, pairs = r["_D"], r["_chosen"], r["pairs"]
    levels = list(plan["levels"])
    fs = plan["fsname"]
    sim = plan.get("backend") == "simulated"
    planned = _planned(plan)
    title = f"{fs}: response curves, {when}" + (" (SIMULATED)" if sim else "")
    maps, vmax = _pair_maps(pairs, chosen, levels)
    peak_by_stim = {s: {p["rec"]: max(p["dev_means"]) for p in pairs if p["stim"] == s} for s in {p["stim"] for p in pairs}}
    chips = [when] + (["SIMULATED"] if sim else []) + [
        "stimulating " + ", ".join(f"e{s}" for s in plan["stims"]), f"trains of {', '.join(map(str, levels))} pulses",
        f"{plan['amplitude_uA']:g} µA × {plan['phase_us']:g} µs", f"{r['n_trials']} of {planned} trials usable",
        f"orgchar {plan.get('code_version', '?')}"]
    body = [f"<h1>{html.escape(fs)}: response curves</h1>",
            "<div class=chips>" + "".join(f"<span class=chip>{html.escape(c)}</span>" for c in chips) + "</div>"]

    # ---- the decision, first
    usable = r["_usable"]
    found = [x for x in F.ROLES if x in chosen]
    lede = (f"{len(usable)} of 3 response types usable for the network"
            + (f": {', '.join(TYPE[x] for x in F.ROLES if x in usable)}." if usable else "."))
    notes = []
    if [x for x in found if x not in usable]:
        notes.append("Found but Poor on test trials, so left out: "
                     + ", ".join(TYPE[x] for x in found if x not in usable) + ".")
    if len(found) < 3:
        notes.append("Not found: " + ", ".join(TYPE[x] for x in F.ROLES if x not in found) + ".")
    if problems:
        notes.append("<b>Incomplete run:</b> " + html.escape("; ".join(problems)) + ".")
    cmd = (f"<pre class=cmd>cd ~/NeuralControlProgram/FuncSolver\npython main.py train {html.escape(r['calibration'])}</pre>"
           if r.get("calibration") else
           "<p>Nothing to hand to the network. Try other stimulating electrodes (<code>--stims</code>) or stronger "
           "trains, after re-checking the screen.</p>")
    body.append(f"<div class=verdict><p class=lede>{html.escape(lede)}</p>"
                + "".join(f"<p class=note>{n}</p>" for n in notes)
                + ("<p>Next: train the network offline on these curves (no hardware needed):</p>" if r.get("calibration") else "")
                + cmd + "</div>")
    body.append("<div class=tiles>" + "".join(_tile(x, r, levels, pairs, peak_by_stim, vmax) for x in F.ROLES) + "</div>")
    body.append(f"<p class=note><b>Good</b>: R² on test trials ≥ {F.GOOD_R2:g}. <b>Fair</b>: ≥ {F.FAIR_R2:g}. "
                "<b>Poor</b>: lower, or the drift-check train moved by more than half the curve's range. Only Good "
                "and Fair curves go to the network. The small map shows where the pair sits on its organoid "
                "(stimulated electrode in black, recording electrode ringed).</p>")

    # ---- checks
    lost = planned - r["n_trials"]
    art = max((p["artifact"] for p in pairs), default=0.0)
    bad_art = [p for p in pairs if not p["artifact_ok"]]
    sh = D[(D.kind == "sham") & (D.partition == "fit")]
    thr = pd.Series(r["thresholds_uV"]).astype(float)
    checks = [(CHECK_BAD if problems else CHECK_OK,
               "; ".join(problems) if problems else f"All {len(plan['segments'])} recordings complete."),
              (CHECK_OK if lost <= 0.1 * planned else CHECK_WARN,
               f"{r['n_trials']} of {planned} trials usable" + (f"; {lost} lost to missing triggers or dropped data."
                                                                 if lost else ".")),
              (CHECK_OK if not bad_art else CHECK_WARN,
               "Stimulation artifact gone before the counting window on every pair." if not bad_art else
               f"Artifact reaches the counting window on {', '.join(_pair(p['stim'], p['rec']) for p in bad_art)}: "
               f"excluded (largest {art:.1f}× its noise floor, limit {F.ARTIFACT_LIMIT:g}×)."),
              (CHECK_OK, f"Sham correction from {sh.train_id.nunique()} no-current trials; spike thresholds "
                         f"{thr.abs().min():.0f}–{thr.abs().max():.0f} µV (5× noise, 300–3000 Hz).")]
    body.append("<h2>Checks</h2><ul class=checks>"
                + "".join(f"<li>{_icon(i)}<span>{html.escape(t)}</span></li>" for i, t in checks) + "</ul>")

    # ---- evidence per chosen pair
    if chosen:
        body.append("<h2>The chosen pairs</h2><p class=note>Each dot is one trial. Fit trials (grey) chose the pair "
                    "and fitted the black curve; test trials (coloured) came later and only score it. On the right, "
                    "the same train at the start, middle and end of the session: if it moves, the organoid drifted.</p>")
    for role in F.ROLES:
        p = chosen.get(role)
        if not p:
            continue
        v = r["roles"][role]
        body.append(f"<div class=card style='border-left:4px solid {TYPE_COLOUR[role]}'><h3>{_type(role)}: "
                    f"{_pair(p['stim'], p['rec'])} &nbsp; {_pill(v['grade'])}</h3><p>{html.escape(v['reason'][0].upper() + v['reason'][1:])}.</p>"
                    f"<div class=fig>{_role_figure(p, role, D, levels, plan['reference_pulses'])}</div>"
                    f"<details><summary>Fitted parameters (as written to calibration.json)</summary><p class=note>"
                    f"Parameters {', '.join(f'{x:.4g}' for x in v['params'])}. R² of the fit-trial means "
                    f"{_fmt(v.get('r2_means'))}; cross-validated R² within the fit trials {_fmt(v.get('cv_r2'))}; "
                    f"sham average {p['control']:.2f} spikes.</p></details></div>")

    # ---- every pair
    body.append("<h2>Every pair measured</h2><p class=note>One block per stimulating electrode. Left: the organoid "
                "on FinalSpark's electrode map, the stimulated electrode in black, every other electrode coloured by "
                "its strongest response. Right: the same responses at each train length, strongest electrode first. "
                "A bright cell on the left of a row is short-pass, in the middle band-pass, on the right long-pass.</p>"
                + maps)
    trs = []
    for p in sorted(pairs, key=lambda q: (q["stim"], q["rec"])):
        best = p.get("shape")
        f = p["fits"].get(best, {}) if best else {}
        trs.append(f"<tr><td>{_pair(p['stim'], p['rec'])}</td><td>{TYPE.get(best, 'no clear type')}</td>"
                   f"<td>{_pill(p['grade'])}</td><td>{_p_text(p['kw_p'])}</td>"
                   f"<td class=n>{_fmt(p['dev_range'], '.1f')}</td><td class=n>{_fmt(p['trial_sd'], '.1f')}</td>"
                   f"<td class=n>{_fmt(f.get('conf_r2'))}</td><td class=n>{_fmt(f.get('conf_rmse'), '.1f')}</td>"
                   f"<td class=n>{_fmt(f.get('drift'), '+.1f')}</td><td class=n>{p['artifact']:.1f}</td>"
                   f"<td>{html.escape(p['grade_reason'])}</td></tr>")
    body.append("<details><summary>All pairs as a table</summary><p class=note>Range: largest minus smallest fit-trial "
                "mean. Trial SD: typical scatter of single trials at one train length. Drift: last drift-check block "
                "minus the first. Artifact: leftover at the start of the counting window, in units of its noise "
                "floor.</p><div class=wrap><table><tr><th>Pair</th><th>Type</th><th>Verdict</th>"
                "<th>Train-length effect</th><th class=n>Range</th><th class=n>Trial SD</th>"
                "<th class=n>R² test</th><th class=n>Error/trial</th><th class=n>Drift</th><th class=n>Artifact</th>"
                "<th>Why</th></tr>" + "".join(trs) + "</table></div></details>")

    # ---- reference material, folded
    body.append("<details><summary>Terms used in this report</summary>" + _terms(TERMS_COMMON + [
        ("Fit and test trials", "Every train length is measured twice. The first pass (fit trials) picks the pairs "
                                "and fits the curves. The second pass (test trials), later and reshuffled, only "
                                "scores the frozen curves."),
        ("Drift check", f"The same {plan['reference_pulses']}-pulse train at the start, middle and end of the session."),
        ("R² on test trials", "The share of the trial-to-trial variation in test trials that the curve explains: 1 = "
                              "every trial on the curve, 0 = no better than the average."),
        ("Error per trial", "How far a single test trial typically falls from the curve (root mean square)."),
        ("p", "Chance of differences across train lengths this large if length had no effect (Kruskal–Wallis on "
              f"fit trials). A pair needs p < {F.MIN_EFFECT_P:g}."),
    ]) + "</details>")
    body.append("<details><summary>How this was measured</summary><ul>"
                "<li>Each train comes from one trigger; the stimulator repeats the pulse, as the network delivers it. "
                "A train's length is part of the uploaded settings, so each length has its own recording.</li>"
                "<li>Order: drift check, fit pass (every length, shuffled), drift check, test pass (reshuffled), "
                "drift check. Trials on one organoid ≥ 4 s apart; on different organoids ≥ 1 s apart.</li>"
                "<li>Spike counting: 300–3000 Hz band-pass; the train and 4 ms after it are blanked; a spike is a "
                "crossing of −5× noise, ≥ 1 ms after the previous one.</li></ul></details>")
    if r.get("calibration") and len(found) < 3:
        body.append("<p class=note>To look for a missing type, re-run this step with other stimulating electrodes "
                    "(<code>--stims</code>) or more current (<code>--amplitude</code>).</p>")
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
    body = [f"<h1>{html.escape(fs)}: screen with pulse trains</h1>",
            f"<p class=meta>{html.escape(when)}{' (SIMULATED)' if sim else ''}; every electrode of the "
            f"organoid(s) on {', '.join(_organoid(8 * o) for o in organoids)}; trains of "
            f"{', '.join(map(str, levels))} pulses, {plan['ipi_ms']:g} ms apart, {plan['amplitude_uA']:g} µA, "
            f"{plan['phase_us']:g} µs per phase; {r['n_trials']} of {planned} trials usable</p>",
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
                        f"<td class=n>{v['diff']:.1f}</td><td class=n>{', '.join(f'{m:.1f}' for m in p['means'])}</td>"
                        f"<td class=n>{v['n_candidates'] - 1}</td></tr>")
        else:
            rows.append(f"<tr><td>{name}</td><td>{_icon(CHECK_BAD)}none</td><td></td><td></td><td></td><td></td></tr>")
    body.append("<div class=wrap><table><tr><th>Response type</th><th>Best pair</th><th class=n>Single trial wins</th>"
                "<th class=n>Difference (spikes)</th>"
                f"<th class=n>Mean evoked spikes at {', '.join(map(str, levels))} pulses</th>"
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
        f"<td class=n>{', '.join(f'{m:.1f}' for m in p['means'])}</td>"
        + "".join(f"<td class=n>{'<b>' if k in p['shapes'] else ''}{p['scores'][k]['win']:.0%}"
                  f"{'</b>' if k in p['shapes'] else ''}</td>" if k in p["scores"] else "<td></td>" for k in F.ROLES)
        + f"<td class=n>{p['artifact']:.1f}</td></tr>" for p in sorted(r["pairs"], key=lambda q: (q["stim"], q["rec"])))
    body.append("<details><summary>All pairs as a table</summary><p class=note>Single trial wins for each type. "
                "Bold means the pair is a candidate for that type. Artifact: the leftover at the start of the "
                "counting window, in units of its noise floor.</p><div class=wrap><table><tr><th>Pair</th>"
                f"<th>Candidate for</th><th class=n>Mean evoked spikes at {', '.join(map(str, levels))}</th>"
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
        fh.write(_page(f"{fs}: screen with pulse trains, {when}", body))
