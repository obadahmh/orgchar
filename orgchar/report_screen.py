"""The screen report, written for readers outside the project: the decision first, then the terms it
uses, then the evidence. Electrode panels follow FinalSpark's electrode map. Figures are SVG, so they
stay sharp at any zoom."""
from __future__ import annotations

import html

import matplotlib.pyplot as plt
import numpy as np

from . import screen as SC
from .report import BLUE, CHECK_BAD, CHECK_OK, CHECK_WARN, GREY, INK, ORANGE, _icon, _page, _svg

TYPE = dict(short="short-pass", band="band-pass", long="long-pass")
TYPE_DESC = dict(short="fewer evoked spikes after longer trains", band="most evoked spikes at a middle train length",
                 long="more evoked spikes after longer trains")
TYPE_COLOUR = dict(short=ORANGE, band="#1baf7a", long=BLUE)
# FinalSpark's electrode map: each organoid is a cross of eight electrodes over three rows.
# Key: electrode number within the organoid (0-7). Value: (column, row) in the cross.
LAYOUT = {0: (1, 0), 7: (2, 0), 1: (0, 1), 2: (1, 1), 5: (2, 1), 6: (3, 1), 3: (1, 2), 4: (2, 2)}


def _jitter(n, width, seed):
    return np.random.default_rng(seed).uniform(-width, width, n)


def _pair(s, c):
    return f"e{s} → e{c}"


def _type(role):
    return f"{SC.SHAPE_SYMBOL[role]} {TYPE[role]}"


def _organoid(e):
    return f"e{8 * (e // 8)}–e{8 * (e // 8) + 7}"


def _terms(items):
    return "<dl class=terms>" + "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in items) + "</dl>"


TERMS_COMMON = [
    ("e8 → e15", "Stimulate electrode 8 and count spikes on electrode 15. Electrode numbers are FinalSpark's. "
                 "Each organoid sits on eight electrodes: e0–e7, e8–e15, e16–e23 and e24–e31."),
    ("Train", "A series of biphasic pulses, 10 ms apart. The stimulator generates the whole train from one trigger."),
    ("Trial", "One train on one stimulating electrode, then counting spikes on the other seven electrodes of the "
              "same organoid. Trials on the same organoid are at least 4 s apart."),
    ("Evoked spikes", "Spikes counted 10 to 210 ms after the end of the train. From this we subtract the spontaneous "
                      "count in the 500 ms before the train (scaled to the same length)."),
    ("Response types", f"<b>{TYPE['short']}</b>: {TYPE_DESC['short']}. <b>{TYPE['band']}</b>: {TYPE_DESC['band']}. "
                       f"<b>{TYPE['long']}</b>: {TYPE_DESC['long']}."),
]


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
                             f"background:{TYPE_COLOUR[k]};vertical-align:-1px'></span> {TYPE[k]}" for k in SC.ROLES)
           + " &nbsp; <span style='display:inline-block;width:12px;height:12px;border-radius:2px;background:#fff;"
           "border:1px solid #c9c8c2;vertical-align:-2px'></span> no clear type")
    return f"<p class=note>{key}</p><div class=maps>{''.join(out)}</div>"


def _candidate_figure(cands, responses, levels):
    n = len(cands)
    ncol = min(4, n)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.0 * nrow), squeeze=False,
                             gridspec_kw=dict(hspace=0.85, wspace=0.4))
    for k, (role, p) in enumerate(cands):
        ax = axes[k // ncol, k % ncol]
        g = responses[(responses.stim == p["stim"]) & (responses.rec == p["rec"])]
        y = g.response
        xs = np.array([levels.index(v) for v in g.pulses])
        ax.scatter(xs + _jitter(len(xs), 0.12, k), y, s=10, color=TYPE_COLOUR[role], alpha=0.7, linewidths=0)
        ax.plot(range(len(levels)), p["means"], color=INK, lw=1.8, marker="o", ms=3.5)
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.tick_params(labelsize=7)
        ax.set_title(f"e{p['stim']} → e{p['rec']}  {TYPE[role]}\n"
                     f"explains {p['explained']:.0%}, {p['scores'][role]['diff']:.1f} spikes apart",
                     fontsize=7.5, loc="left")
        if k % ncol == 0:
            ax.set_ylabel("evoked spikes per trial", fontsize=7.5)
        if k // ncol == nrow - 1:
            ax.set_xlabel("pulses in the train", fontsize=7.5)
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    return _svg(fig)


def screen_page(path, plan, r, when):
    levels = list(plan["levels"])
    fs = plan["fsname"]
    sim = plan.get("backend") == "simulated"
    best, stims = r["best"], plan["stims"]
    planned = r["n_requests"]
    organoids = sorted({e // 8 for e in stims})
    body = [f"<h1>{html.escape(fs)}: screen with pulse trains</h1>",
            f"<p class=meta>{html.escape(when)}{' (SIMULATED)' if sim else ''}; every electrode of the "
            f"organoid(s) on {', '.join(_organoid(8 * o) for o in organoids)}; trains of "
            f"{', '.join(map(str, levels))} pulses, {plan['ipi_ms']:g} ms apart, {plan['amplitude_uA']:g} µA, "
            f"{plan['phase_us']:g} µs per phase; {r['n_trials']} of {planned} trains usable</p>",
            f"<p>This step is a quick scan. It stimulates every electrode with {len(levels)} train lengths and looks "
            "for pairs that might show one of the three response types. The solve step then measures the steadiest "
            "pair's curve in full, trains the network on it and runs it on the tissue.</p>"]
    top = SC.steadiest(best) if best else None
    if top:
        lede = (f"Next, solve measures {_pair(top['stim'], top['rec'])}, the steadiest pair. Candidates found for "
                + ", ".join(TYPE[k] for k in SC.ROLES if k in best) + "."
                + ("" if len(best) == 3 else " None for " + ", ".join(TYPE[k] for k in SC.ROLES if k not in best) + "."))
    else:
        lede = "No electrode gives a short-pass, band-pass or long-pass response."
    if r["verdict"] == "CONTROLS_FAILED":
        lede = "Controls failed: too many pairs across organoids pass. Do not trust the candidates below."
    body.append(f"<p class=lede>{html.escape(lede)}</p>")
    rows = []
    for role in SC.ROLES:
        v = best.get(role)
        name = f"<b>{_type(role)}</b><br><span class=note>{TYPE_DESC[role]}</span>"
        if v:
            p = next(q for q in r["pairs"] if q["stim"] == v["stim"] and q["rec"] == v["rec"])
            rows.append(f"<tr><td>{name}</td><td>{_pair(v['stim'], v['rec'])}</td><td class=n>{v['win']:.0%}</td>"
                        f"<td class=n>{v['explained']:.0%}</td><td class=n>{v['diff']:.1f}</td><td class=n>{', '.join(f'{m:.1f}' for m in p['means'])}</td>"
                        f"<td class=n>{v['n_candidates'] - 1}</td></tr>")
        else:
            rows.append(f"<tr><td>{name}</td><td>{_icon(CHECK_BAD)}none</td><td></td><td></td><td></td><td></td><td></td></tr>")
    body.append("<div class=wrap><table><tr><th>Response type</th><th>Best pair</th><th class=n>Single trial wins</th>"
                "<th class=n>Train length explains</th><th class=n>Difference (spikes)</th>"
                f"<th class=n>Mean evoked spikes at {', '.join(map(str, levels))} pulses</th>"
                "<th class=n>Other candidates</th></tr>" + "".join(rows) + "</table></div>")

    body.append("<h2>Terms used in this report</h2>" + _terms(TERMS_COMMON + [
        ("Single trial wins", "Take one trial at the train length that should give more spikes (2 pulses for "
                              "short-pass, 20 for long-pass, the middle lengths for band-pass) and one at the length "
                              "that should give fewer. This is how often the first has more spikes. 50% means no "
                              "difference. 100% means always."),
        ("Train length explains", "The share of a pair's trial-to-trial variation that the train length accounts "
                                  "for (omega squared). The rest is noise the network cannot use. The best pair of "
                                  "each type is the candidate with the highest share; solve takes the highest of those."),
        ("Candidate", f"A pair whose single trial wins clearly (one-sided rank test, p below {SC.SCOUT_MAX_P:g}) and "
                      f"whose means differ by at least {SC.SCOUT_MIN_EVENTS:g} spikes. A pair is dropped if a pair "
                      "from another organoid does as well on the same recording electrode."),
        ("p", "The probability of a difference at least this large if train length had no effect."),
        ("Control pairs", "A stimulating electrode paired with recording electrodes on another organoid. A train "
                          "cannot reach another organoid through tissue, so these pairs should never pass."),
    ]))

    c = r["controls"]
    body.append("<h2>Checks</h2><ul class=checks>")
    for w in r["warnings"]:
        body.append(f"<li>{_icon(CHECK_WARN)}<span>{html.escape(w[0].upper() + w[1:])}.</span></li>")
    lost = planned - r["n_trials"]
    body.append(f"<li>{_icon(CHECK_OK if lost <= 0.1 * planned else CHECK_WARN)}<span>"
                + (f"{lost} of {planned} trains could not be used, even when requested again: their trigger was not "
                   "found, or a gap in the raw data cut into their counting window." if lost else
                   f"All {planned} trains usable.") + "</span></li>")
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
    body.append("</ul>")

    body.append("<h2>Every electrode pair</h2><p class=note>The large layout is FinalSpark's electrode map. At each "
                "electrode sits a small copy of its organoid's map, showing what stimulating that electrode did. "
                "Each square in the small map is a recording electrode, coloured by the response type it gave. "
                "Hover a square for the numbers.</p>" + _shape_maps(r["pairs"], stims))
    cands = []
    for role in SC.ROLES:
        cands += [(role, p) for p in sorted([q for q in r["pairs"] if role in q["shapes"]],
                                            key=lambda q: -q["explained"])[:4]]
    if cands:
        body.append("<h2>The candidates</h2><p class=note>Up to four per response type, best first. Dots are "
                    "single trials. The line joins the means.</p>"
                    f"<div class=fig>{_candidate_figure(cands, r['_responses'], levels)}</div>")
    trs = "".join(
        f"<tr><td>{_pair(p['stim'], p['rec'])}</td><td>{', '.join(TYPE[k] for k in p['shapes']) or '–'}</td>"
        f"<td class=n>{', '.join(f'{m:.1f}' for m in p['means'])}</td>"
        + "".join(f"<td class=n>{'<b>' if k in p['shapes'] else ''}{p['scores'][k]['win']:.0%}"
                  f"{'</b>' if k in p['shapes'] else ''}</td>" if k in p["scores"] else "<td></td>" for k in SC.ROLES)
        + f"<td class=n>{p['artifact']:.1f}</td></tr>" for p in sorted(r["pairs"], key=lambda q: (q["stim"], q["rec"])))
    body.append("<details><summary>All pairs as a table</summary><p class=note>Single trial wins for each type. "
                "Bold means the pair is a candidate for that type. Artifact: the leftover at the start of the "
                "counting window, in units of its noise floor.</p><div class=wrap><table><tr><th>Pair</th>"
                f"<th>Candidate for</th><th class=n>Mean evoked spikes at {', '.join(map(str, levels))}</th>"
                "<th class=n>Short-pass wins</th><th class=n>Band-pass wins</th><th class=n>Long-pass wins</th>"
                "<th class=n>Artifact</th></tr>" + trs + "</table></div></details>")

    body.append("<h2>For the operator</h2>")
    if top:
        body.append(f"<pre class=cmd>./orgchar.sh solve --fsname {html.escape(fs)} --dry-run</pre><p class=note>This "
                    f"is the same as <code>--pairs {top['stim']}:{top['rec']}</code>; the dry run prints the plan and "
                    "its duration.</p>")
    else:
        body.append("<p>Nothing to measure further with these trains. Try more current (<code>--amplitude 3</code>), "
                    "or re-check the census, because the organoids may be quiet today.</p>")
    with open(path, "w") as fh:
        fh.write(_page(f"{fs}: screen with pulse trains, {when}", body))
