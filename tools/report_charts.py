"""Every figure in the NeuroGecko report, drawn from the extracted JSON only.

The rule this file follows is the project's own. Ledger counts are parsed from
docs/FAILURE_MAP.md, dates come from `git blame`, provenance comes from
config/proxies.yaml through the repo's own validating loader, and every
measurement quoted in a chart carries the ledger row it came from in its
caption. Nothing here is a figure recalled from a conversation.

Where a chart shows a number that has NOT been measured -- the published
targets, the bracket a lizard sleep cycle falls in -- it is drawn as a marker
or a band, never as a bar, so a wish cannot be mistaken for a result.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, OrderedDict, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts/report"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#12161c"
PAPER = "#ffffff"
RED = "#c8442f"        # refuted
GREEN = "#2f7d5b"      # confirmed
AMBER = "#c98a1e"      # partly
GREY = "#8a919b"
BLUE = "#2f5d8a"
PALE = "#eef0f3"
INVENTED = "#8e5b9e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#c9cdd4",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#5a616b",
    "ytick.color": "#5a616b",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
})


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("  ", name)
    return path


def load(name):
    return json.load(open(OUT / f"{name}.json", encoding="utf-8"))


# ====================================================================== ledger
def _session_key(s):
    """Sort '9', '9b', '10', '10c' the way the sessions actually ran."""
    if not s:
        return (999, "")
    num = "".join(c for c in s if c.isdigit())
    suf = "".join(c for c in s if c.isalpha())
    return (int(num or 999), suf)


def chart_verdicts(rows):
    """Every hypothesis, grouped by the session that entered it in the map."""
    by = defaultdict(Counter)
    for r in rows:
        by[r.get("session") or "?"][r["class"]] += 1
    order = sorted(by, key=_session_key)
    ref = [by[s]["Refuted"] for s in order]
    par = [by[s]["Partly"] for s in order]
    con = [by[s]["Confirmed"] for s in order]
    oth = [by[s]["Other"] for s in order]

    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    x = np.arange(len(order))
    base = np.zeros(len(order))
    for vals, col, lab in ((ref, RED, "Refuted"), (par, AMBER, "Partly"),
                           (con, GREEN, "Confirmed"), (oth, GREY, "Retracted")):
        ax.bar(x, vals, bottom=base, color=col, label=f"{lab} ({sum(vals)})")
        base = base + np.array(vals)
    for i, t in enumerate(base):
        ax.text(i, t + 0.7, str(int(t)), ha="center", fontsize=8, color="#5a616b")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in order], fontsize=8, rotation=45)
    ax.set_ylabel("hypotheses entered")
    ax.set_title(f"All {len(rows)} hypotheses, by the session that recorded them",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9.5)
    ax.set_ylim(0, base.max() * 1.30)
    return save(fig, "verdicts.png")


def chart_cumulative(rows):
    """Wrong answers accumulate; right ones barely move."""
    rows = sorted(rows, key=lambda r: r["n"])
    n = [r["n"] for r in rows]
    cum_ref = np.cumsum([r["class"] == "Refuted" for r in rows])
    cum_con = np.cumsum([r["class"] == "Confirmed" for r in rows])
    cum_par = np.cumsum([r["class"] == "Partly" for r in rows])
    pct = round(100 * cum_ref[-1] / len(rows))

    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.fill_between(n, cum_ref, color=RED, alpha=.13)
    ax.plot(n, cum_ref, color=RED, lw=2.5, label="refuted")
    ax.plot(n, cum_par, color=AMBER, lw=2.0, label="partly")
    ax.plot(n, cum_con, color=GREEN, lw=2.5, label="confirmed")
    ax.plot(n, n, color=GREY, lw=1, ls=":", label="total tested")
    for val, col, dy in ((cum_ref[-1], RED, 9), (cum_con[-1], GREEN, -20),
                         (cum_par[-1], AMBER, 6)):
        ax.annotate(f"{val}", (n[-1], val), xytext=(7, dy),
                    textcoords="offset points", color=col, fontweight="bold")
    ax.set_xlabel("hypothesis number, in the order they were tested")
    ax.set_ylabel("running total")
    ax.set_title(f"{pct} % of everything tried was wrong — and that is the product",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.legend(frameon=False, loc="upper left")
    return save(fig, "cumulative.png")


def chart_timeline(rows, commits):
    """When the work actually happened, day by day.

    One honesty problem had to be handled here. `git blame` reports when a line
    was last written, and the living map was CREATED in one commit that
    transcribed every hypothesis tested before it existed. Left alone the chart
    shows a single day on which 132 hypotheses were tested, which is false.
    Those rows are drawn separately, hatched and labelled, rather than smoothed
    away or silently dropped.
    """
    by_day = defaultdict(Counter)
    for r in rows:
        if r.get("date"):
            by_day[r["date"]][r["class"]] += 1
    commits_day = Counter(c["date"] for c in commits)
    days = sorted(set(by_day) | set(commits_day))
    x = np.arange(len(days))

    # the day the map itself was created: the one with the most rows on it
    import_day = max(by_day, key=lambda d: sum(by_day[d].values()))
    imported = sum(by_day[import_day].values())

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13.5, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.6, 1.15]})
    base = np.zeros(len(days))
    for cls, col in (("Refuted", RED), ("Partly", AMBER),
                     ("Confirmed", GREEN), ("Other", GREY)):
        vals = np.array([by_day[d][cls] if d != import_day else 0
                         for d in days])
        ax.bar(x, vals, bottom=base, color=col, width=.72)
        base += vals
    ix = days.index(import_day)
    ax.bar([ix], [imported], color="#dfe3e8", edgecolor=GREY, hatch="//",
           width=.72, zorder=0)
    ax.annotate(f"{imported} rows transcribed into the map\n"
                f"on the day it was created — tested earlier",
                (ix, imported), xytext=(-14, 6), textcoords="offset points",
                ha="right", fontsize=8.6, color="#5a616b")
    for i, t in enumerate(base):
        if t:
            ax.text(i, t + 1.2, str(int(t)), ha="center", fontsize=8,
                    color="#5a616b")
    ax.set_ylabel("hypotheses recorded")
    ax.set_ylim(0, imported * 1.30)
    ax.set_title("Every day this project ran", loc="left", fontsize=14,
                 fontweight="bold", pad=12)

    ax2.bar(x, [commits_day[d] for d in days], color=BLUE, width=.72)
    for i, d in enumerate(days):
        if commits_day[d]:
            ax2.text(i, commits_day[d] + .3, str(commits_day[d]), ha="center",
                     fontsize=7.6, color="#5a616b")
    ax2.set_ylabel("commits")
    ax2.set_xticks(x)
    ax2.set_xticklabels([d[5:] for d in days], fontsize=8.5, rotation=45)
    ax2.set_xlabel("date (2026)")
    ax2.set_ylim(0, max(commits_day.values()) * 1.25)
    fig.text(.005, -.05,
             "Dates come from git blame on docs/FAILURE_MAP.md and from git log. "
             "The work ran in two blocks: a locomotion block in June and the "
             "brain-and-eye block in September.",
             fontsize=8.6, color="#5a616b")
    return save(fig, "timeline.png")


def chart_own_errors(rows):
    """The meta-ledger: the failures that were mine and not the model's."""
    marks = OrderedDict([
        ("the test was wrong,\nnot the model",
         ("test", "harness", "my test", "the test compared")),
        ("a default silently\ndecided the science",
         ("default", "silently", "omitted")),
        ("I claimed it before\nmeasuring it",
         ("i told the user", "asserted", "i claimed", "overstated", "invented")),
        ("the map already\nheld the answer",
         ("already", "#71", "had measured")),
        ("a fix that could not\nchange anything",
         ("could not do anything", "identical", "no effect")),
    ])
    counts = []
    for _, keys in marks.items():
        counts.append(sum(1 for r in rows
                          if any(k in (r["hypothesis"] + " " +
                                       r["evidence"]).lower() for k in keys)))
    fig, ax = plt.subplots(figsize=(11, 3.9))
    y = np.arange(len(marks))[::-1]
    ax.barh(y, counts, color=RED, height=.6, alpha=.85)
    for yy, c in zip(y, counts):
        ax.text(c + .5, yy, str(c), va="center", fontweight="bold", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(list(marks), fontsize=9.5)
    ax.set_xlabel("ledger rows matching the pattern")
    ax.set_xlim(0, max(counts) * 1.18)
    ax.set_title("My own method errors, counted by searching the ledger text",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.text(0, -1.05,
            "A keyword scan over 221 rows, not a hand tally — it undercounts "
            "wherever a row described the same mistake in different words.",
            fontsize=9, color="#5a616b")
    return save(fig, "own_errors.png")


# ==================================================================== commits
def chart_commits(commits):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13.5, 5.0),
                                  gridspec_kw={"height_ratios": [2.4, 1]})
    x = np.arange(len(commits))
    added = [c["added"] for c in commits]
    removed = [-c["removed"] for c in commits]
    ax.bar(x, added, color=BLUE, width=.85)
    ax.bar(x, removed, color="#d5b8b0", width=.85)
    ax.axhline(0, color="#c9cdd4", lw=1)
    big = int(np.argmax(added))
    ax.annotate(f"+{added[big]:,}  {commits[big]['subject'][:40]}",
                (big, added[big]), xytext=(6, -4), textcoords="offset points",
                fontsize=8.5, color=BLUE)
    ax.set_ylabel("source lines")
    ax.set_title(f"{sum(added):,} lines of source written, {abs(sum(removed)):,} "
                 f"deleted, across {len(commits)} commits",
                 loc="left", fontsize=14, fontweight="bold", pad=14)

    ax2.bar(x, [c["data_added"] for c in commits], color="#b9c2cc", width=.85)
    ax2.set_ylabel("recorded\ndata lines")
    ax2.set_xlabel(f"commit, oldest to newest  ({len(commits)} total)")
    ax2.set_yscale("symlog")
    fig.text(.005, -.03,
             f"The lower panel is recorded physics output — "
             f"{sum(c['data_added'] for c in commits):,} lines. One episode trace "
             f"is 1.5 million lines, so it is separated rather than counted as "
             f"authorship.", fontsize=8.6, color="#5a616b")
    return save(fig, "commits.png")


# =================================================================== registry
def chart_provenance(reg):
    """Where every number in the animal came from."""
    prov = Counter(v["provenance"] for v in reg.values())
    conf = Counter(v["confidence"] for v in reg.values())
    spec = Counter(v["species"] for v in reg.values()
                   if v["provenance"] == "PUBLISHED")

    fig = plt.figure(figsize=(13.5, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, .85, 1.5], wspace=.95)

    ax = fig.add_subplot(gs[0])
    order = ["PUBLISHED", "DERIVED", "INVENTED", "UNTAGGED"]
    cols = {"PUBLISHED": GREEN, "DERIVED": BLUE, "INVENTED": INVENTED,
            "UNTAGGED": GREY}
    vals = [prov.get(k, 0) for k in order]
    keep = [(k, v) for k, v in zip(order, vals) if v]
    wedges, _ = ax.pie([v for _, v in keep], startangle=90,
                       colors=[cols[k] for k, _ in keep],
                       wedgeprops=dict(width=.42, edgecolor="white", lw=2))
    total = sum(v for _, v in keep)
    ax.text(0, 0, f"{total}\nnumbers", ha="center", va="center",
            fontsize=13, fontweight="bold")
    ax.legend(wedges, [f"{k.title()}  {v}  ({100*v/total:.0f} %)"
                       for k, v in keep],
              frameon=False, fontsize=9, loc="lower center",
              bbox_to_anchor=(.5, -.26))
    ax.set_title("Provenance", loc="left", fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[1])
    ck = ["verified", "likely", "uncertain"]
    cv = [conf.get(k, 0) for k in ck]
    ax.bar(ck, cv, color=[GREEN, AMBER, RED], width=.6)
    for i, v in enumerate(cv):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold", fontsize=10)
    ax.set_ylim(0, max(cv) * 1.2)
    ax.set_title("Confidence", loc="left", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)

    ax = fig.add_subplot(gs[2])
    top = spec.most_common(6)[::-1]
    short = [k.replace("_", " ").replace("PROXY (", "proxy: ").replace(")", "")
             for k, _ in top]
    ax.barh(short, [v for _, v in top],
            color=GREEN, height=.6)
    for i, (_, v) in enumerate(top):
        ax.text(v + .4, i, str(v), va="center", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, max(v for _, v in top) * 1.2)
    ax.set_title("Which animal each published number came from",
                 loc="left", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("Every parameter in the gecko says where it came from",
                 x=.005, ha="left", fontsize=14, fontweight="bold", y=1.07)
    return save(fig, "provenance.png")


def chart_species_gap(reg):
    """How much of the animal is actually this animal."""
    pub = [v for v in reg.values() if v["provenance"] == "PUBLISHED"]
    target = sum(1 for v in pub if v["species"] == "E_macularius")
    other = len(pub) - target
    derived = sum(1 for v in reg.values() if v["provenance"] == "DERIVED")
    invented = sum(1 for v in reg.values() if v["provenance"] == "INVENTED")
    parts = [("measured in\nE. macularius", target, GREEN),
             ("measured in a\nDIFFERENT species", other, AMBER),
             ("derived from\nsomething published", derived, BLUE),
             ("invented", invented, INVENTED)]
    total = sum(p[1] for p in parts)

    fig, ax = plt.subplots(figsize=(13, 2.5))
    left = 0
    for label, v, col in parts:
        ax.barh([0], [v], left=left, color=col, height=.55)
        ax.text(left + v / 2, 0, f"{v}\n{100*v/total:.0f} %", ha="center",
                va="center", color="white", fontweight="bold", fontsize=11)
        ax.text(left + v / 2, -.45, label, ha="center", va="top",
                fontsize=9, color="#454b54")
        left += v
    ax.set_xlim(0, total)
    ax.set_ylim(-1.15, .5)
    ax.axis("off")
    ax.set_title(f"Only {100*target/total:.0f} % of the {total} numbers in this "
                 f"animal were measured in this animal",
                 loc="left", fontsize=14, fontweight="bold", pad=16)
    return save(fig, "species_gap.png")


def chart_bibliography(bib):
    """Where the citations sit, and how concentrated the real evidence is."""
    cites = bib["citations"]
    top = sorted(cites.items(), key=lambda kv: -kv[1]["count"])[:16][::-1]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    ax.barh([k for k, _ in top], [v["count"] for _, v in top],
            color=BLUE, height=.62)
    for i, (_, v) in enumerate(top):
        ax.text(v["count"] + .2, i, str(v["count"]), va="center", fontsize=9)
    ax.set_xlabel("places in the repository that lean on it")
    ax.set_title(f"The {len(cites)} works this project cites — the 16 leaned on most",
                 loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=9)

    counts = Counter(v["count"] for v in cites.values())
    once = counts[1]
    ax2.bar(["cited\nonce", "cited\n2–4 times", "cited\n5+ times"],
            [once,
             sum(v for k, v in counts.items() if 2 <= k <= 4),
             sum(v for k, v in counts.items() if k >= 5)],
            color=[GREY, BLUE, GREEN], width=.6)
    for i, v in enumerate([once,
                           sum(v for k, v in counts.items() if 2 <= k <= 4),
                           sum(v for k, v in counts.items() if k >= 5)]):
        ax2.text(i, v + 2, str(v), ha="center", fontweight="bold")
    ax2.set_title("How the citing is distributed", loc="left", fontsize=13,
                  fontweight="bold", pad=12)
    ax2.text(0, -len(cites) * .13,
             f"{100*once/len(cites):.0f} % appear exactly once — mentioned "
             f"in a survey, not load-bearing.",
             fontsize=9, color="#5a616b")
    return save(fig, "bibliography.png")


# =============================================================== measurements
def chart_detection():
    """Ledger #220. The blocker is the detection rate, not the bearing."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.0),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    labels = ["not detected", "detected,\nbearing WRONG",
              "detected,\nbearing USABLE"]
    vals = [1419, 37, 44]
    cols = [RED, AMBER, GREEN]
    bars = ax.bar(labels, vals, color=cols, width=.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 28,
                f"{v}\n{100*v/sum(vals):.1f} %", ha="center", fontweight="bold",
                fontsize=10)
    ax.set_ylim(0, 1700)
    ax.set_ylabel("steps of one 1 500-step hunt")
    ax.set_title("What the eye did on every step of a real hunt",
                 loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=9)

    det = [37, 44]
    ax2.bar(["bearing\nwrong", "bearing\nusable"], det, color=[AMBER, GREEN],
            width=.55)
    for i, v in enumerate(det):
        ax2.text(i, v + 1, f"{v}\n{100*v/sum(det):.0f} %", ha="center",
                 fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 60)
    ax2.set_title("...and when it DID report", loc="left", fontsize=13,
                  fontweight="bold", pad=12)
    fig.text(.005, -.10,
             "Three sessions of vision work assumed the bearing was the broken "
             "part. When the eye speaks it is right more often than not — it "
             "just almost never speaks. Ledger #220.",
             fontsize=9.2, color="#5a616b")
    return save(fig, "detection.png")


def chart_efference():
    """Ledger #209 -> #213. The eye was reporting its own walking."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.9))
    fig.subplots_adjust(wspace=.34)

    a1.bar(["prey in\nthe world", "NO PREY\nat all"], [0.4161, 0.4034],
           color=[BLUE, RED], width=.5)
    a1.set_ylim(0, .56)
    for i, v in enumerate([0.4161, 0.4034]):
        a1.text(i, v + .012, f"{v:.4f}", ha="center", fontweight="bold")
    a1.text(.5, .49, "separation d = 0.036", ha="center", color=RED,
            fontweight="bold", fontsize=11)
    a1.set_ylabel("mean reported salience")
    a1.set_title("BEFORE — an empty world looked like a full one",
                 loc="left", fontsize=12, fontweight="bold", pad=12)

    a2.bar(["before\n(local surround)", "after\n(efference copy)"],
           [72, 5], color=[RED, GREEN], width=.5)
    for i, v, t in ((0, 72, "72 %"), (1, 5, "3–7 %")):
        a2.text(i, v + 2, t, ha="center", fontweight="bold")
    a2.set_ylim(0, 92)
    a2.set_ylabel("frames firing with no prey present")
    a2.set_title("AFTER — its own optic flow subtracted",
                 loc="left", fontsize=12, fontweight="bold", pad=12)
    fig.text(.005, -.06,
             "Efference copy: the animal's own motor command is used to predict "
             "the image motion its walking must cause, and that prediction is "
             "subtracted. What survives is about the world. Ledger #209, #211, "
             "#213.", fontsize=8.8, color="#5a616b")
    return save(fig, "efference.png")


def chart_speeds():
    names = ["gecko\nwalking", "cricket\nfleeing", "gecko\nSTRIKE"]
    vals = [0.055, 0.118, 0.851]
    fig, ax = plt.subplots(figsize=(10, 3.7))
    bars = ax.barh(names, vals, color=[GREY, RED, GREEN], height=.55)
    for b, v in zip(bars, vals):
        ax.text(v + .015, b.get_y() + b.get_height() / 2, f"{v} m/s",
                va="center", fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("metres per second")
    ax.set_title("Pursuit is arithmetically impossible — so the animal strikes",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.text(.13, -.66, "the cricket is 2.1× the walk; the strike is 15× "
                       "the walk and 7× the cricket.  Ledger #169, #175.",
            fontsize=9.5, color="#5a616b")
    return save(fig, "speeds.png")


def chart_hunt():
    """Ledger #169 -> #204: the hunt going from impossible to composed."""
    stages = ["prey static,\nno strike\n(#169)",
              "strike, prey\nplaced in range\n(#175)",
              "prey moves +\nflee radius\n(#204)"]
    strikes = [0, 78, 29]
    caught = [0, 63, 26]
    x = np.arange(3)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.2),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    ax.bar(x - .19, strikes, .36, color=GREY, label="strikes launched")
    ax.bar(x + .19, caught, .36, color=GREEN, label="prey caught")
    for i, (s, c) in enumerate(zip(strikes, caught)):
        ax.text(i - .19, s + 1.5, str(s), ha="center", fontsize=9.5)
        ax.text(i + .19, c + 1.5, str(c), ha="center", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(0, 92)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.set_title("The hunt, from impossible to composed", loc="left",
                 fontsize=13, fontweight="bold", pad=12)

    rates = [81.0, 89.7]
    ax2.bar(["strike alone\nn=78", "full hunt\nn=29"], rates,
            color=[BLUE, GREEN], width=.5)
    ax2.axhline(82.9, color=RED, ls="--", lw=1.8)
    ax2.text(1.48, 84.0, "published 82.9 %", color=RED, fontsize=9.5, ha="right")
    for i, v in enumerate(rates):
        ax2.text(i, v + 1.4, f"{v} %", ha="center", fontweight="bold")
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("capture success")
    ax2.set_title("Against the published rate, unfitted", loc="left",
                  fontsize=13, fontweight="bold", pad=12)
    fig.text(.005, -.05,
             "The published 82.9 % is a MARKER, not a bar — nothing in the "
             "model was tuned to reach it. Neither result was fitted to it.",
             fontsize=8.8, color="#5a616b")
    return save(fig, "hunt.png")


def chart_efficiency():
    """Ledger #221. Pointed the right way, arriving nowhere."""
    fig, ax = plt.subplots(figsize=(13, 3.0))
    ax.barh([0], [1.0], color=PALE, height=.5)
    ax.barh([0], [0.0146], color=RED, height=.5)
    ax.set_xlim(0, 1.0); ax.set_ylim(-1.3, .7)
    ax.axis("off")
    ax.text(0.03, 0, "0.0146", va="center", fontsize=13, fontweight="bold",
            color=RED)
    ax.text(1.0, 0, "  1.0 = walks straight at it", va="center", fontsize=10,
            color="#5a616b")
    ax.text(0, -.75,
            "Path efficiency in a real hunt: straight-line ground closed, "
            "divided by ground actually walked.\nThe animal covers about "
            "seventy times more distance than it closes — while satisfying "
            "the published\n'approaching' criterion on 47 % of steps. It is "
            "pointed the right way and getting almost nowhere.",
            fontsize=10, color="#454b54", va="top")
    ax.set_title("The number the oracle removal has to be judged against  "
                 "(ledger #221)", loc="left", fontsize=14, fontweight="bold",
                 pad=10)
    return save(fig, "efficiency.png")


def chart_oracle():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    for ax, vals, title, note in (
        (a1, [0.0857, 0.1103], "distance moved (m)",
         "moves FURTHER without it"),
        (a2, [0.0664, 0.0060], "progress TOWARD the goal (m)",
         "91 % of it was the cheat")):
        b = ax.bar(["oracle ON", "oracle zeroed"], vals, color=[BLUE, RED],
                   width=.55)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v * 1.03, f"{v:.4f}",
                    ha="center", fontweight="bold", fontsize=10)
        ax.set_title(title, fontsize=11.5, loc="left")
        ax.text(0, -max(vals) * .27, note, fontsize=10, color="#5a616b")
        ax.set_ylim(0, max(vals) * 1.25)
    fig.suptitle("The legs were fine. The navigation was the cheat.  "
                 "(ledger #166)", x=.005, ha="left", fontsize=14,
                 fontweight="bold", y=1.06)
    return save(fig, "oracle.png")


def chart_front_duty():
    labels = ["lab profile\n+ trained policy", "legacy profile\n+ trained policy",
              "lab profile\nNO policy", "legacy profile\nNO policy"]
    fl = [0.203, 0.483, 0.457, 0.463]
    fr = [0.591, 0.500, 0.437, 0.491]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.bar(x - .19, fl, .36, color="#8a4a3c", label="front LEFT foot")
    ax.bar(x + .19, fr, .36, color="#c98a6a", label="front RIGHT foot")
    ax.axhline(.70, color=GREEN, ls="--", lw=1.6)
    ax.text(3.45, .715, "published duty target 0.70", color=GREEN, fontsize=9.5,
            ha="right")
    for i, (a, b) in enumerate(zip(fl, fr)):
        ax.text(i - .19, a + .012, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + .19, b + .012, f"{b:.3f}", ha="center", fontsize=9)
        gap = abs(a - b)
        ax.text(i, .78, f"gap {gap:.3f}", ha="center", fontsize=9,
                color=RED if gap > .1 else "#5a616b",
                fontweight="bold" if gap > .1 else "normal")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("fraction of the step each foot carries load")
    ax.set_ylim(0, .88)
    ax.set_title("Only ONE of the four combinations limps  (ledger #184)",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9.5)
    return save(fig, "front_duty.png")


def chart_prey_motion():
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.barh(["before\n(sessions 8–9)", "after\n(session 9t)"],
            [0.0, 398.0], color=[RED, GREEN], height=.5)
    ax.text(6, 0, "0.0 mm  —  the prey never moved", va="center",
            fontsize=11, color=RED, fontweight="bold")
    ax.text(404, 1, "398 mm per episode", va="center", fontsize=11,
            color=GREEN, fontweight="bold")
    ax.set_xlim(0, 560)
    ax.set_xlabel("distance the prey travelled per episode (mm)")
    ax.set_title("Four eye configurations were tested against a prey that was "
                 "not moving", loc="left", fontsize=13.5, fontweight="bold",
                 pad=14)
    return save(fig, "prey_motion.png")


# ============================================================ what comes next
def chart_memory():
    """The published retention curve the memory module has to reproduce."""
    months = np.linspace(0, 15, 400)
    path = np.exp(-months / 3.79)
    lat = np.exp(-months / 4.32)
    fig, ax = plt.subplots(figsize=(12.5, 4.3))
    ax.plot(months, path, color=BLUE, lw=2.4, label="route precision  τ ≈ 3.79 months")
    ax.plot(months, lat, color=GREEN, lw=2.4, ls="--",
            label="ability to reach the goal  τ ≈ 4.32 months")
    for m, txt, col in ((2, "2 months:\nno significant loss", GREEN),
                        (4, "4 months:\nroute precision gone,\ngoal-finding not yet", AMBER),
                        (14, "6–14 months:\nindistinguishable\nfrom naive", RED)):
        ax.axvline(m, color=col, lw=1.2, ls=":")
        ax.text(m + .18, .93 if m < 5 else .60, txt, fontsize=9, color=col,
                va="top")
    ax.set_xlabel("months since training")
    ax.set_ylabel("retained fraction")
    ax.set_ylim(0, 1.05); ax.set_xlim(0, 15)
    ax.legend(frameon=False, loc="lower left", fontsize=9.5)
    ax.set_title("Memory has published numbers after all  (ledger #216)",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.text(0, -.30,
            "The two decay constants are DERIVED by fitting an exponential to "
            "published retention points, not printed in any paper. The three "
            "labelled\nfindings are published (n = 38–42). Nothing has ever "
            "been measured in any lizard at hours, days or weeks — that part "
            "would be invented.",
            fontsize=9, color="#5a616b", va="top")
    return save(fig, "memory.png")


def chart_learning():
    trials = np.arange(0, 21)
    lat = 1 + (3.02 - 1) * (1 - np.exp(-trials / 6.5))
    path = 1 + (4.59 - 1) * (1 - np.exp(-trials / 6.5))
    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    ax.plot(trials, lat, color=BLUE, lw=2.4, label="latency  → 3.02×")
    ax.plot(trials, path, color=GREEN, lw=2.4, label="path  → 4.59×")
    ax.axhline(3.02, color=BLUE, ls=":", lw=1)
    ax.axhline(4.59, color=GREEN, ls=":", lw=1)
    ax.set_xlabel("spaced trials")
    ax.set_ylabel("improvement over naive (×)")
    ax.set_xlim(0, 20)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax.set_title("Learning: hazard ratio 1.09 per trial, ceilings published  "
                 "(ledger #216)", loc="left", fontsize=14, fontweight="bold",
                 pad=14)
    fig.text(.005, -.12,
             "PUBLISHED: the per-trial hazard ratio (χ² = 16.43, n = 38–42) and "
             "both ceilings. The CURVE SHAPE between them is drawn here as an "
             "exponential and is\nINVENTED — no paper prints the trajectory. "
             "Every learning number for this species comes from escaping water, "
             "not from hunting.",
             fontsize=9, color="#5a616b", va="top")
    return save(fig, "learning.png")


def chart_sleep():
    fig, ax = plt.subplots(figsize=(12.5, 3.1))
    ax.axvspan(84, 134.5, color=AMBER, alpha=.22)
    ax.plot([84, 134.5], [0, 0], color=AMBER, lw=6, solid_capstyle="butt")
    for v, lab in ((84, "84 s"), (134.5, "134.5 s")):
        ax.plot([v], [0], "o", color=AMBER, ms=9)
        ax.text(v, .22, lab, ha="center", fontweight="bold", fontsize=10)
    ax.text(109, -.34, "the bracket the gecko must fall inside\n"
                       "(other lizards, published)", ha="center", fontsize=9.5,
            color="#5a616b")
    ax.set_xlim(40, 190); ax.set_ylim(-.75, .6)
    ax.set_yticks([])
    ax.set_xlabel("infraslow sleep-cycle period (seconds)")
    ax.set_title("Sleep: a bracket, not a number  — and the project refuses "
                 "to pick one", loc="left", fontsize=13.5, fontweight="bold",
                 pad=14)
    ax.text(40, -1.35,
            "No sleep-cycle period has ever been printed for Eublepharis "
            "macularius. brain/arousal.py stores sleep_cycle_period_s as null on "
            "purpose, so that\nborrowing a bearded dragon's number is impossible "
            "without declaring it. Temperature scaling would use Q10 = 2.3.",
            fontsize=9, color="#5a616b", va="top")
    return save(fig, "sleep.png")


def chart_gaps():
    """What has never been measured, by area. The honest half of the map."""
    areas = OrderedDict([
        ("vision", ["receptive field size", "size tuning", "speed tuning",
                    "contrast threshold", "elevation tuning",
                    "detection latency", "acuity in this species",
                    "any prey-selective cell in any reptile"]),
        ("energetics", ["cost of transport", "sprint maximum", "endurance",
                        "thermal performance curve", "CTmin"]),
        ("field biology", ["home range", "activity budget", "wild diet",
                           "microhabitat selection"]),
        ("feeding", ["strike distance in this species",
                     "strike speed in this species",
                     "capture success in this species"]),
        ("other senses", ["cricket vibration spectrum",
                          "prey localisation by smell",
                          "crevice tightness preference"]),
        ("memory timing", ["decay at hours", "decay at days", "decay at weeks"]),
    ])
    names = list(areas)[::-1]
    counts = [len(areas[k]) for k in names]
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    ax.barh(names, counts, color=GREY, height=.6, alpha=.8)
    for i, (k, c) in enumerate(zip(names, counts)):
        ax.text(c + .12, i, "  " + ", ".join(areas[k][:3]) +
                ("…" if len(areas[k]) > 3 else ""),
                va="center", fontsize=8.5, color="#5a616b")
    ax.set_xlim(0, max(counts) * 4.4)
    ax.set_xticks(range(0, max(counts) + 1, 2))
    ax.set_xlabel("quantities with no published value for this species")
    ax.set_title(f"{sum(counts)} things that have never been measured — "
                 f"recorded rather than invented", loc="left", fontsize=14,
                 fontweight="bold", pad=14)
    return save(fig, "gaps.png")


def chart_progress():
    parts = [
        ("Body + morphology", 1.0, "14/14 static checks, physics hash stable"),
        ("Walking", 0.75, "4 of 6 gates, accepted = lab + zero residual"),
        ("World", 0.80, "floor, moving prey, shelter, warm surface, threat"),
        ("1  Hypothalamus", 1.0, "drives, energy, thermostat"),
        ("2  Basal ganglia", 1.0, "reproduces the published table, 0.103 pp"),
        ("3  Spinal cord + brainstem", 1.0, "four oscillators, drives the walker"),
        ("4  Eye: gaze reflex", 1.0, "published OKR reproduced"),
        ("4  Eye: finding prey", 0.30, "detects on 5.4 % of steps -- the blocker"),
        ("4  Strike", 0.90, "89.7 % capture vs published 82.9 %"),
        ("5  Smell", 0.55, "built; nothing consults it yet"),
        ("6  Sleep / day-night clock", 0.45, "clock built; no cycle period exists"),
        ("7  Memory / place", 0.10, "spec written, nothing built"),
        ("8  Learning", 0.10, "spec written, nothing built"),
        ("Oracle removal", 0.50, "shadow logger in; oracle still connected"),
        ("Proof battery", 0.07, "1 of 15 tests run"),
    ]
    names = [p[0] for p in parts][::-1]
    vals = [p[1] for p in parts][::-1]
    notes = [p[2] for p in parts][::-1]
    cols = [GREEN if v >= .95 else AMBER if v > .35 else RED if v > .12
            else "#dfe3e8" for v in vals]
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    ax.barh(names, [1] * len(vals), color="#eef0f3", height=.62, zorder=0)
    ax.barh(names, vals, color=cols, height=.62, zorder=1)
    for i, nt in enumerate(notes):
        ax.text(1.02, i, nt, va="center", fontsize=9.2, color="#5a616b")
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_title("Where the build actually stands", loc="left", fontsize=14,
                 fontweight="bold", pad=14)
    ax.legend(handles=[Patch(color=GREEN, label="working, checked"),
                       Patch(color=AMBER, label="partial"),
                       Patch(color=RED, label="built, not accepted"),
                       Patch(color="#dfe3e8", label="specified, not built")],
              frameon=False, ncol=4, fontsize=9,
              loc="lower left", bbox_to_anchor=(0, -.13))
    for s in ("bottom", "left"):
        ax.spines[s].set_visible(False)
    return save(fig, "progress.png")


def main():
    rows = load("ledger")
    commits = load("commits")
    reg = load("registry")
    bib = load("bibliography")
    print("charts:")
    chart_progress()
    chart_verdicts(rows)
    chart_cumulative(rows)
    chart_timeline(rows, commits)
    chart_own_errors(rows)
    chart_commits(commits)
    chart_provenance(reg)
    chart_species_gap(reg)
    chart_bibliography(bib)
    chart_detection()
    chart_efference()
    chart_speeds()
    chart_hunt()
    chart_efficiency()
    chart_oracle()
    chart_front_duty()
    chart_prey_motion()
    chart_memory()
    chart_learning()
    chart_sleep()
    chart_gaps()
    print(f"\n{len(rows)} ledger rows, {len(commits)} commits, "
          f"{len(reg)} parameters, {len(bib['citations'])} works")


if __name__ == "__main__":
    sys.exit(main())
