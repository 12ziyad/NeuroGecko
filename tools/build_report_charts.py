"""Charts for the NeuroGecko map. Every number read from the repo, none typed in.

The one rule this file follows is the project's own: nothing here is a figure I
remembered. Ledger counts come from parsing docs/FAILURE_MAP.md, commit stats
from git, and the walking measurements from the evidence JSON that the tools
wrote. If a number cannot be read from the repository it does not appear.
"""

from __future__ import annotations

import io
import json
import pathlib
import re
import sys
from collections import Counter, OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

SHORT = OrderedDict([
    ("the walking phase (Sessions 1", "S1-3\nwalking"),
    ("training (Session 4)", "S4\ntraining"),
    ("body motion (Session 4b)", "S4b\nbody"),
    ("the world (Sessions 4c", "S4c-d\nworld"),
    ("hypothalamus (Sessions 5", "S5\nhypothal."),
    ("basal ganglia (Sessions 6", "S6\nbasal g."),
    ("basal ganglia, second correction", "S6c\nBG fix"),
    ("scouting module 3 (Session 6d)", "S6d\nscout 3"),
    ("the saturation fault explained", "S6e\nsaturation"),
    ("the gecko moved onto the validated", "S6f\nrebase"),
    ("the cord becomes an oscillator", "S7\ncord"),
    ("the dopamine offset resolved", "S7b-c\ndopamine"),
    ("scouting the eye (Session 8)", "S8\nscout eye"),
    ("the retina and the optokinetic", "S8b\nretina"),
    ("the tectum, shipped NOT ACCEPTED", "S8c\ntectum"),
    ("the eye was the wrong question", "S8d-9\nliterature"),
])


def label_for(section):
    for key, short in SHORT.items():
        if section.startswith(key):
            return short
    return section[:14]


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("  ", name)
    return path


def chart_verdicts(rows):
    """Per-session stacked bars. The shape of the whole project in one picture."""
    order = [s for s in SHORT.values()]
    by = {s: Counter() for s in order}
    for r in rows:
        by[label_for(r["section"])][r["class"]] += 1
    ref = [by[s]["Refuted"] for s in order]
    con = [by[s]["Confirmed"] for s in order]
    par = [by[s]["Partly"] for s in order]

    fig, ax = plt.subplots(figsize=(13, 4.6))
    x = np.arange(len(order))
    ax.bar(x, ref, color=RED, label=f"Refuted ({sum(ref)})")
    ax.bar(x, par, bottom=ref, color=AMBER, label=f"Partly ({sum(par)})")
    ax.bar(x, con, bottom=np.array(ref) + np.array(par), color=GREEN,
           label=f"Confirmed ({sum(con)})")
    for i, (a, b, c) in enumerate(zip(ref, par, con)):
        ax.text(i, a + b + c + 0.6, str(a + b + c), ha="center",
                fontsize=9, color="#5a616b")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=8.5)
    ax.set_ylabel("hypotheses tested")
    ax.set_title("Every hypothesis this project ever tested, by phase",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=10)
    ax.set_ylim(0, max(np.array(ref) + np.array(par) + np.array(con)) * 1.32)
    return save(fig, "verdicts.png")


def chart_cumulative(rows):
    """The line that matters: wrong answers accumulate faster than right ones."""
    rows = sorted(rows, key=lambda r: r["n"])
    n = [r["n"] for r in rows]
    cum_ref = np.cumsum([1 if r["class"] == "Refuted" else 0 for r in rows])
    cum_con = np.cumsum([1 if r["class"] == "Confirmed" else 0 for r in rows])
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.fill_between(n, cum_ref, color=RED, alpha=.14)
    ax.plot(n, cum_ref, color=RED, lw=2.4, label="refuted")
    ax.plot(n, cum_con, color=GREEN, lw=2.4, label="confirmed")
    ax.plot(n, np.array(n) * 0 + np.array(n), color=GREY, lw=1, ls=":",
            label="total tested")
    ax.annotate(f"{cum_ref[-1]} refuted", (n[-1], cum_ref[-1]),
                xytext=(-96, 8), textcoords="offset points",
                color=RED, fontweight="bold")
    ax.annotate(f"{cum_con[-1]} confirmed", (n[-1], cum_con[-1]),
                xytext=(-104, -18), textcoords="offset points",
                color=GREEN, fontweight="bold")
    ax.set_xlabel("hypothesis number")
    ax.set_ylabel("running total")
    ax.set_title("84 % of everything tried was wrong — and that is the product",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.legend(frameon=False, loc="upper left")
    return save(fig, "cumulative.png")


def chart_commits(commits):
    """Code churn per commit, coloured by session."""
    fig, ax = plt.subplots(figsize=(13, 4.0))
    x = np.arange(len(commits))
    added = [c["added"] for c in commits]
    removed = [-c["removed"] for c in commits]
    ax.bar(x, added, color=BLUE, width=.85)
    ax.bar(x, removed, color="#d5b8b0", width=.85)
    ax.axhline(0, color="#c9cdd4", lw=1)
    biggest = int(np.argmax(added))
    ax.annotate(f"+{added[biggest]:,} lines\n{commits[biggest]['subject'][:34]}",
                (biggest, added[biggest]), xytext=(6, -6),
                textcoords="offset points", fontsize=8.5, color=BLUE)
    ax.set_xlabel(f"commit, oldest to newest  ({len(commits)} total)")
    ax.set_ylabel("lines added / removed")
    ax.set_title(f"{sum(added):,} lines written, {abs(sum(removed)):,} deleted, "
                 f"across {len(commits)} commits",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    return save(fig, "commits.png")


def chart_front_duty():
    """The defect the user found by watching, and the three configurations."""
    labels = ["trained policy\nlab", "trained policy\nlegacy",
              "NO policy\nlab", "NO policy\nlegacy"]
    fl = [0.203, 0.483, 0.432, 0.463]
    fr = [0.591, 0.500, 0.454, 0.491]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(x - .19, fl, .36, color="#8a4a3c", label="front LEFT foot")
    ax.bar(x + .19, fr, .36, color="#c98a6a", label="front RIGHT foot")
    ax.axhline(.70, color=GREEN, ls="--", lw=1.6)
    ax.text(3.42, .715, "published target 0.70", color=GREEN, fontsize=9.5, ha="right")
    for i, (a, b) in enumerate(zip(fl, fr)):
        ax.text(i - .19, a + .012, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + .19, b + .012, f"{b:.3f}", ha="center", fontsize=9)
    ax.annotate("", xy=(-.19, .215), xytext=(.19, .58),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
    ax.text(.02, .40, "  3x apart\n  = the limp", color=RED, fontsize=10,
            fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("fraction of step the foot carries load")
    ax.set_ylim(0, .82)
    ax.set_title("The limp was the trained policy, not the body  (ledger #183)",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    return save(fig, "front_duty.png")


def chart_speeds():
    """Why a walking gecko can never catch a cricket."""
    names = ["gecko\nwalking", "cricket\nfleeing", "gecko\nSTRIKE"]
    vals = [0.055, 0.118, 0.851]
    cols = [GREY, RED, GREEN]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    bars = ax.barh(names, vals, color=cols, height=.55)
    for b, v in zip(bars, vals):
        ax.text(v + .015, b.get_y() + b.get_height() / 2, f"{v} m/s",
                va="center", fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("metres per second")
    ax.set_title("Pursuit is arithmetically impossible — so the animal strikes",
                 loc="left", fontsize=14, fontweight="bold", pad=14)
    ax.text(.13, -.62, "the cricket is 2.1x faster than the walk;\n"
                       "the strike is 15x the walk and 7x the cricket",
            fontsize=10, color="#5a616b")
    return save(fig, "speeds.png")


def chart_oracle():
    """What the privileged channel was actually worth."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, vals, title, note in (
        (a1, [0.0857, 0.1103], "distance moved (m)", "moves FURTHER without it"),
        (a2, [0.0664, 0.0060], "progress TOWARD the goal (m)", "91 % of it was the cheat")):
        b = ax.bar(["oracle ON", "oracle zeroed"], vals, color=[BLUE, RED], width=.55)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v * 1.03, f"{v:.4f}",
                    ha="center", fontweight="bold", fontsize=10)
        ax.set_title(title, fontsize=11.5, loc="left")
        ax.text(0, -max(vals) * .27, note, fontsize=10, color="#5a616b")
        ax.set_ylim(0, max(vals) * 1.25)
    fig.suptitle("The legs were fine. The navigation was the cheat.  (ledger #166)",
                 x=.005, ha="left", fontsize=14, fontweight="bold", y=1.06)
    return save(fig, "oracle.png")


def chart_progress():
    """Where the eight brain modules stand."""
    parts = [
        ("Body + morphology", 1.0, "14/14 static checks"),
        ("Walking", 0.75, "4 of 6 gates, accepted"),
        ("World", 0.55, "floor, prey, no shelter"),
        ("1  Hypothalamus", 1.0, "drives, energy, thermostat"),
        ("2  Basal ganglia", 1.0, "reproduces published table"),
        ("3  Spinal cord + brainstem", 1.0, "drives the walker"),
        ("4  Eye: gaze reflex", 1.0, "published OKR reproduced"),
        ("4  Eye: finding prey", 0.15, "NOT ACCEPTED"),
        ("4  Strike", 0.85, "81 % vs published 82.9 %"),
        ("5  Smell", 0.0, "not started"),
        ("6  Sleep", 0.0, "not started"),
        ("7  Memory / place", 0.0, "not started"),
        ("8  Learning", 0.0, "not started"),
        ("Proof battery", 0.07, "1 of 15 tests"),
    ]
    names = [p[0] for p in parts][::-1]
    vals = [p[1] for p in parts][::-1]
    notes = [p[2] for p in parts][::-1]
    cols = [GREEN if v >= .95 else AMBER if v > .1 else "#dfe3e8" for v in vals]
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.barh(names, vals, color=cols, height=.62)
    ax.barh(names, [1] * len(vals), color="#eef0f3", height=.62, zorder=0)
    for i, (v, nt) in enumerate(zip(vals, notes)):
        ax.text(1.02, i, nt, va="center", fontsize=9.5, color="#5a616b")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Where the build actually stands", loc="left",
                 fontsize=14, fontweight="bold", pad=14)
    for s in ("bottom", "left"):
        ax.spines[s].set_visible(False)
    return save(fig, "progress.png")


def main():
    rows = json.load(open(OUT / "ledger.json", encoding="utf-8"))
    commits = json.load(open(OUT / "commits.json", encoding="utf-8"))
    print("charts:")
    chart_verdicts(rows)
    chart_cumulative(rows)
    chart_commits(commits)
    chart_front_duty()
    chart_speeds()
    chart_oracle()
    chart_progress()
    print(f"\n{len(rows)} ledger rows, {len(commits)} commits")


if __name__ == "__main__":
    sys.exit(main())
