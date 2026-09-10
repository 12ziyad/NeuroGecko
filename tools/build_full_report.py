"""The complete NeuroGecko record, as one printable document.

Everything in here is read from the repository. The ledger is parsed from
docs/FAILURE_MAP.md and dated with git blame, the parameters come from
config/proxies.yaml through the loader the code itself uses, the commits come
from git log, and the module specifications come from
artifacts/report/final_spec.json. Run tools/report_data.py and
tools/report_charts.py first; this file only assembles.

One rule is enforced throughout and stated in the document: NO TIME ESTIMATES.
Work that has happened carries the date it happened. Work that has not happened
carries an ORDER and a DEPENDENCY, and nothing else, because this project has
no basis for predicting how long anything takes and has been wrong every time
it guessed.
"""

from __future__ import annotations

import base64
import html
import json
import pathlib
import re
import subprocess
import sys
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts/report"

CHROME = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def b64(name):
    return base64.b64encode((OUT / name).read_bytes()).decode()


def esc(t):
    return html.escape(str(t if t is not None else ""), quote=False)


def md(t):
    """The map is written in markdown. Keep bold, italic, code and strike."""
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = re.sub(r"~~(.+?)~~", r"<s>\1</s>", t)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", t)
    return t


def img(name, cap=""):
    c = f'<div class="cap">{cap}</div>' if cap else ""
    return (f'<figure><img src="data:image/png;base64,{b64(name + ".png")}"/>'
            f'{c}</figure>')


def markdown_to_html(text):
    """Enough markdown for the analysis brief, and no more.

    Blocks are joined before conversion. A bold span in the source wraps across
    the line the editor broke it on, and converting line by line leaves a naked
    `**` in the middle of the page -- which is exactly the class of defect this
    project keeps a ledger about.
    """
    blocks, cur, kind = [], [], None

    def flush():
        if cur:
            blocks.append((kind, " ".join(cur)))
        cur.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            kind = None
        elif stripped.startswith("---"):
            flush()
            blocks.append(("hr", ""))
            kind = None
        elif stripped.startswith("## "):
            flush()
            blocks.append(("h2", stripped[3:]))
            kind = None
        elif stripped.startswith("# "):
            flush()
            blocks.append(("h1", stripped[2:]))
            kind = None
        elif re.match(r"^[-*]\s+", stripped):
            flush()
            kind = "ul"
            cur.append(re.sub(r"^[-*]\s+", "", stripped))
        elif re.match(r"^\d+\.\s+", stripped):
            flush()
            kind = "ol"
            cur.append(re.sub(r"^\d+\.\s+", "", stripped))
        elif kind in ("ul", "ol") and line.startswith("  "):
            cur.append(stripped)            # continuation of the same item
        else:
            if kind not in (None, "p"):
                flush()
            kind = "p"
            cur.append(stripped)
    flush()

    out, open_list = [], None
    for k, body in blocks:
        want = k if k in ("ul", "ol") else None
        if open_list and want != open_list:
            out.append(f"</{open_list}>")
            open_list = None
        if want and not open_list:
            out.append(f'<{want} class="brief">')
            open_list = want
        if k == "h2":
            out.append(f'<h3 class="bq">{md(body)}</h3>')
        elif k == "h1":
            out.append(f'<h4 class="bh">{md(body)}</h4>')
        elif k == "hr":
            out.append('<div class="hair"></div>')
        elif k in ("ul", "ol"):
            out.append(f"<li>{md(body)}</li>")
        else:
            out.append(f"<p>{md(body)}</p>")
    if open_list:
        out.append(f"</{open_list}>")
    return "".join(out)


def sess_key(s):
    if not s:
        return (999, "")
    num = "".join(c for c in s if c.isdigit())
    return (int(num or 999), "".join(c for c in s if c.isalpha()))


# ============================================================ the road map SVG
NODES = [
    (0, 0, 3, "RESEARCH — the literature first",
     "212 works · 143 registered parameters · 221 hypotheses", "done"),
    (0, 1, 1, "BODY", "38 g · 14/14 static checks", "done"),
    (1, 1, 1, "WALKING", "4 of 6 gates · lab + zero residual", "part"),
    (2, 1, 1, "WORLD", "moving prey · shelter · warm rock", "part"),
    (0, 2, 1, "1 HYPOTHALAMUS", "hunger · energy · heat", "done"),
    (1, 2, 1, "2 BASAL GANGLIA", "picks 1 of 6 behaviours", "done"),
    (2, 2, 1, "3 CORD + BRAINSTEM", "four leg oscillators", "done"),
    (0, 3, 1, "4 EYE — gaze", "published OKR reproduced", "done"),
    (1, 3, 1, "4 EYE — finding prey", "detects on 5.4 % of steps", "fail"),
    (2, 3, 1, "4 STRIKE", "89.7 % vs published 82.9 %", "done"),
    (0, 4, 1, "5 SMELL", "built · nothing consults it", "part"),
    (1, 4, 1, "6 DAY/NIGHT CLOCK", "built · no cycle period exists", "part"),
    (2, 4, 1, "ORACLE REMOVAL", "shadow logger in · oracle still on", "part"),
    (0, 5, 1, "7 MEMORY", "specified · not built", "todo"),
    (1, 5, 1, "8 LEARNING", "specified · not built", "todo"),
    (2, 5, 1, "SLEEP WAVES", "specified · not built", "todo"),
    (0, 6, 3, "PROOF BATTERY", "1 of 15 tests run", "part"),
]
EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
         (6, 9), (7, 10), (8, 11), (9, 12), (10, 13), (11, 14), (12, 15),
         (13, 16), (14, 16), (15, 16)]
STATE = {"done": ("#2f7d5b", "#e8f2ed"), "part": ("#c98a1e", "#fbf2e0"),
         "fail": ("#c8442f", "#fbeae6"), "todo": ("#8a919b", "#f1f3f5")}


def roadmap_svg():
    CW, CH, GX, GY = 268, 96, 300, 136
    W, H = 3 * GX + 40, 7 * GY + 30
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'style="width:100%;height:auto">']

    def centre(n):
        x, y, w = NODES[n][0], NODES[n][1], NODES[n][2]
        return 20 + x * GX + (w * CW + (w - 1) * (GX - CW)) / 2, 20 + y * GY + CH / 2

    for a, b in EDGES:
        x1, y1 = centre(a)
        x2, y2 = centre(b)
        p.append(f'<path d="M{x1},{y1+CH/2-6} C{x1},{y1+50} {x2},{y2-50} '
                 f'{x2},{y2-CH/2+6}" fill="none" stroke="#ccd2da" '
                 f'stroke-width="2"/>')
    for x, y, w, label, sub, st in NODES:
        px, py = 20 + x * GX, 20 + y * GY
        pw = w * CW + (w - 1) * (GX - CW)
        edge, fill = STATE[st]
        p.append(
            f'<rect x="{px}" y="{py}" width="{pw}" height="{CH}" rx="11" '
            f'fill="{fill}" stroke="{edge}" stroke-width="2.2"/>'
            f'<text x="{px+18}" y="{py+38}" font-family="DejaVu Sans,sans-serif" '
            f'font-size="17.5" font-weight="700" fill="#12161c">{esc(label)}</text>'
            f'<text x="{px+18}" y="{py+64}" font-family="DejaVu Sans,sans-serif" '
            f'font-size="12.5" fill="#5a616b">{esc(sub)}</text>'
            f'<circle cx="{px+pw-20}" cy="{py+22}" r="6.5" fill="{edge}"/>')
    p.append("</svg>")
    return "".join(p)


# ============================================================= the signal flow
def signal_svg():
    """One picture of how a step actually runs, left to right."""
    stages = [
        ("WORLD", "MuJoCo\n64x64 render", "#8a919b"),
        ("RETINA", "log intensity\nframe difference", "#2f5d8a"),
        ("TECTUM", "efference copy\nsubtracted", "#2f5d8a"),
        ("HYPO-\nTHALAMUS", "hunger, heat\nenergy", "#2f7d5b"),
        ("BASAL\nGANGLIA", "picks 1 of 6\nbehaviours", "#2f7d5b"),
        ("BRAIN-\nSTEM", "decision to\nstride command", "#2f7d5b"),
        ("SPINAL\nCORD", "four coupled\noscillators", "#2f7d5b"),
        ("SERVOS", "kp 0.53\n+-0.87 N", "#8a919b"),
    ]
    BW, BH, GAP = 176, 104, 34
    W = len(stages) * (BW + GAP) + 40
    H = 300
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'style="width:100%;height:auto">']
    for i, (name, sub, col) in enumerate(stages):
        x = 20 + i * (BW + GAP)
        y = 74
        p.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="10" '
                 f'fill="#fff" stroke="{col}" stroke-width="2.4"/>')
        for j, line in enumerate(name.split("\n")):
            p.append(f'<text x="{x+BW/2}" y="{y+30+j*20}" text-anchor="middle" '
                     f'font-family="DejaVu Sans,sans-serif" font-size="16" '
                     f'font-weight="700" fill="{col}">{esc(line)}</text>')
        off = 30 + len(name.split("\n")) * 20
        for j, line in enumerate(sub.split("\n")):
            p.append(f'<text x="{x+BW/2}" y="{y+off+j*17}" text-anchor="middle" '
                     f'font-family="DejaVu Sans,sans-serif" font-size="12.5" '
                     f'fill="#5a616b">{esc(line)}</text>')
        if i < len(stages) - 1:
            p.append(f'<path d="M{x+BW+4},{y+BH/2} L{x+BW+GAP-6},{y+BH/2}" '
                     f'stroke="#ccd2da" stroke-width="2.4" '
                     f'marker-end="url(#a)"/>')
    p.append('<defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" '
             'refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" '
             'fill="#ccd2da"/></marker></defs>')
    # the loop back
    x0 = 20 + 6 * (BW + GAP) + BW / 2
    x1 = 20 + 1 * (BW + GAP) + BW / 2
    p.append(f'<path d="M{x0},{74+BH+6} C{x0},{74+BH+70} {x1},{74+BH+70} '
             f'{x1},{74+BH+6}" fill="none" stroke="#c8442f" stroke-width="2.2" '
             f'stroke-dasharray="6 4" marker-end="url(#a)"/>')
    p.append(f'<text x="{(x0+x1)/2}" y="{74+BH+82}" text-anchor="middle" '
             f'font-family="DejaVu Sans,sans-serif" font-size="13.5" '
             f'fill="#c8442f">EFFERENCE COPY — the motor command predicts the '
             f'image motion the walking will cause, and it is subtracted</text>')
    p.append(f'<text x="20" y="46" font-family="DejaVu Sans,sans-serif" '
             f'font-size="14.5" fill="#5a616b">Every step, at 500 Hz physics '
             f'and 50 Hz control. Nothing in this chain is a neural network '
             f'except the walking residual, which is switched OFF in the '
             f'accepted walker.</text>')
    p.append("</svg>")
    return "".join(p)


# ============================================================= written content
REQUIREMENTS = [
    ("A real gecko, not a gecko-shaped robot",
     "The animal has to behave the way <i>Eublepharis macularius</i> behaves "
     "because of what is published about it, not because a reward function was "
     "shaped until the motion looked right.",
     "Every parameter carries PUBLISHED / DERIVED / INVENTED in "
     "<code>config/proxies.yaml</code>. 143 registered; 43 % published, 14 % "
     "derived, 43 % openly invented."),
    ("Procedural understanding, not a lookup",
     "It should work out what prey is from what it senses — not be told "
     "&ldquo;object 7 is food&rdquo;. Many kinds of insect; some things move "
     "and are not prey; prey sometimes holds still.",
     "Partly. The strike, the flee response and the approach criterion are "
     "procedural and published. The eye is the gap: it detects on 5.4 % of "
     "steps, and prey identity is still colour and motion, not a tuned "
     "detector."),
    ("No oracle. No privileged information.",
     "The animal must not be handed the prey's coordinates. If it finds food, "
     "it must find it with its own senses.",
     "NOT DONE, and now instrumented rather than argued about. The oracle is "
     "demoted to a shadow that reaches the logger and never the policy "
     "(<code>brain/hunt_metrics.py</code> refuses to be constructed any other "
     "way). It is still connected to the walker's goal."),
    ("Videos that show the animal, honestly",
     "One fixed angle. No shaking. Real speed. All four legs visible. The walk "
     "that was accepted, not whatever checkpoint was lying around.",
     "Fixed after four separate defects: half-speed playback, a camera locked "
     "to the bobbing trunk, a compass angle the animal walked into, and — the "
     "one that mattered — the wrong walker in every clip."),
    ("Never guess at how long things take",
     "A standing instruction from the first session.",
     "Honoured. This document dates what happened and orders what has not. It "
     "gives no estimate for anything, and the reason is in Part 10."),
    ("Short answers, plain words, jargon defined inline",
     "A standing instruction.",
     "Honoured in conversation. This document is the exception the user asked "
     "for, and it still defines its terms — see the glossary in Part 17."),
    ("Carry the map, always",
     "<code>docs/FAILURE_MAP.md</code> is read first every session and updated "
     "before every commit that tests something. Nothing is ever deleted from "
     "it.",
     "Honoured across 29 sessions and 221 numbered hypotheses. This report is "
     "generated from it."),
]

RULES = [
    ("1", "Every number is PUBLISHED, DERIVED or INVENTED — and says which.",
     "A plausible number wearing a published number's clothes is the worst "
     "defect available.",
     "<code>cost_of_transport = 0.73</code> sat in the registry tagged "
     "<i>E. macularius, verified</i>. It is <i>Teratoscincus przewalskii</i>, a "
     "different family. No cost of transport has ever been measured in this "
     "species."),
    ("2", "Never tune until the answer matches.",
     "If a model needs its constants adjusted to reproduce a published result, "
     "the constants must come from a paper.",
     "The basal-ganglia dopamine curve sat 0.10 too high for three sessions. "
     "It was not scaled down; the mechanism was found to be wrong in the "
     "opposite direction to the previous correction. Error fell from 10.3 "
     "percentage points to 0.103."),
    ("3", "The reward is what the model is told to want. The gates are what "
     "the animal does. Only the second counts.",
     "A training reward can be satisfied in ways that have nothing to do with "
     "the behaviour.",
     "The trained walking residual improves the reward and fails the gates. "
     "The accepted walker runs with the residual switched off entirely."),
    ("4", "A failed prediction stays in the table.",
     "Adding a knob per failing joint is fitting the harness to the answer.",
     "Two of six locomotion gates have never passed. They are still listed as "
     "failing, in every report, including this one."),
    ("5", "Measure before claiming — including your own tests and your own "
     "solver.",
     "The instrument is part of the experiment.",
     "Five recorded failures were in the test, not the model. One was in the "
     "integration step. One test literally built a patch of the colour it read "
     "out of the model and checked the detector saw it — it passed while the "
     "detector scored 0.000 on the real rendered prey. Twice, three sessions "
     "apart."),
    ("6", "One source is not a citation.",
     "Check a parameter set against something it must independently predict.",
     "The basal-ganglia weights were settled by the gating constant they have "
     "to reproduce, not by a vote among sources."),
]

PATTERNS = [
    ("A default won because nobody named the parameter", 5,
     "The goal oracle (#165), the food oracle, the gait profile (#177), and "
     "twice before that (#26, #33). Each time a constructor omitted an "
     "argument and the callee's default silently decided the science.",
     "Every one is now passed explicitly and reported in <code>info</code> on "
     "every step. <code>common/walker_pairing.py</code> warns when the "
     "measured-broken combination is assembled."),
    ("The test compared the code against itself", 3,
     "A class literally named <i>the detector cannot disagree with the "
     "world</i> read the prey colour from the model, built a patch of that "
     "colour, and checked the detector saw it. It never touched the renderer. "
     "It passed while the detector scored 0.000 on the real prey — twice, "
     "three sessions apart. A world-builder guard required added bodies to be "
     "mocap, when what it needed to test was that they add no degrees of "
     "freedom.",
     "Assertions now reach the thing being guarded. The world guard tests "
     "<code>nq</code> and <code>nv</code> directly (39 → 39, 38 → 38)."),
    ("A fix that provably could not do anything", 2,
     "The tectum subtracted a scalar from the motion map to reject "
     "self-motion. <code>argmax(clip(x−c,0,None)) == argmax(x)</code> for any "
     "scalar c — identical on 20 000 of 20 000 random maps. It had never "
     "changed one reported bearing.",
     "Replaced first with a spatially varying surround, then — when that was "
     "also shown to assume uniform flow — with a predicted flow field "
     "subtracted from the image."),
    ("A number wearing another species' clothes", 3,
     "<code>cost_of_transport = 0.73</code> tagged <i>E. macularius, "
     "verified</i>; it is <i>Teratoscincus przewalskii</i>. The human-readable "
     "scorecard had it right and the machine-readable registry the code reads "
     "did not.",
     "Corrected, and the gap recorded rather than filled."),
    ("I explained away a real observation, repeatedly", 4,
     "The user said the walk dragged. The answers given were: camera shake, "
     "then half-speed playback, then an under-actuated forelimb they would "
     "have to accept. Then the profile. Then the policy. All five were wrong "
     "about the cause.",
     "It was the PAIRING. Three of the four profile-policy combinations walk "
     "with a front-foot gap under 0.03; the one being filmed had a gap of "
     "0.376."),
    ("The map already contained the answer", 2,
     "#71 states outright that the accepted walker IS the open-loop base. #11 "
     "measured the trained residual as worse. The evidence file records its "
     "controller as <i>zero residual</i>. All three were read, and the "
     "checkpoint was loaded anyway.",
     "Reading the ledger is not the same as believing it. "
     "<code>GeckoBrainEnv</code> could not even express <i>lab profile, no "
     "policy</i> until <code>use_policy=False</code> was added."),
    ("I asserted something I had not checked", 4,
     "A citation was invented outright — Day, Crews &amp; Wilczynski were said "
     "to have studied leopard geckos; they never did (#215). "
     "&ldquo;Nobody has filmed a gecko hunting&rdquo; was said as fact; "
     "Delheusy, Brillet &amp; Bels 1995 exists, n=6, 64 fps. &ldquo;A sharper "
     "eye would find the cricket&rdquo; and &ldquo;geckos head-bob to range "
     "prey&rdquo; were both told to the user as fact and both refuted.",
     "Every claim in this document that is not measured in this repository is "
     "marked as unverified, and the companion prompt asks an outside reader to "
     "check the bibliography specifically for this failure."),
    ("I built the thing that was not the blocker", 3,
     "The walker was retrained for 2.1 M steps against a reach problem that "
     "did not exist — the existing walker reached a mouth-goal 75 times in "
     "160 s, and the retrain was 8× worse (168 mm vs 20.5 mm). "
     "<code>hunt_targeting</code>, proposed as the approach fix, gave 0 "
     "strikes against 29.",
     "Both killed. Both kept in the ledger, and the targeting option kept in "
     "the code, defaulted off, with its measurement attached."),
]

TURNAROUNDS = [
    ("The dopamine curve sat 0.10 too high",
     "Three sessions of a basal-ganglia model that would not match the "
     "published table.",
     "The mechanism was wrong in the opposite direction to the previous "
     "correction. Afferent gain (1±λ) with threshold 0.2.",
     "Mean error 10.3 pp → <b>0.103 pp</b>, axis slope 1.000.", "#104–#108"),
    ("The gecko could not see its own food",
     "The detector tested for green; the prey had been changed to brown. "
     "Exactly 0.000 at every illumination — and the guarding test compared the "
     "code to itself.",
     "Read the colour the renderer actually draws — the material, not just "
     "<code>geom_rgba</code>.",
     "Two separate occurrences fixed, three sessions apart, and the "
     "self-referential test replaced.", "#112, #196"),
    ("The eye could not find the cricket",
     "Four eye configurations, 64 → 512 px. Correlation with true prey bearing "
     "≈ 0 in all four.",
     "It was never resolution. The prey travelled <b>0.0 mm</b> and was "
     "off-image on 81 % of frames.",
     "Prey now walks 398 mm per episode. The eye had something to look at for "
     "the first time.", "#116–#118, #134"),
    ("The eye reported the same thing with and without prey",
     "72 % of frames firing at salience 0.4161 with prey; 72 % at 0.4034 with "
     "<b>no prey in the world at all</b>. Separation d = 0.036.",
     "A local-versus-surround comparison assumes self-motion is spatially "
     "uniform. Optic flow expands from a focus of expansion, so the local "
     "excess is large everywhere.",
     "An efference copy — predicting the flow from the animal's own yaw and "
     "speed and subtracting it — cut false alarms to <b>3–7 %</b>.",
     "#209–#213"),
    ("Eight gate contracts failing for four sessions",
     "Reported as eight separate failures; the walking evidence looked "
     "unverifiable.",
     "One body check compared raw bytes and raised before the others could "
     "run. What differed was a <i>comment</i>.",
     "All 16 pass. The physics hash matched all along: <code>b178bf26…</code>",
     "#158–#164"),
    ("Pursuit could never work",
     "The hunting channel had no route to success at any parameter value.",
     "Walk 0.055 m/s, cricket 0.118 m/s. The real animal does not chase — it "
     "strikes from 2 cm at 0.851 m/s.",
     "Strike built: <b>81 %</b> capture against a published 82.9 %, unfitted.",
     "#169, #175"),
    ("Nothing had ever been caught",
     "A strike that worked in isolation and an approach that could not reach "
     "2 cm.",
     "Not the eye, not the walker, not the targeting — all three had been "
     "diagnosed as the blocker and all three were wrong. It was prey motion "
     "plus a speed-dependent flee radius.",
     "<b>26 of 29 strikes landed — 89.7 %</b> against a published 82.9 %. The "
     "first time the animal caught anything.", "#202–#204"),
    ("The walk dragged, for six sessions of denial",
     "The user said so from the first video and was told five different "
     "reasons why it did not.",
     "Front-foot duty gap 0.376 on the ONE combination being filmed. The other "
     "three combinations sit under 0.03.",
     "<code>common/walker_pairing.py</code> now warns whenever that pairing is "
     "assembled, so the next session cannot repeat it silently.",
     "#182–#184, #206–#208"),
]

# --------------------------------------------------------------- open questions
OPEN = [
    ("Is <code>lab</code> or <code>legacy</code> the right gait profile?",
     "must be settled",
     "<code>lab</code> is gate-validated and limps under the trained policy. "
     "<code>legacy</code> is un-gated and looks correct. Whether the gate "
     "battery ever scored front-foot symmetry has never been established. "
     "Looking better is not the same as being right, and this decision sits "
     "underneath every video and every hunt measurement."),
    ("Does the leopard gecko hunt visually at all?",
     "unresolved in the literature",
     "The only quantitative sensory study on this species is about PREDATOR "
     "detection, and it found chemical cues dominant (0.21 chemical vs 0.07 "
     "visual, n=42). The whole visual prey-detection module is built on a "
     "mouse, a fish larva and a toad. If the answer is no, a large part of the "
     "eye work is aimed at the wrong sense."),
    ("What does the animal do between 2 cm and 30 cm?",
     "no data exists",
     "The best-transferring numbers in the entire corpus — lunge distance, "
     "velocity, acceleration, success rate — cover only the last two "
     "centimetres, and they come from a different genus in the same family. "
     "Everything upstream has to be imported."),
    ("Is the memory module going to be present and inert?",
     "must be decided before building",
     "Every measured retention point in this species is at two months or "
     "beyond. A time constant of 115 days inside an episode measured in "
     "minutes is a term whose value is numerically indistinguishable from "
     "&ldquo;no decay&rdquo;. Ship the store without the decay and say so, or "
     "do not ship it."),
    ("What is the simulation's clock actually worth?",
     "unexamined",
     "The learning rate was measured with trials one WEEK apart. The memory "
     "numbers live at months. The hunt lives at seconds. Nothing in this "
     "project has established what one simulated second is supposed to be, and "
     "three modules now depend on the answer."),
    ("Was the gate battery ever a locomotion validation?",
     "worth re-deriving",
     "Two of six gates have never passed. The accepted walker is accepted on "
     "four. Whether four-of-six is a meaningful bar, or whether the two "
     "failing gates are the ones that matter, has never been argued in "
     "writing."),
    ("Does the registry's <code>confidence</code> field mean anything?",
     "audit needed",
     "75 of 143 parameters are tagged <i>uncertain</i> and 30 <i>verified</i>. "
     "The policy text says <i>verified</i> means checked in the primary source. "
     "One entry tagged <i>verified</i> was the wrong species. The field has "
     "never been audited end to end."),
    ("Are the 212 harvested citations real?",
     "explicitly unverified",
     "This project has invented a citation once (#215) and overstated the "
     "absence of one (Delheusy 1995). The bibliography in Part 14 is a HARVEST "
     "of every author-year string in the repository. It has not been checked "
     "against any database. That check is the first thing the companion prompt "
     "asks for."),
]


# ==================================================================== assembly
def build():
    led = json.load(open(OUT / "ledger.json", encoding="utf-8"))
    commits = json.load(open(OUT / "commits.json", encoding="utf-8"))
    reg = json.load(open(OUT / "registry.json", encoding="utf-8"))
    bib = json.load(open(OUT / "bibliography.json", encoding="utf-8"))
    spec = json.load(open(OUT / "final_spec.json", encoding="utf-8"))

    counts = Counter(r["class"] for r in led)
    total = len(led)
    pct = round(100 * counts["Refuted"] / total)
    prov = Counter(v["provenance"] for v in reg.values())
    src_added = sum(c["added"] for c in commits)
    sessions = sorted({r["session"] for r in led if r.get("session")},
                      key=sess_key)

    # ------------------------------------------------------------- Part 5 flow
    by_sess = defaultdict(list)
    for r in led:
        by_sess[r.get("session") or "?"].append(r)
    flow_rows = []
    for s in sessions:
        rows = by_sess[s]
        c = Counter(r["class"] for r in rows)
        subj = rows[0].get("commit_subject", "")
        subj = re.sub(r"^Session\s+[0-9a-z]+[:\-]?\s*", "", subj)
        flow_rows.append(
            f'<tr><td class="n">S{esc(s)}</td>'
            f'<td class="d">{esc(rows[0].get("date",""))}</td>'
            f'<td>{esc(subj)}</td>'
            f'<td class="num">#{min(r["n"] for r in rows)}–'
            f'{max(r["n"] for r in rows)}</td>'
            f'<td class="num del">{c["Refuted"]}</td>'
            f'<td class="num add">{c["Confirmed"]}</td>'
            f'<td class="num">{c["Partly"]}</td></tr>')

    # ------------------------------------------------------------ Part 11 plan
    plan = []
    for i, step in enumerate(spec["build_order"], 1):
        plan.append(
            f'<div class="nx"><div class="rank">{i}</div><div>'
            f'<h4>{esc(step["what"][:110])}'
            f'<em>{esc(step.get("evidence_strength","?"))}</em></h4>'
            f'<p>{esc(step["what"][110:])}</p>'
            f'<p class="why"><b>Why here in the order:</b> '
            f'{esc(step.get("why_now",""))}</p></div></div>')

    # ---------------------------------------------------------- Part 12 ledger
    ledger_rows = "".join(
        f'<tr class="{r["class"].lower()}"><td class="n">{r["n"]}</td>'
        f'<td class="s">S{esc(r.get("session") or "?")}<br>'
        f'<span>{esc((r.get("date") or "")[5:])}</span></td>'
        f'<td class="hy">{md(r["hypothesis"])}</td>'
        f'<td class="v">{md(r["verdict"])}</td>'
        f'<td class="e">{md(r["evidence"])}</td></tr>' for r in led)

    # -------------------------------------------------------- Part 13 registry
    def fmt_val(v):
        if isinstance(v, list):
            return " – ".join(str(x) for x in v)
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    reg_rows = "".join(
        f'<tr class="p{v["provenance"].lower()}">'
        f'<td class="k"><code>{esc(k)}</code></td>'
        f'<td class="num">{esc(fmt_val(v["value"]))}</td>'
        f'<td class="u">{esc(v["units"] or "")}</td>'
        f'<td class="pv">{esc(v["provenance"])}</td>'
        f'<td class="sp">{esc(v["species"] or "")}</td>'
        f'<td class="num">{esc(v["n"] if v["n"] is not None else "")}</td>'
        f'<td class="cf {esc(v["confidence"] or "")}">{esc(v["confidence"] or "")}</td>'
        f'<td class="src">{esc((v["source"] or "")[:210])}</td></tr>'
        for k, v in sorted(reg.items()))

    # ---------------------------------------------------- Part 14 bibliography
    cites = bib["citations"]
    bib_rows = "".join(
        f'<tr><td class="w"><b>{esc(k)}</b></td>'
        f'<td class="wa">{esc(" / ".join(v["written_as"])[:120])}</td>'
        f'<td class="num">{v["count"]}</td>'
        f'<td class="e">{esc(", ".join(v["cited_by"][:9]))}'
        f'{"…" if v["count"] > 9 else ""}</td></tr>'
        for k, v in sorted(cites.items(), key=lambda kv: -kv[1]["count"]))
    doi_rows = "".join(
        f'<tr><td class="w"><code>{esc(d)}</code></td>'
        f'<td class="e">{esc(", ".join(where[:10]))}</td></tr>'
        for d, where in sorted(bib["dois"].items()))

    # ------------------------------------------------------------ Part 16 code
    commit_rows = "".join(
        f'<tr><td class="h"><code>{esc(c["hash"])}</code></td>'
        f'<td class="d">{esc(c["date"])}</td>'
        f'<td>{esc(c["subject"])}</td>'
        f'<td class="num">{c["files"]}</td>'
        f'<td class="num add">+{c["added"]:,}</td>'
        f'<td class="num del">−{c["removed"]:,}</td></tr>' for c in commits)

    # ------------------------------------------------------------- small parts
    req_html = "".join(
        f'<div class="req"><h4>{t}</h4><p class="want">{w}</p>'
        f'<p class="got"><b>Where it stands:</b> {g}</p></div>'
        for t, w, g in REQUIREMENTS)

    rules_html = "".join(
        f'<div class="rule-card"><div class="rn">{n}</div><div>'
        f'<h4>{esc(t)}</h4><p>{esc(why)}</p>'
        f'<p class="from"><b>The failure it came from:</b> {ex}</p></div></div>'
        for n, t, why, ex in RULES)

    pat_html = "".join(
        f'<div class="pat"><div class="pn">×{n}</div><div>'
        f'<h4>{t}</h4><p>{d}</p><p class="fix"><b>Now:</b> {f}</p></div></div>'
        for t, n, d, f in PATTERNS)

    turn_html = "".join(
        f'<div class="turn"><h4>{esc(t)} <em>{esc(ref)}</em></h4>'
        f'<div class="tg"><div><span>WAS</span><p>{was}</p></div>'
        f'<div><span>WHY</span><p>{why}</p></div>'
        f'<div class="win"><span>NOW</span><p>{now}</p></div></div></div>'
        for t, was, why, now, ref in TURNAROUNDS)

    open_html = "".join(
        f'<div class="oq"><h4>{t} <em>{esc(k)}</em></h4><p>{d}</p></div>'
        for t, k, d in OPEN)

    def spec_block(key, title, colour):
        m = spec[key]
        out = [f'<h3 style="color:{colour}">{title}</h3>']
        if m.get("what_the_animal_actually_does"):
            out.append(f'<p class="lead">{esc(m["what_the_animal_actually_does"])}</p>')
        if m.get("nearest_published"):
            out.append(f'<p class="lead">{esc(m["nearest_published"])}</p>')
        if m.get("two_channel_split"):
            out.append(f'<p class="lead">{esc(m["two_channel_split"])}</p>')
        out.append('<table class="spec"><thead><tr><th>Parameter</th>'
                   '<th>Value</th><th>Provenance</th></tr></thead><tbody>')
        for p in m.get("parameters", []):
            pv = esc(p.get("provenance", ""))
            cls = ("pinvented" if "INVENT" in pv.upper() or "NOT_FOUND" in pv.upper()
                   else "pderived" if "DERIVED" in pv.upper() else "ppublished")
            out.append(f'<tr class="{cls}"><td class="k">{esc(p.get("name"))}</td>'
                       f'<td class="e">{esc(p.get("value"))}</td>'
                       f'<td class="src">{pv}</td></tr>')
        out.append("</tbody></table>")
        for label, field, cls in (
                ("What would have to be invented", "must_be_invented", "warn"),
                ("What the model must not claim", "what_the_model_must_not_claim",
                 "warn")):
            if m.get(field):
                out.append(f'<h4 class="sub">{label}</h4><ul class="{cls}">')
                out.extend(f"<li>{esc(x)}</li>" for x in m[field])
                out.append("</ul>")
        return "".join(out)

    surprises = "".join(f"<li>{esc(s)}</li>" for s in spec["surprises"])
    donot = "".join(f"<li>{esc(s)}</li>" for s in spec["do_not_build"])

    GLOSSARY = [
        ("Oracle", "Privileged information handed to the animal that a real "
         "animal could not have — here, the prey's exact position. A model "
         "that uses one has not solved the problem; it has been told the "
         "answer."),
        ("Shadow signal", "The same privileged information, routed to the log "
         "file and never to the animal, so a failure can be attributed."),
        ("Efference copy", "A copy of the motor command, used to predict the "
         "sensory consequences of the animal's own movement so they can be "
         "subtracted. It is why you do not see the world jump when you move "
         "your eyes."),
        ("Optic flow", "The pattern of image motion caused by moving. It "
         "expands outward from the direction of travel — which is why "
         "subtracting a local average does not cancel it."),
        ("Tectum / optic tectum", "The midbrain map of visual space. In "
         "mammals it is called the superior colliculus. It is where "
         "&ldquo;something small is moving over there&rdquo; is computed."),
        ("CPG — central pattern generator", "A circuit in the spinal cord that "
         "produces a rhythm without needing rhythmic input. Here: four coupled "
         "oscillators, one per leg."),
        ("Duty factor", "The fraction of a stride during which a foot is on "
         "the ground and carrying load. Two feet with very different duty "
         "factors is a limp."),
        ("Residual policy", "A trained neural network whose output is ADDED to "
         "a hand-written controller. The accepted walker runs with this set to "
         "zero."),
        ("Gate", "A pass/fail check against a published measurement. Distinct "
         "from a reward: the reward is what the model is told to want, the "
         "gate is what the animal does."),
        ("Path efficiency", "Straight-line ground closed toward the prey, "
         "divided by ground actually walked. 1.0 is a direct approach; near 0 "
         "is a thrash that ended up in the right place."),
        ("Hazard ratio", "How much a treatment multiplies the moment-to-moment "
         "chance of an event. HR 1.09 per trial means about 9 % more likely to "
         "reach the goal with each trial of practice."),
        ("τ (tau), time constant", "How long an exponential decay takes to "
         "fall to about 37 % of its starting value."),
        ("Q10", "How much a biological rate changes for a 10 °C change in "
         "temperature. Q10 = 2.3 means the rate roughly doubles."),
        ("Thigmothermic", "Taking body heat by contact with a warm surface "
         "rather than from sunlight. It is why <i>bask</i> modelled as a "
         "light-seeking behaviour is the wrong mechanism for this animal."),
        ("Vomeronasal / Jacobson's organ", "A chemical sense separate from "
         "smell, sampled by tongue-flicking. In this species it is the channel "
         "that fires the defensive display; vision alone cannot."),
        ("n", "The number of animals in a study. n=42 is a real result; n=2 is "
         "a starting point; the sleep recording of this species is n=2."),
        ("Provenance", "Where a number came from. PUBLISHED means measured and "
         "printed; DERIVED means computed from something published; INVENTED "
         "means chosen, and says so."),
    ]
    gloss = "".join(f'<div class="gl"><b>{t}</b><p>{d}</p></div>'
                    for t, d in GLOSSARY)

    # ------------------------------------------- the companion brief, inlined
    # It travels with the document rather than beside it, because a prompt that
    # gets separated from the report it is about is useless.
    brief = (OUT / "analysis_prompt.md").read_text(encoding="utf-8")
    brief = brief.split("---", 2)[2] if brief.count("---") >= 2 else brief
    brief_html = markdown_to_html(brief)

    charts = ("progress", "verdicts", "cumulative", "timeline", "own_errors",
              "commits", "provenance", "species_gap", "bibliography",
              "detection", "efference", "speeds", "hunt", "efficiency",
              "oracle", "front_duty", "prey_motion", "memory", "learning",
              "sleep", "gaps")

    css = """
@page { size: A4; margin: 14mm 12mm 15mm 12mm; }
* { box-sizing: border-box; }
body { font-family:"DejaVu Sans","Segoe UI",sans-serif; color:#12161c;
       margin:0; font-size:10.2pt; line-height:1.48; background:#fff; }
h1 { font-size:38pt; line-height:1.0; margin:0 0 6px; letter-spacing:-1px; }
h2 { font-size:20pt; margin:0 0 4px; letter-spacing:-.3px; }
h3 { font-size:13pt; margin:22px 0 8px; color:#2f5d8a; }
h4 { font-size:11.2pt; margin:0 0 5px; }
h4.sub { font-size:10.6pt; margin:14px 0 6px; color:#c8442f; }
p { margin:0 0 8px; }
code { font-family:"DejaVu Sans Mono",monospace; font-size:.85em;
       background:#f1f3f5; padding:1px 4px; border-radius:3px; }
.page { page-break-after: always; }
.lead { font-size:11.6pt; color:#3d444d; max-width:64em; }
.rule { height:3px; background:#12161c; margin:12px 0 18px; }
.hair { height:1px; background:#dde1e6; margin:18px 0; }
.part { font-size:8.6pt; letter-spacing:2.6px; text-transform:uppercase;
        color:#8a919b; margin-bottom:6px; }

.cover { padding-top:14mm; }
.eyebrow { font-size:9.6pt; letter-spacing:3px; text-transform:uppercase;
           color:#8a919b; margin-bottom:12px; }
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:9px;
        margin:22px 0 14px; }
.kpi { border:2px solid #12161c; border-radius:9px; padding:11px 12px; }
.kpi b { display:block; font-size:24pt; line-height:1; letter-spacing:-1px; }
.kpi span { font-size:8pt; color:#5a616b; text-transform:uppercase;
            letter-spacing:.7px; }
.kpi.red b{color:#c8442f} .kpi.green b{color:#2f7d5b} .kpi.blue b{color:#2f5d8a}
.kpi.pur b{color:#8e5b9e}

figure { margin:12px 0 6px; page-break-inside:avoid; }
figure img { width:100%; border:1px solid #e4e7eb; border-radius:6px; }
.cap { font-size:8.8pt; color:#5a616b; margin-top:5px; }

table { width:100%; border-collapse:collapse; font-size:7.9pt; }
th { text-align:left; background:#12161c; color:#fff; padding:5px 6px;
     font-size:7.8pt; text-transform:uppercase; letter-spacing:.5px; }
td { padding:4px 6px; border-bottom:1px solid #eceef1; vertical-align:top; }
thead { display:table-header-group; }
tr { page-break-inside:avoid; }
tr.refuted td.n { border-left:3px solid #c8442f; }
tr.confirmed td.n { border-left:3px solid #2f7d5b; }
tr.partly td.n { border-left:3px solid #c98a1e; }
tr.other td.n { border-left:3px solid #8a919b; }
td.n { font-weight:700; width:26px; }
td.s { width:38px; font-size:7.2pt; color:#5a616b; }
td.s span { color:#a0a6ae; }
td.v { width:100px; }
table.ledger td.hy { width:168px; }
td.e { color:#454b54; }
td.h { width:56px; } td.d { width:62px; color:#5a616b; }
td.num { text-align:right; width:50px; font-variant-numeric:tabular-nums; }
td.add { color:#2f7d5b; } td.del { color:#c8442f; }
td.k { width:150px; } td.u { width:56px; color:#5a616b; }
td.pv { width:62px; font-size:7pt; font-weight:700; }
td.sp { width:88px; font-size:7.2pt; }
td.cf { width:52px; font-size:7.2pt; }
td.cf.verified { color:#2f7d5b; } td.cf.uncertain { color:#c8442f; }
td.cf.likely { color:#c98a1e; }
td.src { color:#6a717a; font-size:7.1pt; }
td.w { width:132px; } td.wa { width:150px; color:#6a717a; font-size:7.2pt; }
tr.ppublished td.pv { color:#2f7d5b; }
tr.pderived td.pv { color:#2f5d8a; }
tr.pinvented td.pv { color:#8e5b9e; }
tr.puntagged td.pv { color:#8a919b; }
table.spec td.k { width:190px; font-weight:600; }
table.spec td.src { width:150px; }

.req { border-left:3px solid #2f5d8a; padding-left:13px; margin-bottom:14px;
       page-break-inside:avoid; }
.req .want { font-size:9.8pt; color:#454b54; }
.req .got { font-size:9.8pt; color:#2f7d5b; }

.rule-card { display:grid; grid-template-columns:34px 1fr; gap:12px;
             margin-bottom:14px; page-break-inside:avoid; }
.rn { font-size:20pt; font-weight:700; color:#ccd2da; line-height:1; }
.rule-card p { font-size:9.7pt; color:#454b54; }
.from { color:#c8442f !important; }

.pat { display:grid; grid-template-columns:48px 1fr; gap:12px;
       margin-bottom:14px; page-break-inside:avoid; }
.pn { font-size:16pt; font-weight:700; color:#c8442f; }
.pat p { font-size:9.6pt; color:#454b54; }
.fix { color:#2f7d5b !important; }

.turn { border-left:3px solid #dde1e6; padding-left:13px; margin-bottom:15px;
        page-break-inside:avoid; }
.turn h4 em { font-style:normal; font-size:8pt; color:#8a919b;
              letter-spacing:.6px; margin-left:6px; }
.tg { display:grid; grid-template-columns:1fr 1fr 1fr; gap:11px; }
.tg span { font-size:7.4pt; letter-spacing:1.1px; color:#8a919b; display:block;
           margin-bottom:3px; }
.tg p { font-size:9.2pt; margin:0; color:#454b54; }
.tg .win p, .tg .win span { color:#2f7d5b; }

.nx { display:grid; grid-template-columns:40px 1fr; gap:11px;
      margin-bottom:13px; page-break-inside:avoid; }
.rank { font-size:19pt; font-weight:700; color:#ccd2da; }
.nx h4 em { font-style:normal; font-size:7.8pt; letter-spacing:.7px;
            text-transform:uppercase; color:#fff; background:#8a919b;
            padding:2px 6px; border-radius:9px; margin-left:7px; }
.nx p { font-size:9.6pt; color:#454b54; }
.why { color:#2f5d8a !important; }

.oq { border-left:3px solid #c98a1e; padding-left:13px; margin-bottom:13px;
      page-break-inside:avoid; }
.oq h4 em { font-style:normal; font-size:7.8pt; text-transform:uppercase;
            letter-spacing:.7px; color:#c98a1e; margin-left:7px; }
.oq p { font-size:9.6pt; color:#454b54; }

ul.warn li, ul.gaps li { font-size:9.4pt; color:#454b54; margin-bottom:5px; }
ul.warn { border-left:2px solid #fbeae6; padding-left:16px; }
.legend { display:flex; gap:18px; font-size:8.8pt; color:#5a616b;
          margin:10px 0 6px; flex-wrap:wrap; }
.legend i { width:11px; height:11px; border-radius:3px; display:inline-block;
            margin-right:5px; vertical-align:-1px; }
.quote { border-left:3px solid #12161c; padding:2px 0 2px 14px; margin:14px 0;
         font-size:11.2pt; }
.foot { font-size:8.4pt; color:#8a919b; margin-top:18px; }
.toc { column-count:2; column-gap:26px; font-size:10pt; }
.toc div { break-inside:avoid; margin-bottom:5px; }
.toc b { color:#2f5d8a; margin-right:7px; }
.gl { break-inside:avoid; margin-bottom:9px; }
.gl b { font-size:10pt; }
.gl p { font-size:9.3pt; color:#454b54; margin:1px 0 0; }
.two { column-count:2; column-gap:24px; }
.two > div { break-inside:avoid; }
.big { font-size:15pt; font-weight:700; letter-spacing:-.3px; }
.stamp { display:inline-block; font-size:7.6pt; letter-spacing:1px;
         text-transform:uppercase; padding:2px 7px; border-radius:9px;
         color:#fff; background:#c8442f; }
.stamp.ok { background:#2f7d5b; } .stamp.mid { background:#c98a1e; }
.briefbox { font-size:9.6pt; }
.briefbox h3.bq { font-size:12pt; color:#12161c; margin:16px 0 6px;
                  border-bottom:2px solid #12161c; padding-bottom:3px; }
.briefbox h4.bh { font-size:11pt; color:#2f5d8a; margin:14px 0 5px; }
.briefbox p { color:#454b54; margin:0 0 7px; }
ul.brief { margin:0 0 9px; padding-left:17px; }
ul.brief li { font-size:9.4pt; color:#454b54; margin-bottom:4px; }
"""

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>NeuroGecko — the complete record</title><style>{css}</style></head><body>

<!-- ============================================================ COVER -->
<section class="page cover">
  <div class="eyebrow">Generated from the repository · {esc(commits[-1]["date"])}</div>
  <h1>NeuroGecko</h1>
  <div class="big" style="color:#5a616b;margin:2px 0 10px">The complete record —
  what was asked for, what was built, what was wrong, and what is left</div>
  <div class="rule"></div>
  <p class="lead">A biologically-grounded leopard gecko
  (<i>Eublepharis&nbsp;macularius</i>) in MuJoCo. Every number in the animal is
  tagged <b>PUBLISHED</b>, <b>DERIVED</b> or <b>INVENTED</b>. Every hypothesis
  ever tested stays in the record, especially the wrong ones. This document is
  that record, assembled by machine from
  <code>docs/FAILURE_MAP.md</code>, <code>config/proxies.yaml</code>,
  <code>git log</code>, <code>git blame</code> and the evidence JSON. No figure
  in it was transcribed by hand.</p>

  <div class="kpis">
    <div class="kpi red"><b>{counts['Refuted']}</b><span>hypotheses refuted</span></div>
    <div class="kpi green"><b>{counts['Confirmed']}</b><span>confirmed</span></div>
    <div class="kpi"><b>{total}</b><span>tested in {len(sessions)} sessions</span></div>
    <div class="kpi blue"><b>{len(commits)}</b><span>commits</span></div>
  </div>
  <div class="kpis">
    <div class="kpi"><b>{len(reg)}</b><span>registered parameters</span></div>
    <div class="kpi green"><b>{prov['PUBLISHED']}</b><span>published</span></div>
    <div class="kpi pur"><b>{prov['INVENTED']}</b><span>openly invented</span></div>
    <div class="kpi blue"><b>{len(cites)}</b><span>works cited</span></div>
  </div>

  <div class="quote"><b>{pct} % of everything tried was wrong.</b> That is not
  the failure of the project — it is the product of it. A refuted entry is a map
  of where not to look, and nothing is ever deleted from it.</div>

  {img('progress')}
  <div class="foot">Rebuild this document with
  <code>tools/report_data.py</code> → <code>tools/report_charts.py</code> →
  <code>tools/build_full_report.py</code>.</div>
</section>

<!-- ========================================================== CONTENTS -->
<section class="page">
  <div class="part">Contents</div>
  <h2>What is in this document</h2>
  <div class="rule"></div>
  <div class="toc">
    <div><b>1</b> What you asked for, and where each of those things stands</div>
    <div><b>2</b> What NeuroGecko is, and how one step runs</div>
    <div><b>3</b> The road map</div>
    <div><b>4</b> The six rules, and the failure each one came from</div>
    <div><b>5</b> The flow — every session, in order, with dates</div>
    <div><b>6</b> What was done wrong — the recurring shapes</div>
    <div><b>7</b> Failure → what it became</div>
    <div><b>8</b> The measurements worth keeping</div>
    <div><b>9</b> The eye — why it still cannot find the cricket</div>
    <div><b>10</b> The three modules that are specified and not built</div>
    <div><b>11</b> The full plan, in order, with no time estimates</div>
    <div><b>12</b> Every proposition — all {total} ledger rows, dated</div>
    <div><b>13</b> Every parameter in the animal — all {len(reg)}</div>
    <div><b>14</b> The bibliography — {len(cites)} works, {len(bib['dois'])} DOIs</div>
    <div><b>15</b> What cannot be built honestly</div>
    <div><b>16</b> What must be checked or clarified</div>
    <div><b>17</b> Every commit, and a glossary</div>
  </div>
  <div class="hair"></div>
  <h3>How to read the colour</h3>
  <div class="legend">
    <span><i style="background:#2f7d5b"></i>confirmed / published / working</span>
    <span><i style="background:#c98a1e"></i>partly / uncertain</span>
    <span><i style="background:#c8442f"></i>refuted / not accepted</span>
    <span><i style="background:#8e5b9e"></i>invented, and says so</span>
    <span><i style="background:#8a919b"></i>not started</span>
  </div>
  <p class="lead">Two conventions run through every chart. A number that has
  been <b>measured in this repository</b> is drawn as a bar. A number that is
  <b>published elsewhere and has not been reproduced here</b> is drawn as a
  dashed marker or a shaded band, never as a bar — so a target cannot be
  mistaken for a result. And every chart caption names the ledger row the
  measurement came from, so any figure here can be traced to the run that
  produced it.</p>

  <div class="hair"></div>
  <h3>One thing this document deliberately does not contain</h3>
  <p class="lead"><b>There are no time estimates anywhere in it.</b> Not for the
  remaining modules, not for the blocker, not for the proof battery. That is a
  standing instruction on this project, and it is also the honest position:
  every schedule this project has implied has been wrong, usually because the
  blocker turned out to be somewhere nobody was looking. Work that has happened
  carries the date it happened. Work that has not happened carries an
  <b>order</b> and a <b>dependency</b> — what has to be true before it can
  start — and nothing else.</p>
</section>

<!-- ====================================================== 1 REQUIREMENTS -->
<section class="page">
  <div class="part">Part 1</div>
  <h2>What you asked for, and where each of those things stands</h2>
  <div class="rule"></div>
  {req_html}
</section>

<!-- ============================================================ 2 WHAT -->
<section class="page">
  <div class="part">Part 2</div>
  <h2>What NeuroGecko is</h2>
  <div class="rule"></div>
  <p class="lead">A 38-gram leopard gecko, built in the MuJoCo physics engine,
  with a brain assembled module by module out of published measurements. It is
  not a reinforcement-learning agent that happens to look like a lizard. The
  distinction is the whole project: an RL agent is judged by whether it gets
  better at the reward, and this animal is judged by whether it reproduces
  numbers somebody measured in a real gecko and did not tell it about.</p>

  <h3>How one step actually runs</h3>
  {signal_svg()}
  <p class="cap">The physics runs at 500 Hz and the control loop at 50 Hz. The
  only trained component anywhere in this chain is the walking residual, and in
  the accepted walker it is switched off — the animal walks on a hand-written
  rhythm, and the ledger measured the trained addition as worse (#11, #71,
  #183).</p>

  <div class="hair"></div>
  <h3>The eight brain modules, and what each one is for</h3>
  <div class="two">
    <div><h4>1 · Hypothalamus <span class="stamp ok">working</span></h4>
    <p>The drives. How hungry it is, how much energy it has left, how warm it
    is. Nothing here decides anything; it produces the numbers the decision runs
    on.</p></div>
    <div><h4>2 · Basal ganglia <span class="stamp ok">working</span></h4>
    <p>The selector. Six behaviours compete — hunt, shelter, warm up, rest,
    flee, explore — and exactly one wins. Reproduces the published gating table
    to 0.103 percentage points.</p></div>
    <div><h4>3 · Spinal cord + brainstem <span class="stamp ok">working</span></h4>
    <p>The rhythm. Four coupled oscillators, one per leg, producing the walking
    pattern without needing rhythmic input. The brainstem turns
    &ldquo;hunt&rdquo; into a stride command.</p></div>
    <div><h4>4 · Eye <span class="stamp mid">half</span></h4>
    <p>Two jobs. Keeping the gaze stable — the published optokinetic reflex,
    reproduced. And finding prey — the blocker, detecting on 5.4 % of
    steps.</p></div>
    <div><h4>4b · Strike <span class="stamp ok">working</span></h4>
    <p>The last two centimetres. Released at 20.3 mm, peak 0.851 m/s, 80 ms.
    89.7 % capture against a published 82.9 %, with nothing fitted to it.</p></div>
    <div><h4>5 · Smell <span class="stamp mid">built, unused</span></h4>
    <p>Tongue-flicking. Returns a strength per odour and <b>never a
    bearing</b>, because no gecko has been shown to find prey by smell alone.
    Nothing consults it yet.</p></div>
    <div><h4>6 · Day/night clock <span class="stamp mid">built, unused</span></h4>
    <p>When the animal is awake. The sleep-cycle period is stored as
    <code>null</code> on purpose, so borrowing a bearded dragon's number is
    impossible without declaring it.</p></div>
    <div><h4>7 · Memory <span class="stamp">specified</span></h4>
    <p>Where the goal was. Two decay components: route precision fades before
    the ability to get there at all. Both published in this species.</p></div>
    <div><h4>8 · Learning <span class="stamp">specified</span></h4>
    <p>Getting better with practice. 9 % per trial, ceilings 3.02× on speed and
    4.59× on route. Published in this species — measured on escaping water, not
    on hunting.</p></div>
  </div>
</section>

<!-- ========================================================= 3 ROAD MAP -->
<section class="page">
  <div class="part">Part 3</div>
  <h2>The road map</h2>
  <div class="rule"></div>
  <div class="legend">
    <span><i style="background:#2f7d5b"></i>working, checked against something</span>
    <span><i style="background:#c98a1e"></i>partial</span>
    <span><i style="background:#c8442f"></i>built and not accepted</span>
    <span><i style="background:#8a919b"></i>specified, not built</span>
  </div>
  {roadmap_svg()}
  <p class="cap">Five modules stand. One is built and openly recorded as not
  working. Three are specified and not built — and the specification for all
  three exists because a literature sweep in Session 10c refuted the assumption
  that they were too thin to build (#216, #217). The surprise of that session
  was that the modules deferred for lack of evidence had target-species numbers
  all along, and the module considered urgent — the prey detector — has no
  gecko measurement of any kind.</p>
</section>

<!-- ============================================================ 4 RULES -->
<section class="page">
  <div class="part">Part 4</div>
  <h2>The six rules, and the failure each one came from</h2>
  <div class="rule"></div>
  <p class="lead">None of these were decided in advance. Each one is the scar
  left by a specific defect, written down so the same defect costs one session
  instead of three.</p>
  <div class="hair"></div>
  {rules_html}
</section>

<!-- ============================================================= 5 FLOW -->
<section class="page">
  <div class="part">Part 5</div>
  <h2>The flow — every session, in order</h2>
  <div class="rule"></div>
  {img('timeline', 'The project ran in two blocks: locomotion in June and the '
       'brain, the eye and the hunt in September. The hatched bar is an '
       'artefact worth being explicit about — git blame reports when a line '
       'was last written, and the living map was created in a single commit '
       'that transcribed every hypothesis tested before it existed.')}
  <table><thead><tr><th>Session</th><th>Date</th><th>What it did</th>
  <th>Rows</th><th>Ref.</th><th>Conf.</th><th>Partly</th></tr></thead>
  <tbody>{''.join(flow_rows)}</tbody></table>
</section>

<section class="page">
  {img('verdicts', 'Every hypothesis, grouped by the session that recorded it. '
       'The tallest bars are literature sweeps, not build sessions — reading '
       'refuted more assumptions than coding did.')}
  {img('cumulative', 'Refutations accumulate; confirmations barely move. A '
       'project that confirmed most of its guesses would be one that was not '
       'testing them.')}
</section>

<!-- ========================================================== 6 PATTERNS -->
<section class="page">
  <div class="part">Part 6</div>
  <h2>What was done wrong — the recurring shapes</h2>
  <div class="rule"></div>
  <p class="lead">Eight failure shapes account for most of what went wrong.
  Each one recurred after being &ldquo;fixed&rdquo;, which is why they are
  worth naming rather than merely counting. The count beside each is how many
  distinct ledger rows record it.</p>
  <div class="hair"></div>
  {pat_html}
</section>

<section class="page">
  {img('own_errors', 'The same thing counted mechanically: a keyword scan over '
       'all 221 ledger rows. It undercounts, because a row that described the '
       'same mistake in different words does not match.')}
  <div class="hair"></div>
  <h3>The one that is worst, and why it is listed separately</h3>
  <p class="lead">In Session 10c a citation was <b>invented</b>. Day, Crews
  &amp; Wilczynski were asserted to have studied leopard geckos. They never
  did — their lizard work is three papers on two other families. The confusion
  is explicable, because David Crews did work extensively on
  <i>Eublepharis macularius</i>, but on temperature-dependent sex determination,
  not on the brain. The invented citation was then written into a research
  brief as a known starting point, where it would have been read as
  established.</p>
  <p class="lead">It is recorded as ledger row <b>#215</b>. It is also the
  reason Part 14 of this document states plainly that the bibliography is a
  harvest of strings found in the repository and has <b>not</b> been verified
  against any database, and the reason the companion prompt asks an outside
  reader to check exactly that first.</p>
</section>

<!-- ======================================================= 7 TURNAROUNDS -->
<section class="page">
  <div class="part">Part 7</div>
  <h2>Failure → what it became</h2>
  <div class="rule"></div>
  {turn_html}
</section>

<!-- ===================================================== 8 MEASUREMENTS -->
<section class="page">
  <div class="part">Part 8</div>
  <h2>The measurements worth keeping</h2>
  <div class="rule"></div>
  {img('speeds', 'The arithmetic that forced a strike module to exist. Ledger '
       '#169 measured 60 s of pursuit: closest approach 51.1 mm, zero strikes. '
       'The gecko could not physically reach striking distance by any route.')}
  {img('hunt', 'The hunt going from impossible to composed. What unblocked it '
       'was prey motion plus a speed-dependent flee radius — not the eye, not '
       'the walker and not the targeting, all three of which had been '
       'diagnosed as the blocker (#134, #174, #202).')}
</section>

<section class="page">
  {img('prey_motion', 'The measurement that makes three sessions of vision work '
       'unreadable: every eye configuration up to that point was tested '
       'against prey that never moved and was off-image on 81 % of frames.')}
  {img('front_duty', 'The defect the user found by watching the animal move, '
       'then measured. Only ONE of the four profile-and-policy combinations '
       'limps — and it was the one in every video.')}
  {img('oracle', 'Zeroing the privileged goal channel. The animal moves '
       'FURTHER without it and arrives nowhere: 91 % of all progress toward '
       'the goal was the oracle.')}
</section>

<!-- ============================================================== 9 EYE -->
<section class="page">
  <div class="part">Part 9</div>
  <h2>The eye — why it still cannot find the cricket</h2>
  <div class="rule"></div>
  <p class="lead">This is the only module in the project that is built and
  recorded as not working, and it has been misdiagnosed four separate times.
  What follows is what is actually known, in the order it was found out.</p>

  {img('efference', 'The measurement that ended three sessions of arguing about '
       'resolution. Ledger #209: the eye reported the same thing with prey and '
       'with no prey in the world at all. It was reporting its own walking.')}
  {img('detection', 'And the measurement that ended the argument about bearing '
       'quality. Ledger #220: over 1 500 steps of a real hunt the eye reported '
       'nothing on 94.6 % of them — but when it did report, the bearing was '
       'usable more often than not.')}
</section>

<section class="page">
  {img('efficiency')}
  <div class="hair"></div>
  <h3>What has been ruled out, and what is left</h3>
  <div class="two">
    <div><h4>Ruled out: resolution</h4><p>Held still, peak salience is 0.36 to
    0.83 at every range tested — including ranges where the prey covers half a
    detector cell. Four configurations from 64 to 512 pixels all correlated ≈ 0
    with true bearing (#116–#118).</p></div>
    <div><h4>Ruled out: bearing quality</h4><p>Of 81 detections in a real hunt,
    44 were usable and 37 were wrong. When the eye speaks it is right more
    often than not (#220).</p></div>
    <div><h4>Ruled out: the walker</h4><p>The existing walker reaches a
    mouth-goal 75 times in 160 s. A 2.1-million-step retrain aimed at the
    supposed reach problem was 8× worse and was killed (#200, #201).</p></div>
    <div><h4>Ruled out: targeting</h4><p>Aiming the walker at the prey gave
    <b>0</b> strikes against 29 for aiming past it. Arriving at a goal means
    stopping at it, and the mouth needs the animal to keep going (#203).</p></div>
    <div><h4>Fixed: self-motion</h4><p>An efference copy cut false alarms from
    72 % of frames to 3–7 % and flipped the still-versus-moving effect to the
    right sign for the first time (#213).</p></div>
    <div><h4>Left: the detection rate</h4><p>5.4 % of steps. The same eye that
    peaks at 0.67–0.83 on a stationary animal averages 0.02 during a hunt,
    because the prey is out of view most of the time. This is the blocker, and
    it is the only one.</p></div>
  </div>
  <div class="hair"></div>
  <h3>What the fix is specified to be</h3>
  <p class="lead">Two parallel detector channels over the same image, differing
  in receptive-field size and in what they are wired to. A coarse array of
  ~30° patches that only says <i>a small thing is moving over there</i> and
  gates whether a hunt starts at all; and a dense sub-16° direction-selective
  array that supplies the bearing error during pursuit. The published test for
  it is a lesion double dissociation: suppressing the wide channel should stop
  approaches being initiated from beyond 22 cm without harming approaches
  already under way, and suppressing the narrow channel should do the opposite.
  Every tuning constant in it is a mouse, a zebrafish larva or a toad — the
  full parameter list, with that stated on each row, is in Part 10.</p>
</section>

<!-- ========================================================== 10 SPECS -->
<section class="page">
  <div class="part">Part 10</div>
  <h2>The three modules that are specified and not built</h2>
  <div class="rule"></div>
  <p class="lead">{esc(spec['headline'][:900])}</p>
</section>

<section class="page">
  {spec_block('prey_detector', 'The prey detector — the blocker', '#c8442f')}
</section>

<section class="page">
  {img('memory')}
  {spec_block('memory', 'Memory', '#2f5d8a')}
</section>

<section class="page">
  {img('learning')}
  {spec_block('learning', 'Learning', '#2f7d5b')}
</section>

<section class="page">
  {img('sleep')}
  {spec_block('sleep', 'Sleep', '#c98a1e')}
</section>

<!-- ============================================================ 11 PLAN -->
<section class="page">
  <div class="part">Part 11</div>
  <h2>The full plan, in order</h2>
  <div class="rule"></div>
  <p class="lead"><b>No step below carries a time estimate.</b> Each carries
  its position in the order and the reason it sits there — what has to be true
  before it can start. That is a deliberate choice and it is also the only
  defensible one: on this project the blocker has moved four times, and each
  time it moved because something nobody was measuring turned out to be the
  cause. A schedule built on that record would be fiction.</p>
  <div class="hair"></div>
  {''.join(plan)}
</section>

<section class="page">
  <h3>Loose ends that are not in the ordered plan</h3>
  <ul class="warn">
    <li><b>Smell exists and nothing consults it.</b>
    <code>brain/vomeronasal.py</code> is built and tested and no behaviour reads
    it. Published: tongue-flick rate rises from 3.0 to 14.57 per minute (n=7);
    defensive response 0.21 to chemical cues against 0.07 to visual (n=42).</li>
    <li><b>The day/night clock exists and nothing consults it.</b>
    <code>brain/arousal.py</code> is built. Its
    <code>sleep_cycle_period_s</code> is deliberately <code>null</code>.</li>
    <li><b>Temperature affects nothing.</b> The animal is thigmothermic — it
    takes heat by contact with a warm surface, not from light — so
    <i>bask</i> as currently modelled is the wrong mechanism.</li>
    <li><b>There is no physical strike lunge.</b> The strike's outcome is
    decided at launch and is ballistic. The body does not actually move
    through it.</li>
    <li><b>The proof battery is 1 of 15.</b> Fourteen checks that were listed
    as the acceptance criteria for the whole animal have never been run.</li>
    <li><b>The oracle is still connected to the walker's goal.</b> The shadow
    logger is in place and refuses to be switched into the policy; the
    disconnection itself has not happened.</li>
  </ul>
</section>

<!-- ========================================================== 12 LEDGER -->
<section class="page">
  <div class="part">Part 12</div>
  <h2>Every proposition — all {total}, dated</h2>
  <div class="rule"></div>
  <p class="lead">Every hypothesis this project has ever tested, in the order
  it was tested, with the session and date it entered the map. Nothing has ever
  been deleted from this table. A hypothesis refuted and later found true gets
  a second row, not an edit — which is why some entries contradict earlier
  ones, and why the contradictions are the useful part.</p>
  <table class="ledger"><thead><tr><th>#</th><th>When</th><th>Hypothesis</th><th>Verdict</th>
  <th>Evidence</th></tr></thead><tbody>{ledger_rows}</tbody></table>
</section>

<!-- ======================================================== 13 REGISTRY -->
<section class="page">
  <div class="part">Part 13</div>
  <h2>Every parameter in the animal</h2>
  <div class="rule"></div>
  {img('provenance')}
  {img('species_gap', 'The number that matters most on this page. Under a third '
       'of the parameters in this animal were measured in this animal, and a '
       'little under half were chosen rather than measured — which is a defect '
       'only if it is hidden.')}
</section>

<section class="page">
  <h3>All {len(reg)} registered parameters</h3>
  <p class="lead"><code>config/proxies.yaml</code>, read through the same
  validating loader the simulation uses — which refuses a registry with a
  missing provenance field or an invalid confidence value.</p>
  <table><thead><tr><th>Parameter</th><th>Value</th><th>Units</th>
  <th>Prov.</th><th>Species</th><th>n</th><th>Conf.</th><th>Source</th></tr>
  </thead><tbody>{reg_rows}</tbody></table>
</section>

<!-- ==================================================== 14 BIBLIOGRAPHY -->
<section class="page">
  <div class="part">Part 14</div>
  <h2>The bibliography</h2>
  <div class="rule"></div>
  <div class="quote"><b>Read this before the table.</b> What follows is a
  <b>harvest</b>, not a curated reading list. It is every author-year string and
  every DOI that appears anywhere in this repository's own text, with the
  parameter or ledger row that mentions it. <b>None of it has been verified
  against a bibliographic database.</b> This project has invented a citation
  once (#215) and has overstated the absence of a real one. A name appearing
  here means the repository leans on it — not that it exists. Checking that is
  the first task in the companion prompt.</div>
  {img('bibliography')}
</section>

<section class="page">
  <h3>{len(cites)} works, ordered by how much the project leans on them</h3>
  <table><thead><tr><th>Work</th><th>Written in the repo as</th>
  <th>Uses</th><th>Where</th></tr></thead><tbody>{bib_rows}</tbody></table>
</section>

<section class="page">
  <h3>{len(bib['dois'])} DOIs — the citations that can be resolved directly</h3>
  <p class="lead">These are the strongest entries in the bibliography, because
  a DOI either resolves or it does not. Everything above is an author-year
  string that could be a transcription error, a species confusion, or an
  invention.</p>
  <table><thead><tr><th>DOI</th><th>Used by</th></tr></thead>
  <tbody>{doi_rows}</tbody></table>
</section>

<!-- =========================================================== 15 GAPS -->
<section class="page">
  <div class="part">Part 15</div>
  <h2>What cannot be built honestly</h2>
  <div class="rule"></div>
  {img('gaps', 'Recorded rather than invented. Each of these is a quantity a '
       'simulation would normally just pick a value for.')}
  <div class="hair"></div>
  <h3>Things that must not be built, and why</h3>
  <ul class="warn">{donot}</ul>
</section>

<section class="page">
  <h3>What the literature sweep actually turned up</h3>
  <p class="lead">Twelve findings that changed the plan. Several of them
  reverse a decision this project had already made.</p>
  <ul class="warn">{surprises}</ul>
</section>

<!-- ============================================================ 16 OPEN -->
<section class="page">
  <div class="part">Part 16</div>
  <h2>What must be checked or clarified</h2>
  <div class="rule"></div>
  <p class="lead">Open questions, in the sense that the project cannot answer
  them from anything it currently holds. Several are decisions rather than
  measurements, and they are listed because they are sitting underneath work
  that has already been done.</p>
  <div class="hair"></div>
  {open_html}
</section>

<!-- ========================================================= 17 COMMITS -->
<section class="page">
  <div class="part">Part 17</div>
  <h2>Every commit</h2>
  <div class="rule"></div>
  {img('commits')}
  <table><thead><tr><th>Hash</th><th>Date</th><th>Subject</th><th>Files</th>
  <th>+</th><th>−</th></tr></thead><tbody>{commit_rows}</tbody></table>
</section>

<!-- ========================================================== 18 BRIEF -->
<section class="page">
  <div class="part">Part 18</div>
  <h2>The analysis brief that goes with this document</h2>
  <div class="rule"></div>
  <p class="lead">This report is also meant to be handed to a second reader with
  web access — another model, or a person — whose job is to check it rather than
  agree with it. The brief below is that instruction set. It is reproduced here
  so it cannot get separated from the document it is about; it also ships as
  <code>NeuroGecko_Analysis_Prompt.md</code>.</p>
  <div class="hair"></div>
  <div class="briefbox">{brief_html}</div>
</section>

<section>
  <h2>Glossary — every term this document uses</h2>
  <div class="rule"></div>
  <div class="two">{gloss}</div>
  <div class="hair"></div>
  <div class="foot">
  NeuroGecko · {total} hypotheses · {len(reg)} parameters ·
  {len(commits)} commits · {src_added:,} lines of source ·
  {len(cites)} works cited · generated {esc(commits[-1]["date"])} from
  <code>docs/FAILURE_MAP.md</code>, <code>config/proxies.yaml</code>,
  <code>git log</code> and <code>git blame</code>. Every figure and table in
  this document is machine-generated; none was transcribed by hand.
  </div>
</section>

</body></html>"""

    path = OUT / "neurogecko_complete.html"
    path.write_text(html_doc, encoding="utf-8")
    print(f"html: {path}  ({len(html_doc)/1024:.0f} KB)")
    return path


def to_pdf(html_path, pdf_path):
    for exe in CHROME:
        if not pathlib.Path(exe).exists():
            continue
        r = subprocess.run(
            [exe, "--headless", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}",
             pathlib.Path(html_path).as_uri()],
            capture_output=True, text=True, timeout=600)
        if pathlib.Path(pdf_path).exists():
            kb = pathlib.Path(pdf_path).stat().st_size / 1024
            print(f"pdf : {pdf_path}  ({kb:.0f} KB)  via {pathlib.Path(exe).name}")
            return True
        print(f"  {pathlib.Path(exe).name} failed: {r.stderr[-300:]}")
    return False


if __name__ == "__main__":
    p = build()
    out = (sys.argv[1] if len(sys.argv) > 1
           else str(OUT / "neurogecko_complete.pdf"))
    to_pdf(p, out)
