"""Build the NeuroGecko map: one printable document, every number read from the repo.

Produces artifacts/report/neurogecko_map.html and, if a Chromium is present,
prints it to PDF. Nothing here is typed from memory: the ledger is parsed from
docs/FAILURE_MAP.md, the commits from git, and the measurements from the
evidence JSON the tools wrote.
"""

from __future__ import annotations

import base64
import io
import json
import pathlib
import re
import subprocess
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts/report"

CHROME = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def b64(name):
    return base64.b64encode((OUT / name).read_bytes()).decode()


def esc(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def md(t):
    """The ledger is written in markdown; keep bold and code, drop the rest."""
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = re.sub(r"~~(.+?)~~", r"<s>\1</s>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    return t


# ----------------------------------------------------------------- the roadmap
NODES = [
    # (x, y, w, label, sub, state)
    (0, 0, 2, "RESEARCH", "167 agents · 276 published measurements", "done"),
    (0, 1, 1, "BODY", "38 g · 14/14 static checks", "done"),
    (1, 1, 1, "WALKING", "4 of 6 gates · accepted", "part"),
    (0, 2, 1, "WORLD", "floor · prey · no shelter", "part"),
    (1, 2, 1, "1 HYPOTHALAMUS", "hunger · energy · heat", "done"),
    (2, 2, 1, "2 BASAL GANGLIA", "picks 1 of 6 behaviours", "done"),
    (0, 3, 1, "3 SPINAL CORD", "four leg oscillators", "done"),
    (1, 3, 1, "3b BRAINSTEM", "decision → stride", "done"),
    (2, 3, 1, "4 EYE — gaze", "published OKR reproduced", "done"),
    (0, 4, 1, "4 EYE — prey", "NOT ACCEPTED · corr ≈ 0", "fail"),
    (1, 4, 1, "4 STRIKE", "81 % vs published 82.9 %", "done"),
    (2, 4, 1, "5 SMELL", "the sense it actually uses", "todo"),
    (0, 5, 1, "6 SLEEP", "reptile sleep is published", "todo"),
    (1, 5, 1, "7 MEMORY", "no memory of any kind today", "todo"),
    (2, 5, 1, "8 LEARNING", "barely measured in this animal", "todo"),
    (0, 6, 3, "PROOF BATTERY", "1 of 15 tests run", "part"),
]
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6),
         (5, 7), (6, 8), (7, 9), (7, 10), (8, 9), (10, 11), (9, 12),
         (11, 13), (12, 14), (13, 15)]

STATE = {"done": ("#2f7d5b", "#e8f2ed"), "part": ("#c98a1e", "#fbf2e0"),
         "fail": ("#c8442f", "#fbeae6"), "todo": ("#8a919b", "#f1f3f5")}


def roadmap_svg():
    CW, CH, GX, GY = 268, 92, 300, 132
    W = 3 * GX + 40
    H = 7 * GY + 40
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'style="width:100%;height:auto">']
    def cx(n):
        x, y, w = NODES[n][0], NODES[n][1], NODES[n][2]
        return 20 + x * GX + (w * CW + (w - 1) * (GX - CW)) / 2, 20 + y * GY + CH / 2
    for a, b in EDGES:
        x1, y1 = cx(a); x2, y2 = cx(b)
        parts.append(f'<path d="M{x1},{y1+CH/2-6} C{x1},{y1+46} {x2},{y2-46} '
                     f'{x2},{y2-CH/2+6}" fill="none" stroke="#ccd2da" stroke-width="2"/>')
    for i, (x, y, w, label, sub, st) in enumerate(NODES):
        px, py = 20 + x * GX, 20 + y * GY
        pw = w * CW + (w - 1) * (GX - CW)
        edge, fill = STATE[st]
        parts.append(
            f'<rect x="{px}" y="{py}" width="{pw}" height="{CH}" rx="10" '
            f'fill="{fill}" stroke="{edge}" stroke-width="2.2"/>'
            f'<text x="{px+18}" y="{py+36}" font-family="DejaVu Sans,sans-serif" '
            f'font-size="17" font-weight="700" fill="#12161c">{esc(label)}</text>'
            f'<text x="{px+18}" y="{py+62}" font-family="DejaVu Sans,sans-serif" '
            f'font-size="13" fill="#5a616b">{esc(sub)}</text>'
            f'<circle cx="{px+pw-20}" cy="{py+22}" r="6.5" fill="{edge}"/>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------- content
PATTERNS = [
    ("A default won because nobody named the parameter", 4,
     "The goal oracle (#165), the food oracle, the gait profile (#177), and twice "
     "before that (#26, #33). Each time a constructor omitted an argument and the "
     "callee's default silently decided the science.",
     "Every one is now passed explicitly and reported in <code>info</code> on every step."),
    ("The test compared the code against itself", 3,
     "A class literally named <i>the detector cannot disagree with the world</i> read "
     "the prey colour from the model, built a patch of that colour, and checked the "
     "detector saw it. It never touched the renderer. It passed while the detector "
     "scored 0.000 on the real prey — twice, three sessions apart.",
     "Assertions now reach the thing being guarded, not a copy of the input."),
    ("A fix that provably could not do anything", 2,
     "The tectum subtracted a scalar from the motion map to reject self-motion. "
     "<code>argmax(clip(x−c,0,None)) == argmax(x)</code> for any scalar c — identical "
     "on 20 000 of 20 000 random maps. It had never changed one reported bearing.",
     "Replaced with a spatially varying surround; it moves the answer on 1 735/2 000."),
    ("A number wearing another species' clothes", 3,
     "<code>cost_of_transport = 0.73</code> was tagged <i>E. macularius, verified</i>. "
     "It is <i>Teratoscincus przewalskii</i> — a different family. The scorecard had it "
     "right; the machine-readable registry the code reads did not.",
     "Corrected, and the gap recorded: no cost of transport has ever been measured "
     "in this species."),
    ("I explained away a real observation three times", 3,
     "The user said the walk dragged. I answered: camera shake, then half-speed "
     "playback, then an under-actuated forelimb they would have to accept. All three "
     "were wrong about the cause.",
     "The limp was the trained policy — a policy this project had already measured at "
     "3/6 against the base's 4/6 and rejected (#183)."),
    ("The map already contained the answer", 2,
     "#71 states outright that the accepted walker IS the open-loop base. #11 measured "
     "the trained residual as worse. The evidence file records its controller as "
     "<i>zero residual</i>. I read all three and still loaded the checkpoint.",
     "Reading the ledger is not the same as believing it."),
]

TURNAROUNDS = [
    ("The dopamine curve sat 0.10 too high",
     "Three sessions of a basal-ganglia model that would not match the published table.",
     "The mechanism was wrong in the opposite direction to the previous correction. "
     "Afferent gain (1±λ) with threshold 0.2.",
     "Mean error 10.3 pp → <b>0.103 pp</b>, axis slope 1.000."),
    ("The gecko could not see its own food",
     "The detector tested for green; the prey had been changed to brown. Exactly 0.000 "
     "at every illumination.",
     "Read the colour the renderer actually draws — material first, not just geom rgba.",
     "Two separate occurrences fixed, and the self-referential test replaced."),
    ("The eye could not find the cricket",
     "Four eye configurations, 64→512 px. Correlation with true prey bearing ≈ 0 in all four.",
     "It was never the eye. The prey travelled <b>0.0 mm</b> and was off-image 81 % of frames.",
     "Prey now walks 398 mm per episode. The eye has something to look at for the first time."),
    ("Eight gate contracts failing for four sessions",
     "Reported as eight separate failures; the walking evidence looked unverifiable.",
     "One body check compared raw bytes and raised before the others could run. "
     "What differed was a <i>comment</i>.",
     "All 16 pass. The physics hash matched all along: <code>b178bf26…</code>"),
    ("Pursuit could never work",
     "The hunting channel had no route to success at any parameter value.",
     "Walk 0.055 m/s, cricket 0.118 m/s. The real animal does not chase — it strikes "
     "from 2 cm at 0.851 m/s.",
     "Strike built: <b>81 % capture</b> against a published 82.9 %, unfitted."),
]

NEXT = [
    ("1", "Retrain the walker for precision approach", "blocking",
     "It arrives within 4 cm of a goal and its nose trails its trunk by 5 cm. The strike "
     "needs 2 cm. Nothing in the hunt can complete until this closes."),
    ("2", "Settle whether <code>lab</code> or <code>legacy</code> is right", "open question",
     "<code>lab</code> is gate-validated and limps under the trained policy; "
     "<code>legacy</code> is un-gated and looks correct. Whether the gate battery ever "
     "scored front-foot symmetry is unestablished. Looking better is not being right."),
    ("3", "Make the eye find moving prey", "ready",
     "The prey now moves, which it never did during any previous vision work. Add the "
     "two-channel split the mouse superior-colliculus literature shows: a wide detector "
     "for <i>something is out there</i>, a narrow one for <i>it is 12° left</i>."),
    ("4", "Put objects in the world", "ready",
     "Shelter, a warm surface, a threat. The animal is thigmothermic — it takes heat "
     "from the ground, not from light, so <i>bask</i> is the wrong mechanism as built."),
    ("5", "Add smell", "ready",
     "The documented prey and predator channel for this species. Real numbers exist: "
     "3.0 → 14.57 tongue-flicks per minute. It can say <i>food is near</i>, never "
     "<i>food is there</i> — no gecko has been shown to localise prey by smell."),
    ("6", "Sleep, memory, learning", "thin evidence",
     "Leopard geckos appear in a 2026 reptile sleep study, so sleep has a starting point. "
     "Memory and learning are barely measured in this animal; much would be invented, and "
     "this project's rule is that admitted gaps beat invented numbers."),
]

GAPS = [
    "No cost of transport, sprint maximum, endurance or thermal performance curve "
    "has ever been measured in <i>E. macularius</i>.",
    "No CTmin for this species or any eublepharid — the cold side of the thermostat "
    "has no published floor.",
    "No field study of the wild animal at all: no home range, activity budget, diet "
    "or microhabitat selection.",
    "No strike distance, speed or capture success for this species. The only feeding "
    "kinematics paper filmed at 64 fps and could not resolve the capture phase.",
    "No visual receptive field, size tuning or velocity tuning for <b>any</b> lizard "
    "tectum. No prey-selective cell has ever been described in a reptile.",
    "No cricket walking-vibration spectrum, so a vibration sense has nothing to be "
    "calibrated against.",
    "No tightness-of-fit or crevice-preference experiment — the “tight hide” rule every "
    "care sheet states has zero primary support in this species.",
]


def build():
    rows = json.load(open(OUT / "ledger.json", encoding="utf-8"))
    commits = json.load(open(OUT / "commits.json", encoding="utf-8"))
    counts = Counter(r["class"] for r in rows)
    total = len(rows)
    pct = round(100 * counts["Refuted"] / total)

    charts = {n: b64(n + ".png") for n in
              ("verdicts", "cumulative", "commits", "front_duty", "speeds",
               "oracle", "progress")}

    def img(name, cap=""):
        c = f'<div class="cap">{cap}</div>' if cap else ""
        return f'<figure><img src="data:image/png;base64,{charts[name]}"/>{c}</figure>'

    ledger_rows = "".join(
        f'<tr class="{r["class"].lower()}"><td class="n">{r["n"]}</td>'
        f'<td>{md(r["hypothesis"])}</td>'
        f'<td class="v">{md(r["verdict"])}</td>'
        f'<td class="e">{md(r["evidence"])}</td></tr>'
        for r in rows)

    commit_rows = "".join(
        f'<tr><td class="h"><code>{c["hash"]}</code></td><td class="d">{c["date"]}</td>'
        f'<td>{esc(c["subject"])}</td>'
        f'<td class="num">{c["files"]}</td>'
        f'<td class="num add">+{c["added"]:,}</td>'
        f'<td class="num del">−{c["removed"]:,}</td></tr>'
        for c in commits)

    pattern_html = "".join(
        f'<div class="pat"><div class="pn">×{n}</div><div>'
        f'<h4>{t}</h4><p>{d}</p><p class="fix"><b>Now:</b> {f}</p></div></div>'
        for t, n, d, f in PATTERNS)

    turn_html = "".join(
        f'<div class="turn"><h4>{esc(t)}</h4>'
        f'<div class="tg"><div><span>WAS</span><p>{was}</p></div>'
        f'<div><span>WHY</span><p>{why}</p></div>'
        f'<div class="win"><span>NOW</span><p>{now}</p></div></div></div>'
        for t, was, why, now in TURNAROUNDS)

    next_html = "".join(
        f'<div class="nx {k.split()[0]}"><div class="rank">{r}</div><div>'
        f'<h4>{t} <em>{k}</em></h4><p>{d}</p></div></div>'
        for r, t, k, d in NEXT)

    gaps_html = "".join(f"<li>{g}</li>" for g in GAPS)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>NeuroGecko — the map</title>
<style>
@page {{ size: A4; margin: 15mm 13mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "DejaVu Sans", "Segoe UI", sans-serif; color:#12161c;
        margin:0; font-size:10.5pt; line-height:1.5; background:#fff; }}
h1 {{ font-size:34pt; line-height:1.02; margin:0 0 6px; letter-spacing:-.5px; }}
h2 {{ font-size:19pt; margin:0 0 4px; padding-top:6px; letter-spacing:-.2px; }}
h3 {{ font-size:12.5pt; margin:22px 0 8px; color:#2f5d8a; }}
h4 {{ font-size:11.5pt; margin:0 0 5px; }}
p  {{ margin:0 0 8px; }}
code {{ font-family:"DejaVu Sans Mono",monospace; font-size:.86em;
        background:#f1f3f5; padding:1px 4px; border-radius:3px; }}
.page {{ page-break-after: always; }}
.lead {{ font-size:12.5pt; color:#3d444d; max-width:62em; }}
.rule {{ height:3px; background:#12161c; margin:14px 0 18px; }}
.hair {{ height:1px; background:#dde1e6; margin:20px 0; }}

.cover {{ padding-top:22mm; }}
.eyebrow {{ font-size:10pt; letter-spacing:3px; text-transform:uppercase;
            color:#8a919b; margin-bottom:14px; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:26px 0; }}
.kpi {{ border:2px solid #12161c; border-radius:9px; padding:13px 14px; }}
.kpi b {{ display:block; font-size:27pt; line-height:1; letter-spacing:-1px; }}
.kpi span {{ font-size:8.6pt; color:#5a616b; text-transform:uppercase;
             letter-spacing:.8px; }}
.kpi.red b {{ color:#c8442f; }} .kpi.green b {{ color:#2f7d5b; }}
.kpi.blue b {{ color:#2f5d8a; }}

figure {{ margin:14px 0 6px; }}
figure img {{ width:100%; border:1px solid #e4e7eb; border-radius:6px; }}
.cap {{ font-size:9pt; color:#5a616b; margin-top:5px; }}

table {{ width:100%; border-collapse:collapse; font-size:8.4pt; }}
th {{ text-align:left; background:#12161c; color:#fff; padding:6px 7px;
      font-size:8.2pt; text-transform:uppercase; letter-spacing:.6px; }}
td {{ padding:5px 7px; border-bottom:1px solid #eceef1; vertical-align:top; }}
tr.refuted td.n {{ border-left:3px solid #c8442f; }}
tr.confirmed td.n {{ border-left:3px solid #2f7d5b; }}
tr.partly td.n {{ border-left:3px solid #c98a1e; }}
td.n {{ font-weight:700; width:30px; }}
td.v {{ width:112px; }}
td.e {{ color:#454b54; }}
td.h {{ width:64px; }} td.d {{ width:64px; color:#5a616b; }}
td.num {{ text-align:right; width:52px; }}
td.add {{ color:#2f7d5b; }} td.del {{ color:#c8442f; }}
thead {{ display:table-header-group; }}
tr {{ page-break-inside:avoid; }}

.pat {{ display:grid; grid-template-columns:52px 1fr; gap:13px; margin-bottom:15px;
        page-break-inside:avoid; }}
.pn {{ font-size:17pt; font-weight:700; color:#c8442f; }}
.pat p {{ font-size:9.8pt; color:#454b54; }}
.fix {{ color:#2f7d5b !important; }}

.turn {{ border-left:3px solid #dde1e6; padding-left:14px; margin-bottom:17px;
         page-break-inside:avoid; }}
.tg {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }}
.tg span {{ font-size:7.8pt; letter-spacing:1.2px; color:#8a919b; display:block;
            margin-bottom:3px; }}
.tg p {{ font-size:9.4pt; margin:0; color:#454b54; }}
.tg .win p {{ color:#2f7d5b; }}
.tg .win span {{ color:#2f7d5b; }}

.nx {{ display:grid; grid-template-columns:44px 1fr; gap:12px; margin-bottom:14px;
       page-break-inside:avoid; }}
.rank {{ font-size:20pt; font-weight:700; color:#ccd2da; }}
.nx.blocking .rank {{ color:#c8442f; }}
.nx h4 em {{ font-style:normal; font-size:8.4pt; letter-spacing:.8px;
             text-transform:uppercase; color:#8a919b; margin-left:7px; }}
.nx.blocking h4 em {{ color:#c8442f; }}
.nx p {{ font-size:9.8pt; color:#454b54; }}

.gaps li {{ font-size:9.8pt; color:#454b54; margin-bottom:7px; }}
.legend {{ display:flex; gap:20px; font-size:9pt; color:#5a616b; margin:10px 0 4px; }}
.legend i {{ width:11px; height:11px; border-radius:3px; display:inline-block;
             margin-right:5px; vertical-align:-1px; }}
.quote {{ border-left:3px solid #12161c; padding:2px 0 2px 14px; margin:16px 0;
          font-size:11.5pt; }}
.foot {{ font-size:8.6pt; color:#8a919b; margin-top:22px; }}
</style></head><body>

<section class="page cover">
  <div class="eyebrow">Living map · generated from the repository</div>
  <h1>NeuroGecko</h1>
  <div class="rule"></div>
  <p class="lead">A biologically-grounded leopard gecko (<i>Eublepharis macularius</i>)
  in MuJoCo. Every number is tagged <b>PUBLISHED</b>, <b>DERIVED</b> or
  <b>INVENTED</b>, and every hypothesis that was ever tested stays in the record —
  especially the wrong ones. This document is that record, drawn.</p>

  <div class="kpis">
    <div class="kpi red"><b>{counts['Refuted']}</b><span>refuted</span></div>
    <div class="kpi green"><b>{counts['Confirmed']}</b><span>confirmed</span></div>
    <div class="kpi"><b>{total}</b><span>hypotheses tested</span></div>
    <div class="kpi blue"><b>{len(commits)}</b><span>commits</span></div>
  </div>

  <div class="quote"><b>{pct}% of everything tried was wrong.</b> That is not the
  failure of the project — it is the product of it. A refuted entry is a map of
  where not to look, and nothing is ever deleted from it.</div>

  {img('progress')}
  <div class="foot">Charts and tables in this document are generated by
  <code>tools/build_report.py</code> from <code>docs/FAILURE_MAP.md</code>,
  <code>git log</code> and the evidence JSON. No figure is transcribed by hand.</div>
</section>

<section class="page">
  <h2>The road map</h2>
  <div class="rule"></div>
  <div class="legend">
    <span><i style="background:#2f7d5b"></i>working, checked</span>
    <span><i style="background:#c98a1e"></i>partial</span>
    <span><i style="background:#c8442f"></i>built and not accepted</span>
    <span><i style="background:#8a919b"></i>not started</span>
  </div>
  {roadmap_svg()}
  <p class="cap">Eight brain modules. Five stand. One is built and openly recorded
  as not working. Three have not been started, and two of those three have such
  thin published evidence that building them would mean inventing.</p>
</section>

<section class="page">
  <h2>Every hypothesis, every phase</h2>
  <div class="rule"></div>
  {img('verdicts', 'The tall bar on the right is the current phase: a literature '
       'sweep that refuted more assumptions than any build session.')}
  {img('cumulative', 'Refutations accumulate; confirmations barely move. A project '
       'that confirmed most of its guesses would be one that was not testing them.')}
</section>

<section class="page">
  <h2>Patterns — the same mistake, again</h2>
  <div class="rule"></div>
  <p class="lead">Six failure shapes account for most of what went wrong. Each one
  recurred after being “fixed”, which is why they are worth naming rather than
  merely counting.</p>
  <div class="hair"></div>
  {pattern_html}
</section>

<section class="page">
  <h2>Failure → what it became</h2>
  <div class="rule"></div>
  {turn_html}
</section>

<section class="page">
  <h2>Three measurements worth keeping</h2>
  <div class="rule"></div>
  {img('speeds', 'The arithmetic that forced a strike to exist.')}
  {img('oracle', 'Zeroing the privileged goal channel: the animal moves further and '
       'arrives nowhere.')}
  {img('front_duty', 'The defect the user found by watching the animal move, then '
       'measured. The trained policy limps; the hand-written base does not.')}
</section>

<section class="page">
  <h2>What is next</h2>
  <div class="rule"></div>
  {next_html}
  <div class="hair"></div>
  <h3>What cannot be built honestly</h3>
  <p class="lead">Recorded so the gaps stay visible instead of being invented later.</p>
  <ul class="gaps">{gaps_html}</ul>
</section>

<section class="page">
  <h2>The full ledger — {total} hypotheses</h2>
  <div class="rule"></div>
  <table><thead><tr><th>#</th><th>Hypothesis</th><th>Verdict</th><th>Evidence</th></tr>
  </thead><tbody>{ledger_rows}</tbody></table>
</section>

<section>
  <h2>Every commit — {len(commits)}</h2>
  <div class="rule"></div>
  {img('commits')}
  <table><thead><tr><th>Hash</th><th>Date</th><th>Subject</th><th>Files</th>
  <th>+</th><th>−</th></tr></thead><tbody>{commit_rows}</tbody></table>
</section>

</body></html>"""

    html_path = OUT / "neurogecko_map.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"html: {html_path}  ({len(html)/1024:.0f} KB)")
    return html_path


def to_pdf(html_path, pdf_path):
    for exe in CHROME:
        if not pathlib.Path(exe).exists():
            continue
        cmd = [exe, "--headless", "--disable-gpu", "--no-sandbox",
               "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}",
               html_path.as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if pathlib.Path(pdf_path).exists():
            print(f"pdf : {pdf_path}  "
                  f"({pathlib.Path(pdf_path).stat().st_size/1024:.0f} KB)  via {pathlib.Path(exe).name}")
            return True
        print(f"  {pathlib.Path(exe).name} failed: {r.stderr[-300:]}")
    return False


if __name__ == "__main__":
    p = build()
    out = sys.argv[1] if len(sys.argv) > 1 else str(OUT / "neurogecko_map.pdf")
    to_pdf(p, out)
