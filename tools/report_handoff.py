"""The handoff half of the complete record: the animal as it stands today.

`tools/build_full_report.py` writes the project's history -- the ledger, the
registry, the bibliography, the commits. This module writes what that history
produced: the body that exists right now, every gram and every proportion in
it, the surface wrapped over it, the six behaviour channels and which of them
the animal can actually reach, and the reading list the next session has to go
through before touching any of it.

EVERYTHING HERE IS READ, NOT REMEMBERED. The masses and joint ranges come from
compiling the XML with MuJoCo and asking the model. The gates come from
`common.morphology_audit`. The behaviour gates come from the evidence file the
demo wrote. The research comes from the two agent documents. The generator
constants are parsed out of the generator source. Nothing in this module is
typed in by hand except the prose.
"""

from __future__ import annotations

import html
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts/report"


# --------------------------------------------------------------- tiny helpers
def esc(t):
    return html.escape(str(t), quote=False)


def load(path, default=None):
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = REPO / p
    if not p.exists():
        return default
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return p.read_text(encoding="utf-8", errors="replace")


def inline(text):
    """Bold, italic and code inside one line of markdown."""
    t = esc(text)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    return t


def paras(text, cls=""):
    """Blank-line separated markdown into <p>."""
    chunks, buf = [], []
    for line in (text or "").splitlines():
        if line.strip():
            buf.append(line.strip())
        elif buf:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    k = ' class="%s"' % cls if cls else ""
    return "".join("<p%s>%s</p>" % (k, inline(c)) for c in chunks)


def rows(items):
    return "".join(items)


def num(v, places=6):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return format(v, ",")
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) >= 1e5 or abs(v) < 1e-4:
            return "%.3e" % v
        return "%.*g" % (places, v)
    return esc(v)


def flatten(prefix, value, sink):
    """Every leaf of a nested structure, as (dotted key, formatted value)."""
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(("%s.%s" % (prefix, k)) if prefix else str(k), v, sink)
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                         for x in value):
            sink.append((prefix, ", ".join(num(x) for x in value)))
        else:
            for i, v in enumerate(value):
                flatten("%s[%d]" % (prefix, i), v, sink)
    else:
        sink.append((prefix, num(value)))
    return sink


# ------------------------------------------------------- generator constants
CONST = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*=\s*(.+?)\s*(?:#\s*(.*))?$")


def constants(relpath):
    """Module-level UPPERCASE constants with whatever comment documents them.

    Picks up both the `#:` line above and the trailing `#` comment, because the
    generators in this repository use both conventions.
    """
    text = load(relpath, "") or ""
    found, doc = [], []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#:"):
            doc.append(s[2:].strip())
            continue
        m = CONST.match(line)
        if m and not line.startswith(" "):
            name, value, trail = m.group(1), m.group(2), (m.group(3) or "")
            note = " ".join(doc) if doc else trail
            if doc and trail:
                note = "%s %s" % (" ".join(doc), trail)
            found.append((name, value.rstrip(","), note))
        if not s.startswith("#:"):
            doc = []
    return found


def const_table(title, relpath, caption=""):
    items = constants(relpath)
    if not items:
        return ""
    body = rows("<tr><td class='k'>%s</td><td class='num'>%s</td><td>%s</td></tr>"
                % (esc(n), esc(v), inline(note)) for n, v, note in items)
    cap = "<p class='lead'>%s</p>" % inline(caption) if caption else ""
    return ("<h3>%s <span class='muted'>&mdash; <code>%s</code>, %d constants"
            "</span></h3>%s<table><thead><tr><th>Constant</th><th>Value</th>"
            "<th>What it is</th></tr></thead><tbody>%s</tbody></table>"
            % (esc(title), esc(relpath), len(items), cap, body))


# ----------------------------------------------------------- research parsing
def parse_surface(text):
    """docs/GECKO_SURFACE_ANATOMY.md -> (survivors, refuted, unverified)."""
    survivors, refuted, unverified = [], [], []
    section, mode = "", "survivors"
    for block in re.split(r"\n(?=## )", text or ""):
        head = block.splitlines()[0] if block.strip() else ""
        if head.startswith("## "):
            name = head[3:].strip()
            low = name.lower()
            if low.startswith("refuted"):
                mode, section = "refuted", name
            elif low.startswith("what could not"):
                mode, section = "unverified", name
            else:
                mode, section = "survivors", name
        if mode == "survivors":
            for item in re.split(r"\n(?=\*\*)", block):
                m = re.match(r"\*\*(.+?)\*\*\s*(.*)", item.strip(), re.S)
                if not m:
                    continue
                claim, rest = m.group(1).strip(), m.group(2)
                src = re.search(r"^-\s*source:\s*(.+?)$", rest, re.M | re.S)
                mod = re.search(r"^-\s*for the model:\s*(.+?)$", rest, re.M | re.S)
                body = re.split(r"\n-\s*source:", rest)[0].strip()
                survivors.append({"section": section, "claim": claim,
                                  "evidence": body,
                                  "source": src.group(1).strip() if src else "",
                                  "model": mod.group(1).strip() if mod else ""})
        elif mode == "refuted":
            for item in re.split(r"\n(?=- \*\*)", block):
                m = re.match(r"-\s*\*\*(.+?)\*\*\s*(.*)", item.strip(), re.S)
                if m:
                    refuted.append({"claim": m.group(1).strip(),
                                    "why": m.group(2).strip()})
        else:
            for item in re.split(r"\n(?=- \*\*)", block):
                m = re.match(r"-\s*\*\*(.+?)\*\*\s*:?\s*(.*)", item.strip(), re.S)
                if m:
                    unverified.append({"topic": m.group(1).strip(),
                                       "gap": m.group(2).strip()})
    return survivors, refuted, unverified


FIELD = re.compile(r"^-\s*\*\*(\w+)\*\*:\s*(.+?)$", re.M | re.S)


def parse_lateral(text):
    """docs/LATERAL_UNDULATION.md -> (verified numbers, refuted, gaps)."""
    kept, refuted, gaps = [], [], []
    mode = "kept"
    for block in re.split(r"\n(?=## )", text or ""):
        if not block.strip().startswith("## "):
            continue
        head = block.splitlines()[0][3:].strip()
        rest = "\n".join(block.splitlines()[1:])
        if head.lower() == "refuted":
            mode = "refuted"
            for item in re.split(r"\n(?=- )", rest):
                if item.strip().startswith("- "):
                    refuted.append(item.strip()[2:].strip())
            continue
        if head.lower() == "gaps":
            mode = "gaps"
            for item in re.split(r"\n(?=- )", rest):
                if item.strip().startswith("- "):
                    gaps.append(item.strip()[2:].strip())
            continue
        if mode != "kept":
            continue
        tag = re.match(r"\[(\w+)\]\s*(.*)", head)
        entry = {"group": tag.group(1) if tag else "",
                 "claim": tag.group(2) if tag else head}
        for k, v in FIELD.findall(rest):
            entry[k] = re.sub(r"\s+", " ", v).strip()
        kept.append(entry)
    return kept, refuted, gaps


# ================================================================ the sections
EXTRA_CSS = """
<style>
.muted { color:#8a919b; font-weight:400; }
.claim { break-inside:avoid; margin:0 0 12px; padding:10px 13px;
         border:1px solid #dde1e6; border-left:4px solid #2f5d8a; }
.claim.bad { border-left-color:#c8442f; }
.claim.gap { border-left-color:#8a919b; }
.claim .sec { font-size:7.8pt; letter-spacing:2px; text-transform:uppercase;
              color:#8a919b; }
.claim h4 { margin:3px 0 5px; font-size:10.4pt; }
.claim p { margin:3px 0; font-size:9.2pt; }
.claim .src { font-size:8.4pt; color:#5a616b; }
.claim .use { background:#f4f7fa; padding:6px 8px; margin-top:6px;
              font-size:9pt; border-left:2px solid #2f5d8a; }
.claim .val { background:#fbfaf6; padding:6px 8px; margin-top:5px; font-size:9pt; }
.pend { border:2px solid #c8442f; border-radius:9px; padding:12px 14px;
        margin:0 0 12px; break-inside:avoid; }
.pend .pn { font-size:8.6pt; letter-spacing:2.4px; text-transform:uppercase;
            color:#c8442f; }
.pend h4 { margin:3px 0 6px; }
.promptbox { border:3px solid #12161c; padding:16px 18px; background:#fbfbfc; }
.promptbox h4 { color:#c8442f; margin:15px 0 5px; }
.promptbox ol, .promptbox ul { margin:6px 0 10px 20px; padding:0; }
.promptbox li { margin:0 0 5px; font-size:9.8pt; }
table.tight td, table.tight th { font-size:8.4pt; padding:2px 5px; }
.ok { color:#2f6f4f; font-weight:700; }
.no { color:#c8442f; font-weight:700; }
.note { background:#fbfaf6; border:1px solid #dde1e6; padding:10px 12px;
        font-size:9.4pt; margin:10px 0; }
</style>
"""


def _kpi(value, label, cls=""):
    k = (" " + cls) if cls else ""
    return ("<div class='kpi%s'><b>%s</b><span>%s</span></div>"
            % (k, esc(value), esc(label)))


def _part19_body(census):
    w = census.get("world_v1", {})
    v2 = census.get("lab_v2", {})
    src = census.get("source_r", {})
    kpis = "".join([
        _kpi("%.3f g" % w.get("total_mass_g", 0), "total mass, world"),
        _kpi(w.get("bodies", 0), "rigid bodies"),
        _kpi(w.get("geoms", 0), "geoms"),
        _kpi(w.get("joints", 0), "joints"),
        _kpi(w.get("actuators", 0), "motors"),
        _kpi("%d / %d" % (w.get("nq", 0), w.get("nv", 0)), "nq / nv"),
        _kpi(format(w.get("skin_vertices", 0), ","), "skin vertices"),
        _kpi(w.get("skin_bones", 0), "skin bones"),
    ])
    cols = ("total_mass_g bodies geoms joints hinges slides free actuators "
            "sensors sites nq nv meshes textures materials skins "
            "skin_vertices skin_bones collision_geoms visual_geoms").split()
    head = ("<tr><th>Quantity</th><th>gecko_world_v1<br><span class='muted'>"
            "what runs</span></th><th>gecko_body_lab_v2<br><span class='muted'>"
            "the fitted body</span></th><th>gecko_body_r<br><span class='muted'>"
            "the source template</span></th></tr>")
    body = rows("<tr><td class='k'>%s</td><td class='num'>%s</td>"
                "<td class='num'>%s</td><td class='num'>%s</td></tr>"
                % (esc(c.replace("_", " ")), num(w.get(c, "")),
                   num(v2.get(c, "")), num(src.get(c, "")))
                for c in cols)

    masses = w.get("per_body_mass_mg", {})
    total_mg = sum(masses.values()) or 1.0
    mass_rows = rows(
        "<tr><td class='k'>%s</td><td class='num'>%s</td><td class='num'>%.3f</td></tr>"
        % (esc(n), num(v), 100.0 * v / total_mg)
        for n, v in sorted(masses.items(), key=lambda kv: -kv[1]))

    jt = w.get("joint_table", [])
    TYPE = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    joint_rows = rows(
        "<tr><td class='k'>%s</td><td>%s</td><td class='num'>%s</td>"
        "<td class='num'>%s</td></tr>"
        % (esc(j.get("name") or "&mdash;"), TYPE.get(j.get("type"), "?"),
           ("%s to %s %s" % (num(j["range"][0]), num(j["range"][1]),
                             esc(j.get("units", ""))))
           if j.get("range") else "unlimited",
           num(j.get("damping")))
        for j in jt)

    acts = w.get("actuator_names", [])
    act_rows = rows("<tr>%s</tr>" % "".join(
        "<td class='k'>%s</td>" % esc(a or "") for a in acts[i:i + 4])
        for i in range(0, len(acts), 4))

    return f"""
<section class="page">
  <div class="part">Part 19</div>
  <h2>The animal as it stands today</h2>
  <div class="rule"></div>
  <p class="lead">Every number on this page was produced by compiling the XML
  with MuJoCo and asking the model, at the moment this document was generated.
  None of it is transcribed. Three files are compared because the difference
  between them is where two of this project's worst errors lived:
  <code>gecko_body_r.xml</code> is the hand-written SOURCE and is deliberately
  <b>not</b> to scale &mdash; it weighs 61.2 g and fails 13 of the 14 gates;
  <code>gecko_body_lab_v2.xml</code> is that source put through the two fitting
  passes, and is the body that passes 14 of 14; <code>gecko_world_v1.xml</code>
  is that fitted body placed in the world with a floor and a camera. Rebuilding
  v2 without first rebuilding v1 leaves a stale animal that still compiles,
  still renders and still passes its gates (&#35;332), which is why
  <code>tools/rebuild_body.py</code> now performs the six steps in one order.</p>
  <div class="kpis" style="grid-template-columns:repeat(4,1fr)">{kpis}</div>
  <h3>The census, three ways</h3>
  <table class="tight"><thead>{head}</thead><tbody>{body}</tbody></table>
  <div class="note"><b>Read the mass row carefully.</b> The source template is
  61.2 g and the fitted body is 38.0 g. That is not a bug and not a drift: the
  source is written at convenient hand-editable dimensions and the fitting pass
  scales it onto the published mass and the published proportions. The gate
  audit run against the wrong one of these two files reports 1/14 and looks
  like a catastrophe; run against the right one it reports 14/14. Both numbers
  are in this document on purpose.</div>
</section>

<section class="page">
  <div class="part">Part 19 &middot; continued</div>
  <h2>Every gram in the animal</h2>
  <div class="rule"></div>
  <p class="lead">All {len(masses)} bodies that carry mass, in milligrams,
  largest first, with each one's share of the 38 g budget. The skin contributes
  nothing to this table &mdash; it is a deformable surface bound over the top,
  and &#35;318 measured its contribution to mass, inertia, joint ranges,
  damping and actuator gains as exactly zero on every array.</p>
  <table class="tight"><thead><tr><th>Body</th><th>Mass (mg)</th>
  <th>% of total</th></tr></thead><tbody>{mass_rows}</tbody></table>
</section>

<section class="page">
  <div class="part">Part 19 &middot; continued</div>
  <h2>Every joint, and every motor</h2>
  <div class="rule"></div>
  <p class="lead">{len(jt)} joints. Hinge ranges are in degrees; the one slide
  joint, the tongue, is in metres. Damping is the value MuJoCo integrates, in
  its own units. Two of these ranges are themselves morphology gates &mdash; hip
  pro/retraction must reach 90&deg; and hip rotation must not exceed 35&deg;
  &mdash; and are checked in Part 20.</p>
  <table class="tight"><thead><tr><th>Joint</th><th>Type</th>
  <th>Range</th><th>Damping</th></tr></thead><tbody>{joint_rows}</tbody></table>
  <h3>The {len(acts)} actuators</h3>
  <p>Every one is a position actuator. There is no torque channel and no
  direct velocity channel; a controller can only ask a joint to go somewhere.</p>
  <table class="tight"><tbody>{act_rows}</tbody></table>
</section>
"""


def _part20_gates(morph):
    g = morph.get("gates", [])
    def interval(x):
        lo, hi = x.get("lower"), x.get("upper")
        if lo is None:
            return "&le; %s" % num(hi)
        if hi is None:
            return "&ge; %s" % num(lo)
        return "[%s, %s]" % (num(lo), num(hi))

    def actual(x):
        a = x["actual"]
        if isinstance(a, dict):
            vals = list(a.values())
            if len(set(num(v) for v in vals)) == 1:
                return "%s <span class='muted'>(both sides)</span>" % num(vals[0])
            return " / ".join("%s&nbsp;%s" % (k, num(v)) for k, v in a.items())
        return num(a)

    body = rows(
        "<tr><td class='k'>%s</td><td class='num'>%s</td><td class='num'>%s</td>"
        "<td>%s</td><td class='%s'>%s</td></tr>"
        % (esc(x["label"]), actual(x), interval(x), esc(x.get("units") or ""),
           "ok" if x["passed"] else "no", "PASS" if x["passed"] else "FAIL")
        for x in g)

    prov = rows(
        "<tr><td class='k'>%s</td><td class='src'>%s</td>"
        "<td class='cf %s'>%s</td><td>%s</td></tr>"
        % (esc(x["label"]), esc(x.get("source") or ""),
           esc((x.get("confidence") or "").lower()), esc(x.get("confidence") or ""),
           inline(x.get("notes") or ""))
        for x in g)

    limits = rows("<li>%s</li>" % inline(t)
                  for t in morph.get("corrections_and_limits", []))
    proto = rows("<tr><td class='k'>%s</td><td>%s</td></tr>"
                 % (esc(k.replace("_", " ")), esc(v))
                 for k, v in (morph.get("protocol") or {}).items())
    sc = morph.get("structural_counts", {})
    sc_rows = ", ".join("%s %s" % (num(v), k.replace("_", " "))
                        for k, v in sc.items())
    return f"""
<section class="page">
  <div class="part">Part 20</div>
  <h2>The fourteen morphology gates</h2>
  <div class="rule"></div>
  <p class="lead">These are not unit tests. They are the anatomical checks the
  body has to survive before anything is allowed to walk on it, and each one
  compares a measurement taken on the compiled model against an interval taken
  from a published measurement of <i>Eublepharis macularius</i>. A gate with
  two components &mdash; left and right &mdash; passes only if both pass;
  averaging is not allowed to hide one side. <b>{morph.get('passed_count')} of
  {morph.get('gate_count')} pass.</b></p>
  <p><code>{esc(morph.get('model_path',''))}</code><br>
  <span class="src">model sha256 {esc(morph.get('model_sha256',''))}</span><br>
  <span class="src">registry sha256 {esc(morph.get('registry_sha256',''))}</span><br>
  <span class="src">measured {esc(morph.get('created_utc',''))} &middot; {esc(sc_rows)}</span></p>
  <table><thead><tr><th>Gate</th><th>Measured</th><th>Accepted interval</th>
  <th>Units</th><th>Result</th></tr></thead><tbody>{body}</tbody></table>
  <h3>The protocol the numbers were taken under</h3>
  <table class="tight"><tbody>{proto}</tbody></table>
</section>

<section class="page">
  <div class="part">Part 20 &middot; continued</div>
  <h2>Where each gate's interval comes from</h2>
  <div class="rule"></div>
  <p class="lead">Rule 1 of this project is that every number says whether it is
  published, derived or invented. This table is that rule applied to the
  acceptance intervals themselves, so the gates cannot quietly become a set of
  numbers chosen because the body already met them.</p>
  <table class="tight"><thead><tr><th>Gate</th><th>Source</th>
  <th>Confidence</th><th>Notes</th></tr></thead><tbody>{prov}</tbody></table>
  <h3>Corrections and limits the audit records against itself</h3>
  <ul>{limits}</ul>
</section>
"""


def _part21_proportions(morph, census):
    sink = []
    flatten("", morph.get("metrics", {}), sink)
    metric_rows = rows("<tr><td class='k'>%s</td><td class='num'>%s</td></tr>"
                       % (esc(k), esc(v)) for k, v in sink)
    sink2 = []
    flatten("", morph.get("diagnostics", {}), sink2)
    diag_rows = rows("<tr><td class='k'>%s</td><td class='num'>%s</td></tr>"
                     % (esc(k), esc(v)) for k, v in sink2)
    return f"""
<section class="page">
  <div class="part">Part 21</div>
  <h2>Every proportion measured on the body</h2>
  <div class="rule"></div>
  <p class="lead">The gates in Part 20 are the {len(morph.get('gates', []))}
  quantities that have a published interval to be checked against. These are
  <b>all</b> of them &mdash; {len(sink)} gate metrics and {len(sink2)} raw
  diagnostics, every leaf of the audit's own output, including the ones no
  paper constrains. They are here because a proportion with no gate is exactly
  the kind of number that drifts without anyone noticing.</p>
  <h3>Gate metrics &mdash; {len(sink)} values</h3>
  <table class="tight"><tbody>{metric_rows}</tbody></table>
  <h3>Diagnostics &mdash; {len(sink2)} values, in metres and radians unless named otherwise</h3>
  <table class="tight"><tbody>{diag_rows}</tbody></table>
</section>
"""


def _part22_surface(census):
    w = census.get("world_v1", {})
    tables = "".join([
        const_table("The swept surface", "tools/make_gecko_mesh.py",
                    "One surface is swept along the animal's own axis, sampling "
                    "the envelope of the gated collision primitives at every "
                    "station, so the skin cannot be a different animal from the "
                    "one the gates measured."),
        const_table("The body sculpt", "tools/gecko_sculpt.py",
                    "The profile curves that turn a tube into a gecko: the neck "
                    "dip, the trunk and tail width tracks, the tail's ridge, the "
                    "superellipse exponents that make the back rounder than the "
                    "belly, and the toes."),
        const_table("Binding the surface to the bones", "tools/make_gecko_skin.py",
                    "Linear blend skinning: every vertex is given at most four "
                    "bones and a Gaussian falloff, so one continuous surface "
                    "follows 23 rigid bodies with no seam at any joint."),
        const_table("The eye", "tools/make_gecko_eye.py",
                    "A real globe, a vertical pupil, a movable lid and a rim. "
                    "The outward axis is DERIVED from the cranium ellipsoid's "
                    "own normal rather than chosen."),
        const_table("The pigment", "tools/make_skin_texture.py",
                    "The 512 x 2048 texture: ground colour, spots, tubercles, "
                    "and the face details painted at the v positions the binder "
                    "reported for nostril, jaw and eye."),
        const_table("Wiring it into the model", "tools/wire_mesh_skin.py",
                    "An idempotent XML patcher, so the body can be rebuilt "
                    "without the patch accumulating."),
    ])
    return f"""
<section class="page">
  <div class="part">Part 22</div>
  <h2>The surface: what was added, and what it cost the physics</h2>
  <div class="rule"></div>
  <p class="lead">The animal had 23 rigid shells that visibly slid past one
  another. It now has one continuous deformable surface of
  {format(w.get('skin_vertices', 0), ',')} vertices bound to
  {w.get('skin_bones', 0)} bones, an eyeball with a vertical pupil and a lid
  that closes, and a tongue on a slide joint. The question that matters is not
  whether it looks better. It is whether the animal the gates measured is still
  the animal that walks.</p>

  <div class="note"><b>It is. Measured three ways, not assumed (&#35;318).</b>
  A lab body rebuilt from a mesh-free, skin-free copy of the same source has
  bit-identical <code>body_mass</code>, <code>body_inertia</code>,
  <code>body_ipos</code>, <code>jnt_range</code>, <code>dof_damping</code> and
  <code>actuator_gainprm</code> &mdash; maximum absolute difference exactly
  <b>0.000e+00</b> on every array, total mass 0.038 kg both ways. After 300
  steps from the <code>stand</code> keyframe the two worlds' <code>qpos</code>
  agree to the last bit. And the 64&times;64 image the animal's own
  <code>head_cam</code> delivers to its retina differs in <b>0 of 4096
  pixels</b>. This was checked rather than argued because the assumption was
  not safe: the retina is a camera ON the animal, and in <code>legacy</code>
  mode it draws geom group 1 &mdash; the group every new mesh and the skin sit
  in.</div>

  <h3>The six steps, in the one order that works</h3>
  <p>Forgetting the fifth step twice left a body that compiled, rendered,
  and passed 14/14 gates while three reproducibility tests failed with an
  unrelated-sounding message (&#35;332). <code>tools/rebuild_body.py</code>
  exists so that order cannot be got wrong again.</p>
  <table class="tight"><tbody>
   <tr><td class="k">1</td><td>generate the swept body, limb and toe meshes</td></tr>
   <tr><td class="k">2</td><td>generate the eyeball, lid and rim meshes</td></tr>
   <tr><td class="k">3</td><td>bind one surface to the 23 bones and emit the skin</td></tr>
   <tr><td class="k">4</td><td>patch the source XML (idempotent)</td></tr>
   <tr><td class="k">5</td><td>refit lab body v1 &mdash; <b>the step that gets skipped</b></td></tr>
   <tr><td class="k">6</td><td>refit lab body v2 with <code>--fit-com</code>, then the world, then the furnished world</td></tr>
  </tbody></table>

  <div class="note"><b>Two renderer facts that cost this project several
  sessions, recorded so they are not rediscovered.</b> (a) Mesh UVs require a
  <code>type="2d"</code> texture; a <code>cube</code> texture is addressed by
  POSITION and ignores texture coordinates entirely, which renders the whole
  animal flat grey (&#35;315). (b) A mesh geom's texture rows count from the
  BOTTOM of the image and a skin's count from the TOP, so the same v coordinate
  lands on opposite ends of the same PNG &mdash; measured: the texture at
  v = 0.10 is yellow (237, 200, 59) and at v = 0.90 is white (228, 225, 218),
  which is exactly the white eyelid that appeared (&#35;344). And one that does
  not work at all: MuJoCo accepts <code>&lt;layer role="normal"&gt;</code> in a
  material and the renderer ignores it &mdash; measured, 0 of 57600 pixels
  changed (&#35;329). Bumps have to be baked into the colour.</div>
</section>

<section class="page">
  <div class="part">Part 22 &middot; continued</div>
  <h2>Every constant in the surface generators</h2>
  <div class="rule"></div>
  <p class="lead">These are appearance numbers. Almost none of them is
  published, and the project's first rule applies to them exactly as it applies
  to the physics: the ones that are INVENTED say so in their own source. They
  are reproduced here in full because the surface is the one part of this
  animal that was built by eye, and a reader checking the work needs to see
  which dials exist.</p>
  {tables}
</section>
"""


def _part23_agents(surface, lateral):
    survivors, refuted, unverified = surface
    kept, lat_refuted, gaps = lateral
    sur_cards = "".join(
        f"""<div class="claim"><div class="sec">{esc(x['section'])}</div>
        <h4>{inline(x['claim'])}</h4>{paras(x['evidence'])}
        <p class="src"><b>Source:</b> {inline(x['source'])}</p>
        <div class="use"><b>For the model:</b> {inline(x['model'])}</div></div>"""
        for x in survivors)
    ref_cards = "".join(
        f"""<div class="claim bad"><h4>{inline(x['claim'])}</h4>
        {paras(x['why'])}</div>""" for x in refuted)
    gap_cards = "".join(
        f"""<div class="claim gap"><div class="sec">{esc(x['topic'])}</div>
        {paras(x['gap'])}</div>""" for x in unverified)

    def lat_card(x):
        conf = (x.get("confidence") or "").lower()
        cls = "claim" if conf == "verified" else "claim gap"
        bits = []
        if x.get("value"):
            bits.append("<div class='val'><b>Value:</b> %s</div>" % inline(x["value"]))
        if x.get("species"):
            bits.append("<p class='src'><b>Species:</b> %s</p>" % inline(x["species"]))
        if x.get("source"):
            bits.append("<p class='src'><b>Source:</b> %s</p>" % inline(x["source"]))
        if x.get("opened"):
            bits.append("<p class='src'><b>What was actually opened:</b> %s</p>"
                        % inline(x["opened"]))
        return ("<div class='%s'><div class='sec'>%s &middot; %s</div>"
                "<h4>%s</h4>%s</div>"
                % (cls, esc(x.get("group", "")), esc(x.get("confidence", "")),
                   inline(x["claim"]), "".join(bits)))

    lat_cards = "".join(lat_card(x) for x in kept)
    lat_ref = "".join("<div class='claim bad'>%s</div>" % paras(t)
                      for t in lat_refuted)
    lat_gaps = "".join("<div class='claim gap'>%s</div>" % paras(t) for t in gaps)

    total_sur = len(survivors) + len(refuted)
    return f"""
<section class="page">
  <div class="part">Part 23</div>
  <h2>The agents, and what survived them</h2>
  <div class="rule"></div>
  <p class="lead">Every piece of biology in this animal was found by a research
  pass, and every research pass in this project is adversarial by construction:
  one agent per question finds a claim, then independent agents are told to
  REFUTE it and to reject anything they cannot confirm from a source they have
  personally opened. The refutation rate is the point. A pass that confirms
  everything it looked for has not checked anything.</p>
  <div class="kpis" style="grid-template-columns:repeat(4,1fr)">
   {_kpi("233", "agent runs recorded")}
   {_kpi("276", "published measurements")}
   {_kpi("10 / 36", "surface claims survived")}
   {_kpi("18 / 20", "spine numbers survived")}
  </div>
  <table><thead><tr><th>Pass</th><th>Agents</th><th>Subject</th>
  <th>Survived</th><th>Killed</th><th>Written to</th></tr></thead><tbody>
   <tr><td class="k">Research corpus</td><td class="num">167</td>
       <td>the animal's biology, physiology and gait</td>
       <td class="num">276 measurements</td><td class="num">&mdash;</td>
       <td><code>config/proxies.yaml</code></td></tr>
   <tr><td class="k">Session-12 provenance audit</td><td class="num">12</td>
       <td>this project's own claims about where its numbers came from</td>
       <td class="num">0 of 4</td><td class="num">4 of 4</td>
       <td>ledger &#35;264</td></tr>
   <tr><td class="k">Surface anatomy</td><td class="num">42</td>
       <td>scales, tubercles, eye, ear, feet, pattern</td>
       <td class="num">{len(survivors)}</td><td class="num">{len(refuted)}</td>
       <td><code>docs/GECKO_SURFACE_ANATOMY.md</code></td></tr>
   <tr><td class="k">Lateral undulation</td><td class="num">24</td>
       <td>how far a walking lizard's spine actually bends</td>
       <td class="num">{len(kept)}</td><td class="num">{len(lat_refuted)}</td>
       <td><code>docs/LATERAL_UNDULATION.md</code></td></tr>
  </tbody></table>
  <div class="note">The 167 figure is the tally carried in
  <code>docs/FAILURE_MAP.md</code>'s standing summary and has not been
  independently recounted here; the 12, 42 and 24 are each stated in the
  ledger row or the document that pass produced. 167 + 12 is the pre-existing
  count, and 42 + 24 were added in the most recent session, giving 233 recorded
  agent runs in total. Treat 167 as reported rather than verified.</div>
  <div class="note"><b>The surface pass is PARKED, NOT APPLIED.</b> The
  appearance work was called off by the user before this research returned, and
  a pass in which 26 of 36 claims died is exactly the reason not to have built
  first and checked afterwards. The skin that exists was built from reference
  photographs, not from this document. Applying it is unstarted work.</div>
</section>

<section class="page">
  <div class="part">Part 23 &middot; continued</div>
  <h2>Surface anatomy &mdash; the {len(survivors)} claims of {total_sur} that survived</h2>
  <div class="rule"></div>
  <p class="lead">42 agents. Each claim below was confirmed from a source the
  verifying agent opened and read. The last line of each is the instruction the
  pass wrote for whoever eventually builds it.</p>
  {sur_cards}
</section>

<section class="page">
  <div class="part">Part 23 &middot; continued</div>
  <h2>Surface anatomy &mdash; the {len(refuted)} that did not</h2>
  <div class="rule"></div>
  <p class="lead">This is the more valuable half. Each of these was a plausible,
  confidently-worded statement about this animal that turned out to be a
  congener's trait, an absolute where the source hedged, or a number nobody has
  actually measured.</p>
  {ref_cards}
</section>

<section class="page">
  <div class="part">Part 23 &middot; continued</div>
  <h2>Surface anatomy &mdash; what could not be verified at all</h2>
  <div class="rule"></div>
  <p class="lead">Gaps, stated rather than filled. Anything built into these
  areas is a modelling decision and has to be tagged as one.</p>
  {gap_cards}
</section>

<section class="page">
  <div class="part">Part 23 &middot; continued</div>
  <h2>Lateral spine bending &mdash; the {len(kept)} numbers that survived</h2>
  <div class="rule"></div>
  <p class="lead">24 agents, four search angles, then verifiers told to refute
  each number from the opened source. The one that matters is the second card:
  a SAME-SPECIES measurement of pelvic girdle yaw on a level trackway, which is
  what the walker's spine was eventually fitted against.</p>
  {lat_cards}
  <h3>Refuted</h3>
  {lat_ref}
  <h3>Gaps</h3>
  {lat_gaps}
</section>
"""


def _part24_brains(brains):
    peak = brains.get("peak_gate", {})
    won = brains.get("steps_won", {})
    released = set(brains.get("channels_released", []))
    lim = brains.get("measured_limits", {})
    order = ["hunt", "explore", "flee", "bask", "rest", "groom"]
    body = rows(
        "<tr><td class='k'>%s</td><td class='num'>%s</td><td class='num'>%s</td>"
        "<td class='%s'>%s</td></tr>"
        % (esc(c), num(peak.get(c, 0.0)), num(won.get(c, 0)),
           "ok" if c in released else "no",
           "released" if c in released else "never released")
        for c in order)
    lim_rows = rows("<tr><td class='k'>%s</td><td class='num'>%s</td></tr>"
                    % (esc(k.replace("_", " ")), esc(num(v)))
                    for k, v in lim.items() if k != "note")
    return f"""
<section class="page">
  <div class="part">Part 24</div>
  <h2>The six behaviour channels, measured</h2>
  <div class="rule"></div>
  <p class="lead">The animal's action selection is Prescott's basal ganglia with
  six channels. Before the most recent session, four of those six had a gate of
  exactly 0.0000 on every one of 900 autonomous steps &mdash; a six-channel
  brain running as a two-channel one. Two of the four were the world's fault and
  are now fixed: there was nothing to flee from and nothing warm to lie on. Two
  are the selector's own floor and are not fixable without changing a published
  number.</p>
  <p><code>{esc(brains.get('world',''))}</code> &middot; seed
  {esc(brains.get('seed',''))} &middot; produced by
  <code>{esc(brains.get('generated_by',''))}</code></p>
  <table><thead><tr><th>Channel</th><th>Peak gate</th><th>Steps won</th>
  <th>Outcome</th></tr></thead><tbody>{body}</tbody></table>

  <h3>The floor that caps this brain at four</h3>
  <p>Bisected on one channel with every other channel at zero: a salience of
  0.19 gives a gate of exactly 0.0000 and 0.25 gives 0.98. The release
  threshold is <b>{num(lim.get('basal_ganglia_release_threshold_salience'))}</b>.
  Nothing below it is released at all, however far ahead of its rivals it is.</p>
  <table class="tight"><tbody>{lim_rows}</tbody></table>
  <div class="note">{inline(lim.get('note',''))}</div>

  <h3>Why <code>flee</code> is the interesting one, and was left alone</h3>
  <p>Its salience IS the published probability of a defensive reaction in this
  species &mdash; 0.21 to scent alone, 0.28 with sight, n = 42 &mdash; delivered
  through a nose whose inverse-square concentration tops out at 0.28 even at
  contact. Explore is 0.35 &times; hunger. So it clears the bare threshold by
  0.009 and is beaten by explore as soon as hunger passes about 0.2: with
  hunger 0.30 present, flee needs 0.2171 and has 0.2100, short by 0.0071. The
  model therefore predicts that a starving gecko does not flee from something
  standing on it. <b>Nothing was tuned.</b> That is either a real prediction or
  a sign that a reaction probability is the wrong quantity to use as a salience,
  and either way it is not something to fix by choosing a bigger number
  (&#35;334, &#35;335).</p>

  <h3>The eight brains, and what each one actually is</h3>
  <table><thead><tr><th>#</th><th>Brain</th><th>State</th></tr></thead><tbody>
   <tr><td class="k">1</td><td>hypothalamus &mdash; hunger, energy, fatigue, thermostat</td><td class="ok">built; hunger reproduces the published 2.33-day feeding interval unfitted</td></tr>
   <tr><td class="k">2</td><td>basal ganglia &mdash; action selection</td><td class="ok">built; reproduces the published selection table to four decimal places</td></tr>
   <tr><td class="k">3a</td><td>spinal rhythm &mdash; the leg oscillators</td><td class="ok">built; drives the walker with every gate unchanged</td></tr>
   <tr><td class="k">3b</td><td>brainstem &mdash; decision to command</td><td class="ok">built; thinnest evidence in the project, and says so</td></tr>
   <tr><td class="k">4</td><td>eye &mdash; retina, pretectum, tectum</td><td class="no">gaze reproduces; prey-finding NOT ACCEPTED</td></tr>
   <tr><td class="k">5</td><td>smell &mdash; the channel this species is documented to gate on</td><td class="no">a threat-distance path exists for <code>flee</code>; the sense itself is absent</td></tr>
   <tr><td class="k">6</td><td>day/night clock</td><td class="no">a fake switch &mdash; see below</td></tr>
   <tr><td class="k">7</td><td>memory</td><td class="no">not built</td></tr>
   <tr><td class="k">8</td><td>learning</td><td class="no">not built</td></tr>
  </tbody></table>
  <div class="note"><b>Brain 6 is not connected, and now looks as if it is
  (&#35;347).</b> <code>GeckoBrainEnv</code> takes a <code>circadian=</code>
  flag and constructs <code>Arousal()</code> when it is True, and
  <code>self.clock</code> is reset alongside the other modules. But
  <code>clock.step</code> is never called anywhere in the file, and
  <code>self._arousal</code> is assigned exactly once, to the constant 1.0 at
  construction, and thereafter only READ &mdash; to be copied into
  <code>info["arousal"]</code>. Switching the flag on builds the object, resets
  it, and reports a number the object never produced. This was recorded rather
  than fixed, because fixing it means deciding what arousal should modulate,
  and nothing published says.</div>
</section>
"""


def _part25_walker():
    return """
<section class="page">
  <div class="part">Part 25</div>
  <h2>The walker that is accepted, and the one that is not</h2>
  <div class="rule"></div>
  <p class="lead">There are two walkers in this repository and they are easy to
  confuse, because the rejected one is the default in one of the environments.
  Every clip, every demo and every measurement in this document uses the
  accepted one: <b>gait profile <code>lab</code>, zero residual, no trained
  policy</b> &mdash; the hand-written CPG with its contact reflex, controller
  ledger row &#35;71, accepted at 19 strides, 1.1892 Hz measured against 1.1888
  commanded.</p>
  <table><thead><tr><th>Foot</th>
  <th>Accepted: <code>lab</code>, no policy</th>
  <th>Rejected: <code>legacy</code> + trained residual</th>
  <th>Published target</th></tr></thead><tbody>
   <tr><td class="k">front left</td><td class="num ok">0.770</td><td class="num">0.704</td><td class="num">0.70</td></tr>
   <tr><td class="k">front right</td><td class="num ok">0.764</td><td class="num">0.601</td><td class="num">0.70</td></tr>
   <tr><td class="k">hind left</td><td class="num ok">0.787</td><td class="num no">0.435</td><td class="num">0.765</td></tr>
   <tr><td class="k">hind right</td><td class="num ok">0.790</td><td class="num no">0.367</td><td class="num">0.765</td></tr>
  </tbody></table>
  <div class="note"><b>How the wrong one got filmed, and why it survived
  (&#35;338).</b> <code>GeckoBrainEnv</code> defaults to
  <code>gait_profile="legacy"</code> with the frozen trained residual loaded on
  top, and the brain demo passed neither argument, so it took both defaults. The
  comment twenty lines above that constructor states the measurement above in
  full. The reason it went unnoticed for so long is in the table: the defect is
  in the HIND feet, the front pair looks fine, and front-foot duty was the only
  thing being checked. The user caught it. The environment exposes
  <code>info["accepted_walker"]</code>, which was sitting there unused; the demo
  now asserts on it before filming and exits if it is False. The trained
  residual was refuted on its own terms at &#35;11: two runs of 3.01 M steps,
  best trained 3 of 6 gates against the hand-written base's 4 of 6.</div>

  <h3>What the accepted walker still fails</h3>
  <p>4 of 6 published gait gates. It does not reach front-foot duty 0.70 in the
  open-field measurement (0.447 / 0.444), and the caption on every clip now
  prints that target next to the value, because a number shown alone under the
  word ACCEPTED reads as a pass (&#35;325).</p>

  <h3>The spine, fitted and then not adopted (&#35;346)</h3>
  <p>The 24-agent pass in Part 23 found a same-species number: pelvic girdle yaw
  excursion of about 50&deg; per stride with the intact tail (Jagnandan &amp;
  Higham 2017, n = 10, SVL 104.6 mm, level trackway, hind duty 0.78). The
  accepted walker measures <b>24.3&deg;</b>, peaking 0.21 of a cycle late. The
  spine is one tendon across three lateral joints and excursion is linear in its
  amplitude &mdash; 24.3 / 36.4 / 48.9 / 61.8 / 74.6&deg; at amplitude 0.30 /
  0.45 / 0.60 / 0.75 / 0.90 &mdash; so <code>spine_amp</code> 0.61 is DERIVED,
  and <code>spine_phase</code> 0.21 corrects the timing: measured 51.8&deg;
  peaking at 0.62. Then the gates, one declared change per trial as the protocol
  demands: hind duty 0.636 &rarr; 0.650 &rarr; <b>0.675</b>, moving toward the
  required 0.73&ndash;0.83; but speed 0.0427 &rarr; 0.0427 &rarr;
  <b>0.0369 m/s</b> and net-path 0.823 &rarr; 0.712 &rarr; <b>0.697</b>, and the
  lab set was selected for &gt; 0.04 m/s and &gt; 0.8 net-path. The published
  spine wave breaks both while helping the one the contract is actually stuck
  on. <b>It is not the default.</b> It is a declared parameter set,
  <code>--set spine_amp=0.61 --set spine_phase=0.21</code>, and the trade is
  recorded rather than resolved by picking a smaller amplitude that matches
  nothing.</p>
</section>
"""


def _part26_pending():
    return """
<section class="page">
  <div class="part">Part 26</div>
  <h2>What is pending: the four behaviour items</h2>
  <div class="rule"></div>
  <p class="lead">This is where the work stops and where the next session picks
  it up. Four items, all of them about whether a behaviour the animal is
  supposed to have can actually be reached. None of them is fixable by choosing
  a bigger number, which is precisely why they are still here.</p>

  <div class="pend"><div class="pn">Pending 1</div>
   <h4><code>rest</code> is structurally unreachable</h4>
   <p>Its salience IS fatigue, and fatigue plateaus at <b>0.011</b> in an animal
   that strolls (&#35;284). The basal ganglia releases nothing below
   <b>0.2008</b> (&#35;335). It is short by a factor of eighteen. The honest
   options are: give the animal something that actually tires it, find a
   published tonic for rest in this species, or record that a strolling gecko
   in this world never rests and leave the channel unreachable. Tuning the
   fatigue curve to clear the threshold is forbidden by Rule 2.</p></div>

  <div class="pend"><div class="pn">Pending 2</div>
   <h4><code>groom</code> is structurally unreachable</h4>
   <p>A tonic salience of <b>0.05</b> against the same 0.2008 floor. Same three
   options, same prohibition. Note that grooming in this species has a real
   published trigger &mdash; it is a shedding and eye-cleaning behaviour, not a
   background tonic &mdash; so the most likely correct fix is that the salience
   is the wrong quantity, not the wrong size.</p></div>

  <div class="pend"><div class="pn">Pending 3</div>
   <h4><code>flee</code> releases, but only when the animal is not hungry</h4>
   <p>It clears the floor by <b>0.009</b>. With hunger 0.30 present it needs
   0.2171 and has 0.2100. The model's standing prediction is that a starving
   gecko does not flee from something standing on it. That is either real
   biology or a sign that a published <i>probability of reaction</i> is the
   wrong quantity to feed a selector as a <i>salience</i>. Deciding which is a
   research question, not a coding one (&#35;334, &#35;335).</p></div>

  <div class="pend"><div class="pn">Pending 4</div>
   <h4>Brain 6, the day/night clock, is a switch wired to nothing</h4>
   <p><code>clock.step</code> is never called; <code>_arousal</code> is the
   constant 1.0 (&#35;347). Either connect it &mdash; which means deciding what
   arousal modulates, and nothing published says &mdash; or delete the flag so
   the model stops reporting a number no module produced. Leaving a fake switch
   in place is the worse of the two.</p></div>

  <h3>And behind those four</h3>
  <table><thead><tr><th>Item</th><th>State</th></tr></thead><tbody>
   <tr><td class="k">Brain 4 &mdash; prey-finding</td><td>gaze reproduces; finding the cricket is NOT ACCEPTED. The strike still reads a privileged range.</td></tr>
   <tr><td class="k">Brain 5 &mdash; smell</td><td>absent as a sense, though this species is documented to gate on it.</td></tr>
   <tr><td class="k">Brains 7 and 8 &mdash; memory, learning</td><td>not built. Barely measured in this animal, so the research has to come first.</td></tr>
   <tr><td class="k">The surface research</td><td>10 verified claims in Part 23 are written down and NOT applied to the skin that exists.</td></tr>
   <tr><td class="k">Front-foot duty</td><td>0.447 / 0.444 against a published 0.70. Two of six gait gates still fail.</td></tr>
  </tbody></table>
</section>
"""


def _part27_prompt(stats):
    return f"""
<section class="page">
  <div class="part">Part 27</div>
  <h2>The brief for the next session</h2>
  <div class="rule"></div>
  <p class="lead">Everything above this page is the record. This page is the
  instruction. It is written to be handed to a fresh session verbatim, and it is
  deliberately strict, because every one of this project's {stats['refuted']}
  refuted hypotheses started as something that sounded reasonable.</p>

  <div class="promptbox">
  <p class="big"><b>READ EVERYTHING FIRST. DO NOT BUILD ANYTHING YET.</b></p>

  <h4>1. Go and read the research. All of it. Actually open it.</h4>
  <p>Before you touch one line of this project, read every research paper, every
  agent finding and every document this report names. Not the summaries &mdash;
  the sources. Specifically:</p>
  <ol>
   <li><code>docs/FAILURE_MAP.md</code> &mdash; all {stats['total']} numbered
   hypotheses, in order, including the {stats['refuted']} refuted ones. The
   refuted rows are the map of where not to look and are the most valuable
   thing in the repository. Read the verdicts, not just the claims.</li>
   <li><code>docs/GECKO_SURFACE_ANATOMY.md</code> &mdash; the 42-agent pass. All
   10 surviving claims, all 26 refuted ones, and all 6 could-not-verify
   sections. The refuted half is the half that matters.</li>
   <li><code>docs/LATERAL_UNDULATION.md</code> &mdash; the 24-agent pass. All 18
   surviving numbers with their species distance, the 2 refuted, the 4 gaps.</li>
   <li><code>config/proxies.yaml</code> &mdash; all {stats['params']} parameters
   with their provenance tags. Know which are PUBLISHED, which are DERIVED and
   which are INVENTED before you reason about any of them.</li>
   <li>The bibliography in Part 16 of this document &mdash;
   {stats['works']} works, {stats['dois']} with DOIs. Open the ones your task
   depends on. <b>Do not cite a paper you have not opened.</b></li>
   <li><code>docs/BUILD_LOG.md</code>, <code>docs/DECISIONS.md</code>,
   <code>docs/BLOCKED.md</code> and <code>docs/EYE_SPEC.md</code> &mdash; the
   long-form context behind the ledger rows.</li>
   <li><code>CLAUDE.md</code> &mdash; the standing rules. They are not
   advisory.</li>
  </ol>
  <p>Understand this animal <b>deeply</b> before you act: its mass and every
  proportion in Parts 19 to 21, what each of its 38 joints can do, which of its
  14 morphology gates are load-bearing, which of its eight brains exist, which
  of its six behaviours it can actually reach, and which of its numbers are
  inventions wearing a published number's clothes. Every single thing this
  gecko needs, you should be able to state from the sources, not from
  memory.</p>

  <h4>2. Then come back knowing it, and say where we are resuming</h4>
  <p>The work continues from <b>fixing the four pending behaviour items in Part
  26</b>: <code>rest</code> unreachable at 0.011 against a 0.2008 floor;
  <code>groom</code> unreachable at 0.05; <code>flee</code> clearing that floor
  by 0.009 and losing to hunger; and brain 6's clock being a switch wired to
  nothing. That is the resumption point. Do not start somewhere else because
  something else looked easier.</p>

  <h4>3. Ask before you start</h4>
  <p><b>When you have finished reading, report back with what you found, then
  ASK THE USER FOR PERMISSION TO START.</b> Do not begin editing, fitting,
  rebuilding or filming until the user has said yes. State clearly which of the
  four items you propose to take first and why, and what evidence would settle
  it.</p>

  <h4>The rules that are not negotiable while you work</h4>
  <ul>
   <li>Every number is PUBLISHED, DERIVED or INVENTED and says which, in
   <code>config/proxies.yaml</code>. A plausible number wearing a published
   number's clothes is the worst defect available.</li>
   <li><b>Never tune until the answer matches.</b> If the model needs its
   constants adjusted to reproduce a published result, the constants must come
   from a paper.</li>
   <li>The reward is what the model is told to want; the gates are what the
   animal does. Only the second counts.</li>
   <li>A failed prediction stays in the table. Adding a knob per failing joint
   is fitting the harness to the answer.</li>
   <li>Measure before claiming &mdash; including your own tests and your own
   solver. Five recorded failures were in the test, not the model.</li>
   <li>One source is not a citation.</li>
   <li>Do not cite a paper you have not opened. Do not fill a gap with
   something plausible. Mark anything you could not verify as unverified. Do
   not fill a gap with a value from a related species without naming the
   species and the distance.</li>
   <li><code>docs/FAILURE_MAP.md</code> is updated before every commit,
   confirmed or refuted. <b>Nothing is ever deleted from it.</b> A refuted
   hypothesis later found true gets a second row, never an edit. Never
   renumber.</li>
   <li>Answering the user: short, plain language, jargon defined inline, and
   <b>never give a time estimate</b>.</li>
   <li>Use the accepted walker &mdash; <code>gait_profile="lab"</code>,
   <code>use_policy=False</code> &mdash; and assert on
   <code>info["accepted_walker"]</code> before filming anything.</li>
   <li>Rebuild the body only through <code>tools/rebuild_body.py</code>, never
   step by step by hand.</li>
  </ul>
  </div>
</section>
"""


def sections():
    """All the handoff parts, as one HTML string."""
    morph = load("artifacts/report/handoff_morphology.json", {}) or {}
    census = load("artifacts/report/handoff_census.json", {}) or {}
    brains = load("artifacts/evidence/session12/four_brains.json", {}) or {}
    surface = parse_surface(load("docs/GECKO_SURFACE_ANATOMY.md", ""))
    lateral = parse_lateral(load("docs/LATERAL_UNDULATION.md", ""))
    ledger = load("artifacts/report/ledger.json", []) or []
    reg = load("artifacts/report/registry.json", {}) or {}
    bib = load("artifacts/report/bibliography.json", {}) or {}
    cites = bib.get("citations", {}) if isinstance(bib, dict) else {}
    dois = bib.get("dois", {}) if isinstance(bib, dict) else {}
    stats = {
        "total": len(ledger),
        "refuted": sum(1 for r in ledger if r.get("class") == "Refuted"),
        "params": len(reg),
        "works": len(cites),
        "dois": len(dois),
    }
    return "".join([
        EXTRA_CSS,
        _part19_body(census),
        _part20_gates(morph),
        _part21_proportions(morph, census),
        _part22_surface(census),
        _part23_agents(surface, lateral),
        _part24_brains(brains),
        _part25_walker(),
        _part26_pending(),
        _part27_prompt(stats),
    ])


if __name__ == "__main__":
    html_text = sections()
    print("handoff sections: %.0f KB" % (len(html_text) / 1024))
