"""Extract every fact the report needs, from the repository only.

Four files, all machine-read, none typed from memory:
  ledger.json        every numbered hypothesis in docs/FAILURE_MAP.md
  commits.json       every commit with churn, from git
  registry.json      every parameter in config/proxies.yaml with its provenance
  bibliography.json  every author-year citation and DOI found in the registry

The project's first rule is that a number must say where it came from. This
file is that rule applied to the report itself.
"""

from __future__ import annotations

import io
import json
import pathlib
import re
import subprocess
import unicodedata
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts/report"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------- the ledger
def ledger():
    text = (REPO / "docs/FAILURE_MAP.md").read_text(encoding="utf-8")
    section = "unsectioned"
    rows = []
    for line in text.splitlines():
        m = re.match(r"^##+\s+(?:Ledger\s+[-—]+\s+)?(.*)$", line)
        if m and line.startswith("## "):
            section = m.group(1).strip()
            continue
        m = re.match(r"^\|\s*(\d+)\s*\|(.*)\|(.*)\|(.*)\|\s*$", line)
        if not m:
            continue
        n, hypo, verdict, evidence = m.groups()
        vplain = re.sub(r"[*`]", "", verdict).strip()
        low = vplain.lower()
        cls = ("Partly" if low.startswith("partly") or "partly" in low[:12]
               else "Refuted" if low.startswith("refuted")
               else "Confirmed" if low.startswith("confirmed")
               else "Other")
        rows.append({"n": int(n), "section": section, "hypothesis": hypo.strip(),
                     "verdict": verdict.strip(), "verdict_plain": vplain,
                     "class": cls, "evidence": evidence.strip()})
    rows.sort(key=lambda r: r["n"])
    return rows


def blame_dates():
    """Which commit first wrote each line of the map, so a hypothesis has a date.

    The ledger's section headings stopped tracking sessions once three sessions
    appended rows to the same table. Rather than guess the boundaries, ask git
    which commit introduced each row. The answer is a fact, not a reading.
    """
    raw = subprocess.run(
        ["git", "blame", "--line-porcelain", "--", "docs/FAILURE_MAP.md"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        errors="replace").stdout
    out, cur = {}, {}
    for line in raw.splitlines():
        if re.match(r"^[0-9a-f]{40} ", line):
            cur = {"sha": line.split()[0][:7]}
        elif line.startswith("author-time "):
            cur["t"] = int(line.split()[1])
        elif line.startswith("summary "):
            cur["subject"] = line[8:]
        elif line.startswith("	"):
            m = re.match(r"^	\|\s*(\d+)\s*\|", line)
            if m:
                out[int(m.group(1))] = dict(cur)
    return out


# -------------------------------------------------------------- the commits
def commits():
    fmt = "%H%x1f%h%x1f%ad%x1f%s"
    raw = subprocess.run(
        ["git", "log", "--reverse", f"--pretty=format:{fmt}", "--date=short",
         "--numstat"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8").stdout
    out, cur = [], None
    for line in raw.splitlines():
        if "\x1f" in line:
            if cur:
                out.append(cur)
            full, short, date, subject = line.split("\x1f")
            cur = {"hash": short, "full": full, "date": date, "subject": subject,
                   "files": 0, "added": 0, "removed": 0,
                   "data_files": 0, "data_added": 0}
        elif line.strip() and cur:
            parts = line.split("\t")
            if len(parts) == 3:
                a, r, path = parts
                add = int(a) if a.isdigit() else 0
                rem = int(r) if r.isdigit() else 0
                # RECORDED OUTPUT IS NOT WRITTEN CODE. A single physics trace is
                # 1.5 million lines; counting it as authorship would turn the
                # churn chart into a picture of how many episodes were logged.
                # Split it out and report both, rather than quietly dropping it.
                if path.startswith("artifacts/") or path.startswith("runs/"):
                    cur["data_files"] += 1
                    cur["data_added"] += add
                else:
                    cur["files"] += 1
                    cur["added"] += add
                    cur["removed"] += rem
    if cur:
        out.append(cur)
    return out


# ------------------------------------------------------------- the registry
def registry():
    """proxies.yaml is deliberately written in the JSON subset of YAML 1.2 so
    that it loads with the standard library and the repo carries no PyYAML
    dependency. `species` is where the PUBLISHED / DERIVED / INVENTED tag
    lives: a real species name means published in that animal, `derived` means
    computed from something published, `INVENTED` says so outright."""
    doc = json.loads((REPO / "config/proxies.yaml").read_text(encoding="utf-8"))
    flat = {}
    for key, node in doc["entries"].items():
        if not isinstance(node, dict):
            continue
        sp = node.get("species")
        flat[key] = {
            "value": node.get("value"),
            "units": node.get("units"),
            "species": sp,
            "provenance": ("INVENTED" if sp == "INVENTED"
                           else "DERIVED" if sp == "derived"
                           else "PUBLISHED" if sp else "UNTAGGED"),
            "n": node.get("sample_size"),
            "temp_C": node.get("recording_temp_C"),
            "confidence": node.get("confidence"),
            "source": node.get("source"),
            "notes": node.get("notes"),
        }
    return flat


# --------------------------------------------------------- the bibliography
AUTHOR_YEAR = re.compile(
    r"([A-Z][A-Za-z\-'À-ɏ]+"
    r"(?:\s*(?:&|and|,)\s*[A-Z][A-Za-z\-'À-ɏ]+)*"
    r"(?:\s+et\s+al\.?)?)[,\s]+\(?((?:19|20)\d{2})\)?")
DOI = re.compile(r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)(10\.\S+?)(?=[\s,;)]|$)", re.I)


#: Words that pass the author-year shape and are not authors. Every one of
#: these was actually produced by the scan; the list is a record of what the
#: regex gets wrong, not a guess at what it might.
NOT_AUTHORS = {
    # ordinary words that begin a sentence ending in a year
    "The", "This", "That", "It", "In", "By", "For", "From", "Session",
    "Sessions", "Phase", "Ledger", "Figure", "Table", "See", "But", "And",
    "A", "An", "As", "At", "On", "One", "Two", "No", "Not", "All", "Both",
    "Published", "Derived", "Invented", "Measured", "Confirmed", "Refuted",
    "Note", "Notes", "Source", "Sources", "About", "After", "Before", "Since",
    "Between", "Its", "Their", "There", "They", "We", "I", "Only", "Same",
    "Used", "Using", "Built", "Added", "Corrected", "Verified", "New",
    # journals and conferences cited without an author -- a venue is not a work
    "Nature", "Science", "PNAS", "Neuron", "Cell", "eLife", "Life", "Biol",
    "JEB", "ICLR", "NeurIPS", "CoRL", "ICRA", "IROS", "RSS", "SAB", "Report",
    "Methods", "Nexus", "Robotics", "Systems", "Communications", "Current",
    "Frontiers", "Proceedings", "Journal", "Scientific", "Trends", "Annual",
    "Xiv", "Rxiv", "Creatures", "Puppeteer", "Laudakia", "SVL",
    # month abbreviations, from "(Jun 2024)"-style access notes
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Sept",
    "Oct", "Nov", "Dec", "January", "February", "March", "April", "June",
    "July", "August", "September", "October", "November", "December",
}

#: Bare initials -- "SR et al. 1985" is a citation whose surname the regex
#: failed to reach, not an author called SR. Dropped rather than guessed at.
INITIALS = re.compile(r"^[A-Z]{1,4}$")

#: Surname prefixes the regex clips off, because they are lower case and the
#: pattern anchors on a capital. Restored so "Damme & Cooper" and "Van Damme &
#: Cooper" do not become two different papers.
PREFIX_FIX = {
    "Damme & Cooper 2017": "Van Damme & Cooper 2017",
    "Hemmen 2001": "van Hemmen 2001",
    "Nardo 1995": "Di Nardo 1995",
}


def _canonical(key):
    """Collapse `X et al. Y` and `X, Y & Z Y` onto the same paper.

    The registry cites the same work both ways in different fields. Left alone
    the bibliography double-counts, which for a document whose whole subject is
    provenance is the wrong kind of error to ship.
    """
    key = PREFIX_FIX.get(key, key)
    m = re.match(r"^(.*?)\s+((?:19|20)\d{2})$", key)
    if not m:
        return key, key
    names, year = m.groups()
    first = re.split(r"\s*(?:,|&|\band\b)\s*", names)[0].strip()
    first = re.sub(r"\s+et\s+al\.?$", "", first).strip()
    # Frydlova and Frýdlová are the same surname typed twice. Fold the accents
    # for the KEY only; the original spelling is kept in `written_as`.
    fold = "".join(c for c in unicodedata.normalize("NFKD", first)
                   if not unicodedata.combining(c))
    return f"{fold} {year}", key


def bibliography(reg, led, extra_docs):
    """Every author-year string and DOI anywhere in the repository's own text.

    This is a HARVEST, not a curated reading list. It reports where each name
    appears so a claim can be traced back to the parameter or the ledger row
    that leans on it. A name here has NOT been verified to exist -- one entry
    in the ledger (#215) is a citation this project invented, and the harvest
    would list that too. Verification is the reader's job, and the companion
    prompt asks for exactly that.
    """
    hits, dois, variants = {}, {}, {}

    def scan(blob, tag):
        if not blob:
            return
        for d in DOI.findall(blob):
            dois.setdefault(d.rstrip(".,);"), set()).add(tag)
        for m in AUTHOR_YEAR.finditer(blob):
            raw = re.sub(r"\s+", " ", f"{m.group(1).strip()} {m.group(2)}")
            lead = m.group(1).split()[0].strip(",&")
            if lead in NOT_AUTHORS or INITIALS.match(lead) or not (6 < len(raw) < 62):
                continue
            canon, full = _canonical(raw)
            hits.setdefault(canon, set()).add(tag)
            variants.setdefault(canon, set()).add(full)

    for k, v in reg.items():
        scan(f"{v.get('source') or ''} {v.get('notes') or ''}", k)
    for r in led:
        scan(f"{r['hypothesis']} {r['evidence']}", f"#{r['n']}")
    for path in extra_docs:
        rel = str(pathlib.Path(path).relative_to(REPO)).replace("\\", "/")
        scan(pathlib.Path(path).read_text(encoding="utf-8", errors="replace"), rel)

    return ({k: {"cited_by": sorted(v)[:24], "count": len(v),
                 "written_as": sorted(variants[k])}
             for k, v in sorted(hits.items())},
            {k: sorted(v) for k, v in sorted(dois.items())})


#: A commit subject reading "Session 10b: ..." names the session. Anything
#: else is left unlabelled rather than assigned by guesswork.
SESSION_RE = re.compile(r"^Session\s+([0-9]+[a-z]*)(?![a-z0-9])", re.I)


def main():
    led = ledger()
    dates = blame_dates()
    for r in led:
        d = dates.get(r["n"])
        if not d:
            continue
        r["commit"] = d["sha"]
        r["date"] = __import__("datetime").datetime.utcfromtimestamp(
            d["t"]).strftime("%Y-%m-%d")
        r["commit_subject"] = d.get("subject", "")
        m = SESSION_RE.match(d.get("subject", ""))
        r["session"] = m.group(1) if m else None
    com = commits()
    reg = registry()
    extra = sorted((REPO / "docs/research").glob("*.md")) + [
        REPO / "docs/EYE_SPEC.md", REPO / "docs/DECISIONS.md",
        REPO / "docs/brain_agwm_final_plan.md", REPO / "docs/BLOCKED.md",
    ]
    cites, dois = bibliography(reg, led, [p for p in extra if p.exists()])

    for name, obj in (("ledger", led), ("commits", com), ("registry", reg),
                      ("bibliography", {"citations": cites, "dois": dois})):
        (OUT / f"{name}.json").write_text(
            json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")

    dated = sum(1 for r in led if r.get("date"))
    print(f"ledger       {len(led):>5} rows   {dict(Counter(r['class'] for r in led))}"
          f"   dated {dated}/{len(led)}"
          f"   sessions {len({r.get('session') for r in led if r.get('session')})}")
    print(f"commits      {len(com):>5}        "
          f"source +{sum(c['added'] for c in com):,} / -{sum(c['removed'] for c in com):,}"
          f"   recorded data +{sum(c['data_added'] for c in com):,}")
    print(f"registry     {len(reg):>5} params  "
          f"{dict(Counter((r['species'] or '?') for r in reg.values()).most_common(4))}")
    print(f"bibliography {len(cites):>5} author-year, {len(dois)} DOIs")


if __name__ == "__main__":
    sys.exit(main())
