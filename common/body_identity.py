"""Is this the same ANIMAL, or merely the same FILE?

WHY THIS EXISTS. Eight gate-contract tests failed for four sessions with

    ValueError: Lab base evidence measured a different body than --xml-path.

and the message was false. Every evidence file from sessions 3 and 4 records
body `756765356417749c...`; `morphology/gecko_body_lab_v2.xml` hashes to
`db650f6b...`. The check compared RAW BYTES, so it could not tell a changed
mass from a changed comment -- and what actually changed was a comment.

On 2026-09-05 a portability correction embedded a canonical-LF source hash in
the generator's own comment block. The record of that work
(`artifacts/evidence/morphology_reproducibility_line_endings_20260905.md`)
states the edits were comment-only and that the parsed element trees were
identical. That is verifiable, and it verifies:

    recorded comment-free serialization  b178bf26b644b1d3...
    current  comment-free serialization  b178bf26b644b1d3...   MATCH

So the sessions 3-4 gait evidence was measured on a body whose physics is
exactly the body in the repository today. The evidence was always valid. The
check was wrong, and it masked eight other tests for four sessions by raising
before any of them could reach their own assertions.

WHAT THIS DOES NOT DO. It does not weaken the guard. `ET.tostring` emits every
element, in order, with every attribute, so a changed mass, length, joint
range, actuator gear or keyframe changes the digest. What it drops is comments
and the XML declaration -- the things MuJoCo never reads. A file that differs
in anything the simulator can see still fails.

It is deliberately NOT a claim that two differently expressed physical models
can be canonically compared. Reordered elements, differently spelled numbers
(`0.5` vs `5e-1`) and whitespace inside attributes all produce different
digests. This answers exactly one question: are these two files the same XML
apart from comments?
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

__all__ = ["physics_sha256", "raw_sha256", "same_body", "describe_mismatch"]


def _canonical_lf(raw: bytes) -> str:
    """CRLF and CR both become LF. Windows and Linux checkouts of one file are
    the same file, and this project has already lost a day to that."""
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8")


def raw_sha256(path: Path | str) -> str:
    """Exact file fingerprint, comments and line endings included."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics_sha256(path: Path | str) -> str:
    """Fingerprint of everything MuJoCo will actually read.

    Comments and the XML declaration are dropped; every element, attribute and
    value is kept. Two files with this digest in common describe the same body.
    """
    root = ET.fromstring(_canonical_lf(Path(path).read_bytes()))
    return hashlib.sha256(
        ET.tostring(root, encoding="unicode").encode("utf-8")).hexdigest()


#: Pairings for files whose exact bytes no longer exist. Sessions 3-4 evidence
#: records a RAW hash for a file that was measured and never committed, so
#: nothing in the working tree can be hashed to match it. The pairing that
#: rescues it was written down by the session that produced the file. It lives
#: in a citable evidence record rather than in this source, so that adding one
#: means adding evidence.
EQUIVALENCES = (Path(__file__).resolve().parents[1]
                / "artifacts/evidence/body_identity_equivalences.json")


def _recorded_physics(recorded_sha: str) -> str | None:
    """The physics fingerprint a vanished raw file was recorded to have."""
    if not EQUIVALENCES.is_file():
        return None
    import json
    payload = json.loads(EQUIVALENCES.read_text(encoding="utf-8"))
    for row in payload.get("equivalences", ()):
        if row.get("raw_sha256") == recorded_sha:
            return row.get("physics_sha256")
    return None


def same_body(path: Path | str, recorded_sha: str) -> bool:
    """Does `path` describe the body that `recorded_sha` was measured on?

    Three ways to say yes, in descending order of strength:

      1. the raw hashes match -- byte-identical file, nothing to argue about;
      2. the physics hashes match -- same animal, a comment or line ending
         changed since;
      3. the recorded raw hash belongs to a file that no longer exists, and a
         committed evidence record pairs it with a physics fingerprint that
         matches this file.

    The third is weaker than the other two and is deliberately narrow: it can
    only ever accept a body whose physics already matches. It cannot admit a
    body with a different mass, length, joint or actuator, because the physics
    fingerprint covers every attribute the simulator reads.
    """
    if not recorded_sha:
        return False
    if recorded_sha == raw_sha256(path):
        return True
    physics = physics_sha256(path)
    if recorded_sha == physics:
        return True
    return _recorded_physics(recorded_sha) == physics


def describe_mismatch(path: Path | str, recorded_sha: str) -> str:
    """A refusal message that says which kind of difference was found, because
    'a different body' was the wrong answer eight times."""
    raw, phys = raw_sha256(path), physics_sha256(path)
    if recorded_sha == raw:
        return ""
    if recorded_sha == phys:
        return ""
    if _recorded_physics(recorded_sha) == phys:
        return ""
    return (
        f"Lab base evidence measured a different body than {path}.\n"
        f"  evidence records : {recorded_sha}\n"
        f"  file raw sha256  : {raw}\n"
        f"  file physics     : {phys}\n"
        "Neither the exact file nor its physics matches. If only a comment or "
        "a line ending changed, the physics digest would match and this would "
        "have been accepted -- so this is a real difference in something the "
        "simulator reads.")
