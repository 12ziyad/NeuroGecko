# Morphology generator portability correction

This is an additive evidence record. Earlier body measurements and their exact
file hashes remain untouched. No morphology, mass, inertia, joint, actuator,
landmark, settled keyframe, or research gate was changed by this correction.

## Observed cause

The AWS test snapshot was inspected read-only at
`/home/ubuntu/GeckoBrain/build_sessions/build-20260904T233438Z/lab-9960117`.
`common.morphology_audit.__file__` and `DEFAULT_XML` both resolved inside that
snapshot, not the parent repository or an earlier experiment. The snapshot source
XML had Windows CRLF line endings, while the local source used LF:

| Source | Bytes | CR / LF counts | Exact raw SHA256 |
| --- | ---: | --- | --- |
| Local legacy XML | 34163 | 0 / 390 | `c4293da9a0e86ed3e43d23c561b55fec32ccc9b32ec99699737c4294345c3c73` |
| AWS snapshot legacy XML | 34553 | 390 / 390 | `92f48b3e30a7d42d65c78fa3f084dfcfd101ddb107f3a07611af389e24f89020` |

Both sources have canonical-LF SHA256
`c4293da9a0e86ed3e43d23c561b55fec32ccc9b32ec99699737c4294345c3c73`.
Canonicalization decodes UTF-8 and maps CRLF or CR to LF, preserving other text
and trailing newlines. It is explicitly not an exact raw-file fingerprint.

## Correction and preserved checks

The generator now embeds an explicitly labeled canonical-LF source hash in its
comment. Its evidence separately records `source_canonical_lf_sha256`,
`source_raw_file_sha256`, and `registry_raw_file_sha256`.

Both saved candidate XML edits were comment-only. Parsed element trees,
including all attributes and the saved stand keyframes, were identical to Git
HEAD `9960117a7cd815947293a4d8c7c9de43b6ae06ff`. ElementTree parsing omits
comments; this comparison is an exact XML-content comparison, not a claim that
all differently expressed physical models can be canonically hashed.

| Candidate | Updated local exact raw SHA256 | Comment-free ElementTree serialization SHA256 (unchanged) |
| --- | --- | --- |
| v1 | `6ee41aa34cd453508e4a199ce7571977ffe37cfa02e518473ef1fc2181721082` | `dbc81102603c4db2f2da319d4d4a6db04ff76c47f1640515e0db06a854a1bd88` |
| v2 | `756765356417749c72150b05da59565ffc97825cf2457e638cd04ea2e1885c4b` | `b178bf26b644b1d38ab9d157e0a3dfc3cb887868896aad0867a9e7f613ebb9be` |

Reproducibility tests still compare element count, order, tags, names, attribute
sets, and every attribute. Numeric XML attributes allow only absolute tolerance
`1e-14` and relative tolerance `2e-13`. Only the solver-generated stand `qpos`
allows absolute tolerance `1e-9`, with no relative tolerance. This corresponds
to one nanometre for translational components and one nanoradian for hinge
components; quaternion components are dimensionless. These are software
serialization/solver reproducibility tolerances, not biological tolerances.

New tests verify identical generated model text from LF and CRLF fixtures,
different raw hashes with identical canonical hashes, rejection of a one-percent
mass change, acceptance of a `1e-11` stand-state perturbation, and rejection of a
`1e-5` perturbation. Existing compiled geometry, inertia, symmetry, topology,
physics-invariance, settled-support, and explicit failed-target checks remain.

Local verification: all 19 morphology tests passed with
`python -m unittest discover -s tests -p test_morphology_audit.py -v`.
The parent task owns the subsequent complete local/AWS suite reruns. This record
does not claim those later runs have passed.
