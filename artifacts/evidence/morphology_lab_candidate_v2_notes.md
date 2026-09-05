# Lab body v2: constrained CoM calibration

Result: **14/14 engineering acceptance bands pass**. This does not mean both
published means were matched exactly, or that per-segment masses were measured.
The candidate stays opt-in; neither the legacy body nor v1 was overwritten.

| Quantity | v1 | v2 | Requested target |
|---|---:|---:|---:|
| Whole mass (g) | 38 | 38 | 38 +/- 4 |
| Tail mass fraction | 0.22 | 0.22 | 0.19–0.25 |
| Intact CoM behind snout / SVL | 0.762056 | 0.678112 | 0.659 +/- 0.020 |
| Tail-excluded CoM / SVL | 0.609153 | 0.527000 | 0.527 +/- 0.020 |
| Settled hip height / SVL | 0.148599 | 0.148877 | 0.150 +/- 0.010 |
| Settled shoulder height / SVL | 0.109811 | 0.109308 | 0.113 +/- 0.010 |

## What was verified in the source

The [primary paper](https://doi.org/10.1242/jeb.110916) reports CoMs of 65.9%
and 52.7% SVL. Its seven adult females averaged **121.9 mm SVL**, unlike our
106 mm model. CoM measurements used frozen, rigid specimens; the original tail
was reattached. The recorded posture is not sufficiently specified to reproduce
it. Applying those dimensionless measurements to a straight neutral model is
an explicit transfer assumption, not a replication of walking CoM.

## The inverse fit

The non-tail fit has 12 symmetric mass groups and two equality constraints
(total mass and longitudinal moment). Minimum relative entropy to v1 chooses
one of many possible solutions, with each group restricted to 0.5–2 times its
prior mass. Left/right limb masses remain equal. These priors and bounds are
engineering choices, not biological measurements.

The five tail masses preserve total tail mass. Each remains positive, at least
0.1 times its v1 mass. A separate effective-density guard limits mass divided
by each fixed capsule volume. The guard is **1.5 times** the approximate bulk
density derived from the source's mean mass, tail fraction and tail volume:
1666.9 versus 1111.3 kg/m^3. It is a sensitivity assumption, not a measured
species-specific material limit. Collision envelopes are not anatomical tissue
segments; their overlapping volumes are an optimistic volume estimate.

## Why the exact pair of means was not forced

With tail fraction 0.22, matching both means requires tail CoM at 1.127 SVL
behind the snout. Existing segment centroids are 1.0929, 1.2655, 1.4181, 1.5541
and 1.6736 SVL. Even placing every other gram at the second centroid requires
at least **80.24% of tail mass in segment 1**: 6.708 g in a 2.569 cm^3 capsule,
an effective density of at least **2611.5 kg/m^3**. Positive distal-mass guards
make that lower bound stricter. No landmarks or geometric centroids were moved.

The selected guard's nearest feasible tail CoM is 1.21387 SVL, producing
whole-model CoM 0.678112 SVL. That is inside the requested band but near its
upper edge. The JSON report records `target_feasible: false` for the exact tail
mean, even though all fourteen acceptance bands pass.

| Effective density cap (kg/m^3) | Tail mass feasible? | Closest intact CoM / SVL |
|---:|:---:|---:|
| 1111.3 | No | — |
| 1500.2 | No | — |
| 1666.9 | Yes | 0.678112 |
| 1800.3 | Yes | 0.673713 |
| 2000.3 | Yes | 0.668246 |
| 2222.6 | Yes | 0.665653 |

All these comparisons condition on the fixed geometry and exact tail-excluded
target. Also, identities formed from separately averaged animal measurements
need not hold exactly when mass/CoM covary between individuals.

## Verification and use

Sixteen morphology/inverse-fit unit tests pass. Tests establish reproducibility,
unchanged body/geom/site geometry and actuator settings, preserved total/tail
mass, bilateral symmetry, and positive finite inertias satisfying triangle
inequalities. Only masses, corresponding geometry-derived inertias, and the
regenerated stand keyframe differ from v1. The 92-observation/25-action
environment runs 100 zero-residual steps without a crash or termination; this
is a smoke test, not locomotion validation.

Reproduce with:

```powershell
.\.venv\Scripts\python.exe -m utils.build_lab_morphology --fit-com --report artifacts/evidence/morphology_lab_candidate_v2_fit.json
.\.venv\Scripts\python.exe -m common.morphology_audit --xml morphology/gecko_body_lab_v2.xml --strict
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_morphology_audit.py -v
```

The fit is **calibration to these measurements**, not held-out validation.
Remaining scientific work includes sensitivity to the mass prior/density guard,
dynamic gait tests and the independent tail-restriction experiment.
