# NeuroGecko — Handoff, 6 September 2026

Repository `C:\Users\ziyad\GeckoBrain` · github.com/12ziyad/NeuroGecko · Apache-2.0
HEAD `ccebd20`. All commits local; nothing pushed.

Readable version: https://claude.ai/code/artifact/66dc41c8-6f61-42ab-a500-74a9f62e7a6e

## Start here

1. `docs/research/` — five read-only documents, 431 KB, from 167 fact-checked
   research agents. Ground truth for every build decision.
2. `docs/BUILD_LOG.md` — the narrative of every session with before/after numbers.
3. `docs/DECISIONS.md` and `docs/BLOCKED.md`.
4. `config/proxies.yaml` — every numeric target with species, sample size,
   confidence and recording temperature. Unsourced values are tagged INVENTED.
5. `realism_metrics.py` — the evaluation tool. It defines the gates. **Any claim
   must be measured through this, not through a reimplementation.**

## State

- **Recovered:** 55 model files (233 MB in `models_recovered/`), including the
  pure-vision brain that ate in 29 of 30 episodes with zero falls.
- **Body:** `morphology/gecko_body_lab_v2.xml`, 38 g, **14/14** morphology
  checks (was 1/14). Opt-in; `gecko_body_r.xml` remains the default.
- **Walk (Gate 2, base controller with the policy off): 4/6.**
  Pass: forward 0.0416, net/path 0.743, hind swing load 0.081, hind duty 0.733.
  Fail: front stance load 0.584 (target 0.65), limb phase 0.631 (target 0.435).

## The most important correction in this record

Gate 2 tests the base with the learned policy switched OFF. It was only ever a
sanity check that the foundation is not broken; it was never meant to be a
finished walker. Treating 4/6 as a blocker and pursuing 6/6 by hand and by
search cost many hours. **The base is sound. Training is the correct next step.**

## Evidence ledger — confirmed

- Hind stance height caused the duty failure. Compensator: 0.641 -> 0.733.
- Limb phase equals the fore/hind touchdown-lateness difference:
  0.320 - 0.123 = 0.197, exactly the measured error, left and right to 3 dp.
- Three structural bugs fixed: inverted hind swing lift, zero front swing
  clearance, front press demanding 14 mm of ground penetration. None was
  reachable by training (`front_lift_residual_scale` was exactly 0.0).

## Evidence ledger — refuted by measurement

| Hypothesis | Evidence |
|---|---|
| Hind stance height causes the phase failure | fixing it moved phase 0.577 -> 0.607, wrong way |
| Fore stance height causes it | forelimb table nearly flat; front load 0.611 -> 0.584 |
| Actuator bandwidth | stiffening cut elbow lag 17.0 -> 2.9 deg; phase 0.6347 -> 0.6283 |
| Spine supplies the missing stride | 4x amplitude: +6.5% excursion, -17% speed |
| Fore/hind gearing mismatch | real and predicts the slip, but equalising it is slower at every step |
| Contact softness | impratio 1 -> 100: slip 13.5 -> 12 mm, flat speed; required mu ~0.01 vs floor 0.9 |
| Commanding an earlier touchdown | commanding 0.197 earlier gave 0.688, worse |
| Parameter search reaches 6/6 | ~200 generations, four searches; corrected objective 23.20 -> 22.92, tiebreaker only |

**Where the ledger points:** both failures are the forefoot. It lands 0.320
cycle late and is still 3.44 mm airborne at touchdown *with near-perfect motor
tracking*, so the commanded pose puts it there, not the plant. The trunk rides
3.87 deg nose-up, which lifts the shoulder and accounts for ~2.7 mm of that.
This is a posture problem, not a controller-tuning problem. Untested.

## My errors, recorded so they are not repeated

Advice that was wrong: widen the camera (should narrow to ~70 deg); unlock the
CPG frequency (load-bearing, -24% to -83% off-frequency); kill the green mask
(it is telemetry, feeds no gradient); "0.2-1.1 m/s" walking speed (treadmill
artefact, voluntary is ~0.129); an inverted femur/tibia ratio in a brief; and
treating stride length as a gate when it is not one of the six and sits inside
the {stride, cadence, speed} triangle the research says cannot be jointly met.

Method errors: a stiff-plant test that raised kp 15x without forcerange, so the
actuator saturated at 0.144 rad; a fitting harness that reimplemented limb phase
and rewarded making the gait irregular (stride CV 0.0149 -> 0.2713, reported
0.4497 where the gate read 0.6534 — claim withdrawn, `tools/fit_gate2_cma.py`
marked defective); fitting at a 12 s window when the gate measures at 20 s; an
objective that summed distance instead of counting gates; and an "in-band" phase
reading that was a window artefact (0.3939 at 12 s, but 0.6221-0.6476 across
seven sliding 7 s windows over 40 s).

## Next

1. **Train the walker.** Two short stages, hard stop at the eval peak — this
   project's own history shows reward peaked at 1.5 M steps and fell 63% by 10 M.
   Needs the instance for Linux parallel envs, not for the GPU. Set
   `device="cuda"`; the last run said `cpu` and wasted the A10G. ~$3-7.
2. **Revisit the forefoot.** Trunk pitch lifting the shoulder is the standing
   hypothesis and has never been tested directly.
3. **Make the world real.** Remove the privileged food vector, texture the floor,
   narrow the camera, evasive crickets, episodes 20 s -> 300 s.
4. **Build the brain module by module.** Tectum, basal ganglia, hypothalamus,
   sleep, memory. The recovered 29/30 visual brain re-attaches here.
5. **Prove it.** Pre-registered lesion predictions, behaviour battery in the
   papers' own event-count units, realism report with an explicit not-real list.

## Instance

    ssh -i /c/Users/ziyad/Downloads/geckobrain-key-2.pem ubuntu@54.196.131.211

Only that key with user `ubuntu` works; the other two .pem files fail.
g5.xlarge, A10G 23 GB, 4 vCPU, 15 GB RAM, us-east-1c. The laptop has 12 cores
against the instance's 4, so it is faster for CPU-bound work. EGL/GLX are absent,
so camera rendering falls back to CPU — fix before any vision training.

**It has been running idle for two days at roughly $24/day. Stop it until step 1.**
