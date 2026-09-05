# NeuroGecko build log

## Session 3 — touchdown hypothesis test, 2026-09-05

**AWS state at session start (11:47 UTC): UNKNOWN.** The read-only SSH
connection to the existing host timed out. Current uptime could not be read.
Last successful check was 11:21 UTC in Session 2; it was running and idle then.
No shutdown, cloud writes, training, CMA-ES, or paid jobs were launched here.
The existing $60 cap remains a ceiling, not a spending target.

The owner explicitly authorized the implementation while correcting the causal
claim: the +0.863 mm frozen commanded touchdown gap is independently defective,
but it does not establish the cause of the three dynamic failures. First verify
and flatten the knee/ankle stance arc; then remeasure all six unchanged gates.
Report frozen target geometry, actual physical contact, and force-threshold
loading separately. All trials are deterministic n=1, not repeated-run SDs.
Gate 2 must pass without sacrificing the three passing gates before any plant
changes. No PPO in this session or the conditional subsequent work.

Session 3 measurements and decisions follow here as they are completed.

## Session 2 — laptop-only controller work, 2026-09-05

**AWS cost notice:** at 10:29 UTC the existing AWS host was reachable and idle;
only OS Python services were running. EC2 remained on and billing. This session
does not require AWS and launches no training, PPO learning, CMA-ES or GPU job.
The owner's existing $60 ceiling is not a spending target.
Rechecked at **11:21 UTC**: still reachable, only the same two OS Python
services, no project training process. The instance has NOT been stopped.

### Session 2 outcome: improved base, Gate 2 FAIL — no retrain

The selected experimental lab base is manual trial `19_front_height`, reproduced
exactly in `final_candidate` and `verified_final`. It moves at **0.042444 m/s**
signed body-forward speed, with **0.825320 net/path**. The comparable 250 Hz
pre-edit reference is 0.011811 m/s and 0.205647. These are one deterministic
20-second run per condition, scoring after 3 seconds, not independent animals.

| Gate 2 requirement | Final measured result | Outcome |
|---|---|---|
| Signed forward >=0.04 m/s | 0.042444 | PASS |
| Net/path >=0.5 | 0.825320 | PASS |
| Hind swing load <10%, each foot | HL 1.290%; HR 1.087% | PASS |
| Front commanded-stance load >=65%, each foot | FL 61.139%; FR 60.641% | FAIL |
| Hind observed duty 0.78 +/-0.05 | HL 0.640784; HR 0.631227 | FAIL |
| Observed limb phase 0.435 +/-0.03 | left 0.577408; right 0.569030 | FAIL |

All four final contact-cycle rates are 1.18869–1.19167 Hz, with period CV
1.67–3.13%; acquisition is genuine 250 Hz. Thus the selected candidate satisfies
the engineering entrainment/regularity check without editing observed events.
The legacy walker still fails that check; its behavior was deliberately preserved.
No falls occurred in the selected 20-second run. The six biological-inspired
engineering requirements do NOT all pass merely because the clock now matches.

The candidate is selected for forward movement, low swing loading and relatively
straight travel, not as a globally optimal gait or biological validation. All
failed alternatives remain saved. The default XML/profile remain legacy; opting
into lab now selects these experimental controller values. Old lab commands can
be reconstructed using the explicit versioned profile/parameters in compatibility
tests and the committed baseline evidence. No existing checkpoint was replaced.

### One-change-at-a-time mechanical trials

Each numbered row changes the preceding row's single named parameter; rows 10
and 11 change timing only. Row 00b separately verifies exact old-lab compatibility.
Every trial reruns legacy fixtures and high-rate isolation tests before stepping.
Values are rounded below; full six checks, seeds, force diagnostics, effective
parameters and source hashes are in each `artifacts/evidence/session2/trials/*/report.json`.
No optimizer, learning algorithm or fitted objective was used.

| Trial/change | Forward m/s | Net/path | Hind swing load HL/HR | Front stance load FL/FR | Hind duty HL/HR | Limb phase L/R |
|---|---:|---:|---|---|---|---|
| 00b exact old reference | .01181 | .206 | .511/.691 | .504/.563 | .485/.567 | .470/.537 |
| 01 knee lift multiplier -1.3333 | .00067 | .093 | .240/.318 | .514/.608 | .487/.391 | .454/.361 |
| 02 mirror left fore-aft signs | .00955 | .284 | .435/.249 | .549/.550 | .529/.501 | .354/.492 |
| 03 shoulder tuck 0 rad | .00557 | .121 | .449/.424 | .544/.481 | .630/.675 | .566/.575 |
| 04 left front press 0 | -.00135 | .019 | .502/.460 | .488/.552 | .560/.538 | .559/.432 |
| 05 right front press 0 | -.00484 | .077 | .490/.517 | .539/.379 | .492/.605 | .513/.570 |
| 06 front seek 0 | -.00666 | .082 | .530/.527 | .540/.345 | .434/.627 | .428/.635 |
| 07 continuous front lift | -.00666 | .082 | .530/.527 | .540/.345 | .434/.627 | .428/.635 |
| 08 front swing delta -.8 | .03351 | .704 | .071/.005 | .651/.570 | .657/.713 | .565/.646 |
| 09 fore/hind angular ratio .54 | .02959 | .655 | .064/.010 | .642/.560 | .655/.707 | .573/.644 |
| 10 commanded hind duty .78 | .02949 | .632 | .059/.003 | .638/.556 | .647/.720 | .575/.652 |
| 11 front touchdown delay .435 | .02993 | .632 | .067/.009 | .634/.552 | .642/.710 | .590/.655 |
| 12 heading gain 1/rad | .03191 | .638 | .061/.018 | .630/.564 | .613/.693 | .581/.654 |
| 13 hind amplitude .9 | .03966 | .737 | .041/.059 | .652/.575 | .621/.663 | .574/.617 |
| 14 front swing delta -.6 | .03850 | .761 | .052/.061 | .657/.582 | .617/.681 | .579/.625 |
| 15 right front press .05 | .03930 | .753 | .046/.047 | .644/.581 | .621/.675 | .585/.609 |
| 16 left front press .05 | .03906 | .758 | .053/.050 | .653/.572 | .653/.685 | .589/.627 |
| 17 hind amplitude .98 | .04105 | .778 | .062/.053 | .661/.581 | .657/.660 | .576/.590 |
| 18 other-channel amplitude 0 | .04619 | .777 | .016/.012 | .582/.571 | .667/.668 | .613/.617 |
| 19 front swing delta -.4; SELECTED | .04244 | .825 | .013/.011 | .611/.606 | .641/.631 | .577/.569 |
| 20 front swing delta -.2; rejected | .03550 | .767 | .157/.139 | .604/.613 | .633/.685 | .635/.562 |
| 21 knee multiplier -1; rejected | .03398 | .793 | .130/.142 | .625/.629 | .657/.689 | .636/.561 |

Rows 00–09 initially used the mathematically ideal 8/9 shoulder/hip range ratio;
a test caught that the legacy shoulder range is rounded, making its actual ratio
0.8888892698450486. The resulting command difference was about 2e-7 rad. An
explicit compatibility branch restores exact old commands, confirmed by 00b and
4,000 old/new full-command comparisons across both bodies/profiles. The .54-ratio
selected candidate is unaffected. Rows 06/07 are exactly equal because zero press
and seek already make the compatibility front curve continuous at its boundaries.

### Geometry, verification and stop decision

Frozen-trunk collision geometry, not site height, confirms selected mid-swing
clearance **+2.065 mm hind / +2.735 mm front**, respectively 2.278/3.495 mm above
mid-stance. All four stance pad sweeps now travel rearward. Front commanded stance
overlap fell from 12.905/14.093 mm to **0.760 mm each**, but did NOT become zero;
hind overlap remains 0.212 mm. Soft-contact force and dynamic support remain
separate measurements. No contact geometry was secretly removed or altered.

Final morphology: **14/14** static gates. Guarded software verification:
**189 executed, 188 passed, one optional SciPy skip**, plus three deliberately
excluded tests (one actual PPO-learning smoke test and two OpenGL pixel tests).
Zero guarded learning, graphics or CUDA calls occurred. Existing camera pixels
were not re-rendered in this no-GPU session. No new video is claimed.

Gate 2 remains red after the bounded mechanical trials. Activation lag,
integrator changes, force-ceiling changes and the optional frequency-gain sweep
were NOT started. `FREQ_HZ` stays 1.1888. The training CLI now explicitly refuses
lab training/resume pending validated support/timing and a complete effective-
controller checkpoint contract; legacy CLI behavior remains unchanged.

Steering verification used the selected parameters without further tuning, with
goal bearings 0/+0.4/-0.4 rad. Relative to center, final-cycle mean body heading
shifted +0.40544/-0.40124 rad; post-settle displacement heading shifted
+0.39000/-0.40674 rad. This verifies directional response in three deterministic
probes, not perturbation robustness. Last-cycle mean absolute target-bearing error
remained about0.087–0.089rad (about5degrees), so heading still oscillates.
See `artifacts/evidence/session2/steering_checks.json` and the complete 27-report
comparison. Trial19, final_candidate and verified_final have exactly equal six
gate results; those verification repeats are NOT three independent samples.

Starting code: `3efa2f6`. Research documents remain read-only. Candidate work is
explicitly V2 + lab; the live default stays legacy body + legacy controller.

Instrument repair: true 250 Hz samples every five 1250 Hz physics steps; the
controller still runs at 50 Hz. Each sample copies `MjData` and runs `mj_forward`
on the copy, aligning sensor/geometry timestamps without touching the live
solver. Raw forces and forty-millisecond debounce are retained. Entrainment uses
an explicitly engineered +/-10% rate tolerance, never phase-locked event picking.
An honest instrument can report a controller's failed entrainment: reducing real
chatter by changing a detector until CV passes would invalidate the experiment.
Deterministic zero-noise requests are executed once and reported as n=1, SD=null.

Source for snapshot timing: https://mujoco.readthedocs.io/en/stable/programming/simulation.html
(mj_step computes forward dynamics then integrates; derived values otherwise lag).

Instrument verification: 43 tests run, 42 pass and one optional SciPy filter test
skipped. The 80-control-step observer-on/off test is bitwise identical for live
qpos, qvel, observation and reward; integrated work is counted once per substep.
Full 20-second CPU-only scorecards with seed 0 completed without falling:

| Condition | Signed forward m/s | net/path | HL/FL/HR/FR contact Hz | HL/FL/HR/FR period CV % |
|---|---:|---:|---|---|
| Legacy + frozen policy | .049339 | .460905 | 4.811 / 2.998 / 2.741 / 3.027 | 33.47 / 65.10 / 50.37 / 37.98 |
| V2 lab base alone | .011807 | .205647 | 3.984 / 3.817 / 3.088 / 2.723 | 28.50 / 25.44 / 31.91 / 37.05 |

Gate 0 acquisition and complete-scorecard requirements PASS; its requested
physical entrainment/CV outcome FAILS. Raising acquisition does not remove real
repeated contact. We proceed to mechanical diagnosis with the failure visible,
not by treating these events as verified animal strides. High-rate path distance
also captures small within-control-step excursions missed by the old sampler;
old and new sampling estimates are not identical protocols.

Effort accounting: 27 focused tests pass, including a real paired frozen-control
alpha ramp. The ramp changes only the reward charge, not qpos, observations or
work. The proposed Gate 1 training correlation is not identified in this no-
training session. Formula and zero-residual base-power charge are verified.

## Brain recovery completed — 2026-09-04 UTC / 2026-09-05 India

Before code changes, connected to the existing instance with the user's supplied
SSH identity and strict known-host verification. The remote checkout was
`ddbb105ff40afa7736916750243e282bafd22b98`; its untracked files were left untouched.

- Archived the entire remote `models/` directory: **112,548,021 bytes** compressed.
- Downloaded the archive and the independently generated per-file SHA256 manifest.
- Matched archive SHA256:
  `dd93767f726dc5e254003b1072da5acebe0e92261e293d4766c794df08db302a`.
- Extracted to `models_recovered/recovery-20260904T233438Z/`, without replacing local models.
- Verified **all 55 recovered files** against the remote manifest.
- Also archived the original tracked source at `ddbb105` beside the recovery.
- Confirmed the named visual student (`final.pt`, 619,103 bytes) and privileged
  teacher (`final.pt`, 625,183 bytes) exist. The remote historical evaluation log
  was found; its scores are not yet a new reproduction.
- `models/INVENTORY.md` catalogs exact hashes and training configurations.

There are now copies on the AWS instance and on the laptop. This is not an S3
backup, and not a guarantee against losing both machines. Checkpoint files remain
excluded from Git; only the human-readable inventory is tracked.

## Scope and cost

The user requested implementation from `CODEX_PROMPT.md` and then directly
authorized **$60 maximum additional AWS cost**. No new instance or paid service
was created. GPU observed: NVIDIA A10G, 23,028 MiB. Existing remote source and
untracked work must be preserved. Remote experiments will use an isolated build
directory. No unattended unbounded training, invented co-author identity, or
publication to GitHub is implied by the brief's blanket-authorization language.

AWS cost estimates will be estimates, not an account billing reading. The
existing instance continues to accrue charges while running. Each launched
experiment must have a finite runtime and recovery path.

## Measurement implementation

Implemented the morphology audit, single-speed gait logger, event-count
ethogram proxies, policy camera isolation, and atomic paired model/normalization
checkpoints. Original physical body parameters, gait frequency and recovered
weights remain unchanged. New body/gait changes are explicit opt-in candidates.

Research files under `docs/research/` remain unchanged. Corrections to the supplied
brief and source interpretations are recorded in `docs/DECISIONS.md`.

### Instrumentation verified

- Added seven measurement sites and moved four footzone markers to diagnostic group 4.
- MuJoCo shape remains nq=39, nv=38, nu=25; 83 sensors unchanged.
- Original versus instrumented XML: 2,500 physics steps produced bit-identical
  qpos, qvel and sensor traces under identical controls.
- Added per-physics-step actuator mechanical work (positive, signed negative,
  absolute). This is not metabolic or electrical energy.
- Before/after work-logger 80-control-step qpos/qvel SHA256 matched exactly:
  `a0edbbd87f82d8d4eb88a1f7e9000144c97cc09c778bfefbab83baefcb3daad1`.
- Morphology baseline: **1/14 engineering gates passes** (tail mass fraction).
  Full actuals and source conventions: `artifacts/evidence/morphology_before.json`.
- Local suite: 68 tests passed/skipped accounting at the initial integration run
  (67 passed, 1 skipped: optional SciPy high-rate filter success path). The
  undersampled-input rejection test runs without SciPy. Subsequent additions are
  reported at final verification.
- Camera tests include actual rendered-pixel invariance for hidden self-appearance.
- Checkpoint tests include a real tiny PPO save/load, identical deterministic
  actions and normalization statistics, then resumed learning.
- Local 1,000-iteration component profile is in
  `artifacts/evidence/profile_local.json`; it is a microbenchmark, not training FPS.

The body correction is being built as a separate opt-in `gecko_body_lab.xml` so
the original walking system remains available for comparisons. No candidate is
promoted merely because it looks better.

## Measured results and safety test

- Saved walker baseline: **20/20 runs completed; zero falls; 400 simulated seconds**.
  Each run used the same zero-noise calibration start, so these are deterministic
  repeats, not independent environments. After discarding the first 3 seconds,
  mean path speed was 0.128484 m/s, but signed body-forward speed only 0.049517 m/s
  and net displacement 1.023632 m over 17 s. Do not confuse path length with
  useful forward progress. Absolute actuator work was 17.565282 J per path metre.
- Observed contact-cycle duty factors HL/FL/HR/FR: 0.530/0.265/0.315/0.414
  (rounded; exact values in report). Same-foot contacts recur much faster than
  the 1.1888 Hz commanded clock. These are operational contact cycles, not
  verified biological strides. The new evaluator warns about gross mismatches
  without filtering away the inconvenient cycles.
- Zero residual plus existing contact reflex: zero falls in one 20 s run,
  path speed 0.074722 m/s but **signed forward speed -0.001373 m/s**. The base
  moves around without useful forward walking; path speed alone is a bad gate.
- Zero lateral tail drive: one 20 s diagnostic, no falls, signed forward speed
  0.025817 m/s versus baseline 0.049517. This is not graphite-rod restriction:
  passive tail motion remains possible. No biological ablation reproduction claimed.
- Recovered visual brain, original XML and camera: seeds 100–102, 1,000 steps
  each, **21 food-proximity events, 3/3 episodes with events, zero falls**.
  Sealed camera on the same seeds: **20 events, 3/3 episodes, zero falls**.
  Small A/B only, not the historical 30-episode evaluation. Privileged input is
  zero and asserted; the eating radius is 0.10 m, not a genuine prey strike.
- Tiny AWS training safety test: 256 aggregate steps, four subprocess workers,
  CUDA PPO. Initial, 128, 256 and final paired checkpoints all reached the
  laptop and passed SHA256 checks before training received its acknowledgement.
  Final pair resumed on laptop CPU from **256 to 320 steps**, optimizer updates
  10 to 20, with source hashes unchanged. No production walker was replaced.
- Raw traces are retained locally and on AWS, excluded from Git to avoid over
  100 MB of repetitive JSON. Compact reports and the 20 s baseline movie are
  versioned. See `artifacts/evidence/` and the updated model inventory.

## Corrected body and explicit lab profile

- `gecko_body_lab.xml` v1 passed 12/14 static engineering bands.
- Separate `gecko_body_lab_v2.xml` passes **14/14 bands**: total mass 38 g,
  SVL 106 mm, tail fraction 22%, tail length 0.73 SVL. Geometry and positive
  inertias were checked; the 92-D observation and 25-D action shapes remain.
- V2 fits non-tail mass distribution with positive bounded changes and a
  documented relative-entropy prior; tail distribution uses an explicitly
  invented generous density guard. Tail-excluded CoM is 0.527 SVL; intact CoM
  **0.678112** passes the 0.659 +/- 0.020 band but does not match its mean.
  This is calibration to targets, not held-out validation. Exact paired means
  are infeasible under that density guard and the retained geometry.
- Primary CoM source used seven frozen adult females with SVL 121.9 mm. Transfer
  to the neutral 106 mm model is an assumption, not a measurement of this body.
  Full constraints, fitted masses and sensitivities are recorded in
  `artifacts/evidence/morphology_lab_candidate_v2_fit.json` and accompanying notes.
- `gait_profile='lab'` shares one touchdown-delay schedule between controller
  and reward: HL 0, FL .44, HR .50, FR .94; hind stance .765, fore .70;
  frequency remains locked at **1.1888 Hz**. Actual control-interval boundary
  tests verify reward timing matches the executed interval. Legacy keeps its
  old behavior bit for bit for recovered checkpoints.
- Lab-only reward/contact guards use actual bodyweight and hip/shoulder/SVL
  landmarks. They do not equate skin tactile sensitivity with ground contact.
  Speed reference is 0.72 SVL * 1.1888 = 0.090729 m/s and slow penalty is zero.
  Resolved parameters and superseded legacy values are recorded; lab resume
  rejects different timing, morphology, calibration or contact settings.
- Read and visually checked Jagnandan & Higham 2018 Tables 1–3 in the supplied
  PDF. `common/emg_pattern.py` implements a standalone timing-prior library,
  including both gastrocnemius bursts and pre-footfall caudofemoralis onset.
  Smoothstep waveform shape is invented; EMG amplitude is not joint angle or
  muscle force. This library is **not yet coupled to the live motor controller**.

## Throughput finding

The initial 1,000-iteration AWS microbenchmark measured 0.065770 ms per physics
substep, 22.304586 ms per 64x64 camera render, and 0.963163 ms per visual-actor
prediction on CUDA. The nominal A10G did **not** accelerate that render:
OpenGL identified **Mesa llvmpipe (software CPU renderer)**. CUDA availability
alone is not rendering evidence. The profiler now records both graphics and
Torch device identities. A subsequent measured local run on Intel UHD 770 gave
0.040979 ms/substep, 4.297575 ms/render, 0.507562 ms/CPU actor prediction.
These separate microbenchmarks are not training FPS and include different devices.

Read-only driver inspection found only Mesa EGL vendor registration and no
NVIDIA EGL/GLX graphics libraries. CUDA compute works, but a process-local
renderer switch cannot provide missing libraries. No driver packages, services
or system configuration were changed. A renderer-identified repeat is saved as
`profile_aws_renderer_verified.json`.

Final local integration at this checkpoint: **120 tests, 119 passed, one optional
SciPy high-rate-filter test skipped**. The actual tiny save/resume test ran;
skipping SciPy did not skip checkpoint recovery, rendering or gait tests.

Long retraining, CMA-ES gait tuning, commanded stop/rest and gaze reassignment
have not been completed. Dynamic candidate tests and final verification are
recorded below when finished; passing static bands alone is insufficient.

## Dynamic candidate checks

Source snapshot `9960117` was transferred into isolated `lab-9960117`; archive
SHA256 matched `5e0afecca1289e24fffc0e5b75e7a0584a197b87f204f0301f735f37e483ab69`.
The recorded v2 XML SHA256 was
`cfc485e6fab752dc4ab25d25beebf549fded5fb241b3da6c828df55156b9c22d`.
Each condition below used one deterministic 20 s run, discarding the first 3 s.
All completed with zero falls. They are diagnostics, not broad validation.

| Condition | Path speed (m/s) | Signed body-forward speed (m/s) | Nose-up pitch (degrees) |
|---|---:|---:|---:|
| Original body + saved walker | 0.128484 | 0.049517 | 7.818 |
| V2 body + saved walker, legacy timing | 0.105698 | 0.033660 | 9.543 |
| V2 body + saved walker, lab timing/calibration | 0.064364 | 0.019909 | 9.407 |
| V2 body + lab rhythm, zero residual with contact reflex | 0.046894 | 0.011736 | 11.334 |

The lab zero-residual contact-cycle duty factors HL/FL/HR/FR were
0.443/0.464/0.549/0.414, despite commanded ratios 0.765/0.700/0.765/0.700.
Its hip height averaged 0.167555 SVL. Its total absolute actuator work was
29.738381 J per path metre. The target gait is **not achieved**. In particular,
static neutral posture passing does not make the dynamic pitch acceptable.

The first AWS integration test caught source-file line endings in generated
XML provenance fingerprints. Corrected using explicitly labeled canonical-LF
text hashes; exact original byte hashes remain in the evidence. The correction
changes candidate comments only. Generated structure and numerical attributes
are still compared tightly; no research tolerance was widened.

## Final verification and handoff

- **Windows: 127 tests; 126 passed, one optional SciPy skip, zero failures.**
- **AWS Linux: 127 tests; 126 passed, the same skip, zero failures.**
- AWS independently reran the strict v2 morphology audit: **14/14 bands pass**.
- Exact source and portability notes remain in
  `morphology_reproducibility_line_endings_20260905.md`. The uploaded portability
  patch archive matched SHA256
  `7b83ccc229030843b4134b8e1af0a36dadca38f92978015e3e75c37f27449f98`.
- Created a clean 20 s human replay of the actual recorded lab-v2 qpos samples,
  500 frames at 25 fps. No physics steps, interpolation or controller execution;
  shadows/reflections/debug sites disabled for this presentation renderer only.
  Source trace and exact XML hashes matched and were unchanged. Policy images
  and original before/after measurements were not modified.
- Final AWS process check found no compute processes and no project Python jobs;
  only the two pre-existing system Python services remained. No training is left
  running. **EC2 itself remains on and billing**, pending the separate stop choice.
- No new cloud resource, production replacement, system-driver change or GitHub
  push was performed. The $60 direct authorization remains the ceiling, not a
  target to spend; the exact account bill was not read.

This is a completed recovery/instrumentation/body-calibration build increment,
not completion of the full research program. The gait remains untuned; see
`docs/BLOCKED.md` for the actual remaining implementation work.

## Session 3 (continued in Claude Code) — hind stance compensation implemented and tested

The Session 3 Codex run ended mid-implementation when the ChatGPT subscription
lapsed. Commit `ad54e9a` (baseline + hypothesis protocol) had landed; the
compensator module `common/hind_stance_geometry.py` was written but uncommitted,
untested and unwired. Work resumed here.

### Solver corrections to the module as written

`HindStanceCompensator` could not build: `_solve` raised at HL hip=0.3068 rad
with an 18 um residual against a 0.1 um failure threshold. Diagnosis before
changing anything: the offset bounds are NOT binding at that pose (reachable
clearance spans +0.03 to -7.72 mm around a -0.60 mm target, and zero offsets
already give -0.68 mm). Two real causes:

1. A 10 nm convergence tolerance probed with 10 urad finite differences that
   move the foot only ~90 nm. Loosened to 2 um convergence / 10 um failure,
   with 100 urad probes. 2 um is 0.3% of the 0.6 mm headroom to either edge of
   the declared [-1.2 mm, 0] band.
2. `clearance()` takes a `min` over several foot collision geoms, so it is
   continuous but not differentiable where the lowest geom switches. The
   Newton line search stalls at that kink. Added a bisection fallback along
   +[knee, ankle], which is monotone (probed: (0,0) -> -0.68 mm,
   (0.35,0.35) -> -7.72 mm) and needs no derivative.

The post-build dense-interpolation guard against the stance band is unchanged.
Built table: max node error 1.98 um; dense-interpolation clearance -1.098 to
-0.594 mm, i.e. in contact at every hip angle.

### Assumption checked, not assumed

The table's reference requires zero base knee/ankle/sprawl/rotation targets.
In the lab profile `other_amplitude == 0.0`, so it holds; the constructor now
asserts this rather than trusting it. Verified against real base commands:

| local phase | base clearance | with compensation |
|---|---|---|
| 0.000 | +0.8629 mm | -0.5984 mm |
| 0.312 | -0.0025 mm | -0.5998 mm |
| 0.780 | -1.1999 mm | -0.9955 mm |

Worst deviation from the -0.600 mm target across stance: 0.396 mm, at the
stance/swing boundary. The commanded foot is now below ground at every phase
of stance instead of 0.863 mm airborne at commanded touchdown.

### Integration

`hind_stance_compensation` is an opt-in constructor flag on
`CPGResidualController`, forwarded by `GeckoWalkEnv` and exposed as
`--hind-stance-compensation` on `realism_metrics.py`. Default off; legacy
raises if it is requested. Offsets are applied after the limb loop (so the hip
command is final) and fade to zero at mid-swing, leaving swing commands intact.

### Gate 2 result — FAIL. Hypothesis partially refuted.

Identical protocol, V2 + lab, zero residual, seed 0, 20 s, 250 Hz. n=1
deterministic runs; the difference is the flag only.

| Metric | Baseline | +Compensation | Target | |
|---|---|---|---|---|
| forward speed m/s | 0.0424 | 0.0406 | >= 0.04 | pass, slightly worse |
| net / path | 0.8253 | 0.7323 | >= 0.50 | pass, worse |
| duty HL | 0.6408 | 0.7001 | 0.73-0.83 | FAIL, improved |
| duty HR | 0.6312 | 0.7013 | 0.73-0.83 | FAIL, improved |
| limb phase HL->FL | 0.5774 | 0.6070 | 0.405-0.465 | FAIL, worse |
| limb phase HR->FR | 0.5690 | 0.6031 | 0.405-0.465 | FAIL, worse |
| trunk pitch deg | 3.298 | 4.078 | 3.11-4.33 | pass |
| hip height SVL | 0.1561 | 0.1589 | 0.140-0.160 | pass |
| shoulder height SVL | 0.1161 | 0.1279 | 0.103-0.123 | now FAIL |
| stride period CV HL | 0.0288 | 0.0193 | single digits | pass, improved |
| duty FL / FR | 0.3766/0.3778 | 0.3886/0.3962 | ~0.70 | FAIL |
| stride length HL SVL | 0.3307 | 0.3174 | 0.62-0.82 | FAIL |

Answering the three questions the brief required:

1. **Direct effect confirmed.** The commanded hind foot is no longer airborne
   at touchdown; clearance is negative through the whole of commanded stance.
2. **Hind duty moved, limb phase did not.** Hind duty rose 0.64 -> 0.70,
   covering about 60% of the gap to the 0.73 floor. Limb phase moved the
   *wrong way*, 0.577 -> 0.607, away from 0.435. **The claim that the airborne
   hind foot explained the limb-phase failure is refuted.** Recorded as a
   finding, not a partial success.
3. **The front-load question is superseded by a larger front failure.**
   Forefoot duty factor is 0.38-0.40 against a commanded 0.70 — the forefoot
   is now the limb that will not stay down. Shoulder height rose 0.116 ->
   0.128 SVL and left its band, and net/path and forward speed both fell
   slightly. The coherent reading is that planting the hind foot through full
   stance levers the front of the body up, unloading the forefeet. That is a
   mechanical coupling, not a threshold-provenance question, so the tail-heavy
   hypothesis is neither supported nor needed here.

Also newly visible, and not previously gated: **stride length is 0.32 SVL
against a published 0.62-0.82.** The animal is taking roughly half-length
steps. This is independent of the compensator (0.3307 baseline) and is a
large, separate defect.

**Gate 2 remains failed. No plant change, CMA-ES or training was started.**
The remaining bottleneck has moved from the hind limb to the forelimb and to
stride length.

## Session 3b — compensation generalised to all four limbs

`HindStanceCompensator` generalised to `StanceCompensator` with a per-limb spec
(alias kept). Hind limb has two free joints (knee, ankle); the forelimb has one
(elbow), because this MJCF has no wrist actuator, so its solve is 1-D. The
solver, bisection fallback, diagnostics and band guard are all now written for a
variable free-joint count. `max_offset_rad` default raised 0.35 -> 0.70; the
hind solution is unchanged at -0.346, so 0.35 was NOT binding after all — that
earlier suspicion was wrong.

Applied as an ABSOLUTE blended target rather than an additive offset. This
matters for the forelimb only: its elbow already carries `front_stance_press`
during stance, so an additive offset would stack two independent height
commands. For the hind limb the two forms coincide (knee "lift" and ankle
"other" with amplitude 0 are both zero-offset from neutral in stance), which is
why the hind numbers below reproduce the additive run's direction.

Built tables (lab v2): HL/HR clearance -1.098..-0.594 mm; FL/FR -0.601..-0.597 mm.
The forelimb table is essentially flat, i.e. the forelimb's frozen-pose geometry
was already fine and had almost nothing to correct — 1.93 um max node error and
offsets of only +/-0.08 rad.

### Gate 2, three configurations, identical protocol (V2 + lab, zero residual, seed 0, 20 s, 250 Hz)

| Check | baseline | hind only | all four | target |
|---|---|---|---|---|
| 1 forward m/s | 0.0424 | 0.0406 | 0.0416 | >= 0.04 |
| 2 net/path | 0.8253 | 0.7323 | 0.7427 | >= 0.50 |
| 3 hind swing load | — | 0.0441 | 0.0806 | < 0.10 |
| 4 front stance load | 0.6114 | 0.6159 | **0.5841** | >= 0.65 |
| 5 hind duty | 0.6408 | 0.7001 | **0.7334 PASS** | 0.73-0.83 |
| 6 limb phase | 0.5774 | 0.6070 | **0.6347** | 0.405-0.465 |
| gates passed | 3/6 | 3/6 | **4/6** | |

Secondary: stride period CV 0.0288 -> 0.0149; trunk pitch 3.30 -> 3.87 deg (in
band); hip height 0.1561 -> 0.1584 SVL (in band); shoulder height 0.1161 ->
0.1249 SVL (left its 0.103-0.123 band); stride length 0.3307 -> 0.3242 SVL
against a published 0.62-0.82, essentially unmoved.

### What this settles

**Gate 5 now passes.** Holding the hind collision foot at constant commanded
height through stance raised hind duty 0.641 -> 0.733. That is the compensator
doing exactly and only what it was built to do.

**The forelimb hypothesis is refuted.** Front duty barely moved (0.377 ->
0.394) and front stance load went the wrong way (0.611 -> 0.584). The forelimb
table being almost flat is the explanation: forelimb frozen-pose geometry was
never the defect. Measured dynamically, the forefoot is unloaded for the first
~30% and last ~10% of its commanded stance while the shoulder girdle rides
11.3-18.0 mm (6.7 mm of vertical travel). The forefoot is losing contact
because the front of the body is moving, not because the leg is folded, and a
frozen-pose table cannot correct a moving reference.

**Limb phase is not a touchdown-height problem at all.** It has moved
monotonically away from target at every step: 0.577 -> 0.607 -> 0.635. Three
independent interventions all pushed it the wrong way. Whatever sets realised
limb phase here, it is not hind or fore stance height, and it should be
diagnosed from first principles rather than by further stance edits.

**Stride length is untouched and large.** 0.32 SVL against 0.62-0.82 published,
unchanged across all three configurations. This is an independent defect of
roughly a factor of two and nothing done so far addresses it.

**Gate 2 remains failed at 4/6.** No plant change, CMA-ES or training started.
The all-four configuration is retained as the better of the two (4/6 vs 3/6)
despite its slightly worse front stance load, and both remain opt-in behind
`--hind-stance-compensation {hind,all}`; the default is still off and legacy is
untouched.

## Session 3c — limb phase and stride length diagnosed; both hypotheses refuted

### Touchdown timing (debounced 40 ms, 20 strides per foot, all-four-compensated run)

| foot | commanded TD | actual TD | error |
|---|---|---|---|
| HL | 0.000 | 0.1234 | +0.123 |
| HR | 0.500 | 0.6244 | +0.124 |
| FL | 0.435 | 0.7551 | +0.320 |
| FR | 0.935 | 0.2552 | +0.320 |

Left and right are identical to three decimals, so this is systematic, not noise.
**The limb-phase error IS the touchdown-lateness difference:** 0.320 - 0.123 =
0.197, against a measured phase error of +0.197. Realised limb phase is late
touchdown, nothing else. The forefoot lands a third of the way into its own
commanded stance.

### Actuator bandwidth measured, then refuted as the cause

Servo dynamics at the stand pose:

| joint | inertia | kp | kv | natural freq | damping ratio |
|---|---|---|---|---|---|
| elbow | 1.20e-3 | 0.310 | 0.0056 | **2.6 Hz** | **0.27** |
| knee | 1.20e-3 | 0.466 | 0.0075 | **3.1 Hz** | **0.26** |

The gait runs at 1.1888 Hz, so the actuators are only ~2x faster than the
command they track and are badly underdamped. Measured tracking error at the
elbow reaches 0.32 rad (18 deg) just before touchdown. This looked like the
cause.

Built `morphology/gecko_body_lab_v2_stiff.xml` with per-joint kp/kv solved from
the actual inertia for 10 Hz natural frequency (8.4x gait) and zeta = 0.90 — a
~15x kp increase. Peak torque for a 0.3 rad error stays inside forcerange for
all limb joints except the ankle, which is noted.

| check | soft | stiff |
|---|---|---|
| 1 forward | 0.0416 PASS | 0.0409 PASS |
| 2 net/path | 0.7427 PASS | 0.8004 PASS |
| 3 hind swing load | 0.0806 PASS | 0.1467 FAIL |
| 4 front stance load | 0.5841 fail | 0.5755 fail |
| 5 hind duty | 0.7334 PASS | 0.7543 PASS |
| 6 limb phase | 0.6347 fail | **0.6396 fail** |
| stride length SVL | 0.3242 | 0.3192 |
| **gates** | **4/6** | **3/6** |

**Limb phase did not move (0.635 -> 0.640) under a 15x stiffer, near-critically
damped plant.** Actuator tracking is therefore NOT the cause of late touchdown.
Three hypotheses have now been tested and refuted for limb phase: hind stance
height, fore stance height, and actuator bandwidth. The stiff variant is
retained as evidence but is NOT adopted — it costs the hind-swing-load gate.

This also reverses the Session 4 plan as previously written: it proposed *adding*
activation lag and *cutting* force ceilings. On a plant already at 2.6 Hz and
zeta 0.27, both would move the wrong way. Do not do that without re-deriving it.

### Stride length cannot be fixed by the legs — arithmetic

| quantity | value |
|---|---|
| hip_proret ctrlrange half | 45.0 deg |
| lab `hind_fa_amplitude` | 0.98 (of 1.0) |
| resulting hip sweep | **88.2 deg peak-to-peak** |
| published femur retraction excursion | 82.57 deg |
| measured stride length | 0.324 SVL |
| target | 0.62-0.82 SVL |

**The hip already sweeps MORE than the published femur excursion, at 98% of
available amplitude, and still delivers less than half the stride length.**

Measured foot-x relative to trunk during HL stance spans -50.3 to -14.1 mm, a
36.3 mm excursion. With an 88.2 deg sweep that implies an effective hip-to-foot
radius of 36.3 / (2 sin 44.1) = 26.1 mm. Reaching 0.72 SVL (76.3 mm) by sweep
alone would require a radius of 54.9 mm — **longer than the entire hind limb
(0.417 SVL = 44.2 mm)**. It is geometrically impossible.

So the missing stride length is not in the limbs at all. In real lizards the
remainder comes from lateral spine bending and pectoral/pelvic girdle rotation,
which is exactly what the standing-wave spine term is for. Forelimb stride is
worse still (0.189 SVL), consistent with the same explanation.

**Conclusion for the next session:** stop tuning limb amplitude for stride
length — it is saturated and the geometry forbids the target. The open items
are (a) spine/girdle contribution to stride, and (b) why the forefoot lands
0.32 cycle late when neither its stance geometry nor the plant bandwidth
explains it. Gate 2 stands at 4/6 with the all-four soft configuration.

## Session 3d — stride length is speed; the spine does not supply it; the hind foot slips

### 1. "Stride length" and "speed" are the same gate at a locked cadence

    speed 0.0416 m/s / 1.1888 Hz = 35.0 mm = 0.330 SVL per stride
    reported stride_length_svl   = 0.324    (the same number)

Stride length is not an independent quantity here. At a fixed 1.1888 Hz it is
speed divided by frequency, so the only way to reach 0.72 SVL is to reach
**0.0907 m/s** — which is exactly the `target_speed` already sitting in the lab
reward calibration (0.090729216 = 0.72 x 0.106 x 1.1888). Current speed is
**46% of it**. Treating stride length and speed as two separate failures was
double-counting one failure.

### 2. The spine does not supply the missing stride — measured, not argued

Direct sweep, all-four compensation, zero action, foot-x excursion relative to
the trunk over the full cycle:

| spine_amp | tail_amp | HL exc mm | FL exc mm | speed m/s |
|---|---|---|---|---|
| 0.00 | 0.15 | 107.16 | 54.07 | 0.0226 |
| 0.30 | 0.15 | 109.33 | 57.23 | 0.0226 |
| 0.60 | 0.15 | 111.63 | 71.24 | 0.0217 |
| 0.90 | 0.15 | 114.82 | 74.11 | 0.0213 |
| 0.90 | 0.45 | 113.87 | 77.27 | 0.0214 |
| 1.20 | 0.45 | 116.41 | 92.34 | **0.0188** |

Quadrupling spine amplitude (0.30 -> 1.20) buys **+6.5% hind foot excursion and
costs 17% of speed.** The Session 3c inference that the missing stride "must
come from spine and girdle rotation" is therefore **refuted for this body**. It
was a reasonable read of the geometry but it does not survive measurement.

(Speeds in this table are lower than the metrics run because this rollout issues
no heading target; the comparison between rows is valid, the absolute value is
not comparable to the gate runs.)

### 3. What is actually lost: hind-foot slip

A planted foot should have near-zero world-frame travel during stance.

| foot | slip per stance | body advance per stride | slip fraction |
|---|---|---|---|
| HL | 14.4 mm | 35.0 mm | **41%** |
| HR | 14.4 mm | 35.0 mm | **41%** |
| FL | 5.0 mm | 35.0 mm | 14% |
| FR | 5.5 mm | 35.0 mm | 16% |

The hind feet slide backward through 41% of the distance the body advances.
That is propulsion being thrown away, and it is the largest single identified
loss between the limbs' motion and the body's motion.

### 4. Contact-cycle mismatch, fore vs hind

Complete strides over the same window: HL 19, HR 20, **FL 28, FR 25**. The
forefeet are registering roughly 1.4x more touchdown-to-touchdown cycles than
the hind at a single commanded cadence, i.e. the forefoot is making extra
contacts within a commanded stance. This is consistent with, and probably the
same phenomenon as, the unexplained 0.32-cycle forefoot touchdown lateness.

### Status

Gate 2 remains 4/6. Hypotheses refuted so far, each by measurement: hind stance
height (for limb phase), fore stance height, actuator bandwidth, and now spine
amplitude (for stride). Confirmed causes: hind stance height did fix hind duty;
stride length is speed; 41% hind slip is real lost propulsion.

Next open items, in order of evidence: (a) hind-foot slip during stance, which
is quantified and large; (b) forefoot extra contacts, which is quantified and
likely the same defect as the touchdown lateness.

## Session 3e — three more refutations, and a correction to my own framing

### Refuted: fore/hind gearing mismatch

The body is rigid, so both girdles' feet must sweep backward at the same rate or
they fight. They do not: hind demands 55.3 mm/s of body speed, fore demands
34.1 mm/s, and the body settles at 41.6 mm/s between them. The predicted slip
from that mismatch matches the measurement well (hind predicted +9.0 mm against
14.4 measured; fore -4.4 against -5.0). The mechanism is real.

The fix is not. Sweeping `fore_hind_amplitude_ratio` to equalise foot travel:

| ratio | speed m/s | stride SVL | slip HL | exc FL mm |
|---|---|---|---|---|
| 0.54 (current) | 0.0226 | 0.179 | 13.47 | 57.2 |
| 0.70 | 0.0217 | 0.173 | 13.72 | 77.8 |
| 0.85 | 0.0194 | 0.154 | 12.65 | 87.1 |
| 0.97 | 0.0189 | 0.150 | 12.13 | 89.3 |
| 1.10 | 0.0187 | 0.148 | 11.99 | 88.2 |

Front excursion rises 57 -> 89 mm as intended, and speed falls monotonically.
Equalising the gearing makes it slower. Refuted.

### Refuted: contact softness

Floor friction is 0.9 and the tangential force a foot needs is ~0.01 of normal,
so this was never Coulomb sliding. `impratio = 1.0` makes MuJoCo's frictional
constraints as soft as normal ones, which is the documented cause of tangential
creep. Sweeping it 1 -> 100 moves hind slip 13.5 -> ~12 mm and leaves speed flat.
Refuted.

### My framing error: stride length is not a Gate 2 check, and it is inside the
### triangle the research explicitly warned about

`docs/research/beyond_statistics_realism.md` states plainly: *"Encode at most one
member of the {stride length, cadence, speed} triangle at a time - they are
mutually unsatisfiable in the source itself."*

The cadence lock at 1.1888 Hz already spends one member. The published pair
(0.129 m/s voluntary, 0.62-0.82 SVL stride) implies a cadence of **1.69 Hz**. At
the locked 1.1888 Hz a 0.72 SVL stride forces 0.0907 m/s, which is 70% of the
published voluntary speed. The three cannot hold together, which is the
documented reason the gate list contains speed but not stride length.

**Stride length is not one of the six Gate 2 checks.** I added it as an extra
column in Session 3b and then treated it as a failure for three sessions,
including inferring a spine contribution (Session 3c) and a gearing fix
(Session 3e) to chase it. Both inferences were wrong and both are now refuted by
measurement. The correct handling is the documented one: report stride length,
do not gate it while the cadence is locked.

### Actual Gate 2 status: 4/6, two real failures

| # | check | value | target | |
|---|---|---|---|---|
| 1 | forward speed | 0.0416 | >= 0.04 | PASS |
| 2 | net/path | 0.7427 | >= 0.50 | PASS |
| 3 | hind swing load | 0.0806 | < 0.10 | PASS |
| 4 | front stance load | 0.5841 | >= 0.65 | **FAIL** |
| 5 | hind duty | 0.7334 | 0.73-0.83 | PASS |
| 6 | limb phase | 0.6347 | 0.405-0.465 | **FAIL** |

Both remaining failures are the same limb. The forefoot lands 0.320 cycle late
(hind lands 0.123 late) and registers 25-28 contact cycles against the hind's
19-20 at one commanded cadence, i.e. it bounces within its own stance. Limb
phase error equals the fore/hind touchdown-lateness difference exactly.

### Refutation ledger

Tested and refuted by measurement: hind stance height as the cause of limb phase;
fore stance height; actuator bandwidth (15x stiffer plant, near-critical damping);
spine amplitude as the source of stride; fore/hind gearing; contact softness.
Confirmed: hind stance height fixed hind duty (0.641 -> 0.733); limb-phase error
is fore/hind touchdown-lateness difference; hind slip is 41% of body advance.

The single unexplained mechanism is the forefoot: late touchdown and extra
contact cycles, with stance geometry, plant bandwidth, gearing and friction all
eliminated.
