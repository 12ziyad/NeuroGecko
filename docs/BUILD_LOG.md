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

## Session 3f — CMA-ES fixes the limb-phase gate that eight hand fixes could not

### Why the method changed

Eight single-parameter interventions were tried and all failed to move realised
limb phase: hind stance height, fore stance height, actuator bandwidth (both
without and, after finding the first test invalid, with force headroom), spine
amplitude, fore/hind gearing, contact softness, and commanded-delay calibration.

The last of those is the decisive one. Commanding the forefoot to land 0.197
cycle earlier, exactly cancelling its measured lateness, moved realised phase
the WRONG way (0.635 -> 0.688). If lateness were a fixed offset that would have
worked exactly. It is a coupled nonlinear plant, so single-parameter reasoning
does not apply and joint search is the correct tool.

An error in my own earlier work, recorded: the first stiff-plant test raised kp
15x but left forcerange unchanged, so the actuator saturated at 0.144 rad of
error and could never deliver the commanded stiffness. That test did not
measure what it claimed. Repeating it with force headroom (kp x15/force x4 and
kp x5/force x3) still did not fix limb phase, so the conclusion survives, but it
was unearned the first time.

### The fit

`tools/fit_gate2_cma.py`. CMA-ES over 10 controller parameters against the six
Gate 2 checks directly, so a pass is a pass on the same measurement rather than
a surrogate. It searches controller values only: no body geometry, no locked
cadence, no published stance ratio or target. 70 generations, popsize 12, on the
laptop's 12 cores. The harness reproduces the `realism_metrics` baseline to
within measurement scatter once two harness bugs were fixed (`commanded_contacts`
returns a dict, so iterating it yielded key strings; and the calibration scenario
sets a distant goal for heading, which the harness had omitted).

| check | hand-tuned | CMA-ES | target | |
|---|---|---|---|---|
| forward speed | 0.0416 | **0.0479** | >= 0.04 | PASS |
| net/path | 0.7427 | 0.5802 | >= 0.50 | PASS |
| hind swing load | 0.0806 | **0.0506** | < 0.10 | PASS |
| front stance load | 0.5841 | 0.6070 | >= 0.65 | FAIL |
| hind duty | **0.7334** | 0.6286 | 0.73-0.83 | FAIL |
| **limb phase** | 0.6347 | **0.4497** | 0.405-0.465 | **PASS** |
| gates | 4/6 | **4/6** | | different four |

**Limb phase is solved.** Stable across episode length, which was checked
explicitly after a scoring discrepancy: 11/14/17/20/25 s give 0.4295, 0.4237,
0.4497, 0.4546, 0.4463 — inside the band at every length. The apparent
instability was my own `--scale-override` changing the objective between the
search and the re-score, not a property of the measurement.

The solution is not one a human would have proposed: front stance press flipped
from +0.05 to -0.39, spine amplitude tripled (0.30 -> 0.94), tail amplitude
quintupled (0.15 -> 0.77), fore/hind ratio raised past 1.0, and the FL touchdown
delay moved to 0.358. Nine interacting changes.

### 6/6 appears unreachable with these parameters

Two further searches were run. A refinement with the two failing checks weighted
2.5x harder reached 1.92 on its own modified objective but scored worse on the
original one and lost the limb-phase gate. A third search warm-started from the
best solution with correct weights plateaued at 2.23 across ~55 generations,
i.e. slightly worse than the 2.10 it started from, and was stopped.

Roughly 150 generations across three independent searches all converge to
2.1-2.3 and none crosses it. On the evidence available this is a genuine limit
of the ten searched controller parameters on this body, not an under-converged
search. **Hind duty and limb phase trade against each other**; the optimiser can
buy either but not both.

### Adopted

The CMA-ES solution is preferred over the hand-tuned one: it wins the limb-phase
gate (the footfall pattern, which is the biologically meaningful one), is 15%
faster, and has lower hind swing load. It loses hind duty and net/path. Both
remain opt-in; the default is unchanged and legacy is untouched.

Gate 2 stands at 4/6 with the two failures documented and their cause
characterised as a parameter-space limit rather than an unfound bug.

## Session 3g — CORRECTION: the CMA-ES limb-phase result was a measurement artefact

The Session 3f claim that CMA-ES solved the limb-phase gate is **withdrawn**. It
did not. The fit optimised a quantity my search harness computed differently
from `realism_metrics`, and the difference is exactly the one that matters on a
non-entrained gait.

### The two definitions

`realism_metrics.py:240-252` walks each hind stride (measured touchdown to
touchdown), finds fore touchdowns falling inside it, and **keeps a stride only
if exactly one fore touchdown lies within it**, scoring
`(t_fore - t_hind_start) / measured_period`.

`tools/fit_gate2_cma.py` took a circular mean of every fore touchdown and every
hind touchdown against the **commanded** 1.1888 Hz, then subtracted.

On an entrained gait these agree. On a gait where the forefoot makes extra
contacts they do not: the official measure discards those strides, mine folds
the extra events into a circular mean, and mine also assumes a period the gait
is not actually holding.

### What the official measure says about the fitted config

| check | reverted (Session 3b) | CMA-adopted | target |
|---|---|---|---|
| forward speed | 0.0416 P | 0.0639 P | >= 0.04 |
| net/path | 0.7427 P | 0.5716 P | >= 0.50 |
| hind swing load | 0.0806 P | 0.0315 P | < 0.10 |
| front stance load | 0.5841 F | 0.5775 F | >= 0.65 |
| hind duty | 0.7334 P | 0.6829 F | 0.73-0.83 |
| limb phase | 0.6347 F | **0.6534 F** | 0.405-0.465 |
| **gates** | **4/6** | **3/6** | |
| stride period CV | 0.0149 | **0.2713** | |
| entrainment | False | False | |

My harness reported limb phase 0.4497 for this config; the official measure
reports 0.6534. The fit had driven the gait into a far less regular state —
stride CV rose 0.0149 -> 0.2713, an 18x increase — which is precisely the regime
where the two definitions diverge. The search was rewarded for making the gait
irregular, because irregularity moved my metric and not the real one.

**The `config/proxies.yaml` change adopting the fitted values has been reverted.**
The Session 3b configuration stands at 4/6 and remains the best measured result.

### What this does and does not change

Still standing, unaffected: the hind stance compensator and the hind-duty fix
(0.641 -> 0.733); the eight refuted hypotheses; that realised limb phase equals
the fore/hind touchdown-lateness difference; the 41% hind slip; that stride
length is speed at a locked cadence.

Withdrawn: "CMA-ES solved the limb-phase gate", and with it the inference that
hind duty and limb phase trade against each other — that conclusion rested on
the same bad metric. The three searches' 2.1-2.3 plateau measured a quantity
that was not the gate, so it says nothing about the gate.

### Lesson for the next fit

Any future search must call the same code path as the gate, not a
reimplementation. The correct move is to import the scoring from
`realism_metrics` (or run it as a subprocess and read its JSON) rather than
re-derive it in the search harness. `tools/fit_gate2_cma.py` is retained with
this defect recorded in its docstring; it must not be used to make a gate claim
until it scores through the official path.

Gate 2 stands at **4/6** with the Session 3b configuration.

## Session 3h — a corrected fit harness, and a second fit failure with a different cause

### The harness now scores through the gate's own code

`tools/fit_gate2_official.py` replaces `tools/fit_gate2_cma.py`. Every candidate
is scored by running `realism_metrics.py` as a subprocess and reading the six
checks out of its JSON. There is no second implementation of any metric, so
there is nothing that can diverge from the gate.

Supporting change: `realism_metrics.py` gained `--lab-params` (a JSON object of
per-run `lab_base_parameters` overrides) and `--fl-touchdown-delay`. Both are
lab-profile only and neither touches the shared registry. `spine_amp`,
`tail_amp` and `tail_phase_lag` were moved into `lab_base_parameters` **at their
existing constructor defaults (0.30 / 0.15 / 0.15)**, so behaviour is unchanged;
they are merely overridable now. The controller reads them from the registry
when present, matching how it already handles the front-press channels.

Verification that the harness is honest — baseline measured both ways:

| | forward | net/path | hind swing | front stance | hind duty | limb phase |
|---|---|---|---|---|---|---|
| via fit harness | 0.0419 | 0.7378 | 0.0740 | 0.5611 | 0.7311 | 0.6390 |
| via realism_metrics | 0.0416 | 0.7427 | 0.0806 | 0.5841 | 0.7334 | 0.6347 |

Guards, as hard rejections rather than scored preferences: the episode must
complete without termination, and HL stride-period CV must stay under the
registry's own 0.10 ceiling. The CV guard directly closes the hole the previous
harness exploited (baseline 0.0149, the bad fit 0.2713). Note that
`entrainment_pass` could **not** be used as a hard requirement: the baseline
itself fails it, so requiring it would reject the incumbent. Caught in a smoke
test before the run.

### The 12 s fit failed too, for an unrelated reason

45 generations at a 12 s evaluation window reached score 1.11 — genuinely better
than the flawed harness ever managed on its own inflated metric. Re-scored at
the gate's 20 s duration it read 4.38. Measured across durations:

| duration | forward | net/path | hind swing | front stance | hind duty | limb phase | gates |
|---|---|---|---|---|---|---|---|
| **CMA fit** 12 s | 0.0406 P | 0.6901 P | 0.0956 P | 0.5484 | 0.7176 | 0.3939 | 3/6 |
| 16 s | 0.0400 P | 0.6889 P | 0.1036 | 0.5577 | 0.7093 | **0.3472** | 2/6 |
| 20 s | 0.0404 P | 0.6885 P | 0.1076 | 0.5491 | 0.7210 | **0.6675** | 2/6 |
| **baseline** 12 s | 0.0419 P | 0.7378 P | 0.0740 P | 0.5611 | 0.7311 P | 0.6390 | 4/6 |
| 16 s | 0.0414 P | 0.7427 P | 0.0847 P | 0.5733 | 0.7340 P | 0.6374 | 4/6 |
| 20 s | 0.0416 P | 0.7427 P | 0.0806 P | 0.5841 | 0.7334 P | 0.6309 | 4/6 |

This is not simple overfitting to a short window. The fitted config's limb phase
reads 0.3939, 0.3472 and 0.6675 at 12/16/20 s — it is bimodal. The baseline's is
0.6390/0.6374/0.6309, stable to three decimals. Stride CV is low in both
(0.004 vs 0.015), so the CV guard did its job; the instability is in the phase
measurement itself on that configuration, not in gait regularity.

Two things this does establish: limb-phase values inside the target band **are**
reachable on this body (0.3939 and 0.3472 were measured, both near or inside
0.405-0.465), and the 12 s fitting window was the wrong choice because the gate
measures at 20 s.

### Current action

Refitting at **20 s**, the gate's own duration, so no window mismatch can exist.
The baseline remains the incumbent at 4/6, stable across every duration tested,
and is not replaced unless a candidate beats it at the gate's own duration.

# Session 4 — the walker is trained, and training does not fix the forefoot

Laptop only. 12 cores, torch 2.12.0+cpu, no AWS, no dollars spent. Measured
throughput 894 fps at 16 subproc envs with `--n-steps 2048`; the 3.01 M-step run
took 4,882 s wall clock at a run-average 617 fps (eval and checkpointing
included). The instance was not started and is not needed: it has 4 vCPU against
this laptop's 12, its own 10 M-step run logged 853 fps, and the policy is a
185,907-parameter MLP that never touched a GPU.

## What was built

**The lab training block is replaced by an evidence contract.** `BLOCKED.md`
asked for two things before any lab training or resume: the failed gait gate
deliberately resolved, and the exact effective lab controller persisted and
compared in checkpoint contracts. `require_lab_training_readiness(args)` now
requires `--lab-base-evidence`, a `realism_metrics` report measuring this body
and base controller with a **zero policy**, and checks that its XML hash matches
`--xml-path`, its `effective_lab_parameters` match the registry, its stance
compensation matches the flag, its duration is the gate's own 20 s, and its
reset noise is zero. The whole contract is written into `train_config.json`.

The reason it is an evidence check and not a pass requirement: **Gate 2 scores
the base with the policy switched off.** It gates the foundation, not the
trained walker. Two of its six checks — front stance load and limb phase — are
precisely what a learned residual exists to attack. They are recorded as unmet,
not waived.

Verified rejections, all against real reports already in the repo:

| Evidence offered | Outcome |
|---|---|
| `gate2_all_limbs.json` (registry params, comp `all`, 20 s) | **accepted** |
| `gate2_with_compensation.json` (hind-only, duty .700) | rejected on hind duty |
| `gate2_baseline.json` (no compensation) | rejected on compensation flag |
| `gate2_ADOPTED.json` (withdrawn CMA fit) | rejected on controller parameters |
| `gate2_baseline12.json` (same base, 12 s window) | rejected on duration |
| `gecko_body_r.xml` (61 g body) | rejected on XML hash |

A pre-Session-3g report records a subset of `lab_base_parameters`, because
`spine_amp`/`tail_amp`/`tail_phase_lag` entered the registry at the controller's
existing constructor defaults. That is accepted only by reading those defaults
back off `CPGResidualController.__init__` and proving each missing key equals
the value the older run actually executed — never by assuming it.

**A defect that would have silently run the wrong experiment.** `make_env` did
not forward `hind_stance_compensation`, so every lab run would have trained on
the **uncompensated** base — hind duty 0.641, a 2/6 foundation — while the
config claimed otherwise. Fixed and pinned by `test_make_env_forwards_the_base_controller_controls`.

**`--target-kl`.** The documented 10 M-step collapse (eval reward 1128.76 at
1.5 M to 422.97 at 10 M) ran `approx_kl` to 67.26. SB3 will stop an epoch loop
on KL if asked; the script never asked. It fired once in the 20 k smoke and not
once in the 3 M run, which is the correct behaviour for a stable run.

**`--front-lift-residual-scale`.** The FL/FR lift lock stays structurally on;
only its scale becomes caller-selected, so the locked channels stay enumerated.

**`envs/gecko_walk_env.py::lab_controller_snapshot`** — a plain picklable record
of the *live* controller in every vector worker: effective lab parameters,
compensation scope, phase offsets, commanded stance, frequency, and the residual
scale of all 25 actuators. `environment_calibration_snapshot` collects it,
requires every worker to agree, and stores it; lab resume refuses a changed one.
Returns `None` for legacy, so no historical run changes.

**`tools/gate_checkpoints.py`** — scores every checkpoint of a run through
`realism_metrics.py` and then `eval.session2_controller.gate2`, the same function
`tests/test_session2_gate.py` pins and this log tabulates. Nothing is re-derived:
Session 3g is the standing warning about a harness that reimplemented limb phase
and optimised a quantity that diverged from the gate. Evaluation flags are read
from the run's own `train_config.json`, so the body and controller a checkpoint
is scored under cannot drift from the ones it was trained under. Reward is
deliberately not a column.

Its zero-residual row reproduces `gate2_all_limbs.json` exactly — 0.0416 /
0.7427 / 0.0806 / 0.5841 / 0.7334 / 0.6347, stride CV 0.0149, 4/6.

224 tests pass, one SciPy skip.

## Run 1 — 3.01 M steps, seed 0, elbow locked at 0.00

`artifacts/evidence/session4/s4_run1_seed0/gate_curve.{json,md}`, 32 rows.

**No trained checkpoint reached 4/6.** The best of 31 is 3/6; the base is 4/6.
Step 0 scores 4/6, which is the setup check passing — an untrained policy must
reproduce the base — and is not a trained result.

| | base | 3.01 M | |
|---|---|---|---|
| signed forward m/s | 0.0416 | **0.0607** | +46 % |
| net / path | 0.7427 | **0.7943** | better |
| hind swing load | 0.0806 | 0.0548 | better |
| front stance load | 0.5841 | **0.5313** | **worse**, target ≥ 0.65 |
| hind duty | 0.7334 | **0.5986** | **lost**, band 0.73–0.83 |
| limb phase | 0.6347 | **0.8848** | **worse**, band 0.405–0.465 |

Eval reward over the same run went 1,490 → 3,120 and was still setting new bests
at 2.5 M. **The reward more than doubled while three gates degraded.** Had this
run been judged by `ep_rew_mean` it would have been recorded as a success.

What it learned is a faster, less gecko-like gait: it buys speed by shortening
hind stance. It also unloaded the fronts, which is the exact escape the
controller's own docstring documents from V4.2.3 — "PPO learned to unload the
fronts because progress rewards the faster front-light crawl". The V4.2.4
per-actuator caps were introduced to close that escape structurally. **They did
not: it re-emerged through the uncapped channels** (hips 0.25, ankle 0.25,
spine_pitch 0.10).

## Why the fronts fail, measured

Final checkpoint, 20 s, first 3 s discarded, contact threshold 0.03502 N,
bodyweight 0.3728 N:

| foot | loaded in commanded stance | contact overall | contact during swing |
|---|---|---|---|
| HL | 0.7723 | 0.6153 | 0.0548 |
| FL | **0.5313** | 0.3732 | **0.0000** |
| HR | 0.8441 | 0.6814 | 0.0924 |
| FR | **0.6145** | 0.4287 | **0.0000** |

Mean front force through commanded stance: FL 0.1123 N, FR 0.1191 N — about
**30 % of bodyweight**. The front feet never touch during swing, so this is not
mistimed contact being scored as absence.

The forefeet are not pressing too weakly. **They are not reaching the ground for
roughly 40 % of the stance they are commanded to hold.** That is geometry, and
it matches the standing hypothesis carried from Session 3: trunk pitch lifting
the shoulder puts the commanded pose above the floor.

And the policy has **no lever on it**. `res_scale_vec[elbow_L] = res_scale_vec[elbow_R] = 0.00`.
Three million steps were spent attacking a forefoot-height problem with hips and
spine because those were the only channels open.

## Next

Run 2b: identical, with `--front-lift-residual-scale 0.08`. The elbow is the only
actuator that sets front foot height directly. This is the plan's fallback
branch, now motivated by measurement rather than by expectation.

The risk is explicit and is the reason the gate table exists: elbow authority can
press the foot down or lift it further, and lifting is the documented escape.
`front_track`/`front_miss` penalise it and `front_factor` throttles progress
(measured front duty score 0.533, so the throttle was active all run and did not
prevent the degradation). Whichever way it goes will be read off the gates.

## Run 2b — elbow authority unlocked to 0.08

`artifacts/evidence/session4/s4_run2b_elbow008/`. 3.01 M steps, seed 0, otherwise
identical to run 1.

**The hypothesis was right and the result is still a failure.** Front stance load
passed its gate for the first time in the project's history — 0.6653 at 700 k,
0.6593 at 1.0 M, **0.6740 at 1.1 M**, 0.6613 at 1.2 M, against a 0.65 target.
Run 1, elbow locked, never exceeded 0.6017. The elbow is the lever.

But every checkpoint that passed it has a wrecked stride rhythm:

| | stride-period CV |
|---|---|
| base | **0.0149** |
| rows that passed front stance load | **0.16 – 0.38** |
| ceiling | **0.10** |

Only 5 of 31 trained checkpoints stayed inside the CV ceiling, and the best of
those is 3/6. The policy raises front load by walking irregularly, not by walking
better — the same shape of exploit as Session 3g, where a harness was rewarded
for driving CV 0.0149 → 0.2713.

### A defect in the scoreboard, found by that result

`tools/gate_checkpoints.py` ranked by gate count and applied **no** CV rejection,
so it selected step 1.1 M — CV 0.3752 — as "best". `tools/fit_gate2_official.py`
rejects CV > 0.10 outright; the new tool did not. That is the Session 3g hole
rebuilt. Fixed: `CV_CEILING` is now imported from the fitter so there is one
definition, irregular rows are hard-rejected from selection and labelled
`REJECTED` in the table, `rows_rejected_for_irregular_gait` is recorded, and two
tests pin it. Both runs were re-scored under the corrected rule:

| run | baseline | best trained | beats base |
|---|---|---|---|
| run 1 (elbow locked) | 4/6 | step 400 k, **3/6** | no |
| run 2b (elbow 0.08) | 4/6 | step 200 k, **3/6** | no |
| | | rejected for irregular gait | 24 and 26 rows |

**No trained checkpoint beats the base in either run.**

## The standing hypothesis is refuted by direct measurement

Carried since the handoff: *trunk pitch rides nose-up, lifting the shoulder, so
the forefoot cannot reach the floor*. It had never been tested directly.

Base controller, 20 s, first 3 s discarded:

| quantity | measured | published | verdict |
|---|---|---|---|
| trunk pitch | **3.87°** | 3.72 ± 0.61° nose-up | **inside the range — correct** |
| shoulder height | **0.125 SVL** | 0.11 SVL | **13.6 % too high** |
| hip height | 0.158 SVL | 0.15 SVL | 5.3 % too high |

**Trunk pitch is not the fault.** The whole body rides too high while walking,
and the shoulder proportionally more than the hip. The forefeet sit **3.14 mm
(FL) and 3.25 mm (FR)** above the floor on the samples where they are commanded
down but carry no load, peaking at 7.7 mm. Front feet touch during commanded
swing **0.0000** of the time, so this is not mistimed contact scored as absence.

Note the static morphology audit passes 14/14 including hip height at the stand
keyframe. **The static pose is right and the walking pose is not**; those are
different measurements and only the first was ever checked.

## Two more refutations

**Pressing the front foot harder does almost nothing, and the reason matters.**
Sweeping `front_stance_press` 0.05 → 0.36 moved front stance load 0.5841 → 0.6253
with the stride staying clean (CV 0.0132), then everything collapsed at 0.44
(0/6, CV 0.46). The weak effect is explained by `base_ctrl`: when compensation is
active the compensator writes an **absolute** elbow target blended by swing
weight, so through stance — the part that is scored — `front_stance_press` is
overwritten entirely. It was a knob disconnected from the thing being measured.

**Aiming the frozen pose deeper fails, both ways.** `StanceCompensator` solves
each foot's pose against `self._stand_qpos`, i.e. with the body at its standing
height; a walking gecko rides higher, so the solved foot lands short. Aiming
deeper is the obvious correction and it does not work:

| target depth | front stance load (uniform) | front stance load (front only) |
|---|---|---|
| −0.60 mm (default) | 0.5841 | 0.5841 |
| −1.50 mm | 0.5801 | 0.5752 |
| −2.50 mm | 0.5231 | 0.5680 |
| −3.50 mm | 0.5028 | 0.5656 |
| −7.00 mm | 0.5339 | 0.5863 |

Uniform depth is confounded — it jams the hind feet down too, hind swing load
0.0806 → 0.2710 — so the front-only column is the clean test. It is flat.

## Why: the forelimb is under-actuated for this correction

Read from `common/hind_stance_geometry.py` `limb_specs` and confirmed on the
built compensator:

| limb | free joints available to the solve |
|---|---|
| HL / HR | **2** — `knee`, `ankle` |
| FL / FR | **1** — `elbow` (this MJCF has no wrist actuator) |

A limb with two free joints can set foot **height** and foot **fore-aft position**
independently. A limb with one cannot: changing the elbow to lower the foot also
swings it fore-aft, so the correction moves where the foot lands instead of
pressing it down. That is why the same compensator fixed hind duty in Session 3
(0.641 → 0.733) and cannot fix front stance load now.

This is a structural limit of the body, not a tuning failure, and it bounds what
any base-controller correction can achieve. `docs/DECISIONS.md` forbids adding a
shoulder joint and requires preserving nq=39/nv=38/nu=25, so adding a wrist is a
morphology decision to be taken deliberately, with its own published
justification, and not as a fix smuggled in to pass a gate.

## Session 4 evidence ledger

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 11 | A learned residual can fix the forefoot | **Refuted** | 2 runs × 3.01 M steps; best trained 3/6 vs base 4/6 |
| 12 | Trunk pitch lifts the shoulder | **Refuted** | pitch 3.87° vs published 3.72 ± 0.61 |
| 13 | Elbow authority is the missing lever | **Confirmed** | front load 0.5841 → 0.6740, unreachable with it locked |
| 14 | ...and it can be had without cost | **Refuted** | every passing row had CV 0.16–0.38 vs 0.10 ceiling |
| 15 | Pressing harder in the base fixes it | **Refuted** | 0.5841 → 0.6253 then collapse; press is overwritten in stance |
| 16 | Aiming the frozen pose deeper fixes it | **Refuted** | flat front-only, worse uniform, across −0.6 to −7.0 mm |
| 17 | The walking posture rides too high | **Confirmed** | shoulder 0.125 vs 0.11 SVL; forefoot 3.1 mm high in stance |
| 18 | The forelimb is under-actuated for it | **Confirmed** | 1 free joint vs the hindlimb's 2, read from `limb_specs` |

227 tests pass, one SciPy skip.
