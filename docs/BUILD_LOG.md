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
