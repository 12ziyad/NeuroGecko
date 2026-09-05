# NeuroGecko build log

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
