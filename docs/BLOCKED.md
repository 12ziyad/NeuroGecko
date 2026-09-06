# Build blockers and deferred work

## Session 2 decision — Gate 2 FAIL, stop before training or plant changes

Selected V2 + lab base, zero policy action, seed0,20s with first3s excluded,
genuine250Hz sampling. Each condition is n=1; repeated verification is not new data.

| Requirement | Measured | Pass? |
|---|---|---|
| Forward >=.04m/s | .042444 | Yes |
| Net/path >=.5 | .825320 | Yes |
| Each hind swing load <.10 | .012903/.010870 | Yes |
| Each front stance load >=.65 | .611390/.606408 | NO |
| Each hind duty within.05 of.78 | .640784/.631227 | NO |
| Each limb phase within.03 of.435 | .577408/.569030 | NO |

Contact cycles are now near1.1888Hz with CV1.67–3.13%, but stance support and
touchdown timing remain wrong. A further reward-trained residual must not be used
to conceal this defect. All failed manual trials are retained in the build log.

Also incomplete: zero commanded front penetration (still0.760mm at the frozen
mid-stance pose), biological angle/waveform validation, and Gate1's proposed
learning-correlation claim (unidentifiable without learning). Mechanical power
accounting itself is implemented and tested; the old residual-only effort term
asserted by the brief did not exist.

No activation lag, integrator/force-ceiling modification, frequency-gain sweep,
PPO, CMA-ES or GPU operation was performed. This is an improved experimental base,
not a validated animal brain or production walker.

Before any future lab training or resume, persist and validate the exact effective
`cpg.lab_parameters` in checkpoint contracts; the older timing-only contract does
not cover the new mechanical controls. The training CLI now fails closed for lab
until that contract and the failed gait gate are deliberately resolved. A reward
override or `--allow-*` resume switch does not bypass this explicit stop.

**RESOLVED, Session 4.** Both conditions are met, not waived. The effective lab
controller is now recorded from the live vector workers
(`GeckoWalkEnv.lab_controller_snapshot`: lab parameters, compensation scope,
phase offsets, commanded stance, frequency, and all 25 residual scales), every
worker must agree, it is written to `train_config.json`, and lab resume refuses a
changed one. The gait gate is resolved by requiring `--lab-base-evidence`: a
zero-policy `realism_metrics` report whose body hash, controller parameters,
stance compensation, 20 s duration and zero reset noise are all checked against
the run being launched. Gate 2 measures the base with the policy OFF, so it gates
the foundation and not the trained walker; the two checks it does not meet
(front stance load, limb phase) are recorded as unmet in the contract rather than
hidden. `artifacts/evidence/session3/gate2_all_limbs.json` is the accepted base;
the hind-only, uncompensated, withdrawn-CMA and 12 s reports are all rejected.
See docs/BUILD_LOG.md Session 4.

## The residual cannot fix the forefoot (Session 4, measured)

3.01 M PPO steps on the corrected base moved **no** checkpoint above the 4/6 the
base already scores; the best of 31 is 3/6, while eval reward more than doubled.
Front stance load went 0.5841 -> 0.5313, away from its 0.65 target.

Measured cause, final checkpoint: the forefeet are loaded for only 0.53/0.61 of
their commanded stance and carry ~30% of bodyweight (0.112/0.119 N against
0.3728 N), with **zero** contact during commanded swing. They are not pressing
too weakly; they are not reaching the floor for roughly 40% of the stance they
are commanded to hold. This is a commanded-pose/geometry problem.

The policy had no lever on it: `res_scale_vec` is 0.00 on `elbow_L`/`elbow_R`.
Session 4 added `--front-lift-residual-scale` and run 2b tested 0.08.

**Resolved, and the answer is structural.** Elbow authority does reach the gate
(front stance load 0.6740 at 1.1 M steps, against 0.65) but only on checkpoints
whose stride-period CV is 0.16-0.38 against a 0.10 ceiling. No trained checkpoint
in either run beats the base's 4/6. The base-controller routes were then swept
and all failed: `front_stance_press` is overwritten by the compensator through
stance, and aiming the frozen pose deeper is flat front-only and worse uniform.

The cause is that **the forelimb has one free joint for this correction and the
hindlimb has two** (`limb_specs` in `common/hind_stance_geometry.py`: elbow alone
versus knee plus ankle; this MJCF has no wrist actuator). One joint cannot set
foot height and foot fore-aft position independently, so lowering the forefoot
also moves where it lands. That is why the same compensator fixed hind duty in
Session 3 and cannot fix front stance load.

## Front stance load: closed as unreachable on this body

**Decision (Session 4): 4/6 is accepted as the walker.** The two unmet checks are
engineering targets recorded in `session2_gate2` as `species: INVENTED`, not
published biological thresholds; the walk has the published hind duty factor,
feet that lift, a straight heading and no falls. Reopening this requires a wrist
actuator, which is a morphology change governed by `docs/DECISIONS.md` (no joint
additions, preserve nq=39/nv=38/nu=25) and would need its own published anatomical
justification and a full retrain. It is not to be reopened as gate-chasing.

Also still open from the walking-posture measurement: shoulder height rides at
0.125 SVL against a published 0.11, and hip at 0.158 against 0.150. The static
morphology audit passes 14/14 at the stand keyframe, so **the standing pose is
correct and the walking pose is not**. That is a separate, unclosed finding and
should be attacked on its own terms rather than through the forefoot gate.

AWS recheck11:21UTC: instance reachable and idle, still on/billing. No remote
project Python process was running; only OS services. The owner can stop it.

Earlier-session notes follow for historical context; their old gait numbers are
superseded by the table above, not silently erased.

No blocker prevented model recovery.

## Completed prerequisites

Recovered and independently checksummed all 55 model files; saved original source;
measured the legacy walker baseline; exercised paired checkpoint publication,
laptop verification and real off-host resume. All cloud experiments had finite
process timeouts. A separate corrected body passes the 14 static tolerance bands.

## Work still required before a new production walker

- Tune the base controller on the corrected body, evaluating signed forward
  progress, actual four-foot contacts, stability and posture rather than path
  speed alone. The first 20 s lab-v2 no-residual run had no falls but only
  0.011736 m/s signed forward speed, 11.33 degrees nose-up pitch and contact-cycle
  duty factors around 0.41–0.55. It does not pass the intended gait targets.
- Couple the verified standalone EMG timing prior to an explicit motor model;
  normalized muscle activity is not a desired joint angle. CMA-ES tuning was
  not run in this increment. No fresh production retrain has been launched.
- Implement and test commanded stop/rest and gaze-actuator ownership in that
  fresh-training path. Neither is added to the recovered walker silently.
- A physical tail-restriction experiment remains outstanding. Zero lateral
  motor drive is implemented/measured, but it does not immobilize the tail.
- A proper longer, varied validation battery is needed. Deterministic repeats
  and three short feeding episodes cannot establish generalization or biology.

These are remaining implementation/calibration tasks, not claims that the
project is impossible. Static fitting is not dynamic validation.

## Infrastructure limits

AWS A10G CUDA compute works. NVIDIA EGL/GLX graphics libraries are absent, so
head-camera rendering uses Mesa llvmpipe on CPU. System driver installation or
replacement was not performed. Training without rendering remains possible;
GPU vision throughput should not be promised from the current setup.

SciPy is absent, so the optional genuinely high-rate strike-filter success test
is skipped. The evaluator rejects 50 Hz traces instead of pretending upsampling
recovers missing 500 Hz data.

The existing instance continues billing while on. Per-process timeouts do not
stop EC2; stopping it was asked separately. No new cloud resources were created.
