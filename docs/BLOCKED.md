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

## Two brain evidence files pin a body that no longer exists (found Session 4d)

`artifacts/evidence/brain_legacy.json` and `artifacts/evidence/brain_sealed.json`
record `xml_sha256 = 58cdb1e8a842...`. No file in the repository hashes to that
value: `gecko_body_r.xml` is `c4293da9a0e8...`, `gecko_body_lab_v2.xml` is
`7567653564177...`, `gecko_world_v1.xml` is `e6bda295a42e...`.

Neither measurement can therefore be reproduced against any body still present,
and neither can be compared to a new one. This predates Session 4 and was not
introduced by it; it is recorded here rather than stepped over.

Not resolved, because resolving it honestly means one of:
- recovering the body that hash names, if it exists in `models_recovered/` or on
  the instance, and committing it; or
- re-measuring both conditions against a body that does exist and superseding the
  files, keeping the originals; or
- marking both files as unreproducible in place.

Choosing among those is a judgement about what the original measurement was for,
and it belongs to whoever knows why those two conditions were recorded.

## The brain environment's remaining shortcuts (Session 4d)

Fixed: the 0.10 m eat radius (against a published 4.07 cm strike distance), the
0.035 m food radius, the food-as-painted-marker, and the inability to remove the
privileged block rather than merely scale it.

Still open and deliberately so:
- **The reward is ground truth.** `r_progress` is 12.0 x the change in mouth-to-food
  distance and `r_eat` is a flat 10.0. A reward is external to the animal by
  construction, so this is not the same shortcut as a privileged observation, but
  it is still an oracle and it is still doing work.
- **`oracle_action()` exists** and returns the exact bearing to food. It is the
  supervision signal the recovered visual student was distilled against.
- **No strike.** Capture is a distance test. The published capture rate comes from
  a 0.851 m/s strike from 2.03 cm, and a 16-20 ms strike is shorter than one 50 Hz
  control step, so scoring one needs 500 Hz sampling. Until that exists, a capture
  rate measured here is not the published quantity.

## Fear cannot be built until a chemosensory channel exists (Session 5b)

`envs/gecko_brain_env.py:546` computes `danger = 0.65*belly_contact + (1.0 if
fallen else 0.0)` -- entirely mechanosensory. The published defensive-response
probabilities for this species, n=40-42, are:

    mechanosensory alone   0     (chi2 < 0.01, P > 0.9)
    visual alone           0
    chemical alone         0.20  (95% CI 0.09-0.40; chi2 = 8.098, P = 0.0044)
    chemical + visual      0.40

So the signal feeding fear is built from the one modality that provably produces
no defensive response, and the modality that does -- smell -- has no input
anywhere in this repository.

`danger` is left in place as a physical-harm penalty, which it legitimately is:
falling and belly-dragging are real failure states and the walker terminates on
them. What is refuted is reading that signal as fear.

Blocked on: a chemosensory input. There is no olfactory or vomeronasal channel in
the observation, no odour field in the world, and MuJoCo has no scent primitive,
so this is a world feature to be designed rather than a parameter to be set. Until
it exists, any fear response in this model is triggered by the wrong sense, and
the Tier A5 predator-cue test cannot be run honestly.

Related and also open: `brain/drives.py` still carries hand-written `fear` and
`danger` integration with invented constants. It is the default path; the
hypothalamus is opt-in. Fear left the hypothalamus deliberately (it belongs to the
tectal escape integrator per best_achievable_brain.md section 3.4) and has not
landed anywhere yet.

## Fatigue recovery rate is invented (Session 5b)

The endurance side is published: t_end = 0.030 * v^-2.07 hours at 25 C, n=25,
Teratoscincus/Coleonyx. How fast a lizard RECOVERS from fatigue appears nowhere in
the corpus for any species. `fatigue_recovery_time_constant_s` is 600 s by
engineering choice and is tagged INVENTED. Any behaviour that depends on recovery
rate -- pacing, rest bouts, whether the animal can sprint twice -- rests on it.

## Basal ganglia: built, not accepted, blocked on a parameter table (Session 6)

`brain/basal_ganglia.py` implements the extended GPR architecture and does NOT
reproduce the Prescott 2024 tonic-dopamine sweep, which is its published
acceptance test. Evidence: `artifacts/evidence/session6/dopamine_sweep.json`.

Reproduced: immobility at lambda <= 0.06; distortion rising with lambda;
disinhibition rather than maximum selection; sigma-pi salience gating.

Not reproduced: the switching pattern is INVERTED. 131 switching bouts at the
0.20 baseline against a published ~7, and 1 at lambda 0.43 against a published
21.3. The published result that GPR dithers MORE than a winner-take-all under
excess dopamine comes out backwards.

Root cause: docs/research/ names the GPR model and gives the behavioural sweep
but does not state the connection weights or thresholds. They were reimplemented
from the equations. Three parameter adjustments were tried; two made things worse
and were reverted, with what was tried recorded in the evidence file so it is not
retried. Further adjustment would be fitting the model to the answer.

Blocked on: the Gurney, Prescott & Redgrave 2001 parameter table from the primary
paper, or a licence for ModelDB 124111 (Girard et al. 2008), which carries none
anywhere and so cannot be vendored into this Apache-2.0 repository. With either,
re-run the sweep unchanged; the harness and the "before" are committed.

Also open, and recorded as an expectedFailure test rather than hidden: the module
oscillates at zero salience. The thalamocortical loop gain was one cause and is
fixed (it was exactly 1.0, the boundary of positive-feedback instability). The
remaining cause is the diffuse STN-GPe loop, whose effective gain grows with
channel count, so the module is less stable the more behaviours the animal has.

NOTHING IN THE REPOSITORY DEPENDS ON THIS MODULE. It is not wired into any
environment and must not be until the sweep reproduces. The if/else arbiter on
`target_interest` remains the live path.
