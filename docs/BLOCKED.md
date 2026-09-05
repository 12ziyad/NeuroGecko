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
