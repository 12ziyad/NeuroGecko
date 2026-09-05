# Build blockers and deferred work

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
