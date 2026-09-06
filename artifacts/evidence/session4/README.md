# Session 4 evidence

`gate_curve.json` / `gate_curve.md` per run: every checkpoint scored through
`realism_metrics.py` and then `eval.session2_controller.gate2`, the official
gate. Reward is deliberately absent — it is not the quantity being claimed.

Rows whose stride-period CV exceeds 0.10 are REJECTED from best-checkpoint
selection, matching `tools/fit_gate2_official.py`. `rows_rejected_for_irregular_gait`
lists them.

## Videos (local only)

`*.mp4` is gitignored here by repository convention, so these are not in the
repo. They are named for what they are rather than for what was hoped, and can
be regenerated with `tools/gate_checkpoints.py --video-best`.

- `s4_run1_seed0/step0_untrained_equals_base.mp4` — the step-0 snapshot, i.e. the
  base controller before any learning. It was briefly selected as "best" by a
  defect in `tools/gate_checkpoints.py` that has since been fixed; step 0 is a
  setup check, never a trained result.
- `s4_run2b_elbow008/rejected_step1100000_irregular_cv0.375.mp4` — the checkpoint
  that passed front stance load (0.6740) with stride-period CV 0.3752 against a
  0.10 ceiling. Retained because it is the evidence for ledger entry 14: the
  forefoot gate is reachable only by walking irregularly.

Neither is a trained walker worth adopting. Under the corrected rule the best
trained checkpoint is 3/6 in both runs, against the base's 4/6.
