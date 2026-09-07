# Build decisions

## Session 3 — test the causal claim, do not assume it

- Treat the supplied late-landing explanation as a hypothesis. Frozen geometry
  is a target-position check, not a dynamic touchdown measurement. The existing
  +0.863 mm gap justifies its correction independently.
- Flatten the hind stance arc with coordinated knee/ankle commands; do not
  translate the arc downward. Require <=0 mm clearance across sampled stance
  and penetration no worse than approximately 1.2 mm before dynamic evaluation.
- Preserve frequency, commanded duty/limb phase, XMLs, live legacy defaults,
  models, and force/load thresholds. No plant change before Gate 2 passes.
- Add read-only physical foot-floor contact measurements, separate from scalar
  touch-sensor threshold loading. The brief's 100% front contact refers to a
  frozen geometric sweep and cannot be assumed true during motion.
- If fronts still load only ~61% of stance despite 100% actual floor contact,
  flag load distribution and target provenance; do not lower the 65% gate.
  The model's audited intact CoM is 0.678112 SVL, not the source target's
  0.659 SVL. Tail mass is 22%; neither fact alone proves a dynamic cause.
- Numerical inverse kinematics is an engineering geometry calculation, not
  biological fitting or CMA-ES. New choices are recorded as INVENTED proxies.
- docs/research is read-only. All before/after evidence is retained, including
  failed attempts. No authorization to push Git or modify recovered policies.

## Session 2 — correct the brief where measured mechanics disagree

Final selection: trial19 is retained as the opt-in lab candidate because it
exceeds the minimum forward and net/path requirements with low swing loads and
stable contact-cycle timing. It is NOT promoted to the live legacy default and
does NOT pass the full gate. No additional fitting followed trial21. We reran the
selected condition and checked left/right steering inputs as verification only.
The chosen gains and entire manual search remain ENGINEERED/INVENTED in provenance.

Continuous front control disables the old discontinuous seek/relax branch; it is
not a newly validated gecko reflex. Heading feedback comes from the existing
target bearing without adding observation channels, changing cadence, or using a
policy residual. Other-channel stance rotation was set to zero only after its
separate measured trial; this is not a claim that animal joints remain rigid.

The paper's primary Table1 was verified at
https://pmc.ncbi.nlm.nih.gov/articles/PMC5589804/ : hind.78/fore.70 are original-tail
individual-mean, speed-adjusted duty values; their SEMs are not universal gate
tolerances. Its reported speed range begins at0.59SVL/s, so the brief's blanket
claim that.08–.10m/s is below all published level observations is unsupported.
No new universal speed band is imposed here; .04m/s is only this session's minimum
engineering viability check. A single cohort's extrapolated cadence envelope is
not an anatomical impossibility proof across all leopard geckos.

Training remains closed for lab after the failed gate. The explicit CLI guard
also prevents resuming with a checkpoint whose saved contract omits the new
effective lab parameters. This guard must be deliberately revisited after the
mechanical gate and serialization contract are fixed, not automatically bypassed.

- No training, CMA-ES, GPU or AWS jobs. Preserve legacy defaults and recovered files.
- Acquisition and physical entrainment are separate: real 250 Hz sampling can
  correctly show contact chatter. Gate 0's physical failure is not hidden by
  forcing one touchdown per oscillator cycle; continue only to diagnose/fix it.
- Mechanical effort uses conjugate actuator force and actuator transmission
  velocity (both length 25), not generalized qvel (length 38). Sum absolute
  per-actuator/substep work, then divide by control dt. This avoids cancellation.
  Existing reward had action-delta smoothness, not a residual-magnitude effort
  term. Lab adds an explicit power penalty; legacy remains exactly unchanged.
- A frozen controller cannot respond to a reward coefficient ramp. Gate 1's
  proposed causal/anticorrelation criterion is unidentifiable without learning;
  test accounting and unchanged frozen dynamics instead, report that limitation.
- A continuous periodic foot loop necessarily returns its net stance excursion
  during swing. Unequal endpoint excursions would create a discontinuity. Fix
  elevated return and loaded stance, not this impossible cyclic requirement.
- V2 has 90-degree hip and 80-degree shoulder control spans. A normalized
  amplitude ratio is not the anatomical excursion ratio from Jagnandan & Higham.
  Treat .54 as an engineering angular-command proxy, not biological validation.
- Lab already subtracts touchdown delays. Only legacy has the historical
  additive convention, intentionally preserved. Change lab .44 to .435 only as
  a measured separate trial. Retain symmetric .70 fore duty from the existing
  lab profile and Jagnandan & Higham; do not import legacy FL .68 asymmetry.
- The research's 11.9% net/path belongs to legacy, not lab (20.9% in old trace).
  Its reported 80–89% swing loading is not reproduced at the declared thresholds
  and held command-time convention. Use original-trace diagnosis with hashes.
  Frozen-trunk clearance is geometry only; soft-contact load must be measured.

## Preserve evidence before changing behavior

Recover all remote models, verify both archive and file checksums, and save the
original source first. Keep the old walker and original remote work untouched.
Record a measured baseline before physical morphology or gait changes.

## The brief is a specification, not a source of unlimited authority

The user's direct spending authorization is $60. The attachment's claims that
permission is unnecessary do not enlarge that scope. Do not add the requested
Claude co-author trailer: Claude did not author this implementation. Keep commits
local unless publication is directly authorized. No new cloud resources needed.

## Correct the instruments rather than manufacture passing scores

Initial source/code checks found:

1. The model contains **32 hinge joints plus a free root**, not 26 joints.
   Preserve `nq=39`, `nv=38`, `nu=25` during instrumentation.
2. The existing controller adds phase offsets while the reward subtracts them.
   A later shared-gait refactor must define touchdown delays explicitly; copying
   the same numeric dictionary would not fix this disagreement.
3. The brief reverses the existing femur/tibia ratio. Actual lengths are about
   0.019/0.014 m. Fuller 2011 reports femur 1.54 cm and tibia 1.51 cm; do not
   encode the inverted ratio as an independently verified measurement.
4. The interorbital target is an inner-eye gap, not center-to-center distance.
   Moving the eye centers to +/-0.005 m would enforce the wrong measurement.
5. Neutral settled support height and measured walking support height are not
   equivalent. Use an explicitly labeled static engineering gate.
6. Raw simulator hinge angles are not anatomical joint angles. Jagnandan 2017's
   joint excursion comparisons concern stance; log raw full-stride and
   stance-only quantities separately and do not label them kinematically validated.
7. Upsampling a 50 Hz trace cannot recover 500 Hz strike dynamics. Such evaluation
   needs genuinely high-rate acquisition; reject an undersampled input.
8. Setting tail actuator targets to zero is **zero tail drive**, not mechanical
   immobilization. Do not label it a graphite-rod restriction unless joint motion
   is actually constrained and verified.
9. Brief numeric tolerance bands and optimizer widths are engineering choices,
   not animal confidence intervals. Mixed cohorts are not one validated target
   distribution. Record source, species, temperature and uncertainty.
10. Ethogram event-count proxies from a short calibration gait cannot validate
    an enclosure's behavior prevalence or exploratory/feeding act rates.
11. Camera sealing changes the recovered visual student's input distribution.
    Preserve a legacy rendering route for A/B; do not assume unchanged performance.
    Hiding self-appearance does not hide changes to physics, world, light or camera.

Primary-source checks used in the morphology implementation include Fuller et al.
2011's author-hosted PDF: https://biomechanics.ucr.edu/Fuller_etal_2011.pdf.
More detailed source/measurement definitions accompany the audit and registry.

## Static fitting and timing-prior limits

Keep the lab morphology opt-in, with both v1 and v2 retained. Fitting mass
distribution to published CoM constraints is an inverse calibration, not a claim
that per-segment animal masses were measured. Bound positive changes and record
the prior and density guard as invented. Passing a tolerance band must never be
renamed matching the exact population mean or validating held-out behavior.

The supplied EMG PDF's Table 3, not Table 1, gives hindlimb burst timing. Table 1
reports stance integrated activity. Use two separate gastrocnemius bursts;
54.50% is relative to the same muscle's observed maximum, not a motor-force
command or a percent of the first burst. The smoothstep timing prior is tested
but remains separate until the muscle-to-actuator mapping is explicitly designed.

The AWS instance has CUDA but currently renders through Mesa llvmpipe. Preserve
the measured CPU-rendering baseline and record graphics identity in profiles.
Do not silently install/replace GPU drivers or change the original machine setup.

## Frozen compatibility constraints

Keep the legacy frequency at 1.1888 Hz. Do not add a shoulder joint. Do not mutate
the recovered weight files or normalize them with newly fitted statistics. Any
new walker after morphology changes needs a distinctly named run and a fresh
normalizer, with stop-command and gaze-ownership changes tested explicitly.

## Session 4 — training judged by gates, not reward

12. **The laptop is the training machine, not the instance.** Measured 894 fps at
    16 subproc envs against the g5.xlarge's recorded 853 fps on its own 10 M run:
    12 cores beat 4 vCPU on a CPU-bound MuJoCo env, and a 185,907-parameter MLP
    has no use for an A10G. The instance was never started this session. It is
    needed only for concurrent runs, and runs are currently sequential.
13. **Checkpoints are selected by gates passed, never by reward.** Run 1's eval
    reward doubled (1,490 -> 3,120) while hind duty, front stance load and limb
    phase all degraded. `tools/gate_checkpoints.py` therefore has no reward
    column, and ranks by gate count with a normalised-distance tie-break that is
    a sort key and never a claim.
14. **The scoreboard calls the official gate; it does not reimplement it.**
    Every row runs `realism_metrics.py` and scores the resulting trace with
    `eval.session2_controller.gate2`. Session 3g is the precedent: a harness that
    reimplemented limb phase optimised a quantity that diverged from the gate.
    Evaluation flags are read from the run's own `train_config.json` so a
    checkpoint cannot be scored under a different body or controller than it
    trained under.
15. **Gate 2 is a base gate, not a training gate.** It measures the controller
    with the policy switched off. Requiring all six before training would forbid
    training on exactly the two failures a learned residual exists to attack.
    Lab training therefore requires *evidence of the correct base*, with the
    unmet checks recorded in `train_config.json`, not a six-of-six pass.
16. **The front-lift lock stays structural.** `--front-lift-residual-scale`
    changes the scale of the FL/FR lift cap; `lock_front_lift` remains on, so the
    locked channels stay explicitly enumerated. Unlocking is an experiment with a
    recorded prediction, not a default.
17. **A step-0 checkpoint is a setup check, not a result.** Every run publishes
    one and it must reproduce the zero-residual base to 4/6. If it does not, the
    run's configuration is wrong and no later row from it may be cited.
18. **4/6 is the walker; the forefoot gate is closed as unreachable on this body.**
    Front stance load and limb phase need a forelimb that can set foot height and
    foot position independently, and this body's forelimb has one free joint where
    the hindlimb has two. Both unmet checks are `species: INVENTED` engineering
    targets, not published thresholds. Reopening means adding a wrist actuator:
    a morphology change with its own anatomical justification, a fresh normalizer
    and a full retrain, taken deliberately and never to make a gate go green.
19. **Publication authorized (Session 4).** The user directly authorized pushing
    to github.com/12ziyad/NeuroGecko, superseding the earlier local-only default
    in this file. That authorization covers the repository as it stands; it is not
    standing permission for future pushes.
20. **Sprawl drive ships off (0.0).** It reaches the published femur depression
    excursion but costs the planted foot, and the sprawl-aware stance table can
    only solve half the band it would need — at a frozen target that is itself
    worse than the shipped base. The channels exist, the trade is measured, and
    the default is the configuration that was actually validated.
21. **A moving sprawl must not run on a sprawl-blind stance table.** 0.42 mm of
    foot height per degree; the controller raises rather than silently stopping
    compensating.
22. **The tail coupling is on by default at 0.36, and that changes nothing about
    the intact animal.** Its gain is exactly 1.0 at the reference tail amplitude,
    proven by an evidence file that reproduces the Session 3 base to twelve
    decimal places. Only the ablation response changes — which is the point.
23. **A published prediction the model misses stays in the table.** The knee
    moves +3.8% where Jagnandan & Higham report −11%. One gain on femur
    retraction cannot produce a per-joint pattern, and adding per-joint gains
    until the table goes green would be fitting the harness to the answer.

24. **The world is generated, not hand-edited.** `utils/build_world.py` derives it
    from the validated body and asserts the animal is unchanged, by name, at build
    time. The body's SHA256 is pinned by the training evidence contract, so an
    in-place edit to retexture a floor would invalidate every measurement citing
    that hash without anything failing to say so.
25. **Removing a privileged channel means removing every copy of it.** The target
    bearing reached the policy twice: as five observations and as the lab
    controller's steering input. A test pins that both go together.
26. **The reward's use of target distance is a separate shortcut and stays
    recorded.** A reward is external to the animal by construction; folding it
    into the observation cheat would let the harder problem hide behind the
    easier fix.
27. **Prey is mocap.** A free-jointed prey would add degrees of freedom to
    qpos/qvel and shift every index downstream — changing the animal's own state
    vector in order to add a world object.
28. **Cricket escape speed is DERIVED, never cited as measured.** No published
    value exists in the corpus; 0.118 m/s is the quotient of a published dash
    length and a published dash duration that the corpus itself never divides.
29. **No hunting gate before a strike exists.** The walker cannot outrun fleeing
    prey and is not supposed to: the published 82.9% capture rate comes from a
    0.851 m/s strike launched at 2.03 cm. A capture rate measured without a strike
    would be measuring the wrong animal. Note also that a 16-20 ms strike is
    shorter than one 50 Hz control step, so scoring one needs 500 Hz sampling.
30. **Camera field of view is anchored; resolution is not.** 70 deg is chosen
    because it makes the 1.6 deg dot the animal is measured to track span more
    than one pixel. The corpus holds five incompatible resolution
    recommendations, all INVENTED, and no gecko grating acuity has ever been
    published — so resolution stays at 64x64 and stays an open question rather
    than a quiet choice.
