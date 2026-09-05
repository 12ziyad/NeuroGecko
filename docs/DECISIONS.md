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
