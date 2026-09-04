# Build decisions

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

## Frozen compatibility constraints

Keep the legacy frequency at 1.1888 Hz. Do not add a shoulder joint. Do not mutate
the recovered weight files or normalize them with newly fitted statistics. Any
new walker after morphology changes needs a distinctly named run and a fresh
normalizer, with stop-command and gaze-ownership changes tested explicitly.
