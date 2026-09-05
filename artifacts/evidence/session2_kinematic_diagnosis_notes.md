# Session 2: frozen collision-foot diagnosis and opt-in controls

Baseline diagnosis was saved before controller changes and committed by the
parent task at `992b062`. The two JSON records are
`session2_kinematics_baseline_lab.json` and
`session2_kinematics_baseline_legacy.json`. Earlier evidence is not overwritten.

## Instrument and scope

`utils/controller_kinematics.py` holds the root, spine, other legs, and passive
joints at the saved stand keyframe. It maps actual zero-residual commands through
the position-servo transmission into the selected limb's joint coordinates, then
runs forward kinematics only. It measures the exact plane clearance of the
collision pad and all five collision claws. The lowest of those six surfaces is
reported. Visual geoms and footzone sites are excluded. This is not a dynamic
rollout, and geometric overlap is not measured ground reaction force.

The environment supplies measured front-contact feedback. Consequently the
diagnostic reports the original open-loop, loaded-stance, and airborne-stance
branches separately, rather than pretending the open-loop branch is always
executed.

## V2 + lab baseline measurements

All distances below are millimetres. Stance and swing clearances are at their
commanded midpoints; stance travel is the collision-pad centre's displacement
along the frozen trunk's forward axis from touchdown to liftoff.

| Foot | Feedback | Stance clearance | Swing clearance | Swing minus stance | Stance forward travel |
| --- | --- | ---: | ---: | ---: | ---: |
| HL | any | -0.212 | -5.011 | -4.798 | +27.835 |
| HR | any | -0.212 | -5.011 | -4.798 | -24.903 |
| FL | none | -12.905 | -12.905 | 0.000 | +21.173 |
| FR | none | -14.093 | -12.905 | +1.188 | -21.173 |
| FL | loaded | -9.347 | -12.905 | -3.558 | +21.175 |
| FR | loaded | -9.853 | -12.905 | -3.052 | -21.175 |

The left legs sweep forward in stance while the right legs sweep backward.
Both sides use +Z protraction hinge axes, so identical angle signs do not produce
mirror-symmetric motion. This is a concrete heading/thrust defect, separate from
the inverted hind toe lift and excessive front tuck/press.

The brief's claim that walking requires unequal stance/swing net excursion is
incorrect for a continuous periodic trajectory. If touchdown is A and liftoff B,
stance displacement is B-A and the return is A-B. Different durations, speeds,
vertical paths, and ground-contact states do not require different endpoints.
Assigning inconsistent endpoints creates a jump instead of solving propulsion.

The brief's equal 54.4-degree front/hind actuator excursion describes the legacy
body, not V2: V2's hip half-range is pi/4, while the inherited shoulder half-range
is the rounded 0.698132 rad. Identical normalized 0.68 therefore produces 61.2
degrees hind and approximately 54.4 degrees fore. These are actuator angle
excursions, not the papers' reconstructed anatomical segment angles.

## Mechanically consistent hypotheses, not measured biological parameters

Keep local phase as `(time * frequency - touchdown_delay) % 1` for lab. This was
already correct; the legacy additive convention is intentionally preserved.

Keep the closed fore-aft triangle: +1 to -1 during stance, -1 to +1 during swing.
For forward travel, mirror the left +Z hip/shoulder command signs. For the body's
+Y knee/elbow hinges, negative swing excursions can lift the collision foot.
The continuous front option uses `press + swing_delta * sin(pi * swing_fraction)`
during swing and `press` during stance. It intentionally disables the original
contact-dependent seek/relax reflex in that mode so the target joins at both
boundaries. Seek/relax remain unchanged in compatibility mode.

Frozen sensitivity gives the following starting hypotheses:

- Hind multiplier -1 gives only +0.767 mm absolute mid-swing clearance; -4/3
  gives +2.065 mm. Multiplier -2 requests normalized -1.2 and saturates at -1,
  producing +3.772 mm. Saturation is reported rather than hidden.
- The 0.30-rad shoulder tuck alone leaves -7.197 mm overlap at zero elbow press.
  With zero tuck and zero press, stance is -0.234 mm and swing delta -0.8 gives
  +7.645 mm clearance. Negative press with zero tuck can leave stance airborne.
- A coordinated positive knee / negative ankle alternative can also work:
  normalized +0.6 knee and -0.8 ankle cancel foot pitch and give +2.637 mm hind
  clearance. This alternative was diagnosed, not added as a controller channel.

All these magnitudes and sweep grids are engineering probes, not animal data.
Mid-swing clearance alone does not guarantee sufficiently long flight, correct
force timing, adequate propulsion, stability, or heading regulation.

## Compatibility and verification

`lab_parameters` is accepted only with the opt-in lab profile. The dictionary is
copied and read-only inside the controller. `heading_error` is a finite signed
angle in radians: a positive error increases right-side fore-aft amplitude and
decreases left-side amplitude. The gain product is clipped to [-1, 1]. Zero gain
does not change commands. Final actuator commands retain their existing total
ctrlrange clip; residual authority and observation/action sizes are unchanged.

The declared ratio 8/9 is a versioned equal-normalized compatibility setting,
not a claim that the inherited rounded XML ranges have an exact 8/9 ratio. The
measured V2 ratio is 0.8888892698450486. Other requested ratios are converted using
the actual model half-ranges. An initial geometric conversion introduced an
approximately 2e-7-radian default discrepancy; the exact baseline test caught it
and the compatibility branch was corrected before final handoff. The parent
preserves initial trial evidence and owns the reference rerun.

After correction, 1,000 full compute outputs with varying residuals and mixed
front-contact branches were compared against source at `992b062` for each of
V2+lab, V2+legacy, legacy-body+lab, and legacy-body+legacy: all 4,000 arrays were
bit-identical on the local runtime, maximum difference zero. Nine dedicated lab
parameter tests, twelve kinematic instrument tests, and fifteen existing gait
and legacy tests passed. These are software/kinematic checks, not Gate 2 gait
validation. No training, GPU use, AWS activity, morphology edits, or frequency
changes were performed by this subtask.
