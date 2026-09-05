# THE SCORECARD

### Calibrating NeuroGecko against published *Eublepharis macularius* measurements

**Version 1.0 — replaces the gecko video you cannot film**

---

## Notation used throughout

Every number carries two tags. Never drop them.

**Species tag**

| Tag | Means |
|---|---|
| **[EM]** | *Eublepharis macularius* — the leopard gecko itself. The target. |
| **[CV]** | *Coleonyx variegatus*, western banded gecko. **Proxy.** Same family (Eublepharidae), sister genus. Closest thing to a leopard gecko that isn't one. Caution: every published prey-capture animal was ~53 mm SVL and ~3.5 g — the size of a leopard gecko *hatchling*. |
| **[TP]** | *Teratoscincus przewalskii* / *scincus*. **Proxy.** Nocturnal, desert, but a different family (Gekkonidae). Used only for energetics. |
| **[TC]** | *Tarentola chazaliae*. **Proxy.** Nocturnal, but spectacled (no eyelids) and a different family. Used only for eye optics. |
| **[GG]** | *Gekko gecko* / *G. subpalmatus*. **Proxy.** Used only for hearing. |
| **[II]** | *Iguana iguana*. **Proxy, not even a gecko.** Used only for hip range-of-motion. |
| **[LIZ]** | Generic sprawling lizard. **Weakest proxy class.** |

**Confidence tag**

- **V (verified)** — I read the number in the primary source's own text or table.
- **L (likely)** — read off a figure, derived by arithmetic from two printed numbers, or read in a secondary source that cites the primary.
- **U (uncertain)** — abstract-only, paywalled, ambiguous notation, or two sources disagree.

**A proxy is never presented as [EM].** If a row says [CV] or [TP], the leopard gecko has never been measured for that quantity and you are choosing to borrow.

**The sim's own constants** (verified by reading the repo this session):
SVL ≈ 0.106 m · total model mass 0.0612 kg · timestep 0.0008 s · `frame_skip` 25 → control 50 Hz · `max_steps` 1000 → **20 s episode** · `FREQ_HZ` 1.1888 · `STANCE` 0.62 (hind) · `FRONT_STANCE` FL 0.68 / FR 0.70 · `PHASE` {HL 0.00, FL 0.25, HR 0.50, FR 0.75} · root spawn z 0.028 m · `eat_radius` 0.10 m · `food_radius` 0.035 m · `head_cam` fovy 120, 64×64 · sensors: 4 foot touch, 2 belly touch, trunk IMU.

---

## 1. What science already measured for you

Between 2008 and 2026, five research groups filmed leopard geckos on instrumented trackways at 250–500 frames per second, stuck EMG electrodes in five of their muscles, weighed their tails, balanced their frozen bodies on threads to find the centre of mass, spun them in optokinetic drums, waved snakes at 585 of them, put them in respirometers, and scored 378 hours of their evening behaviour by hand. **Those measurements are your video.** They cover: how fast the animal walks and how it changes speed; the exact fraction of each stride each foot spends on the ground and the phase order of the four footfalls; the maximum and swept range of every major limb joint; when each muscle fires relative to foot contact; how high the hips and shoulders ride and how flat the back stays; how much the tail weighs, where the whole body balances, and what breaks when the tail is immobilised; what the eye can and cannot resolve, and how the head stabilises gaze; which sensory channel actually triggers fear (it is not vision); how often the animal walks versus stops; how a strike, a shake and a meal are timed; and how much energy all of it costs. What they do *not* cover is listed honestly in §6 — and the gaps are smaller than you would fear, but they are in awkward places.

The single most valuable property of this collection: **many of these numbers are ratios and phases, not absolute magnitudes.** A ratio survives the fact that your MJCF gecko is not exactly a real gecko. That is what makes them a scorecard rather than trivia.

---

## 2. The gait scorecard

Ordered by consequence. **Rows 1–10 contradict what the build currently does.** Rows 11–15 vindicate it. Rows 16–24 are targets the sim simply has not measured yet.

### 2A. The contradictions — fix these first

| # | Quantity | Published value | Species / conf. | What the sim does now | Parameter / file it drives | Source |
|---|---|---|---|---|---|---|
| **1** | **Hindlimb duty factor** (fraction of stride the hind foot is on the ground) | **0.78 ± 0.01** (n=10); 0.75 ± 0.01 (n=7); median 0.79 (N=2); 0.70–0.72 (force-plate). **Range 0.70–0.79, best estimate 0.765** | [EM] **V** — four independent datasets | **0.62** (`STANCE`) | `envs/cpg_residual_controller.py` → `STANCE` | Jagnandan & Higham 2017 T1; Jagnandan+ 2014 T1; Usherwood & Self Davies 2017 T1; McElroy+ 2008 T1 |
| | **What is wrong:** the hind foot leaves the ground far too early. The animal is a high-duty-factor walker that keeps three or four feet down almost all the time. A 0.62 hind duty factor is a *trotting* animal's number. This is the single highest-leverage change in the whole document and it is a one-line edit. It should also directly raise the weak foot-contact signal (`contact_thresh` 0.0564). | | | | | |
| **2** | **Fore/hind duty-factor asymmetry — the sim has it BACKWARDS** | Hind (0.78) is **higher** than fore (0.70). Difference +0.08 in favour of the hindlimb. | [EM] **V** | Fore 0.68–0.70, hind 0.62 → hind is **0.07 LOWER** than fore. Sign inverted. | `STANCE` + `FRONT_STANCE` | Jagnandan & Higham 2017 T1 |
| | **What is wrong:** the forelimb value (0.68–0.70) is *already correct* — leave it alone. The error is entirely in the hindlimb, and because the hind is set below the fore, the model's weight-support pattern is qualitatively inverted relative to the animal. Set `STANCE` (hind) = 0.765, keep `FRONT_STANCE` ≈ 0.70. | | | | | |
| **3** | **Limb phase** — how far into the stride the ipsilateral **fore** foot lands after the reference **hind** foot | **43–44 %** (44 ± 1.1 walking, 43 ± 1.8 running, McElroy; median 43, SD 3.5, Usherwood). Footfall order **RH → RF → LH → LF**. | [EM] **V** — two labs, two methods, agree to 1 % | **25 %** (`PHASE` HL 0.00 → FL 0.25) — the textbook lateral-sequence walk offset | `PHASE` dict | McElroy+ 2008 T1; Usherwood & Self Davies 2017 T1 |
| | **What is wrong:** 0.25 is the *generic* lateral-sequence value from a robotics/comparative-anatomy prior, not this animal. The gecko walks with near-diagonal couplets — close to a trot but not a trot. **Set `PHASE = {"HL": 0.00, "FL": 0.44, "HR": 0.50, "FR": 0.94}`.** Sanity check: Usherwood's cross-tetrapod law `phase% = 130·DF − 66` predicts 35 % at DF 0.78; the measured value is 43–44 %. **Use the measured 0.44**; treat the law only as a guide when extrapolating to duty factors nobody measured. | | | | | |
| **4** | **The "0.2–1.1 m/s" speed figure is not a walking speed** | Zaaf+ 2001 is the only source printing 1.1 m/s. It is the *range sampled on a motorised treadmill*, chosen deliberately to push the animal past voluntary behaviour. Fuller+ 2011 say so explicitly. **Abstract only — closed access, could not read the body.** | [EM] **U** (abstract only) | If any reward target, curriculum ceiling or "max speed" constant in the repo derives from 0.2–1.1 m/s, it is chasing a treadmill artefact | `rewards/walk_reward.py`, curriculum config | Zaaf, Van Damme, Herrel & Aerts 2001, JEB 204:1233 |
| | **What is wrong:** 1.1 m/s is ≈ 10 SVL/s. No leopard gecko has ever been observed doing that voluntarily. **Delete it as a target.** Keep it only as a never-exceed sanity bound in an episode-validity check. Also: **ignore pet-care websites claiming 15–20 mph** — that is 30–100× the highest peer-reviewed figure and is fabricated. | | | | | |
| **5** | **Hip height** (pelvis clearance above ground during steady walking) | **0.15 ± 0.00 SVL** = 15.9 mm at SVL 106 mm. Independently: hip height = 36.0 ± 1.8 % of total hindlimb length. Two labs agree to 1 %. | [EM] **V** | Root spawns at z = 0.028 m = **0.264 SVL** — the model stands ~75 % too tall | `morphology/gecko_body_r.xml` root pos + a posture reward term | Jagnandan+ 2014 T1; Fuller+ 2011 §3.3 |
| **6** | **Shoulder height — and the fact that it is LOWER than the hip** | **0.11 ± 0.00 SVL** shoulder vs **0.15 SVL** hip. Ratio shoulder/hip = **0.73**. | [EM] **V** | In the MJCF, `humerus_L/R` and `femur_L/R` attach at the *same* z offset (−0.002). No fore/hind height asymmetry exists in the body at all. | `gecko_body_r.xml` girdle attachment z; posture reward | Jagnandan+ 2014 T1 |
| | **What is wrong:** the real animal carries its chest measurably closer to the ground than its pelvis. Encode **both** heights, not just one, or the trunk will settle at the wrong pitch and every downstream joint angle will be biased. | | | | | |
| **7** | **Stride length — the sim over-strides and under-steps** | 0.82 SVL (8.74 ± 0.27 cm at 0.18 m/s, Fuller); 0.70–0.73 SVL (2014 cohort); 0.62–0.63 SVL (2017 cohort). **Range 0.62–0.82 SVL.** | [EM] **V** | 0.11 m/s ÷ 1.1888 Hz = **0.0925 m = 0.873 SVL** — above the top of the published range | CPG frequency ↔ commanded speed coupling | Fuller+ 2011 §3.2; Jagnandan+ 2014 T1; Jagnandan & Higham 2017 T1 |
| | **What is wrong:** the sim reaches its speed by taking *long slow steps*. The animal reaches speed by taking *more steps*. To land on 0.73 SVL at the sim's current 0.11 m/s, the CPG must run at **≈1.42 Hz**, not 1.1888. **Note a published inconsistency:** the 2017 table's stride length (0.62 SVL), stance time (0.64 s) and speed (1.23 SVL/s) are not arithmetically self-consistent, because that paper reports speed-corrected "means + residuals". The only internally consistent published triple is Fuller's: **2.03 Hz × 0.0874 m = 0.177 m/s ≈ 0.18 m/s reported.** Use that triple as your self-consistency test on any tuned gait. | | | | | |
| **8** | **How speed is modulated: frequency, not length** | Stride frequency vs speed **r² = 0.98**; stride length vs speed **r² = 0.53**. Both rise, but frequency dominates. Corroborated by Zaaf+ 2001 abstract. | [EM] **V** | `FREQ_HZ` is a **constant**. Speed can only be changed by the PPO residual lengthening steps — exactly the wrong mechanism. | CPG frequency must become a function of commanded speed | Fuller+ 2011 Fig. 6 |
| | **The fix:** make CPG frequency near-linear in commanded speed and let stride length rise only weakly. Anchor: **1.4 Hz @ 0.11 m/s → 2.03 Hz @ 0.18 m/s.** Then *penalise* the residual for achieving speed changes by step lengthening. | | | | | |
| **9** | **Total body mass** | 36.3 ± 1.9 g at SVL 104.6 mm (n=10); 39.4 ± 1.2 g at SVL 121.9 mm (n=7); 30.7 ± 1.7 g at SVL 110 mm (n=10). **Wild adults: 26–27 g** (n=2). | [EM] **V** | **61.2 g** (sum of 44 hand-assigned `mass=` attributes) at SVL 106 mm | `gecko_body_r.xml` all `mass=` attributes | Jagnandan & Higham 2017; Jagnandan+ 2014; Jagnandan & Higham 2018; Rawat+ 2019 |
| | **What is wrong:** the sim gecko is ~1.6× too heavy *for its length*. Every ground reaction force, joint torque and `forcerange` is therefore inflated by the same factor. The 60–65 g figure is a real published number — but it is **captive pet-trade adult males** (median 79.6 g, range 46–91 g), not the lab animals whose duty factors and hip heights you are trying to hit. **Decide explicitly which population you are modelling.** Recommended: the Jagnandan cohorts, so target **0.036–0.040 kg**. | | | | | |
| **10** | **Tail length relative to SVL** | Tail = **0.73 SVL** (87 mm / 119 mm, wild adult); 0.77 SVL (juveniles); tail = **41 % of total body length** (n=89). | [EM] **V** (though the adult ratio rests on n=2 wild specimens) | Five tail capsules sum to **0.110 m** against SVL 0.106 m → **1.04 SVL**, and 50.9 % of total length | `gecko_body_r.xml` tail1–5 `fromto` | Rawat+ 2019 T1; Higham & Russell 2012; McLean & Vickaryous 2011 |
| | **What is wrong:** the tail is ~40 % too long. Since the tail is a quarter of the mass and swings laterally every stride, this corrupts tail inertia, the tail-bend moment arms and the whole-body centre of mass. Shorten the chain to ≈0.078 m total. | | | | | |

### 2B. What the build already gets right — do not "fix" these

| # | Quantity | Published value | Species / conf. | What the sim does now | Verdict | Source |
|---|---|---|---|---|---|---|
| **11** | Base CPG frequency for a *voluntary walk* | Derived stride period 0.821 s = **1.22 Hz** (hind, 2017 cohort: 0.64 s stance ÷ 0.78 DF); 1.30 Hz fore; 1.74 / 2.12 Hz in the 2014 cohort | [EM] **L** (derived — neither paper printed stride frequency) | `FREQ_HZ` = 1.1888 → period 0.841 s | **Legitimate.** 1.1888 Hz lands almost exactly on the derived voluntary-walk period. It is not a magic number; it is a real slow-walk value. Keep it as the *low end* of a speed-dependent schedule. | Derived from Jagnandan & Higham 2017 T1 and Jagnandan+ 2014 T1 |
| **12** | Forelimb duty factor | **0.70 ± 0.01** (n=10, unchanged across three tail treatments); 0.70 ± 0.03 (n=7) | [EM] **V** | `FRONT_STANCE` = 0.68 / 0.70 | **Correct.** Leave it. | Jagnandan & Higham 2017 T1 |
| **13** | Walking speed | Voluntary: **1.23 ± 0.16 SVL/s = 0.129 m/s**; full voluntary range 0.59–3.36 SVL/s (0.062–0.351 m/s); second cohort 0.23–2.4 SVL/s (0.028–0.293 m/s) | [EM] **V** | Measured **0.11 m/s = 1.04 SVL/s** | **Inside the published band**, ~15 % below the voluntary mean. Set the reward *peak* at 0.13 m/s and stop there. Do not chase speed — fix duty factor and frequency instead. | Jagnandan & Higham 2017 T1; Jagnandan+ 2014 |
| **14** | Tail mass as a fraction of body mass | **22 %** (n=7, adults); ~25 % (n=10); 19.27 % actually shed at autotomy (n=10). **Adult target 0.22, pass band 0.19–0.25.** Juveniles are only 11.5 % — do not use that. | [EM] **V** | tail1–5 = 0.005+0.004+0.003+0.002+0.001 = 0.015 kg = **24.5 %** | **Fraction correct.** The absolute is not (0.015 kg vs 22 % of 39 g = 8.7 g), but that falls out of fixing row 9. | Jagnandan+ 2014; Jagnandan & Higham 2017 |
| **15** | Footfall sequence class | Limb phase < 50 % = **lateral sequence** | [EM] **V** | "lateral-sequence phase order" | **Qualitatively right.** Only the numeric offset is wrong (row 3). | McElroy+ 2008; Usherwood & Self Davies 2017 |

### 2C. Targets the sim has never measured — instrument these

| # | Quantity | Published value | Species / conf. | Sim now | Parameter / file | Source |
|---|---|---|---|---|---|---|
| **16** | **Trunk pitch during steady walking** | **3.72 ± 0.61°** nose-up (2.44 ± 0.77 autotomised, 2.78 ± 0.80 regenerated; n.s.). Corroborated qualitatively: "ran with a horizontal body angle". | [EM] **V** | unmeasured | Trunk IMU (`imu` site) reward: hold +3.7° ± 1°; penalise beyond 8°. Also constrains `spine_pitch` neutral offset. | Jagnandan+ 2014 T1 |
| **17** | **Femur retraction excursion** — the main propulsive DOF | Max **55.00 ± 2.18°**; **excursion 82.57 ± 2.29°** peak-to-peak. Largest sweep of any hindlimb joint. | [EM] **V** | unmeasured | `hip_proret` ctrlrange must allow **≥90°**, and this is the DOF the CPG's main oscillator should drive. Repo currently ±40° = 80° total — marginal. | Jagnandan & Higham 2017 T1 |
| **18** | **Femur depression (sprawl) — CONVENTION CONFLICT** | **Jagnandan convention:** max 49.44 ± 2.49°, excursion **52.42 ± 3.25°**. **Fuller convention:** max **8.8 ± 1.2°** (0° = femur perpendicular to body axis; the knee sits *above* the hip for much of the stride, "hyper-sprawled", vs 32.8° in a sympatric gecko). | [EM] **V** for both; **the two cannot be the same angle** | unmeasured | Pick **one** convention, write it in an XML comment, and **never average them.** Recommendation: use **Jagnandan's excursion (52.4°)** to size `hip_sprawl` range, because all the tail-ablation ratios come from that lab and are self-consistent. Then use Fuller's finding *qualitatively*: the knee must be allowed **above** the hip — most quadruped priors forbid this. Speed-invariant (P = 0.751), so **do not modulate `hip_sprawl` with speed.** | Jagnandan & Higham 2017 T1; Fuller+ 2011 §3.3 |
| **19** | **Knee trajectory** | Max (extended) **164.49 ± 0.93°**; min (flexed) **69.0 ± 1.5°**; **at footfall 101.6 ± 1.8°**; excursion **98.83 ± 1.39°** | [EM] **V** | unmeasured | Complete knee spec: land at 102°, flex to 69° through early stance, extend to 164°. Encode as the knee CPG waveform, not a sinusoid. | Jagnandan & Higham 2017 T1; Fuller+ 2011 |
| **20** | **Ankle excursion** | Max **139.65 ± 2.03°**; excursion **85.19 ± 2.38°**. Speed-invariant (P = 0.313). | [EM] **V** | unmeasured | Combined range of the two ankle actuators. Hold amplitude constant across the command range. | Jagnandan & Higham 2017 T1 |
| **21** | **Forelimb sweeps are much smaller than hindlimb** | Humerus retraction excursion **44.41 ± 1.80°** — *half* the hindlimb's 82.6°. Humerus depression excursion **101.38 ± 15.88°** (very noisy). Elbow excursion **92.49 ± 2.27°**. Wrist excursion **71.93 ± 3.09°**. | [EM] **V** | Symmetric fore/hind CPG amplitudes | `shoulder_proret` ≈ 44° p-p, `shoulder_sprawl` ≈ 100° p-p, `elbow` ≈ 92° p-p. **Do not use a symmetric fore/hind amplitude.** The MJCF has no wrist actuator — either add one (72°) or fold it into distal compliance. | Jagnandan & Higham 2017 T1 |
| **22** | **CAUDOFEMORALIS EMG timing** — the only published constraint on CPG *phase* rather than amplitude | Onset **−1.68 ± 2.26 %** of stride (fires just *before* footfall); burst duration **81.45 ± 2.00 %**; peak at **14.01 ± 3.08 %**; peak amplitude 53.7 % of muscle max. Internally consistent: −1.7 + 81.5 = 79.8 %, matching hind DF 0.78. | [EM] **V** | unmeasured | **This is the ground truth for `hip_proret` phase.** Begin retraction 1.7 % of a cycle *before* contact, ramp to peak at 14 %, sustain through 80 %. This muscle originates on the proximal caudal vertebrae — its 2.5× collapse after tail loss is *why* `tail_bend_L/R` must be coupled to `hip_proret`. | Jagnandan & Higham 2018 T1, T3 |
| **23** | **GASTROCNEMIUS: two bursts per stride** | **Burst 1** (stance): onset 6.40 ± 2.30 %, duration 64.41 ± 4.26 %, peak at 33.79 ± 5.49 %. **Burst 2** (late-swing pre-landing stiffening): onset **85.06 ± 2.31 %**, duration 13.23 ± 1.56 %, peak at **94.74 ± 1.28 %**, amplitude 54.5 % — *higher* than burst 1. | [EM] **V** | Single sinusoidal ankle drive | **A single sinusoid cannot produce this.** Add the second burst (or raise ankle gain/stiffness in late swing). Authors interpret burst 2 as stiffening the ankle for landing. **This is a strong candidate explanation for the sim's weak foot contacts** (`contact_thresh` 0.0564). | Jagnandan & Higham 2018 T3 |
| **24** | **Centre-of-mass energy mechanics** | **Walking:** phase shift **151 ± 6.6°**, recovery **32 ± 4.3 %**, PE/KE 0.98, at 0.24 m/s. **Running:** 8 ± 3.2°, 16 ± 3.9 %, at 0.29 m/s. (180° = perfect inverted pendulum, 0° = perfect bounce.) | [EM] **V** | unmeasured | **A scorecard metric no joint-angle reward can fake.** Compute trunk KE and gravitational PE, take the phase difference. A correct slow gecko walk gives ~151° and ~32 %. Diagnostic, not a reward term. | McElroy+ 2008 T1 |
| **25** | **Stance/swing durations in absolute seconds** | Hind stance **0.64 ± 0.03 s**, fore stance **0.54 ± 0.03 s** (2017); hind 0.43 ± 0.02 s, fore 0.33 ± 0.04 s (2014). Derived swing: hind 0.18 s, fore 0.23 s. | [EM] **V** (stance printed; swing derived) | unmeasured | At 50 Hz, a hind stance should span **21–32 control steps** and swing **7–9 steps**. Assert per-foot contact durations from the 4 touch sensors fall in this band. | Jagnandan & Higham 2017 T1; Jagnandan+ 2014 T1 |
| **26** | **Leopard geckos do NOT drop duty factor to run** | One of only three "wide foragers" that shift to **high duty factor (67–72 %) and low speed (0.16–0.24 m/s) running**. Duty factor is essentially flat across the whole voluntary speed range. | [EM] **V** | n/a | **Do not add a duty-factor-decreases-with-speed schedule.** Keep `stance_ratio` near-constant across 0.06–0.32 m/s. This is counter to every quadruped robotics prior. | McElroy+ 2008, Discussion |
| **27** | **Tail behaviour during walking** | Tail is **lifted clear of the ground** and swung laterally. Base flexes **toward the protracted (swing-phase) hindlimb**; motion propagates caudally as a travelling wave. Tail-tip lateral displacement **decreases** with speed (F₁,₈ = 5.870, P = 0.042, R² = 0.423). Tail **height is independent of speed** (P = 0.759). Frequency **~2 Hz** — one full cycle per stride, phase-locked to pelvic yaw. No amplitude printed (figure only, y-axis ~0–0.12 SVL). | [EM] **V** for the relationships, **L** for the ~2 Hz (derived from stride frequency, not spectrally measured), **U** for amplitude | `tail_amp` 0.15 constant, `tail_phase_lag` 0.15 | Three rules: (1) `tail_lift` = **constant, speed-independent** setpoint; (2) `tail_bend` phase locked so the base swings toward the *protracting* hindlimb, i.e. antiphase with that side's `hip_proret` retraction; (3) **`tail_bend` amplitude DECREASES with speed** — the opposite of the usual robotics intuition. Tail CPG frequency must be tied **1:1** to the limb CPG, never free-running. | Jagnandan & Higham 2017; Higham & Russell 2012 |
| **28** | **The tail-restriction ablation — your free experiment** | A <1 g graphite rod glued along the tail blocked lateral undulation while still allowing lifting. **It reproduced almost every autotomy effect with no mass removed.** Hindlimb: femur depression excursion 52.4→30.6°, femur retraction 82.6→65.3°, knee 98.8→88.1°, ankle 85.2→70.4°. Hind step length −17 %. **Duty factor unchanged** (0.78 → 0.78, P = 0.583). **Forelimb entirely unchanged** (all P ≥ 0.052). | [EM] **V** | not run | **The single most valuable falsifiable test available to you, and it costs nothing.** Clamp `tail_bend_L/R` to zero, keep all tail mass and `tail_lift` free. If the hindlimb collapses in these proportions while the forelimb and duty factor stay put, your model has captured tail *movement*, not merely tail *mass*. If "restricted" looks like "intact", the tail is a passive pendulum and the coupling is missing. Full gate list in §7. | Jagnandan & Higham 2017 |
| **29** | Body-wave / spine coordination | Fore- and hindlimb movement **in phase** with shoulder and hip bending respectively (stronger for hind/hip). Maximum curvature convex toward a leg at that leg's touchdown. Fully-limbed lizards sit at the **standing-wave** end of the continuum. | **[LIZ] L** — *Brachymeles*, *Uma*, *Sceloporus*. **No gecko was measured.** | `spine_amp` 0.30, `spine_phase` 0.0 | `spine_bend` = **standing** wave locked in phase with `hip_proret` at the pelvic end and `shoulder_proret` at the pectoral end; `tail_bend` = **travelling** wave. That split is exactly what Jagnandan & Higham describe for [EM]. **Mark as proxy-derived in the code comment.** | Chong+ 2022 PNAS 119:e2118456119 |
| **30** | Aerobic speed ceiling | Max aerobic speed **0.075 m/s at 15 °C**; VO₂max reached at 0.100–0.158 m/s at 25 °C. Sustained >60 min at 0.050 m/s. Above VO₂max: fatigue in under 4 min. | **[TP] / [CV] V** — *Coleonyx variegatus* (9 g) and *Teratoscincus*. **Not [EM].** | flat `energy` drain | The sim's 0.11 m/s is at or below the warm-animal aerobic ceiling — **locomotion at the current speed is fully sustainable.** See §4 for what this does to `drives.py`. | Autumn, Weinstein & Full 1994 |
| **31** | Experimental temperature — the validity envelope of this entire document | **~30 °C** (Jagnandan 2014/2017/2018); 34–35 °C (Fuller); 26–30 °C (McElroy). **No [EM] gait kinematics exist below 26 °C.** All on level, high-friction substrate (sandpaper or cork), trackways 1.0–5.2 m. | [EM] **V** | floor friction 0.9 | Record this in the calibration config. Every target above applies to a **warm gecko on flat high-friction ground**. Nobody may later compare a cold-gecko or rough-terrain run against these numbers. Justifies the high MuJoCo friction for the calibration scenario. | Methods sections of all five papers |

---

## 3. The senses table

| Quantity | Published value | Species / conf. | Sim now | Parameter / file | Source |
|---|---|---|---|---|---|
| **Retinal specialisation** | **No fovea.** Strictly nocturnal geckos completely lack foveae. | [EM] **V** | uniform 64×64, no foveation | **Already correct.** Do *not* add a foveal crop, centre-weighting, or log-polar transform. A flat uniform grid is the biologically right model. | Masseck, Röll & Hoffmann 2008 |
| **Photoreceptor pooling** | ~**1–2 photoreceptors per ganglion cell**, uniform across the retina, no area centralis | [EM]/nocturnal geckos **L** | uniform | Confirms uniform sampling; resolution is set by receptor spacing, not pooling. | Higham & Schmitz 2019; Röll 2000/2001 |
| **Camera resolution — theoretical ceiling** | Focal length **3.5 ± 0.1 mm**, cone outer segments **30–40 µm long × ~10 µm wide** → Nyquist ≈ **2–3 cyc/deg**. Must sit *below* the 1.5–2 cyc/deg measured behaviourally in a small diurnal lacertid. | **[TC] V** for optics; **[LIZ] L** for the acuity ceiling. **No gecko acuity has ever been published.** | **64×64 over 120° = 0.27 cyc/deg** — roughly 5× under-sampled | Defensible upgrade: **128×128** (0.53 cyc/deg). Anything above ~256 is biologically unjustifiable. Write the derivation as an XML comment next to `head_cam` so the number is traceable. | Roth+ 2009 J. Vision 9(3):27; Fleishman+ 2024 |
| **Camera field of view** | **~30° binocular field** — and that is *Lygodactylus*, reported as an unpublished observation inside another paper's introduction. Nocturnal geckos converge highly movable eyes forward to boost sensitivity. | **[LIZ]/gecko U** — the weakest number in this table | **fovy 120** | `head_cam` fovy 120 has **no biological basis**. Document it as a deliberate engineering choice. Optionally define a central ~30° "binocular" sub-window as the region that gates the `engage` channel. | Masseck+ 2008, Introduction |
| **Camera update rate** | No CFF has ever been measured in any gecko from a readable primary source. *Anolis*: 55–70 Hz at high light, higher for long wavelengths. CFF falls in dim light. | **[LIZ] U**. The widely-repeated "~20 Hz for geckos" could not be traced to any primary measurement and is **excluded here.** | implicitly 50 Hz | Justified design: render `head_cam` every 2nd–5th control step (**10–25 Hz**) and hold the last frame. Cuts laptop CPU cost too. **Flag as a guess.** | Fleishman+ 2024, citing Fleishman+ 1995 |
| **Light level — the two levels [EM] was actually tested at** | **Bright 21,662 lx**, **dark 0.25 lx**, measured mid-trackway. Antipredator arenas used a single 25 W blue bulb. | [EM] **V** | not modelled | Add a `light_lux` field with these two exact settings. **Use 0.25 lx as the nominal training condition** — ecologically correct for a crepuscular animal. | Higham & Schmitz 2019 |
| **Light level — the negative control** | The three nocturnal species (incl. [EM]) **maintained sprint performance at both light levels.** Only a secondarily-diurnal gecko slowed in the dark. | [EM] **V** | n/a | **A gate, not a target.** Dropping illumination from 21,662 to 0.25 lx must **not** degrade the gait (speed, stride frequency, duty factor invariant). If the trained policy slows in the dark, it has over-fitted to vision for locomotion — which this animal does not do. Locomotion should ride on IMU + touch + CPG; vision should only feed the 4-D command. | Higham & Schmitz 2019 |
| **Colour** | Nocturnal gecko cones: **UV 363–366 nm, blue 452–470 nm, green 520–533 nm** (the blue-sensitive S class and rod opsin RH1 were lost). Colour discrimination demonstrated down to **0.002 cd/m²** (dim moonlight) — 350× more light-sensitive at colour threshold than a human eye. | **[TC]/nocturnal geckos L.** **No λmax has ever been measured for [EM].** | human RGB | Keep RGB (this clade genuinely uses chromatic information at night — do not switch to greyscale). If you reinterpret channels: R→520–533, G→452–470, B→363–366 UV. **Make the food object discriminable on the 520-vs-460 nm axis**, not on human-green. | Roth+ 2009; Kelber & Roth 2006; Fleishman+ 2024 |
| **Pupil / exposure** | Pupil area dynamic range **100–150×** (*Gekko gecko* 300×; human 16×). Constricted→round transition takes **~1 hour**. | **[TC] V** — a four-pinhole spectacled pupil, anatomically unlike a leopard gecko's eyelidded slit | fixed exposure | Scale rendered image gain by an effective aperture spanning ~100×, saturating at both ends. **Adaptation time constant ~3600 s ≫ any episode → treat sensitivity as CONSTANT within an episode**, set once from the episode's light level. Do not model fast adaptation. | Roth+ 2009 |
| **Gaze stabilisation (optokinetic reflex)** | **Binocular hOKR gain: 0.9 at 20°/s, 0.8 at 30°/s, 0.7–0.8 at 40°/s.** Monocular temporo-nasal: 0.7 / 0.6 / 0.4. **Monocular naso-temporal: ZERO** — no nystagmus could be elicited at any velocity. ~**80 % of gaze stabilisation is head movement**, not eye movement. | [EM] **V** — *this is the best-matched published sensorimotor number for the target species* | absent | Build an OKR loop driving `neck_yaw` + `head_yaw` from optic flow on `head_cam`; tune until head angular velocity ÷ world angular velocity = 0.9 at 20°/s. **Critical: half-wave rectify the flow per hemifield so backward (naso-temporal) flow gives ZERO head-turn command** — otherwise the CPG's own forward walking drives the reflex into runaway. Because 80 % of real stabilisation is head rotation, a camera rigidly fixed to a servo'd head is a faithful model. Only validate at 20/30/40°/s; refuse to extrapolate. | Masseck, Röll & Hoffmann 2008 |
| **Chemoreception — tongue-flick rate** | **Stationary 0.020/s (1.2/min); moving 0.044/s (2.6/min); moving after eating 0.076/s (4.5/min).** Movement effect F = 12.62, P < 0.001. Independent lab, 30-min trials: baseline 0.035–0.053/s, threatened 0.088–0.093/s. **Ceiling under maximal multimodal threat: 0.146/s.** Two labs, two decades, converge on **1.2–8.8 flicks/min.** | [EM] **V** | absent | Add a discrete `tongue_flick` action (or autonomous oscillator) at base 0.020/s stationary, 0.044/s moving, 0.076/s post-feed, clamped ≤0.15/s. **Gate on locomotion state, not hunger.** This replaces the invented curiosity constant with a measured, species-correct sampling rate — and gives the reward a principled small cost per flick. | Cooper, DePerno & Steele 1996; Landová+ 2016; Frýdlová+ 2026 |
| **Chemoreception — labial licking** | **Stationary + recently fed: 0.165/s (9.9/min)** — by far the highest. Moving + fed 0.019/s. Moving + unfed: **0.000** (no lizard ever did it). Eating × movement interaction F = 36.87, P < 0.001. Uncorrelated with tongue-flicking in every condition. | [EM] **V** | absent | A second oral behaviour with the **opposite gate** to tongue-flicking. The zero correlation means the two must not share a latent variable. This is the strongest published justification for a **post-feeding stationary state**: the real animal stops and licks rather than resuming the gait. | Cooper+ 1996 Fig. 2 |
| **Chemoreception — latency** | Labial-lick rate rises **immediately** (<60 s). **Tongue-flick rate does not rise until the THIRD minute (~180 s)** after prey-chemical exposure. | [EM] **V** | n/a | Two time constants, not one. **The slow channel's 180 s dead time is 9× the current 20 s episode** — it cannot be exercised at all. Either lengthen episodes or model only the fast channel and document the omission. | Cooper+ 1996, Discussion |
| **Chemoreception — range** | Every published bioassay presented the stimulus at **3 mm to 1 cm from the snout**, or required substrate contact. | [EM] **V** (of the protocol) | n/a | Model odour as a **contact-range** scalar at `nose_tip` (a few cm), sampled **only at flick events** — not as a continuously readable long-range gradient. Note: the current `eat_radius` of 0.10 m is **33× the published chemosensory range.** | Frýdlová+ 2026; Cooper+ 1996 |
| **Vibration** | Saccular vibration pathway (nucleus vestibularis ovalis) responds **50–200 Hz, peaking ~100 Hz** — well below cochlear hearing. **No threshold in physical units is readable** (paywalled). | **[GG] L** (the nucleus is present across lepidosaurs, so very likely present in [EM]) | absent | **No new MuJoCo sensor needed:** band-pass the existing trunk IMU accelerometer at 50–200 Hz. The 0.0008 s timestep (1250 Hz) resolves 200 Hz comfortably, but **the 50 Hz control loop does not — the filter must run in the physics substeps.** Feed the result in as a *small additive modulator only* (see next row). | Han & Carr 2024 Curr. Biol. 34:5017 |
| **Vibration — the behavioural verdict** | 4 firm taps at **~80 dB** at the gecko's head, producing substrate vibration: **P(antipredator reaction) = 0, χ² < 0.01, P > 0.9.** | [EM] **V** | n/a | **Vibration alone triggers nothing.** ~80 dB is ~40 dB above threshold and still produced no defence. Set the vibration weight in the danger term to **zero** on behavioural grounds, not sensory ones. | Frýdlová+ 2026 |
| **Hearing** | Spontaneous otoacoustic emissions **1–4.5 kHz** (temperature-dependent, 54–107 Hz/°C). Audiogram: flat **~40 dB SPL from 1–3.5 kHz**, steep roll-off above 4 kHz. Auditory nerve CFs 200–3200 Hz, bimodal at 400 and 2000 Hz, median threshold 36 dB SPL. Ear is internally coupled: up to **30 dB** ipsi/contralateral difference below 3–4 kHz — a very strong azimuth cue. | SOAE band is **[EM] L**; audiogram is *G. subpalmatus* **V**; nerve/directionality is **[GG] V** | absent (no ear sites) | **Deprioritise hearing.** It is audible but not behaviourally decisive for this animal. If ever added: pass-band 0.2–3.5 kHz, floor ~40 dB SPL, and model the ears as an internally-coupled pair (not two independent mics) — that 30 dB directional contrast would give `dir_x`/`dir_y` a cheap non-visual source. Also note the 54–107 Hz/°C temperature dependence argues for not adding hearing before thermal state exists. | Manley, Gallo & Köppl 1996; Chen+ 2016; Christensen-Dalsgaard+ 2021 |
| **Touch — the biggest sensor gap in the model** | **Dorsal tail 59.34–76.12 sensilla/mm²** — the highest density anywhere on the animal. Best non-tail dorsal site 33.51/mm²; upper labials 34.67/mm²; ventral tail 7.28–25.51/mm². Body-vs-tail Mann-Whitney P = 0.0045. Regenerated tail collapses to 4.35–24.65/mm². | [EM] **V** | **4 foot + 2 belly touch sensors. ZERO tail sensors.** | The tail is by ~2× the most densely innervated skin on the animal, and the model has no sensor on it at all. **Add touch sensors to the tail segments, dorsally weighted at ~2–3× the belly sensors' weighting**, and feed them to the policy. | Russell, Lai, Powell & Higham 2014 J. Morphol. 275:961 |
| **Touch — thresholds** | **Tail base 0.11 ± 0.009 g (~1.1 mN)** — most sensitive. Forelimb 0.13 g. Tail tip 0.14 g. **Hindlimb 0.19 g (~1.9 mN)** — least sensitive. Tail sensors are **~1.7× more sensitive** than foot sensors. | [EM] **L** (publisher abstract/first page, not full PDF) | one global `contact_thresh` = 0.0564 (unitless) | Split into at least two thresholds: tail ≈1.1 mN, foot ≈1.9 mN. | JEB 2021, 224:jeb234054 |
| **Visual fixation is a discrete, rare event** | "Binocular fixation" totalled only **~2.2–3.4 s per 1800 s trial**, in ~2.8 discrete events. In a gaze-following study, sustained look-ups (≥10 s each) occurred **5.5 per 60 s** with a salient target vs 0.5–2.0 without. | [EM] **V** | continuous camera stream | Head-camera-directed *staring* should be **rare, not the default**. Model an "attend" state: fixation duration ≥10 s, onset ~0.09/s with a salient target, ~0.008–0.03/s without. Use these as the prior for the `engage` channel and as a check on whether the trained policy's dwell statistics are gecko-like at all. | Landová+ 2016; Simpson+ 2019 Anim. Cogn. 22:145 |

---

## 4. The behaviour and physiology table

### 4A. Time budget and activity

| Quantity | Published value | Species / conf. | Sim now | Implication |
|---|---|---|---|---|
| **Behaviour budget** (proportion of 25,366 scored behavioural elements, with live prey present) | position alteration (mostly isolated head movements) **46.5 %** · walk around **22.7 %** · change of body orientation **8.2 %** · sensory exploration (look/smell/tongue-flick) **7.2 %** · rest **6.9 %** · **foraging 4.1 %** · interests 3.3 % · basic needs 0.7 % · atypical 0.3 % · distress 0.2 % | [EM] **V** (n=18, 378 h; the 4.1 % foraging band was validated against the paper's own stated figure) | Effectively 100 % food-directed locomotion | **THE PRIMARY SCORECARD for the brain.** Nearly half of all behaviour is head/body micro-movement without a step; only 23 % is walking; **foraging is 4 %.** Cap the food/approach reward so it cannot dominate; give exploration terms ~10× its weight. |
| **Baseline budget** (no live prey) | sensory exploration **45.7 %** · walking **19.4 %** · interest **15.5 %** · resting **14.8 %** · other 4.6 % | [EM] **V** (n=18, 12,519 elements) | — | Two different target histograms for two drive states: "food present" vs "food absent". Note live prey did **not** suppress resting in absolute terms (1848 → 1734 counts) — it *added* activity on top. So `engage` should be **additive**, not a mode switch. |
| **Behavioural act rate** | **1.66 acts/min** baseline (one act every 36 s); **3.35 acts/min** with live prey (one every 18 s) | [EM] **V** (derived from 696/1409 elements ÷ 420 min) | — | **The episode-length killer.** At 20 s the sim contains **0.55–1.1 real behavioural acts.** Any reward that scores "behaviour" cannot see a behaviour. |
| **Behavioural diversity** | **~4 distinct behaviour categories per 5-minute interval** (3.64 baseline, 4.26 with prey); 19–21 of 31 categories realised per animal per 14-day set | [EM] **V** | one gait attractor | A directly codeable auxiliary reward: entropy over behaviour labels, target ~4 classes per 300 s. This is what pushes PPO off the single-gait fixed point. |
| **Percent time moving** | **33.2 %** — the only gecko known to exceed the 30 % active-forager threshold | **[CV] L** — *Coleonyx variegatus*, reached only through two secondary accounts citing Kingsbury 1989. **The single most load-bearing proxy in this document.** | CPG runs continuously | Target ~33 % walking, 67 % stationary. Cross-checks against [EM]'s 22.7 % walking *events*. **A CPG cycling continuously is ~3× too much locomotion.** Add a stop/start gate above the oscillator. |
| **Locomotor style** | "characterised by slow movements interrupted by numerous pauses" [EM]; "forages actively, but moves more slowly and deliberately than typical active foragers, and tongue-flicks at lower rates" [EM] | [EM] **V** (authors' characterisation) | continuous CPG | Argues for **intermittent** locomotion architecture: move–pause–move. |
| **Distress / stereotypy** | **0.2 %** of behaviour ("all reduced wellbeing": pane scratching, vertical pane standing, wriggling at pane) plus 0.3 % atypical, in a well-furnished enclosure | [EM] **V** | — | **A negative gate.** Wall-directed repetitive motion should be <0.5 % of sim behaviour. If PPO converges on pressing against an arena boundary, that is the published signature of a *bad enclosure*, not a solution. |
| **Activity rhythm** | Peak **6–7 pm**, "roughly evenly decreasing" to 11 pm. Effect of hour P = 0.001. | [EM] **V** *within the observers' pre-chosen 6–11 pm window only* | none | A circadian gain: monotonic decay over a 5-h window triggered by lights-off. **Do not claim to reproduce a circadian rhythm** — nobody watched outside that window. |
| **Non-effects (free parameters you can delete)** | Terrarium size vs activity **P = 0.926**. Body temperature vs activity **P = 0.322**. Sex vs activity **P = 0.648**. Sex vs defence **P = 0.45**. Body weight vs defence **P = 0.13**. | [EM] **V** | — | **Four invented constants you never have to write.** Arena size, a temperature term, sex, and body mass can all be left out of `drives.py` without loss of realism, within the tested ranges. |

### 4B. Prey capture

Every kinematic number here is **[CV]** — no prey-capture kinematics have ever been published for *E. macularius*.

| Quantity | Published value | Species / conf. | Sim now | Implication |
|---|---|---|---|---|
| **Approach is a two-mode state machine, not a scalar** | **Non-evasive prey:** approach closely, use "movements of mainly the head, neck, and forelimbs to manoeuvre the jaws into position" — no hindlimb lunge. **Evasive prey:** "typically stopped at a greater distance", then "pushed off with the hindlimbs to lunge forward", propelling ~2 cm. Independently corroborated: "carefully approaching… once within a few centimetres, they struck by pushing forwards with their hindlimbs and rapidly straightening their spine". | **[CV] V**, two labs | single scalar `engage` | **Replace `engage` with a 3-state machine: STALK → HALT+AIM → LUNGE.** The animal *stops* before striking — insert an explicit locomotion-halt state. The lunge is driven by `hip_proret` + `spine_pitch` (spine straightening), **not** by the CPG. |
| **Strike trigger distance** | **Evasive prey 2.03 ± 0.08 cm = 0.384 SVL.** Non-evasive **~1.19 cm = 0.225 SVL.** Independent lab: 14.6 mm (arthropods), 21.2 mm (scorpion). | **[CV] V** (evasive) / **L** (non-evasive, figure-read) | **`eat_radius` = 0.10 m** | **The capture radius is ~5× too generous.** Scale to your MJCF's SVL: **0.22 × SVL for a stationary target, 0.38 × SVL for a moving one.** Also separate "fixate" distance from "capture" distance. |
| **Strike peak velocity** | Evasive **0.851 m/s** (16.1 SVL/s); non-evasive **~0.325 m/s** (6.1 SVL/s) — 38 % of the evasive value. Independent lab, per-trial maxima: 0.24–1.10 m/s. | **[CV] V** / **L** | instant teleport at 0.10 m | Two setpoints, not one gain. **Since the sim's food is a STATIC sphere, the mealworm number (~0.33 m/s scaled) is the correct target today.** Reward the *range* 0.25–1.1 m/s scaled, not a single deterministic speed, or the policy overfits one ballistic pattern. |
| **Strike acceleration and braking** | Max accel **46.6 ± 1.97 m/s² (4.75 g)**; max **decel −74.7 ± 2.54 m/s²**. Independent lab: 30.0 and 36.9 m/s². **The animal brakes ~60 % harder than it launches.** | **[CV] L** | — | **The braking phase is the binding constraint.** The sim needs enough antagonist torque to decelerate at ~1.6× the launch acceleration or the head overshoots the prey. |
| **Strike timing — a hard simulator constraint** | Max acceleration **16.34 ± 6.71 ms BEFORE** contact; max velocity 9.03 ms before; max deceleration 0.2 ms **after**. Total propulsive phase ≈ **16–20 ms**. | **[CV] L** (500 fps = 2 ms resolution) | 50 Hz control = **20 ms/step** | **The entire strike fits inside ONE control step. A 50 Hz PPO residual physically cannot shape it.** Three options: (a) run the strike as an open-loop ballistic primitive scripted at the 0.0008 s physics timestep; (b) raise control rate to ≥500 Hz while `engage`=1; (c) accept that only pre-strike posture and trigger distance are learnable and hard-code the lunge. **Do not try to learn the strike at 50 Hz.** |
| **Lunge posture signature** | Body height start → max: pectoral 14.2 → 16.4 mm (**+15 %**); **mid-back 12.5 → 16.3 mm (+30 %)**; pelvic 10.2 → 13.5 mm (+32 %). SVL-relative: mid-back 0.236 → 0.308 SVL. | **[CV] L** | — | **The mid-back rise is the `spine_pitch` signature and the single most diagnostic variable for whether the lunge looks real.** Reward a +30 % mid-back height rise over the pre-strike value. |
| **Capture success** | Evasive prey **82.9 %** overall (77.0 % intact, 78.6 % post-autotomy — not significantly different). Non-evasive **100 %** (53/53). | **[CV] V** | — | **Do not push a policy to 100 % on moving prey — the real animal misses ~1 in 5.** Set the curriculum advance gate at ~80 % for moving food, 100 % for the current static sphere. **If the sim exceeds 85 % on moving prey, the prey's evasion model is too weak.** |
| **Prey shaking** | **12.30 ± 2.06 shakes/s** intact (independently corroborated at 13.06 ± 2.0 Hz). Max shake velocity **1.18 ± 0.17 m/s**, amplitude **10.78 ± 1.57 mm = 0.204 SVL**. Head sweeps ±90–110° from midline; peak angular velocity **127.9 rad/s**, angular acceleration **16,512 rad/s²**. Bout length: harmless prey 5 ± 1 cycles over 0.3 ± 0.1 s (shaken in only 30 % of trials); dangerous prey 31 ± 7 cycles over 1.4 ± 0.5 s (100 % of trials). | **[CV] V** for magnitudes; the *direction* of the autotomy change is **contradictory in the source paper's own text vs its table** — use 12.3 Hz for intact and distrust the post-autotomy value in either direction | absent | A separate oscillator at ~10× the locomotion frequency — script it at the physics timestep, not the 50 Hz loop. **The shake reaches HIGHER snout speed (1.18 m/s) than the strike itself (0.325 m/s), so the shake — not the strike — is the binding actuator-torque constraint for the neck joints.** Verify `neck_yaw`/`spine_bend` ranges permit ±90–110° and that 0.0008 s is stable at 16,500 rad/s². Give the food object a "danger" attribute that scales shake duration — the natural hook connecting `fear`/`danger` to feeding. |
| **Foot anchoring during the shake — a clean tail test** | Feet spend 0.008–0.065 s off the ground per shake. **Tail intact: aerial time is UNCORRELATED with shake velocity (R² = 0.0002).** After tail loss it becomes correlated (R² = 0.23). Hindlimbs and tail "appear to serve as anchors". | **[CV] V** | — | **The cleanest published pass/fail check on the 5-segment tail's dynamics.** A correctly-modelled intact gecko keeps hind feet planted through the shake. If your sim shows the *tailless* signature with the tail attached, the tail counterbalance is wrong. |
| **Ingestion / handling time** | Harmless prey "consumed within **10–20 s**" [CV]. Three adult crickets from forceps: **41 s to 4 min 2 s** [EM]. | **[CV] V** / **[EM] V** | instant deletion | Replace the teleport with a **10–20 s occupied state** (locomotion suppressed). Gives `drives.py` a real time constant for the hunger step change. Note the [EM] feeding bout (41–242 s) is **2× to 12× the entire current episode.** |
| **Prey size** | Lab prey 15–20 mm for a ~50 mm SVL gecko = **0.30–0.40 SVL** [CV]. Enrichment prey ≤1 cm [EM]. | **[CV] V** / **[EM] V** | `food_radius` = 0.035 m (7 cm diameter) | Set food diameter to **~0.3 × SVL** (≈3 cm for the current model, or ≤1 cm to match the [EM] enrichment protocol). Sizing it correctly makes the `nose_tip` alignment requirement realistic. |
| **Prey motion matters — a lot** | Live free-roaming prey **roughly DOUBLED** total behavioural output (696 → 1409 elements/animal). Preference by prey motion style: desert locust 19.95 > housefly 17.73 > house cricket 13.45 behavioural units (P = 0.001). | [EM] **V** | **static green sphere** | **A static sphere is the least-stimulating possible prey.** Give the food a random walk or a startle-jump on approach; expect ~2× change in engagement, with 1.48× spread as the measurable size of the motion-style effect. |
| **Inter-strike refractory** | Prey re-offered **~5 minutes** after each consumption, until the gecko stopped pursuing. | **[CV] V** (of the protocol) | none | Use ~5 min as the minimum inter-strike interval; terminate the foraging episode on a satiation threshold rather than a fixed food count. |

### 4C. Fear and danger — the section that rewrites `drives.py` most

Current code:
```python
self.danger += (danger_signal - self.danger) * min(4.0 * dt, 1.0)
self.fear   += 0.75 * self.danger * dt
```
Every constant here is invented, and the *channel* feeding `danger_signal` is wrong.

| Quantity | Published value | Species / conf. | Implication for `drives.py` |
|---|---|---|---|
| **Which modality actually triggers defence** | P(overt antipredator reaction): **chemical alone 0.20** (95 % CI 0.09–0.40) · **chemical + visual 0.40** (0.21–0.62) · chemical + mechanosensory 0.23 · all three 0.29 · **visual alone 0** · **mechanosensory alone 0** · visual + mechanosensory 0. Factorially: **chemical χ² = 8.098, P = 0.0044; visual χ² < 0.01, P > 0.9; mechanosensory χ² < 0.01, P > 0.9.** Integration additive, **no synergy.** | [EM] **V** (n = 40–42) | **`fear = w_chem · odour + noise`, with `w_visual = w_mech = 0` as the DEFAULT.** Any nonzero visual/vibration weight on the *fear* channel is an assumption beyond the data and must be flagged in a comment. |
| **A real, visible, live snake** | **P(defence) = 0.07** (CI 0.02–0.20) — **not significantly different from the 0.02 no-cue control.** | [EM] **V** (n = 42) | **The most important single number for this architecture.** `head_cam` is the only exteroceptive predator channel the MJCF has, and a live snake seen through glass barely moves the needle. Set the vision-driven fear gain so a visible threat produces P(defence) ≈ 0.07 — essentially a weak prior. **Do not train a strongly-rewarded vision-triggered flight reflex; the animal does not have one.** |
| **Fear floor** | **0.02** (CI 0.0033–0.15) with no cue present. Independently: 2 defensive reactions in 78 trials with a scentless plastic object touched to the snout = **0.026**. | [EM] **V**, two experiments | The fear ODE must **rest near 0.02, not 0.** Use as the target false-positive rate: a gate firing >2–3 % on empty scenes is mis-tuned. Note a non-threat object physically touching the nose still gave only 2.6 % defence — **proximity alone must not drive fear; only cue identity should.** |
| **Fear ceiling** | **0.40** — the maximum across all eight treatments, at chemical + visual. Three cues gave *less* (0.29) than two. | [EM] **V** | **Clamp the fear output so P(defensive action) never exceeds ~0.40.** And because three cues < two cues, **fear must NOT be a monotonic sum of channels** — implement additive-with-noise or a single-dominant-channel rule. |
| **Vision drives CURIOSITY, not fear** | Tongue-flick counts per 90 s: control 3.66 → chemical 8.69 → visual+mechanosensory (no odour) **6.73** → chem+visual 10.75 → all three 13.18. **Visual effect on flicking χ² = 14.50, P = 0.0001** — even though visual effect on *defence* is null. | [EM] **V** | **This is the split to encode: vision → curiosity; odour → fear.** A purely visual stimulus raises investigation 1.8× while raising defence not at all. |
| **Curiosity dynamic range** | Max/baseline flick ratio = **3.60×** (13.18 ÷ 3.66). Ceiling replicates independently at 13.22/90 s = 0.146/s. | [EM] **V**, two experiments | Clamp the curiosity ODE so max/rest ≤ **3.6**. Hard-code **0.15/s** as saturation. |
| **Defence rate limit** | Discrete defensive postures: at most **~6.9 per 30 min** even under strong threat (0.82/30 min for a harmless control). Tail waving 5.62–5.67 per 30 min under threat vs ~3.2 control; adults wave ~2.4× less than subadults. Freezing bouts 1.4 (control) to 3.29 (threat). | [EM] **V** (n = 585) | **At most ~0.004 defensive events per second even under threat.** A policy flipping into a defensive posture more than once every ~4 minutes is behaviourally wrong. Tail-waving is a **rare discrete bout**, not a continuous behaviour — gate it on a high danger threshold and grade it by threat identity. |
| **Threat multiplies sensing, it does not suppress it** | Tongue-flicks rose **2.3–2.7×** above the harmless-control rate when confronted with snake predators. | [EM] **V** | **Encode danger as a GAIN on the sensing behaviours, not as a flee-only switch.** The gecko investigates more, not less. |
| **The energy↔fear link** | Choice of deterrent **threat–vocalise–bite** vs **escape** is affected by age **and body condition**; juveniles threaten, adults flee, and the two are **mutually exclusive**. | [EM] **U** — n = 552, but paywalled with no OA copy; per-age percentages unobtainable | **The single cleanest published link between the energy drive and the fear drive, and it is absent from `drives.py`.** Treat "threaten" and "flee" as mutually exclusive branches selected by **one switch gated on body condition (energy)** — low condition → threaten, high → flee. |
| **Fear rise/decay time constants** | **Unpublished for this species.** The 2026 paper explicitly abandoned latency (too zero-inflated) and durations ("would not yield a biologically meaningful measure"). Best surrogates: **binocular-fixation bouts ~180 s** out of a 30-min trial [EM]; reptile plasma corticosterone detectably elevated at **2.5–4 min** after disturbance [proxy: rattlesnake, iguana, cottonmouth]. | **U** | **Label these as INVENTED in the code.** Suggested architecture: **two** fear variables — a fast behavioural one (sub-second, invented) and a slow physiological arousal one (τ ≈ 200 s, weakly anchored). |
| **Sensor geometry for threat** | Chemical stimulus at **~3 mm** from the snout. Visual threat at **30 cm**. | [EM] **V** | Odour channel radius = **millimetres**, not the 0.10 m used for food. Visual threat channel ~0.30 m. |

### 4D. Metabolism, energy and hunger — the numbers that break the current ODE

Current code:
```python
self.hunger += 0.015 * dt          # saturates in 67 s
self.energy += 0.020 * dt * (1 - moving)
self.energy -= 0.045 * dt * moving # empties in 22 s
```

| Quantity | Published value | Species / conf. | Implication |
|---|---|---|---|
| **Resting metabolic rate** | **0.075 ± 0.011 mL O₂ g⁻¹ h⁻¹** at 30 °C. Scaled to a 40 g adult (M^−0.22): 0.064 mL O₂ g⁻¹ h⁻¹ = **2.54 mL O₂/h = 51.1 J/h = 14.2 mW.** | [EM] **V** (n = 6) — **the only direct metabolic measurement ever made on this species** | The anchor for everything below. |
| **Cost of transport** | **1.5 mL O₂ kg⁻¹ m⁻¹** level, 2.7 uphill (50°), vertical efficiency 37 % [CV, 4.2 g]. Independently **0.73 mL O₂ kg⁻¹ m⁻¹** [TP, 11.2 g], temperature-independent. **Mass-scaled to 40 g: Cmin ≈ 0.6 ± 0.15 mL O₂ kg⁻¹ m⁻¹.** | **[CV] L / [TP] V**; the mass-scaling exponent is **my assumption** | Nocturnal geckos are **genuinely cheap movers** — ~a third the cost of a diurnal lizard of the same mass. **The energy penalty for locomotion should be SMALL**, not the dominant term. At 0.11 m/s, walking costs ~4× resting. |
| **Maximum aerobic speed** | **0.075 m/s at 15 °C**; VO₂max reached across **0.100–0.158 m/s at 25 °C**. Sustained >60 min at 0.050 m/s. | **[TP] V** | **The sim's 0.11 m/s is at or below the warm-animal aerobic ceiling — locomotion at the current speed is fully sustainable indefinitely.** Energy should drain at essentially the resting rate. Implement the drain as a **hinge at MAS**: flat below ~0.13 m/s, rising steeply above. **The current flat 0.045/s is the wrong SHAPE, not merely the wrong magnitude.** |
| **Endurance as a function of speed** | **t_end = 0.030 · v^−2.07 hours** at 25 °C (r² = 0.87); **t_end = 0.013 · v^−1.70 h** at 15 °C (r² = 0.83), v in km/h. | **[TP] V** (n = 25 / 24) | **This is the equation the fast variable should implement.** At the sim's 0.11 m/s (= 0.396 km/h): **t_end = 12.2 min = 735 s** warm, 226 s cool. Honest fatigue drain = **1/735 = 0.00136 /s** — **10× to 33× slower** than the current 0.045/s. |
| **Meal size and interval** | Fed 3×/week; offered 5 % of body weight, actually consumed **54.7 ± 12.2 %** of it = **2.74 % of body weight per feeding**. Mean inter-meal interval **2.33 days = 201,600 s.** | [EM] **V** (n = 24 adult males) | A hunger drive rising 0→1 between meals rises at **4.97 × 10⁻⁶ per second** — **~3,000× slower than the current 0.015/s.** Within a 90 s episode a real gecko's hunger changes by 4 × 10⁻⁴. **Hunger cannot be a per-step integrated variable; it must be sampled once per episode as an initial condition and carried ACROSS episodes.** |
| **Digestibility and meal energy** | Apparent dry-matter digestibility **71 ± 4.7 %** (crude protein 81 %, fat 65–74 %, calcium only 43 %) [EM]. Cricket meal ~17.7 kJ/g dry matter [rooster assay proxy]. **Derived: a 40 g gecko's 1.10 g wet meal yields ~4.1 kJ metabolizable ≈ 81 hours = 3.4 days of resting metabolism.** | Digestibility [EM] **V**; energy density **U** (avian assay). **But the derivation independently reproduces the observed 2.33-day feeding interval — two unrelated sources agree, so the budget is trustworthy.** | **Eating one food item should add about 3 days' worth of resting energy — NOT reset energy to 1.0.** |
| **Fasting tolerance** | **28 days of complete fasting** produced no significant change in distal intestine morphology (all P ≥ 0.84) and no change in gut microbial diversity — uniquely stable among five vertebrate classes. | [EM] **V** | **28 days = 2.42 × 10⁶ s.** A hunger drive reaching "starving" faster than that is unfaithful. The current 0.015/s reaches 1.0 in 67 s — **36,000× too fast.** |
| **Tail as an energy reserve** | Tail = 22 % of body mass, and is a lipid store. **Derived: ~140 days of resting metabolism** for a 40 g gecko (8.8 g tail at an *assumed* 50 % lipid, 39 kJ/g, against 51.1 J/h). | **U** — the lipid *fraction* was not obtained from a primary source and **sets the entire energy timescale** | **The headline implication.** If `energy = 1.0` means a full tail fat reserve, the resting drain is **8.3 × 10⁻⁸ /s** — the current 0.045/s is ~500,000× too fast. **Two honest options: (a) keep `energy` as a real physiological reserve at ~1e-7/s and carry it across episodes; or (b) RENAME the fast variable `fatigue` / `aerobic_debt` and drive it from the endurance equation at 0.00136/s. Option (b) is the honest reading of what 0.045/s was trying to be.** |
| **Cost inversion you would never guess** | Postprandial metabolic rate peaks at **3.7–7.3× pre-feeding** 15–33 h after a meal, returning to baseline at 62–170 h. | **[LIZ] U** (*Eulamprus tympanum*, paywalled) | **Digesting a meal is far more expensive than walking** (4–7× resting vs ~4× resting). That inverts the usual RL intuition that movement is what costs energy. |
| **Voluntary daily locomotor distance** | **124 m/day** on a running wheel (SD 177, range 0–560); 11 wheel episodes/day; **133 min total engagement/day**. Activity concentrated in the first 60 min after lights-out. | [EM] **L** — **n = 1**; the authors themselves caution against generalising | ~100 m/night in ~130 min of intermittent activity ≈ **1.6 cm/s averaged over the active period.** Calibrate hunger/energy decay and the engage-vs-rest threshold against this rather than inventing constants. |
| **Preferred body temperature** | **29.5–31.9 °C** (midpoint 30.7 °C) in a 5–45 °C gradient. Separately **25.8 °C** in a thigmothermal gradient with no day/night difference. Body temperature tracks *substrate* temperature, r² = 0.97 — this is a **thigmotherm, not a heliotherm.** CTmax 41.07 ± 0.89 °C. | [EM] **V** | Every metabolic number above is a 30 °C number; scale with Q₁₀ ≈ 2.5 if temperature is ever modelled. If it is: drive it from the **belly touch sensors' contact surface**, not a radiant source — the MJCF already has the physiologically correct channel. Low priority: activity did not correlate with body temperature (P = 0.322). |

### 4E. The realistic episode timescale — the headline recommendation

Every quantity in §4 was measured over one of these windows. **The sim's episode is 1000 steps × 0.02 s = 20 s.**

| Published timescale | What it is | Steps at 50 Hz | vs the sim's 20 s |
|---|---|---|---|
| **20 s** | *the current episode* — contains 0.55–1.1 behavioural acts | 1,000 | — |
| **90 s** | Standard antipredator trial (Frýdlová 2026) | 4,500 | 4.5× |
| **180 s** | Dead time before prey chemicals raise tongue-flick rate | 9,000 | 9× |
| **260–485 s** | **Latency to first contact with a novel object** | 13,000–24,000 | 13–24× |
| **41–242 s** | One feeding bout (3 crickets) | 2,000–12,000 | 2–12× |
| **300 s** | **The published focal-sampling quantum** — ~4 distinct behaviour categories | **15,000** | 15× |
| **600 s** | Standard social-encounter test | 30,000 | 30× |
| **1800 s** | Standard antipredator trial and daily focal session | 90,000 | 90× |
| **2700 s** | Enrichment session (3–6 object interactions) | 135,000 | 135× |
| **5 h / 7 h** | The 6–11 pm activity window; a thermoregulation trial | — | — |

**The recommendation:**

- **Keep 1000 steps (20 s) ONLY for CPG/gait tuning.** At a 0.84 s stride, 20 s gives ~24 strides — plenty for duty factor, phase, joint excursions and stride length. **The gait scorecard in §2 is fully measurable at 20 s.** Nothing in this section requires you to change episode length for CMA-ES gait calibration.
- **Use 4,500 steps (90 s) for anything involving fear or `danger`** — that is exactly the trial length every fear number was measured over, so sim and paper become directly comparable.
- **Use 15,000 steps (300 s) as the literature-matched minimum for `drives.py`, foraging, exploration, curiosity or `engage`.** Below 300 s the sim physically cannot produce a behaviour budget comparable to any published number.
- **Use 90,000 steps (1800 s) to match the antipredator and daily-session protocols**, if you can afford it.
- **Hunger and energy must be carried ACROSS episodes**, not reset at `env.reset()`. Their real time constants are days.

---

## 5. The body check

Model total mass **0.0612 kg** (sum of 44 `mass=` attributes) · SVL ≈ **0.106 m** · tail chain **0.110 m** · root spawn z **0.028 m**.

### Supported by literature — do not change

| Repo feature | Value | Published | Species / conf. |
|---|---|---|---|
| Tail mass **fraction** | 0.015 kg / 0.0612 = **24.5 %** | 22 % (adults, n=7); ~25 % (n=10); pass band **0.19–0.25** | [EM] **V** |
| Tail **volume** | 5 capsules ≈ 6.87 cm³ | **7.8 × 10³ mm³** original | [EM] **V** |
| Head length | 0.029 m capsule = **0.27 SVL** | **0.27–0.30 SVL** (32/119, 33/109 mm) | [EM] **V** |
| Eye diameter | spheres size 0.0033 = **6.6 mm** | **6–7 mm** (0.055–0.064 SVL) | [EM] **V** — dead centre |
| Snout-to-eye position | eye anterior margin **8.9 mm** behind snout tip | **SEL = 8 mm** in both specimens | [EM] **V** |
| Max tail width | tail1 collision radius 0.0055 → **11 mm** base diameter | **9 mm** intact, 11 mm regenerated | [EM] **V** — at the top of range, trim slightly |
| Padless, clawed feet; no adhesion | 5 massless claw capsules per foot, floor friction 0.9, no tendon adhesion model | [EM] is explicitly one of two "desert-dwelling **padless** geckos"; keys out by tuberculated subdigital lamellae | [EM] **V** — **correct, and it also means never reward vertical climbing** |
| Belly touch sensors | 2 sites | Dermis carries Merkel mechanoreceptors; skin is tuberculate (rough), supporting friction 0.9 | [EM] **L** |
| 3-segment lateral spine + separate neck DOFs | `spine_lat_1/2/3`, `neck_yaw/pitch` | Presacral vertebrae group into **cervical / sternal / dorsal** compartments; cervical growth is constrained by head mechanics | [EM] **V** (abstract) |
| `hip_proret` range ±40° = 80° total | | **In vivo 79.3°** pro/retraction during locomotion | **[II] V** — *Iguana*, not a gecko |
| `hip_sprawl` range ±35° = 70° total | | **In vivo 71.1°** ab/adduction | **[II] V** — *Iguana* |

### Off — measurably wrong

| Repo feature | Value | Published | Error | Fix |
|---|---|---|---|---|
| **Total mass** | 0.0612 kg at SVL 0.106 m | 0.0363 kg at SVL 0.1046 m [EM] **V** | **~1.6× too heavy for its length.** Inflates every GRF, torque and `forcerange`. | Rescale all `mass=` by ~0.60, or better: **replace all 44 hand-assigned masses with a single uniform `density` ≈ 1100 kg/m³**, derived from the published tail mass ÷ tail volume (8.67 g / 7.8 cm³). That one number fixes the 2.18 g/cm³ tail and the 0.41 g/cm³ trunk simultaneously and makes the distribution self-consistent. |
| **Tail length** | 0.110 m = **1.04 SVL**, 50.9 % of total length | **0.73 SVL**; 41 % of total length | **~40 % too long** | Shorten chain to ≈0.078 m (e.g. 0.020/0.017/0.016/0.014/0.011). Changes tail inertia, `tail_bend` moment arms, and the CoM. |
| **Standing height** | root z = 0.028 m = **0.264 SVL** | hip **0.15 SVL**, shoulder **0.11 SVL** | **~75 % too tall** | Lower the spawn and add a posture reward on hip and shoulder site z. |
| **Girdle height symmetry** | humerus and femur attach at the **same** z (−0.002) | shoulder/hip height ratio **0.73** | Fore/hind asymmetry entirely absent | Lower the shoulder attachment (or shorten the forelimb) so the settled ratio is 0.73. |
| **Head shape** | visual ellipsoid 0.0253 m wide (**0.24 SVL**), 0.0209 m tall (**0.20 SVL**) | head width **0.18–0.19 SVL**, height **0.10–0.11 SVL** | ~30 % too wide, **~2× too tall** | Flatten to ≈ `size 0.0155 0.0098 0.0058`. Affects where `head_cam` sits and whether the head scrapes during belly contact. |
| **Eye separation** | orbit-to-orbit gap 0.0136 m; eyes protrude beyond head width | interorbital distance **10 mm** in both specimens | ~36 % too far apart | Move to ±0.0083 m. Changes the effective binocular overlap `head_cam` is meant to represent. |
| **Forelimb length** | 0.0483 m = **0.456 SVL**, essentially equal to the hindlimb's 0.474 SVL | Total hindlimb **0.417 SVL** (derived: hip height 0.15 SVL ÷ 36.0 % of limb length). Nepal specimens read as a sum: FLL 0.29 SVL, HLL 0.32 SVL | Both limbs ~14 % long; **and forelimb ≈ hindlimb, which no lizard shows** | Shorten the forelimb chain relative to the hindlimb. Prefer shrinking the **pes** (currently 0.0172 m = 34 % of the whole limb) rather than the femur. |
| **`hip_rot` range** | ±35° = **70° total** | **25–30°** long-axis rotation in vivo **[II] V** | **~2.3× too wide** | Tighten `hip_rot_L/R` to ≈±15° and the actuator ctrlrange to ≈±0.26 rad. |
| **Hindlimb segment proportions** | crus/thigh = 0.014/0.019 = **0.74** | **Femur 1.54 ± 0.04 cm and tibia 1.51 ± 0.04 cm are essentially EQUAL** (ratio 0.98). Segments as SVL fractions: femur 0.145, tibia 0.142, tarsal+meta 0.059, toe 0.070 | Model uses a mammalian femur-longer-than-tibia prior | Set femur ≈ tibia. **Fuller+ 2011 also prints "relative hind limb length 35.6 % SVL", which does NOT equal the arithmetic 4.41/10.65 = 41.4 % — that figure is size-corrected via regression residuals. Use the absolute segment lengths.** |
| **No tail touch sensors** | 4 foot + 2 belly only | Dorsal tail **59–76 sensilla/mm²** — highest on the body, ~2× the best non-tail site | The most densely innervated skin on the animal has no sensor | Add dorsally-weighted tail touch sensors at ~2–3× the belly weighting. |
| **Single contact threshold** | `contact_thresh` = 0.0564, unitless, global | Tail base **1.1 mN**, hindfoot **1.9 mN** — tail is 1.7× more sensitive | One threshold where the animal has at least two | Split into tail and foot thresholds. |

### No published basis at all — these are invented, and must be labelled so

| Repo feature | Status |
|---|---|
| **All 44 hand-assigned `mass=` values** (trunk 8 g each, head 5 g, pelvis 5 g, femur 1.6 g…) | Published mass partitioning stops at "tail = 22–25 %, rest = body". **No published head, neck, trunk, girdle or per-limb mass fraction, and no segment moment of inertia, exists for this species.** Mitigation: a single uniform density (above) at least makes them self-consistent. |
| **Girdle widths** (±0.011 m shoulder, ±0.012 m hip offsets) | Nothing published on pectoral or pelvic girdle width, glenoid-to-glenoid or acetabulum-to-acetabulum spacing. Only indirect bound: head width 0.18–0.19 SVL caps trunk width from above. |
| **5 tail segments** | No published caudal vertebra count for [EM] — only the qualitative fact that *every* caudal except the first and last carries an intravertebral fracture plane. 5 segments is an engineering choice. **But note: the standard experimental protocol breaks the tail at the PROXIMAL-most plane, i.e. at the vent — so a tail-loss curriculum should detach the whole chain at the pelvis, matching how the papers you are scoring against did it.** |
| **Tail taper profile** | Only max width and total length are published. The diameter-vs-distance profile setting the 5 capsule radii is invented. Anchor it on two published shape facts: the regenerated tail is 61 % the length but **131 % the max diameter**, and max diameter is reached proximally — so mass is **strongly proximally biased.** A monotonic proximal-to-distal taper summing to 22 % is the honest choice. |
| **All claw dimensions** (0.0004 m capsules) | No claw length, curvature or tip radius published for [EM]. |
| **All joint stiffness, damping, `gainprm` and `armature`** | Nothing published for this species. GRFs exist but the magnitudes appear only in figure axes. Only empirical ceiling: **tail musculature can drive 8 Hz** (autotomised tails swing at 4–8 Hz) — **do not set tail damping so high that 8 Hz is unreachable.** |
| **Every joint range except the hip** | There is **no** cadaveric or in-vivo ROM study for [EM] or any eublepharid on **any** joint. The hip has an *Iguana* anchor; knee, ankle, wrist, elbow, shoulder, spine and tail ROM have **no** published sprawling-lizard cadaveric equivalent — only in-vivo walking excursions (which are §2C's targets, and are *narrower* than anatomical limits). |
| **Floor friction 0.9 0.005 0.0001** | No skin friction coefficient against sand, rock or cork published for this species. Only qualitative support: tuberculate skin is rough. |
| **`head_cam` fovy 120** | See §3. No published FOV for any eublepharid. |
| **Forearm 27 % LONGER than the humerus while the crus is 26 % SHORTER than the femur** | An internally inconsistent pattern, but **no published [EM] humerus/radius length pair exists to test it against.** Flagged as an open question, not a confirmed error. |

**One free, high-value validation you can run today, with no new data:** compute the model's centre of mass with `mj_forward` in the neutral keyframe. It must sit at **0.659 × SVL behind the `nose_tip` site** (= 0.0699 m at SVL 0.106). Then delete `tail1..tail5` and re-compute: it must move to **0.527 × SVL**. **Unit warning: the widely-quoted "13 % shift" is 13 PERCENTAGE POINTS OF SVL, not a 13 % relative change** — the target is 0.527, not 0.659 × 0.87. Assert both in the same test.

---

## 6. What nobody has measured

For each gap: what is missing, and the **safest proxy**.

### Gaps that block gait calibration

| Gap | Safest proxy |
|---|---|
| **Lateral trunk (spine) bending amplitude.** No number, in any unit, for [EM] or any gecko. Jagnandan & Higham describe a standing wave in the trunk becoming a travelling wave in the tail, but cite a *bipedal locomotion review* rather than a measurement. Ritter's lizard axial-muscle papers are EMG/denervation studies with no printed bending amplitude. | **Let the measured downstream variables determine the unmeasured upstream one.** Set `spine_bend` as a standing wave and tune its amplitude until hind step length and femur retraction excursion hit their measured targets (0.06 SVL and 82.6°). Do not pick the amplitude directly. |
| **Pelvic girdle yaw excursion in degrees.** Jagnandan & Higham *measured* it, proved it drops with tail restriction (t = 2.287, P = 0.048) and autotomy (t = 3.129, P = 0.012), and plotted the full within-stride trace — but **printed no numeric value anywhere.** Only the effect exists, not the magnitude. | Invent it, then **validate with the ratio**: restricting the tail must reduce it. This is the published causal chain (tail undulation → pelvic yaw → femur retraction → step length), so wire `spine_bend` in phase with `tail_bend_L/R` and let step length *emerge* rather than commanding it. |
| **Pectoral girdle rotation.** Complete blank — the 2017 paper marked the pectoral girdle centre but reported only pelvic rotation. | Set to zero and note it. The forelimb is decoupled from the tail, so nothing downstream depends on it. |
| **Tail-tip lateral displacement magnitude.** The *sign* and *slope* are published (negative with speed, R² = 0.423, P = 0.042); the amplitude lives only in a figure with a y-axis running ~0–0.12 SVL. Same for tail height. | Set peak tail-tip lateral offset from the pelvis to **~0.10 SVL at slow speed falling toward ~0.05 SVL at 0.32 m/s**, and **treat the DOWNWARD SLOPE as the thing being matched** — that is what is actually published. The 0.12 SVL ceiling is a figure read; treat as a soft bound. |
| **Tail travelling-wave parameters.** No published wavenumber, wave speed, or inter-segment phase lag for the leopard gecko tail. Only qualitative anchors: base flexes toward the protracted hindlimb; wave travels caudally. | Borrow the **autotomised-tail EMG timing**: contralateral onset delay ~130 ms proximal vs ~90 ms distal for rhythmic swings means the wave travels distally. Set a per-segment phase lag consistent with that ~40 ms proximal-to-distal gradient. **[EM] V** for the delays — but measured on a shed tail, not a walking one. |
| **Phase of tail flexion relative to the stride.** Nobody has published the lag between peak tail bend and hindlimb footfall in any gecko. | Physically defensible default: **peak tail bend toward the SWING-side hindlimb**, i.e. ~180° out of phase with ipsilateral `hip_proret` retraction; lock to `spine_bend` at zero lag (standing wave) for the slow speeds these papers cover. |
| **Duty factor and stride length as explicit functions of speed.** Zaaf+ 2001 measured exactly this and is the one paper that would give regression coefficients — **closed access, no repository copy anywhere.** Only the abstract is readable. | You have duty factor at **two** speeds (McElroy: 0.72 at 0.24 m/s, 0.70 at 0.29 m/s) plus the strong finding that duty factor is essentially flat. **Interpolate flat.** For stride length, use the published frequency-dominance (r² 0.98 vs 0.53) to make length rise only weakly. |
| **Full within-stride joint-angle trajectories.** Every paper reports only maxima, minima and excursions — never a time series. One figure plots a representative stride but with no tabulated points. | **Synthesise waveform SHAPE from the EMG onset/duration/peak-time numbers** (§2C rows 22–23), which *are* published as percentages of stride. That is strictly better than guessing a sinusoid. |
| **Stride-to-stride variability.** No paper reports the CV of stride period or stride length. Without a variability target, a suspiciously metronomic sim gait cannot be flagged. | **Invent a CV band (I suggest 0.03–0.30) and label it invented.** A CPG with zero jitter is a detectable tell, but nothing published tells you how much jitter is right. |
| **Femur depression convention mismatch.** Fuller reports 8.8°, Jagnandan reports 49–66°, for nominally the same angle, because the zero reference and marker set differ. **No published reconciliation exists.** | **Pick Jagnandan's convention** (all the tail-ablation ratios use it and are self-consistent), state it in the MJCF comment, and **never average across labs.** |
| **Foot and toe kinematics.** No data on digit angles, foot contact area, foot roll, or slip for [EM]. | Duty factor is your only constraint on what a correct contact profile looks like. |
| **Joint torques, moment arms and actuator gains.** Nothing published. Ground reaction forces were measured but the magnitudes appear only in figure axes. | The 26 position actuators' `kp`/damping are uncalibrated and must be tuned by CMA-ES against the kinematic scorecard, not set from data. |
| **Head/neck kinematics during walking.** No published data on head stabilisation, gaze holding, or neck/head joint excursions in a walking leopard gecko — **and this directly determines the `head_cam` image stream the brain sees.** | Use the **optokinetic gains** (§3) as the only published constraint: the head should stabilise gaze at gain ~0.9 for slow whole-field rotation. That is a *behavioural* target, not a kinematic one, but it is real and it is [EM]. |
| **True maximum sprint speed.** No dedicated sprint protocol has ever been run. Every published maximum is a by-product: 0.32 m/s (prompted), 0.32 m/s (tail-pinch-induced), 1.1 m/s (motorised treadmill, abstract only). | **The true maximum is unknown and lies somewhere between 0.32 and 1.1 m/s.** Treat 0.32 m/s as the practical ceiling for the command range. |

### Gaps that block the brain / drives

| Gap | Safest proxy |
|---|---|
| **No duration-based time budget exists for this species.** Every published leopard gecko "time budget" is an **event-count** budget — the authors state explicitly they recorded "only the occurrence of a behaviour within the observation interval and not its length." There is no published answer to "what percent of TIME is spent hiding/exploring/immobile." | **Score the simulator in EVENT COUNTS to match.** That is the honest option, and it is easy: apply the paper's own thresholds (see next row) and count. Do not invent duration priors. |
| **How to turn simulator state into published labels** — solved, actually. | Published thresholds: **"walk around" = displacement ≥ ONE BODY LENGTH; "change of body orientation" = displacement < one body length with at least one leg moved; "rest" = not physically active for ≥ 3 s.** [EM] **V.** Code these directly against SVL and joint velocity. |
| **No PTM or moves-per-minute has ever been published for [EM] itself.** The 33.2 % figure is *Coleonyx*, from a 1989 paper reachable only through secondary accounts that omit n. | **The single most load-bearing proxy in this document.** Use 33 % but flag it; cross-check against [EM]'s 22.7 % walking events. |
| **No move-bout and pause-bout durations in seconds for any eublepharid.** The only bout-structure result is *Teratoscincus* from a paywalled abstract. | Borrow the one structural insight that is published: **pause length should be sampled conditional on the preceding move length**, not i.i.d. Also **[TP]:** moonlight reduced moves/min by 16.7 % and activity by 27 % — a light-level modulation for `head_cam`. |
| **No fear rise or decay time constant.** See §4C. | Two variables: fast behavioural (invented), slow arousal (τ ≈ 200 s from corticosterone proxies). Label both. |
| **No freezing duration, no tail-wave waveform, no escape speed, no flight-initiation distance.** Tail-waving was scored **binary only** — no Hz, no amplitude, no elevation angle, no bout duration. | The tail actuators have a published *function* (defence) and a published *rate* (~3–5.7 bouts per 30 min) but **no waveform.** Invent it. |
| **No autotomy force threshold.** Every paper induces it by "gently pinching the base of the tail." No newton value published for any gecko. | Trigger on **sustained touch-sensor contact on a proximal tail segment**, not a global damage counter. Break at the most proximal segment whose sensor fired, defaulting to segment 1. |
| **No prey detection range or sensory trigger.** No paper reports at what distance a gecko first orients to prey, the visual angle at detection, or detection probability for moving vs stationary prey. The prey-type difference is measured only *after* detection. | The `head_cam` parameters and any detection gate must be invented. Nearest anchors: prey ≤1 cm elicits behaviour; binocular fixation occurs ~2.8 times per 30 min. **Separate "fixate" distance from "capture" distance and document both as choices.** |
| **No latencies:** detection→orient, orient→approach, halt→lunge. The 16 ms figure covers only the ballistic phase. | STALK and HALT+AIM dwell times must be invented. |
| **No approach (stalk) speed** — described only as "slow stalking movements." | Nothing to compare the sim's 0.11 m/s against for stalking specifically. |
| **The prey-capture scaling rule is undecided and changes the answer by 60 %.** The proxy animals are hatchling-sized relative to an adult leopard gecko. Isometric scaling of the 0.851 m/s cricket strike gives **2.09 m/s**; Froude/dynamic-similarity gives **1.33 m/s**. Nobody has published which applies to lizard strikes. | **PICK ONE, write it in a constants file with a comment, and use it consistently.** Cleaner alternative: **build the strike model at Coleonyx size and use the numbers raw.** |
| **Shake frequency direction is contradictory in the published paper itself** (Table 2 says 12.3→19.3 after autotomy; the Discussion says "significantly lower"). | Use **12.3 Hz for intact** — independently corroborated at 13.06 ± 2.0 Hz by another lab. **The post-autotomy value cannot be trusted in either direction.** |
| **No social/agonistic display rates.** The literature establishes *which* behaviours occur (rapid tail vibration, high posture, scent marking) and that incubation temperature modulates aggression, but **no accessible source gives displays per minute.** | Any social term is unavoidably invented. |
| **No jaw.** "Mouth-open threat" and "biting" are two of the seven published defensive behaviours, and the MJCF has **no jaw joint among its 110 joints.** No gape angle, gape-cycle duration or bite force has been published for any eublepharid either. | **Two of seven defensive behaviours are structurally unrepresentable.** Document the omission; do not fake them with head pitch. |
| **No evaporative water loss / thirst constant, no home range, no distance-moved-per-night for wild [EM], no field body temperature, no thermal performance curve for speed, no actograph.** | This species is described in the literature as an **"accidental model organism"** — abundant in labs, almost never studied in the field. Do not invent a water drive; leave it out. |
| **Lipid fraction of the tail** — sets the entire energy timescale (§4D) and rests on an assumption. | Verify before coding. Until then, prefer the `fatigue` framing (option b) which does not depend on it. |
| **Blink rate.** [EM] is one of the very few geckos that can close its eyes — its signature feature — and no blink rate has ever been published. | If `head_cam` occlusion is modelled, it is pure invention. |

---

## 7. How to actually use this — the fitness function

Two separate artefacts. **Do not merge them.**

### 7.1 `tests/test_morphology_audit.py` — a static gate, run once, before any optimisation

Assertions on the loaded MJCF (`mj_forward` in the neutral keyframe). All are pass/fail. **Optimisation must not begin until these pass**, because a wrong body makes every gait target unreachable.

| Assertion | Target | Tolerance | Source |
|---|---|---|---|
| total model mass | 0.038 kg | ±0.004 | Jagnandan cohorts |
| tail mass ÷ total mass | 0.22 | 0.19–0.25 | Jagnandan+ 2014/2017 |
| tail chain length ÷ SVL | 0.73 | ±0.04 | Rawat+ 2019 |
| whole-model CoM, distance behind `nose_tip` ÷ SVL | 0.659 | ±0.02 | Jagnandan+ 2014 |
| same, with `tail1..tail5` deleted | 0.527 | ±0.02 | Jagnandan+ 2014 |
| settled hip site z ÷ SVL | 0.150 | ±0.010 | Jagnandan+ 2014 |
| settled shoulder site z ÷ SVL | 0.113 | ±0.010 | Jagnandan+ 2014 |
| shoulder z ÷ hip z | 0.73 | ±0.05 | derived |
| femur length ÷ tibia length | 0.98 | ±0.08 | Fuller+ 2011 T1 |
| total hindlimb length ÷ SVL | 0.417 | ±0.03 | derived |
| `hip_proret` range | ≥ 90° | one-sided | Jagnandan & Higham 2017 |
| `hip_rot` range | ≤ 35° total | one-sided | Arnold+ 2014 **[II]** |
| head width ÷ SVL | 0.185 | ±0.02 | Rawat+ 2019 |
| interorbital distance | 0.010 m | ±0.001 | Rawat+ 2019 |

### 7.2 `realism_metrics.py` — the CMA-ES objective

**Measurement protocol (this part is mandatory, or the comparison is invalid).**

1. Run the controller at commanded speeds **v ∈ {0.08, 0.11, 0.14, 0.18} m/s**, level ground, friction 0.9, ambient equivalent to ~30 °C conditions (i.e. the calibration scenario, not a randomised one).
2. **20 s per speed** (1000 steps). Discard the first 3 s as settling. That leaves ~20 strides per speed — sufficient for every metric below.
3. Log at the 50 Hz control rate: trunk position and quaternion, all `qpos`, the 6 touch sensors, `nose_tip` site position, hip and shoulder site z, tail-tip and pelvis site positions.
4. **Contact events:** a foot is in stance when its touch sensor exceeds threshold. **Debounce: discard any contact or flight interval shorter than 40 ms (2 control steps)**, or contact chatter will destroy the duty-factor estimate.
5. **For any strike/shake metric, resample `nose_tip` to 500 Hz and apply a zero-lag 50 Hz low-pass Butterworth BEFORE differentiating.** Both prey-capture labs independently chose exactly this filter. **Comparing raw 1250 Hz MuJoCo velocities against published filtered values will over-read the sim's peak velocity and acceleration by a large margin and make the sim look faster than the animal.** Put the filter in the evaluation script, never in the controller.

**Metrics, targets and weights.** `z_i = (x_i − μ_i) / σ_i`.

| # | Metric | μ | σ | w | σ source |
|---|---|---|---|---|---|
| 1 | hind duty factor | 0.765 | 0.020 | **3** | between-study spread 0.70–0.79 |
| 2 | fore duty factor | 0.700 | 0.020 | **3** | published SEM ×2 |
| 3 | limb phase, hind→ipsilateral fore (%) | 43.5 | 2.0 | **3** | two-lab spread |
| 4 | stride length ÷ SVL | 0.73 | 0.06 | **3** | between-study spread 0.62–0.82 |
| 5 | hip height ÷ SVL (dynamic mean) | 0.150 | 0.010 | **3** | invented (SEM was 0.00) |
| 6 | shoulder height ÷ SVL | 0.113 | 0.010 | **3** | invented |
| 7 | stride frequency at 0.18 m/s (Hz) | 2.03 | 0.18 | **2** | published SD |
| 8 | freq-vs-speed regression r² | 0.98 | 0.05 (penalise below only) | **2** | published |
| 9 | stride-length-vs-speed r² | 0.53 | 0.20 | 1 | published |
| 10 | trunk pitch, nose-up (deg) | 3.7 | 1.0 | **2** | SEM ×1.6 |
| 11 | `hip_proret` excursion (deg) | 82.6 | 4.0 | **2** | SEM ×1.7 |
| 12 | `hip_sprawl` excursion (deg) | 52.4 | 5.0 | **2** | SEM ×1.5 |
| 13 | knee excursion (deg) | 98.8 | 5.0 | **2** | SEM ×3.6 |
| 14 | knee angle at footfall (deg) | 101.6 | 4.0 | 1 | SEM ×2.2 |
| 15 | ankle excursion (deg) | 85.2 | 5.0 | **2** | SEM ×2.1 |
| 16 | `shoulder_proret` excursion (deg) | 44.4 | 4.0 | **2** | SEM ×2.2 |
| 17 | `shoulder_sprawl` excursion (deg) | 101.4 | 16.0 | 1 | published SEM 15.9 — genuinely noisy |
| 18 | elbow excursion (deg) | 92.5 | 5.0 | 1 | SEM ×2.2 |
| 19 | hind stance duration at 0.13 m/s (s) | 0.64 | 0.06 | 1 | SEM ×2 |
| 20 | fore stance duration at 0.13 m/s (s) | 0.54 | 0.06 | 1 | SEM ×2 |
| 21 | COM KE↔PE phase shift (deg) | 151 | 15 | **2** | SEM ×2.3 |
| 22 | COM percent recovery (%) | 32 | 6 | **2** | SEM ×1.4 |
| 23 | tail-tip lateral excursion ÷ SVL | 0.10 | 0.03 | 1 | **figure-read, soft bound** |
| 24 | d(tail lateral)/d(speed), normalised slope | −1.0 | 0.6 | 1 | **sign is published, magnitude invented** |
| 25 | d(tail height)/d(speed), normalised slope | 0.0 | 0.3 | 1 | published null result |
| 26 | mean forward speed (m/s) at v_cmd = 0.13 | 0.129 | 0.020 | 1 | published SEM |

**σ values marked "invented" or "figure-read" are mine, not published.** Keep them in a single dict so they can be revised in one place.

**Objective:**

```
J = Σ_i w_i · min(z_i², 9) / Σ_i w_i
```

The `min(·, 9)` clip (3σ) stops one catastrophically-wrong metric from swamping the gradient early in the search — CMA-ES needs to see improvement in the other 25. **CMA-ES minimises J.**

**Hard gates (boolean vetoes — if any fails, return J = 10.0 regardless):**

- **G1** belly touch sensors never fire. *(The paper states the body "did remain elevated above the substrate.")*
- **G2** all four feet make contact every stride — no dragged or hopping limb.
- **G3** no flip: trunk up-vector z > 0.5 throughout.
- **G4** mean speed inside **0.028–0.35 m/s** (the full published voluntary range).
- **G5** tail never contacts the ground.
- **G6** stride-period coefficient of variation inside **0.03–0.30**. *(Both bounds invented — see §6. This catches a metronomic CPG, which is a visible tell.)*

**Ablation gates — run the identical protocol with `tail_bend_L/R` clamped to zero, all tail mass and inertia retained, `tail_lift` free.** This is the graphite-rod experiment, and it is free.

| Gate | Requirement | Published |
|---|---|---|
| **A1** | `hip_sprawl` excursion drops **40–60 %** | 52.4° → 30.6° |
| **A2** | `hip_proret` excursion drops **15–25 %** | 82.6° → 65.3° |
| **A3** | knee excursion drops **8–14 %** | 98.8° → 88.1° |
| **A4** | ankle excursion drops **14–20 %** | 85.2° → 70.4° |
| **A5** | hind step length drops **15–20 %** | 0.06 → 0.05 SVL (use the **ratio only** — the absolute is arithmetically ambiguous in the source) |
| **A6** | hind **duty factor UNCHANGED** within ±0.02 | 0.78 → 0.78, P = 0.583 |
| **A7** | **forelimb excursions UNCHANGED** within ±10 % | all P ≥ 0.052 |

**A6 and A7 are the negative controls, and they are the ones that catch a fake.** A model that simply couples everything to the tail will fail them. If the tail-frozen run looks like the intact run, the tail is a passive pendulum and the coupling is missing. **This test alone distinguishes a gecko-shaped controller from a generic quadruped one, and it requires no video.**

**What CMA-ES should search over** (~22–26 parameters, comfortable for popsize 14–20 on CPU):

- frequency schedule `f = a + b·v` (2)
- `stance_ratio_hind`, `stance_ratio_fore` (2)
- fore-vs-hind phase offset (1)
- per-joint CPG amplitude: `hip_proret`, `hip_sprawl`, `knee`, `ankle`, `shoulder_proret`, `shoulder_sprawl`, `elbow` (7)
- per-joint CPG phase offset, **seeded from the EMG percentages** in §2C rows 22–23 and allowed to drift ±10 % (7)
- `spine_bend` amplitude (1)
- `tail_bend` amplitude schedule `A = c − d·v` (2)
- `tail_lift` constant setpoint (1)
- global actuator `kp` and damping scale (2)

**Compute budget:** 4 speeds × 20 s = 80 s of simulated time per candidate. Parallelise across cores; a popsize-16 generation is 16 candidates. Trim to 3 speeds × 15 s if generations are too slow — the phase and duty-factor metrics are the least data-hungry, and the two r² regression metrics are the ones that need the multiple speeds.

**What a passing score looks like:**

| Tier | Criterion | Meaning |
|---|---|---|
| **Tier 1 — "gecko-shaped"** | J ≤ 2.0, all hard gates G1–G6 | The average metric sits within ~1.4σ of its published mean. The gait is recognisably this species. |
| **Tier 2 — "matches the literature"** | **J ≤ 1.0**, all hard gates, **all ablation gates A1–A7** | The average metric sits **exactly 1σ from the published mean — i.e. inside the published SEM/SD band.** This is the target. Claim the sim is calibrated to the literature only at Tier 2. |
| **Tier 3 — "indistinguishable on this scorecard"** | J ≤ 0.5, all gates | Diminishing returns; further tuning is fitting noise in σ values that are partly invented. |

**Report, but do not optimise:** COM phase shift and percent recovery (metrics 21–22) are already in the objective at weight 2, which is right — but also print them separately every generation, because they are the one pair that no joint-angle reward can fake and they are the best single indicator that the gait is *mechanically* a gecko walk rather than a kinematic imitation of one.

**And print, always, alongside J:** the sim's speed in **both m/s and SVL/s**, and the calibration scenario's temperature and friction. Every target in this document is valid only for a **warm (26–35 °C) gecko on flat, high-friction ground.** Reporting in SVL/s is also what prevents the "0.2–1.1 m/s" class of unit error from ever recurring.

---

## 8. Reading list

Ranked by value to this build. **"Free"** means I read the full text or the specific table at that URL this session.

**Tier 1 — read these five; they contain ~80 % of the scorecard**

1. **Jagnandan, K. & Higham, T. E. (2017).** *Lateral movements of a massive tail influence gecko locomotion: an integrative study comparing tail restriction and autotomy.* **Scientific Reports 7:10865.** — **FREE, fully open access (CC-BY).** The single richest table in the literature: Table 1 gives stride length, step length, stance time, duty factor and twelve joint-angle maxima and excursions for fore- and hindlimb across three tail treatments, n = 10 [EM]. It is also the source of the tail-restriction ablation that becomes your §7 gate list.
   → https://pmc.ncbi.nlm.nih.gov/articles/PMC5589804/ (author PDF: https://biomechanics.ucr.edu/Jagnandan_Higham_2017.pdf)

2. **Jagnandan, K., Russell, A. P. & Higham, T. E. (2014).** *Tail autotomy and subsequent regeneration alter the mechanics of locomotion in lizards.* **J. Exp. Biol. 217:3891–3897.** — **FREE author PDF.** The morphology bible: tail mass 22 %, tail volume, **centre of mass at 65.9 → 52.7 → 60.5 % SVL**, hip height 0.15 SVL, shoulder height 0.11 SVL, body pitch 3.72°, plus a second independent duty-factor and joint-angle cohort and the ground-reaction-force results. This is the paper that gives you the free MJCF mass audit.
   → https://biomechanics.ucr.edu/Jagnandan_etal_2014.pdf

3. **Jagnandan, K. & Higham, T. E. (2018).** *Neuromuscular control of locomotion is altered by tail autotomy in geckos.* **J. Exp. Biol. 221:jeb179564.** — **FREE author PDF.** In-vivo EMG from five muscles, giving burst **onset, duration and peak time as percentages of the stride relative to footfall.** **This is the only published source that constrains CPG PHASE rather than just amplitude**, and it is what turns a joint-range table into an actual controller.
   → https://biomechanics.ucr.edu/Jagnandan-Higham-JEB-2018.pdf

4. **Fuller, P. O., Higham, T. E. & Clark, A. J. (2011).** *Posture, speed, and habitat structure: three-dimensional hindlimb kinematics of two species of padless geckos.* **Zoology 114:104–112.** — Nominally Elsevier-paywalled but **the author's copy is freely hosted by the Higham lab.** The only source for hindlimb **segment lengths**, hip height scaled to limb length, absolute **stride frequency 2.03 Hz**, stride length 8.74 cm, the internally-consistent (frequency × length = speed) triple, and the "hyper-sprawled, knee above hip" posture finding.
   → https://biomechanics.ucr.edu/Fuller_etal_2011.pdf

5. **Frýdlová, P. et al. (2026).** *Response of leopard geckos (Eublepharis macularius) towards a multimodal cue simulating a predator.* **Behavioral Ecology 37(4):arag062.** — **FREE, open access.** Modality-by-modality dissection of fear in n = 42 [EM] with response probabilities and tongue-flick counts. **This paper alone should dictate the sensory weighting in `drives.py`** — including the finding that a live snake seen through glass barely raises defence above baseline.
   → https://pmc.ncbi.nlm.nih.gov/articles/PMC13294456/ · publisher: https://academic.oup.com/beheco/article/37/4/arag062/8702772

**Tier 2 — read when you get to that subsystem**

6. **Krönke, F. & Xu, L. (2023).** *Sensory Stimulation as a Means of Sustained Enhancement of Well-Being in Leopard Geckos.* **Animals 13(23):3595.** — **FREE.** The full **31-category ethogram with definitions**, 378 h of focal observation on n = 18, the behaviour budget (Table 4, Figure 9), the 5-minute observation quantum, the 6–11 pm activity window. The primary scorecard for the brain, and the source of the codeable "one body length" motion thresholds.
   → https://pmc.ncbi.nlm.nih.gov/articles/PMC10705344/

7. **Cooper, W. E. Jr., DePerno, C. S. & Steele, L. J. (1996).** *Effects of movement and eating on chemosensory tongue-flicking and on labial-licking in the leopard gecko.* **Chemoecology 7:179–183.** — **FREE author PDF.** The only source giving absolute tongue-flick and labial-lick **rates per second** for [EM] as a function of locomotion and feeding state, n = 16, plus the 180 s chemosensory lag.
   → https://faculty.cnr.ncsu.edu/christophersdeperno/wp-content/uploads/sites/8/2016/01/PR8EublepharisChemoecology.pdf

8. **Masseck, O. A., Röll, B. & Hoffmann, K.-P. (2008).** *The optokinetic reaction in foveate and afoveate geckos.* **Vision Research 48:842–849.** — **FREE author PDF.** The **only paper with quantitative sensorimotor measurements on [EM] itself**: hOKR gains at 20/30/40°/s, complete naso-temporal monocular asymmetry, and the ~80 % head / 20 % eye split. Everything in §3's gaze-stabilisation row comes from here.
   → https://homepage.ruhr-uni-bochum.de/klaus-peter.hoffmann/pdf_hoffmann/masseck_vis_res_08.pdf

9. **Vollin, M. F. & Higham, T. E. (2023).** *The tailless gecko gets the worm: prey type alters the effects of caudal autotomy on prey capture and subjugation kinematics.* **Frontiers in Behavioral Neuroscience 17:1173065.** — **FREE, fully open access.** Strike velocity and distance for evasive vs non-evasive prey, prey-shake velocity/amplitude/frequency, the aerial-time regression, and the explicit two-mode approach description. **[CV] proxy** — read the scaling warning in §6 first.
   → https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2023.1173065/full · mirror https://pmc.ncbi.nlm.nih.gov/articles/PMC10484749/

10. **Landová, E. et al. (2016).** *Antipredatory reaction of the leopard gecko Eublepharis macularius to snake predators.* **Current Zoology 62(5):439–450.** — **FREE.** n = 585 geckos, 30-min trials: absolute counts of tongue-flicks, tail waving, freezing, defensive postures and binocular fixation against nine snake species and a harmless control. **The best-powered behavioural dataset on the species** and the only calibration source for a fear/danger drive over a long horizon.
    → https://academic.oup.com/cz/article/62/5/439/2196971

**Tier 3 — specific single numbers**

11. **Usherwood, J. R. & Self Davies, Z. T. (2017).** *Work minimization accounts for footfall phasing in slow quadrupedal gaits.* **eLife 6:e29495.** — **FREE.** Independent confirmation of [EM]'s duty factor (median 0.79) and limb phase (median 43 %) from a different lab and method, plus the general law `phase% = 130·DF − 66`.
    → https://pmc.ncbi.nlm.nih.gov/articles/PMC5599235/

12. **McElroy, E. J., Hickey, K. L. & Reilly, S. M. (2008).** *The correlated evolution of biomechanics, gait and foraging mode in lizards.* **J. Exp. Biol. 211:1029–1040.** — Closed access, but Table 1 has [EM] directly: duty factor, limb phase, **COM phase shift, percent recovery, PE/KE** for both walking and running mechanics from force-plate data. The definitive source for §2C row 24 and for the "runs at high duty factor and low speed" finding.
    → https://journals.biologists.com/jeb/article/211/7/1029/18146/

13. **Autumn, K., Weinstein, R. B. & Full, R. J. (1994).** *Low cost of locomotion increases performance at low temperature in a nocturnal lizard.* **Physiological Zoology 67:238–262.** — **FREE PDF.** Cost of transport, maximum aerobic speed, and the **endurance-vs-speed equations** that should replace the flat energy drain. **[TP] proxy.**
    → https://polypedal.berkeley.edu/publications/039_Autumn_LowCostofLocomotionIncreasesPerformancePhysiolZool_1994.pdf

14. **Ha, J. et al. (2017).** *Behavioral hypothermia of a domesticated lizard under treatment of the hypometabolic agent 3-iodothyronamine.* **Experimental Animals 66(2):99–105.** — **FREE, machine-readable PDF.** **The only direct metabolic-rate and preferred-body-temperature measurement ever made on [EM]** (RMR 0.075 mL O₂ g⁻¹ h⁻¹, PBT 29.5–31.9 °C, ventilation 11.6/min).
    → https://www.jstage.jst.go.jp/article/expanim/66/2/66_16-0070/_pdf

15. **Russell, A. P., Lai, E. K., Powell, G. L. & Higham, T. E. (2014).** *Density and distribution of cutaneous sensilla on tails of leopard geckos in relation to caudal autotomy.* **J. Morphol. 275:961–979.** — **FREE author PDF.** Tail mechanoreceptor density (59–76/mm² dorsal tail vs 33/mm² best body site) and the 1:1 scale-whorl-to-fracture-plane mapping. **Directly motivates the tail touch sensors the model lacks.**
    → https://biomechanics.ucr.edu/Russell_etal_2014.pdf

16. **Rawat, Y. B., Thapa, K. B., Bhattarai, S. & Shah, K. B. (2019).** *First records of the Common Leopard Gecko, Eublepharis macularius, in Nepal.* **Reptiles & Amphibians 26(1):58–61.** — **FREE.** The **only source with a complete head, eye and tail morphometric table** for wild adults: head length/width/height, eye diameter, snout-eye, interorbital, tail width and length. n = 2 — small, but it is all there is.
    → https://journals.ku.edu/reptilesandamphibians/article/view/14342

17. **Roth, L. S. V. et al. (2009).** *The pupils and optical systems of gecko eyes.* **Journal of Vision 9(3):27.** — **FREE PDF.** Focal length 3.5 mm, pupil area range 100–150×, cone outer segments, multifocal zones, and the 0.002 cd/m² colour-vision threshold. **[TC] proxy** — everything optical in §3 rests on this and no eublepharid equivalent exists.
    → https://pdfs.semanticscholar.org/97f8/45c73bf60cb196dd2e60521aea52efff691a.pdf

18. **Arnold, P., Fischer, M. S. & Nyakatura, J. A. (2014).** *Soft tissue influence on ex vivo mobility in the hip of Iguana.* **J. Anatomy 225:31–41.** — **FREE.** The best available published anchor for the repo's transferred joint ranges: in-vivo hip excursions of 79.3° pro/retraction, 71.1° ab/adduction and 25–30° long-axis rotation, plus cadaveric limits with soft tissue progressively removed. **[II] — not a gecko.** But it is what vindicates two of your three hip ranges and condemns the third.
    → https://pmc.ncbi.nlm.nih.gov/articles/PMC4089344/

**One paper worth chasing down if you ever get institutional access:**
**Zaaf, Van Damme, Herrel & Aerts (2001), J. Exp. Biol. 204:1233–1246.** It is the one study that measured duty factor and stride length as explicit *functions* of speed in this species — exactly the regression coefficients §6 lists as missing. It is closed-access with no repository copy anywhere (OpenAlex `oa_status: closed`, no PMC, no preprint, publisher behind Cloudflare). It is also the origin of the "0.2–1.1 m/s" figure that must not be used as a walking target.

---

## Appendix: the six things to do first

1. `STANCE` 0.62 → **0.765** (hind only; leave `FRONT_STANCE` at 0.70). One line. Biggest single effect.
2. `PHASE` → `{"HL": 0.00, "FL": 0.44, "HR": 0.50, "FR": 0.94}`. One line.
3. Make `FREQ_HZ` a function of commanded speed: **1.4 Hz @ 0.11 m/s → 2.03 Hz @ 0.18 m/s.**
4. Write the **morphology audit test** (§7.1). It runs in a second, needs no training, and the CoM check alone will tell you whether the body is salvageable.
5. Run the **tail-restriction ablation** (§7.2, gates A1–A7) against the *current* controller, before changing anything. It costs nothing and tells you immediately whether the tail in the model is doing real work or is a passive pendulum.
6. Rename `drives.py`'s `energy` to `fatigue`, set its drain to **0.00136 /s**, and move `hunger` out of the per-step loop entirely.
