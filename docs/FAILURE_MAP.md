# NeuroGecko — the map

**One document. Everything tried, everything that failed, why, and what fixed it.**
Past, present, future. Updated every session; nothing removed.

Last updated: Session 8c.

---

## Why this document exists

A model that only records its successes is a sales pitch. This project's own
research corpus calls the evidence ledger *"the most valuable part of the
record — it is the map of where **not** to look."*

Every entry below is a claim that was **measured**, not argued. A refuted entry
is not a mistake to be embarrassed by; it is a piece of the map that is now
drawn. The expensive thing is not being wrong — it is being wrong twice.

**Rule: nothing is deleted from this file.** A hypothesis that was refuted and
later turned out to be right gets a second row, not an edit.

---

## Where the project is

| Stage | State |
|---|---|
| **Research** — 167 agents, 276 published measurements, fact-checked | ✅ done |
| **Body** — 38 g lab morphology, 14/14 static checks | ✅ done |
| **Walking** — base controller, 4/6 gates, accepted | ✅ done |
| **World** — no cheat, textured floor, fleeing prey, **shelter + warm surface + threat** | ✅ done |
| **Brain 1/8 — hypothalamus** (hunger, energy, fatigue, thermostat) | ✅ done |
| **Brain 2/8 — basal ganglia** (action selection) | ✅ reproduces the published table, mean error 0.10 pp |
| **Brain 3a/8 — spinal rhythm** (the leg oscillators) | ✅ drives the walker, every gate unchanged |
| **Brain 3b/8 — brainstem** (decision → command) | ✅ built; thinnest evidence in the project, declared |
| **Brain 4/8 — eye** (retina, pretectum, tectum) | 🟡 gaze reproduces; **prey-finding NOT ACCEPTED** |
| **Strike** — the thing that actually catches prey | ✅ built, 81 % capture vs published 82.9 %. **The approach cannot reach it** — #174 |
| **Smell** — the channel this species is documented to gate on | ❌ absent |
| **Brain 6/8 — day/night clock** | ✅ circadian gate. Sleep architecture left open — #198 |
| Brains 7–8 — memory, learning | ❌ barely measured in this animal |
| **Proof** — 15-test battery | 🟡 1 run (A3, failed then fixed) |
| **Gate contracts** | ✅ 16/16. One check compared bytes instead of physics and masked seven others — #167, #168 |

**Roughly 50 % done.** The irreversible parts — the research and the body — are
behind us.

---

## What works — the successes

The ledger below is mostly failures, because that is what a map of where not to
look is made of. This is the other half: what is standing, and what it is
checked against.

| What | Checked against | Status |
|---|---|---|
| **Research corpus** — 276 published measurements, 167 agents, fact-checked | the literature | standing |
| **Provenance registry** — every number tagged published / derived / invented, with species and source | itself, by test | standing |
| **Body** — 38 g lab morphology | 14 of 14 static anatomical checks | standing |
| **Walker** — hand-written CPG + stance compensator | 4 of 6 published gait gates | accepted |
| **Tail coupling** — blocking the tail collapses the hind legs | published ablation **−21% / −17%**; ours **−20.7% / −16.5%** | reproduced |
| **Limb phase = late touchdown** | 0.320 − 0.123 = **0.197**, exactly the measured error | confirmed |
| **No-cheat world** — no privileged channels, real textured floor, fleeing prey | the animal's own senses | standing |
| **Hunger** — from the animal's own metabolism | **70%** hungry at the published 2.33-day feeding interval, unfitted | reproduced |
| **Energy budget** — metabolism and feeding schedule agree | 79 h per meal against a 56 h interval, from **separate papers** | reproduced |
| **Fatigue** — from the published endurance curve | >60 min at the published speed | reproduced |
| **Selection** — Gurney 2001b selection sequence | published output values to **four decimal places** | reproduced |
| **Dopamine** — the published disembodied sweep | akinesia **exact**; peak clean selection **79.9 vs 78.6** | reproduced |
| **The gecko decides** — six behaviours, sigma-pi salience | behaviour, by test | standing |
| **Brains 1 + 2 running together** in the live simulation | the seam, by test | standing |
| **385 tests**, no expected failures | themselves | green |

**The single best result** is the energy budget. Metabolism and the feeding
schedule were measured by different scientists, decades apart, for unrelated
reasons. One meal covers 79 hours; geckos are fed every 56. Nobody arranged
that agreement — it is the model's own physiology coming out right.

---

## The flow — every session, start to now

Where the work went, in order, and where it stopped each time. **71 commits.**

### Phase 1 — locomotion, before the ledger existed

| | What was done | Outcome |
|---|---|---|
| V4.2.2–4.2.9 | front-foot correction, front-load controller, escape-proof gate, knee residual cap, duty reward gate, contact reflex curriculum | iterative, no accepted base |
| V4.3–4.6 | velocity tracking, matched stance cadence, front de-stick | each refuted as the fix |
| V4.5B | — | **accepted as the locomotion champion** |

### Phase 2 — the first brain, since superseded

| | What was done | Outcome |
|---|---|---|
| AGWM | brain environment wiring, trainable visual policy, curriculum, privileged food taper, mouth-distance reward, oracle eval, recurrent PPO, behaviour cloning, DAgger, visual distillation, drive-based controller | produced the recovered brains; the drive module here is the one `hypothalamus.py` later replaced |

### Phase 3 — rebuilding on evidence

| Session | What was done | Where it stopped |
|---|---|---|
| recovery | verified the cloud model lineage before touching anything | lineage confirmed |
| baselines | measured baselines, camera isolation, verified checkpoint recovery | — |
| morphology | constrained lab morphology, shared gait calibration | — |
| step 0 | recorded true 250 Hz physics | **reported failed entrainment honestly** rather than hiding it |
| step 1–2 | charge effort on actuator work; force and collision diagnosis | 19 force-only tests |
| step 3 | improve the lab base | **stopped training on failed gates** |
| **3** | hind stance compensation, implemented and tested | Gate 2 still failed |
| **3b** | generalised compensation to all four limbs | **Gate 2 reaches 4/6** |
| **3c** | limb phase identified as late touchdown | stride length judged geometrically unreachable |
| **3d** | stride length is speed; spine refuted as its source | 41% hind foot slip found |
| **3e** | gearing and contact softness refuted | corrected my own framing of stride length |
| **3f** | CMA-ES appears to solve the limb-phase gate | 6/6 declared unreachable |
| **3g** | **the 3f claim WITHDRAWN** — it had optimised the wrong metric | a success retracted, not buried |
| **3h** | fit harness rescored through `realism_metrics` | a 12 s fit rejected as unstable |
| **3i** | gate-count objective, 20 s fit evidence | **searches stopped** — the space was exhausted |
| handoff | `docs/HANDOFF.md`, the full record of research and build | — |

### Phase 4 — training, body, and the world

| Session | What was done | Where it stopped |
|---|---|---|
| **4** | trained the walker, 6M steps over two runs | **the forefoot gate is unreachable on this body** — it needs a wrist joint the animal lacks |
| **4b** | drove the sprawl degrees of freedom; coupled the tail to the hindlimb | the tail ablation **reproduces published values** |
| **4c** | the no-cheat world; prey that flees | the walker **cannot catch it, and is not meant to** — real geckos strike |
| **4d** | finished the no-cheat work in the environment that has eyes | a privileged channel had survived in the other environment |

### Phase 5 — the first brain module

| Session | What was done | Where it stopped |
|---|---|---|
| **5** | hypothalamus: hunger from the animal's own metabolism | reproduces the scorecard four ways |
| **5b** | fatigue from the published endurance curve | **fear is wired to the wrong sense** — still blocked |

### Phase 6 — the second brain module

| Session | What was done | Where it stopped |
|---|---|---|
| **6** | basal ganglia built from the equations | **shipped NOT ACCEPTED** — the sweep failed and the commit said so |
| **6b** | found the parameter table; two of three errors were structural | 5 of 7 properties |
| **6c** | **this document created**; 6b's parameters turned out to be a variant's | the oscillation fixed — it had three causes |
| **6d** | scouted module 3 with six agents | **the reported failure was not a failure** — the comparison had no published counterpart |
| **6e** | found the authors' own source on disk | **the whole model was the wrong one**; reimplemented and reproduced |
| **6f** | migrated the gecko onto the validated model | brains 1 and 2 **running together in the simulation** |

### Phase 7 — the eye, and then the literature (Sessions 8–9)

The retina and pretectum landed and the pretectum reproduces the one published
sensorimotor measurement that exists for this species. The tectum landed and
**does not work**: it finds a synthetic cricket perfectly and a real one not at
all. Session 8d tried to fix that with better optics and failed.

Session 9 stopped building. Six literature searches asked what a real leopard
gecko actually does and senses, and refuted sixteen of seventeen assumptions —
including that the animal chases prey, that it basks, that it flees what it
sees, and that stationary prey is invisible to a visual predator. Ledger
133–149.

### ← Where we are paused, right now

- **Working tree is uncommitted.** Five modified files from session 8d plus the
  session 9 provenance correction and this map entry.
- **8 tests fail** — `test_session2_gate.GateContracts`. They fail in a clean
  worktree at `e4196d8` too, so they pre-date session 8d. Earlier sessions
  reported 456 passing; **that report was wrong.** Ledger #149.
- **Underneath it:** the sessions 3–4 evidence records a body hash no committed
  version of `gecko_body_lab_v2.xml` produces. The accepted 4/6 walker's
  evidence points at a body that is not in the repository. **Unresolved, and
  everything else stands on the locomotion work.**
- **Brain 3 is finished.** The cord drives the walker with every gate
  unchanged; the brainstem turns a decision into a stride frequency and a turn.
- **Brain 4 is half done.** Gaze reproduces. Prey-finding does not, and session
  9 explains why: the prey never moves and is off-image 81 % of the time, so
  the detector was never the problem.
- **What is now known to be missing:** a strike. The animal walks at 0.055 m/s
  and the cricket escapes at 0.093–0.143 m/s, so pursuit cannot work and the
  real animal does not attempt it — it strikes from ~2 cm at 0.851 m/s. Until a
  strike exists the hunting channel cannot succeed by any route.
- **Next decision, not yet taken:** chase the body-hash mismatch first, or
  build the world and the strike and come back to it.

---

## The five rules this project runs on

Each was learned by getting it wrong. They are the real output of the ledger.

1. **Every number is published, derived, or INVENTED — and says which.**
   A plausible number wearing a published number's clothes is the worst defect
   available, because nothing downstream can detect it.
2. **Never tune until the answer matches.** If a model needs its constants
   adjusted to reproduce a published result, the constants must come from
   somewhere else — a paper, not a search.
3. **The reward is what the model is told to want. The gates are what the
   animal does.** They diverge. Only the second counts.
4. **A failed prediction stays in the table.** Adding a knob per failing joint
   until the table goes green is fitting the harness to the answer.
5. **Measure before claiming, including your own tests.** Four entries below are
   failures of the *test*, not the model.

---

## Ledger — the walking phase (Sessions 1–3)

The forefoot problem: the front feet land late and carry too little weight. Ten
hypotheses, eight refuted.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | Hind stance height causes the duty failure | **Confirmed** | compensator: 0.641 → 0.733 |
| 2 | Limb phase = fore/hind touchdown-lateness difference | **Confirmed** | 0.320 − 0.123 = 0.197, exactly the measured error |
| 3 | Hind stance height causes the *phase* failure | **Refuted** | fixing it moved phase the wrong way, 0.577 → 0.607 |
| 4 | Fore stance height causes it | **Refuted** | forelimb table nearly flat; front load fell 0.611 → 0.584 |
| 5 | Actuator bandwidth is the cause | **Refuted** | stiffening cut elbow lag 17.0° → 2.9°; phase moved 0.6347 → 0.6283 |
| 6 | Spine bending supplies the missing stride | **Refuted** | 4× amplitude: +6.5 % excursion, **−17 % speed** |
| 7 | Fore/hind gearing mismatch | **Refuted** | real, predicts the slip, but equalising it is slower at every step |
| 8 | Contact softness lets the feet creep | **Refuted** | impratio 1 → 100: slip 13.5 → 12 mm. Required μ ≈ 0.01 against a floor of 0.9 |
| 9 | Commanding an earlier touchdown fixes phase | **Refuted** | commanding 0.197 earlier gave 0.688 — worse |
| 10 | Parameter search can reach 6/6 | **Refuted** | ~200 generations, four searches; 23.20 → 22.92, tiebreaker only |

---

## Ledger — training (Session 4)

**6 million steps. Two runs. Neither beat the hand-written base.**

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 11 | A learned residual can fix the forefoot | **Refuted** | 2 × 3.01 M steps; best trained 3/6 against base 4/6 |
| 12 | Trunk pitch lifts the shoulder | **Refuted** | measured **3.87°** against a published 3.72 ± 0.61 — it was correct all along |
| 13 | Elbow authority is the missing lever | **Confirmed** | front load 0.584 → **0.674**, unreachable with the elbow locked |
| 14 | …and it can be had without cost | **Refuted** | every passing row had stride CV 0.16–0.38 against a 0.10 ceiling |
| 15 | Pressing harder in the base fixes it | **Refuted** | 0.584 → 0.625 then collapse; the press is *overwritten* during stance |
| 16 | Aiming the frozen pose deeper fixes it | **Refuted** | flat front-only, worse uniform, across −0.6 to −7.0 mm |
| 17 | The walking posture rides too high | **Confirmed** | shoulder 0.125 SVL against a published 0.11; forefoot 3.1 mm airborne |
| 18 | **The forelimb is under-actuated** | **Confirmed** | **1 free joint against the hindlimb's 2** |

**#12 is the expensive one.** That hypothesis had been carried for three sessions
and steered the work. Nobody had measured it. One measurement killed it.

**#18 is the answer.** One joint cannot set foot height *and* foot position
independently. It is a body limit, not a tuning failure — which is why 6 M steps
of training could never have found it.

**Decision: 4/6 accepted as the walker.** Both unmet checks are engineering
targets tagged `species: INVENTED`, not published biology.

---

## Ledger — body motion (Session 4b)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 19 | The limbs are barely articulated | **Confirmed** | 6 of 8 excursions at 0.00–0.37 of published |
| 20 | Sprawl can reach the published excursion | **Confirmed** | 55.25° against a published 52.42 |
| 21 | …without losing the planted foot | **Refuted** | hind duty **0.631 → 0.094** across the sweep |
| 22 | A sprawl-aware stance table rescues it | **Refuted** | ±15° solvable — half what is needed — and 2/6 at zero amplitude |
| 23 | The tail was a passive pendulum | **Confirmed** | every hindlimb excursion moved **< 1 %** when tail drive was removed |
| 24 | A caudofemoralis coupling reproduces the ablation | **Confirmed** | **−20.7 % / −16.5 %** against a published −21 % / −17 % |
| 25 | One gain reproduces the whole per-joint pattern | **Refuted** | knee +3.8 % against a published −11 % |

**#24 is the best result in the project.** Blocking the tail now collapses the
hind legs the way it does in a real gecko — and the intact animal is *bit-identical*
to before, proven to twelve decimal places.

**#25 stayed in the table.** Per-joint gains would have made it green. That is rule 4.

---

## Ledger — the world (Sessions 4c–4d)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 26 | The food vector is the only privileged channel | **Refuted** | the controller was steered by the same bearing; both had to go |
| 27 | The floor needed a texture | **Refuted** | it had one — at 0.64 squares per body length. The **scale** was the fault |
| 28 | ~70° camera is a guess | **Refuted** | it makes the 1.6° tracked dot span 1.46 px where 120° gives 0.85 |
| 29 | The corpus specifies the camera | **Refuted** | five incompatible recommendations, all INVENTED |
| 30 | Cricket escape speed is published | **Refuted** | absent — only dash length and duration, never divided |
| 31 | A walking gecko can catch fleeing prey | **Refuted** | 0.055 m/s against 0.118 m/s; never captured in 40 s |
| 32 | A strike can be scored at the control rate | **Refuted** | a 16–20 ms strike is **under one 50 Hz step**; 500 Hz needed |
| 33 | The no-cheat work landed where it mattered | **Refuted** | the vision env had its own privileged channel, untouched |
| 34 | The eat radius was roughly right | **Refuted** | 0.10 m against a published **4.07 cm** strike distance |
| 35 | Scaling the cheat to zero removes it | **Refuted** | five slots remain in the observation |
| 36 | The camera saw a real food object | **Refuted** | a green sphere **painted on after rendering** |
| 37 | The brain evidence files are reproducible | **Refuted** | both pin a body hash no file in the repo has |

**#31 is a design constraint, not a bug.** Real geckos catch 82.9 % of crickets by
**striking at 0.851 m/s from 2 cm** — not by chasing. No hunting gate means
anything until a strike exists.

**#33 was my own claim of completion, caught by an audit.** I said the cheat was
gone. It was gone from the *walking* environment, not the one with eyes.

---

## Ledger — hypothalamus (Sessions 5–5b)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 38 | The old drives were roughly right | **Refuted** | hunger saturated in **67 s** where the animal takes days |
| 39 | Anchoring the allometry at 40 g is fine | **Refuted** | inflates metabolism 17 %; measured mass is **19.5 g** |
| 40 | Hunger measured against the energy reserve | **Refuted** | 1.7 % per inter-meal interval — the animal would never eat |
| 41 | Hunger measured in **meals** matches the animal | **Confirmed** | **70 %** at the published 2.33-day interval, unfitted |
| 42 | The metabolism and the feeding schedule agree | **Confirmed** | 79 h per meal against a 56 h interval, from separate papers |
| 43 | A clipped drive is harmless | **Refuted** | zero reward for eating when several meals in debt |
| 44 | n = m is an acceptable drive exponent | **Refuted** | drive goes linear; a meal worth the same at any hunger |
| 45 | The thermostat can be built now | **Refuted** | no temperature field exists anywhere |
| 46 | Session 5's module was complete | **Refuted** | fatigue absent; the second channel carried no information |
| 47 | The endurance velocity is in m/s | **Refuted** | gives 15 h at 0.050 m/s against a published **>60 min** |
| 48 | A walking gecko accumulates fatigue | **Refuted** | 0.055 m/s is under the aerobic ceiling — sustainable indefinitely |
| 49 | The danger channel drives fear correctly | **Refuted** | it is **mechanosensory**, and mechanosensory-alone defence is **0** |
| 50 | `curiosity` had no published basis | **Refuted** | tongue-flick 3.60×, ceiling 0.146/s — right to delete, wrong about why |

**#42 is the finest moment in the project.** Metabolism and the feeding schedule
were measured by different scientists, decades apart, for unrelated reasons. One
meal covers 79 hours; geckos are fed every 56. Nobody arranged that.

**#40 → #41 is the most instructive pair.** Two hungers exist. Between meals a
gecko burns **1.7 % of its tail reserve** but **71 % of a meal**. Measured against
the tail, the animal is never hungry. Measured in meals, it gets hungry exactly on
schedule. Same physiology, one wrong denominator.

**#49 is unresolved and serious.** The gecko is frightened by the one sense that
provably does nothing to it.

---

## Ledger — basal ganglia (Sessions 6–6b)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 51 | The architecture can be built from the corpus alone | **Refuted** | the weights are not in it |
| 52 | Loop gain 1.0 is acceptable in positive feedback | **Refuted** | oscillated at zero input, gates cycling 0 → 0.34 |
| 53 | Diffuse drive can be summed regardless of channel count | **Refuted** | effective gain scales with N |
| 54 | Averaging the diffuse drive fixes it | **Refuted** | cures oscillation, **destroys discrimination** |
| 55 | Self-calibrating the gate is safe | **Refuted** | collapsed the range; every dopamine level identical |
| 56 | Selection is disinhibition, not a maximum | **Confirmed** | losers are *partially* released; a WTA cannot do that |
| 57 | Sigma-pi salience gates hunting on hunger AND prey | **Confirmed** | a sated gecko ignores visible prey |
| 58 | The GPR parameters are unobtainable | **Refuted** | Table 5 of arXiv cs/0601004, plus c = 0.169 from a CC BY paper |
| 59 | The oscillation was an unfixable structural flaw | **Refuted** | it was the **slopes**: VL 0.62 keeps the loop gain under one |
| 60 | The striatum is driven by cortex | **Refuted** | Eq 14 drives cortex from thalamus *alone*; salience goes direct |
| 61 | 825 switches per 120 s was a model failure | **Refuted** | 50 Hz white-noise salience — the test measured its own noise |
| 62 | Distortion can be counted | **Refuted** | the count saturates; deepening needs a continuous measure |
| 63 | Published parameters make the sweep reproduce | **Partly** | 5 of 7 properties; the switching ratio stays inverted |

**#51 → #58 is the lesson of the session.** I declared a wall, wrote it up
honestly, and the wall was not there — the parameters were in an open-access
paper. **Recording a blocker is not the same as verifying one.**

**#59 is the sharpest finding.** The oscillation I could not fix, and had called
structural, was one number: the output slopes are not all 1, and the 0.62 on the
thalamus is exactly what holds the loop below instability.

**#60 was worse than any constant.** I had the wiring backwards. No parameter
would have fixed it.

**#63 was superseded within the session** — see below. The parameters it
called published were a re-tuned variant's.

---

## Ledger — basal ganglia, second correction (Session 6c)

A background search finished after Session 6b was committed and showed that
6b's "fix" was **half a regression**. It had found a real bug and introduced a
new one.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 64 | Girard 2005 Table 5 holds the GPR parameters | **Refuted** | it is a re-tuned *robotic variant*; six independent sources give 0.9 / 0.3, not 0.8 / 0.4 |
| 65 | The oscillation at zero salience had one cause | **Refuted** | it had **three**, and each hid the next |
| 66 | The caller's control rate can be the solver's rate | **Refuted** | at 50 Hz the resting output oscillates over **0.143–0.205**; at 200 Hz it is exact |
| 67 | The diffuse STN drive caused the instability | **Refuted** | the drive is published and deliberate; the **integration step** was the fault |
| 68 | A guard pinned to one publisher's name is a good guard | **Refuted** | it failed when the citation was *corrected*, which is backwards |
| 69 | The sweep evidence was reproducible | **Refuted** | generated by a throwaway script; could not be re-run when parameters changed |
| 70 | Canonical parameters make the sweep reproduce | **Partly** | 6 of 7; switching ratio improved 0.3× → 0.55×, still inverted |

### #64 — the arithmetic that settled it

Prescott 2024 does not merely *state* the gating constant c = 0.169. It
**defines** c as the model's own resting output. That makes c an independent
test of the weights, and the two candidate sets are not close:

| weight set | resting output | matches published c = 0.169? |
|---|---|---|
| **0.9 / 0.3** (canonical) | **0.16953** | **yes, to three decimals** |
| 0.8 / 0.4 (Girard variant) | 0.14286 | no, at any channel count |

The published gating constant and the published weights are **the same fact
seen twice**. With Session 6b's weights every channel sat permanently 14.3 %
released at zero salience, and the module named a winner when nothing at all
was salient.

### #65 — three causes, found one at a time

The oscillation-at-zero-salience defect had been carried as an
`expectedFailure` test for two sessions. It is now fixed and the test is a
requirement. All three had to go:

1. **The loop gain sat at exactly 1.0** — the boundary of positive-feedback
   instability. Cause: the output slopes are not all 1; the published 0.62 on
   the thalamus supplies the margin. *(Found in 6b.)*
2. **The weights were a variant's** — resting output 0.143 where the published
   gate needs 0.169, so the gate could never close. *(Found in 6c.)*
3. **The solver ran at the caller's rate.** The STN–GPe loop is negative
   feedback with gain 0.9 × channels — 5.4 for six behaviours — so a 20 ms step
   against a 40 ms time constant is an unstable discrete map. *(Found in 6c.)*

Each one masked the next. Fixing any single one would not have closed the gate.

### What the module now reproduces exactly

Independent integration of the published equations, matched to four decimals —
the five steps of Gurney 2001b Fig. 2a:

| salience | published output | ours |
|---|---|---|
| rest | gate closes | **0.16953 → gate exactly 0** |
| ch1 = 0.4 | 0.0850, 0.3290 | **0.0850, 0.3290** |
| ch1 0.4, ch2 0.6 | 0.2335, 0.0415, 0.4775 | **0.2335, 0.0415, 0.4775** |
| both 0.6 | 0.1225, 0.1225, 0.5585 | **0.1225, 0.1225, 0.5585** |

**Still not reproducing:** the switching-rate comparison. Published ≈ 3× more
switching at excess dopamine than at baseline; measured 0.55×. Improved from
0.3× and still inverted. The published counts are from an embodied robot
foraging task and this is a disembodied competition — plausible, **not
demonstrated**, and the module stays unwired until it is.

---

## Ledger — scouting module 3 (Session 6d)

Six agents swept for the published basis of the next brain module and every
load-bearing claim was adversarially checked. Twelve claims survived, twelve
were refuted. **The most important finding is that the basal-ganglia failure
this project has reported for three sessions is not a failure.**

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 71 | The open-loop CPG contributes ~2% of the walking | **Refuted** | that figure is the *legacy* profile on the wrong body. The accepted walker **is** the open-loop base: 19 strides, 1.1892 Hz measured against 1.1888 commanded, duty 0.733 |
| 72 | The brainstem's declared output has somewhere to land | **Refuted** | the cord is `(t·freq + offset) % 1.0` — closed form, no phase state, no drive input. **1 of 4 drive channels is consumable** |
| 73 | The switching-bout definition needs the supplementary materials | **Refuted** | it is in the **main text**, §3.1.3, freely readable — a wall recorded without being verified, for the second time |
| 74 | The 0.55× switching ratio is a failed reproduction | **Refuted** | the published number counts bouts over two named task sequences against a **task-topology minimum of 7**, and the authors explicitly declined fixed-interval counting — which is what our tool does |
| 75 | A published *disembodied* acceptance test exists | **Confirmed** | Prescott 2024 Study 1 workbook: 61 dopamine levels × five selection classes, CC BY, obtained |
| 76 | The module reproduces that table | **Partly** | every ordering reproduces and distortion onset lands on the published step; partial selection 17–47 points high, clean 15–31 low |
| 77 | The module computes the paper's distortion | **Refuted** | ours divides losers by the **winner**; Eq. 3 divides by the **total** and doubles it |
| 78 | The registry holds a published stride frequency | **Refuted** | 116 entries, and the only frequency is the invented 1.1888 Hz lock |
| 79 | This species changes gait with speed | **Refuted** | relative phase not significant over 0.2–1.1 m/s; 44±1.1% walking against 43±1.8% running |
| 80 | The published EMG pattern is wired into the controller | **Refuted** | `common/emg_pattern.py` is imported by tests and nothing else |
| 81 | MLR stimulation has never been done in a gecko | **Refuted** | walking, phonation and left/right turns elicited from the midbrain of *Gekko gecko* — the one study in the target family |
| 82 | The scout's settle loop converged *(a scout's own method error)* | **Refuted** | breaking on two equal gate vectors fired during the clipped-at-zero transient and reported 100% no-selection at every level |
| 83 | The scout's first walker measurement used the accepted body *(same)* | **Refuted** | it used the default body and legacy profile; the accepted body is identified only by a hash inside a gate evidence file |

### #74 — the failure that was not a failure

For three sessions this project reported an inverted switching ratio as its one
open basal-ganglia failure. It was measuring something the paper never
published.

- **Published:** switching bouts counted over *the first avoidance sequence and
  first foraging sequence of each trial*, against a floor of 7 — which is the
  **number of sub-behaviours in the task**, not a rate.
- **Ours:** switches counted in a fixed 120-second window.
- The authors say in the same sentence that they preferred their measure to
  *"counting bouts (or switches) within a fixed time interval"* — precisely
  what our tool does.

Disembodied, the denominator does not exist. **0.55× is not an inverted
reproduction; it is a quantity with no published counterpart.** The right
response is to retire the comparison, not to chase it.

### #75 → #76 — and the test that replaces it

The same paper's supplementary workbook holds the **non-embodied** model's own
selection statistics at 61 dopamine levels — no robot, no task, no bouts. A
like-for-like test for a like-for-like model, and it was free the whole time.

Against it, every ordering reproduces and the onset of distortion lands on the
published step. One fault remains, with a single clean signature: **our winner
does not saturate.** Published clean selection at baseline is 78.6%; ours is
47.4%, with the difference sitting in "partial". A competition our module calls
partial, the paper calls clean.

That is a diagnosis, not a tuning target. Rule 2 applies.

---

## Ledger — the saturation fault explained (Session 6e)

The fault was **not in our arithmetic**. We had built the Gurney, Prescott &
Redgrave 2001 model and were scoring it against a 2024 paper that uses a
**different dopamine mechanism**. The comparison could not have succeeded
however well either model worked.

The paper's own supplementary archive — its C++ source and the data behind
Figure 5 — was obtained and read.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 84 | The winner failing to saturate was a defect in our model | **Refuted** | it was a **model mismatch**; every constant we had was right for a different model |
| 85 | The 2024 paper uses the 2001 (1±λ) dopamine mechanism | **Refuted** | dopamine modulates the striatal output **slope** about a pivot, not the input gain. The authors call it "new DA" in their own source |
| 86 | The striatal threshold is 0.2 | **Refuted** | **0.1** in the basic variant, **0.15** in the extended |
| 87 | Salience reaches the striatum directly | **Refuted** | through a leaky cortical relay — and the extended model splits it **0.5/0.5** with the thalamic return |
| 88 | The supplementary materials were unobtainable | **Refuted** | on disk: the authors' **C++ source** and the **Figure 5 data workbook** |
| 89 | Figure 5 came from the basic variant | **Refuted** | the shipped harness sets `EXTENDED 0`, but the basic variant peaks at **8%** clean selection where the extended reaches **79%** |
| 90 | The published distortion carries the paper's factor of two | **Refuted** | the authors' code has **no factor of two**, and the published range 0–0.217 fits the unfactored form |
| 91 | A faithful implementation reproduces the published table | **Partly** | akinesia **exact**, peak clean **79.9 against 78.6**, efficiency **0.996 against 0.999** — but the curve sits **+0.10 higher in dopamine** |
| 92 | The residual disagreement is a shape error | **Refuted** | axis fit slope **1.08** — a constant shift, not a scaling or a deformation |
| 93 | The sampling behind Figure 5 is recoverable | **Refuted** | percentages exact to 1e-5 imply **100,000 competitions per level**; the shipped harness sweeps 202 ramp points |
| 94 | The winner-take-all comparison arm reported its own result | **Refuted** | `gates()` returned all zeros one line after `step()` named a winner |

### What changed, and what it bought

| | before | after |
|---|---|---|
| Winner efficiency | tops out ~0.40 | **0.996** (published 0.999) |
| Peak clean selection | 47.4% | **79.9%** (published 78.6%) |
| Akinesia at zero dopamine | not tested | **100.0%** — exact |
| Mean error across the table | — | 10.3 points, down from 20.4 |

**The one thing still open** is a constant offset of about **+0.10** along the
dopamine axis: our curve looks like the published one shifted. Every constant
has been checked against the authors' source and matches, so this is **not**
closed by adjusting one. Recorded, not tuned.

---

## Ledger — the gecko moved onto the validated model (Session 6f)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 95 | The environment has a behaviour arbiter to replace | **Refuted** | it has none. It is reward-driven, and the `target_interest` the plan named lives in the **superseded** drive module |
| 96 | The selector can be given control of the animal now | **Refuted** | only one behaviour has a controller. A six-way chooser on a one-behaviour body would look like integration and mean nothing |
| 97 | Persistence needs an invented stickiness constant | **Refuted** | the published cortico-thalamic loop supplies it **structurally** — the invented gain of 0.15 is deleted, not re-tuned |
| 98 | The two brain modules can run together on live data | **Confirmed** | hypothalamus → selector every control step inside the simulation; a tired gecko rests, and a rested hungry one explores |

**#97 is the quiet win.** Migrating to the published model *removed* a number
nothing could check. The old selector made a behaviour sticky by feeding its own
output back through a gain someone chose; the published model gets the same
effect from wiring that is in the literature.

**#96 is the discipline.** The selector is wired in and **reports only** — its
choice appears in the info dictionary and steers nothing. It cannot steer
anything honestly until the cord and brainstem exist to carry a decision to the
legs.

---

## Ledger — the cord becomes an oscillator (Session 7)

The leg rhythm was a closed-form clock: no state, no coupling, and **no input
for a descending command**. Three of the brainstem's four output channels had
nowhere to land. So the cord had to become a real oscillator first — the
reverse of the order the plan proposed.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 99 | The plan's "bit-identical at zero coupling" test is achievable | **Refuted** | an accumulated phase and a closed-form one differ in the last bits, and the clock snaps to exact cycle boundaries an accumulator never lands on |
| 100 | An integrated oscillator can reproduce the accepted walker's rhythm | **Confirmed** | max deviation **3.3 × 10⁻¹¹ of a stride** over 60 s of walking |
| 101 | The foot order is the obvious one | **Refuted** | it is **HL, FL, HR, FR** — restating it would have planted a silent index mismatch |
| 102 | A single-step check shows the oscillator and the clock differ *(my own test error)* | **Refuted** | they agree **exactly** for the first few steps; divergence needs accumulation, so the test proved nothing |

**#99 is the honest downgrade.** The plan asked for a bit-identical trace and
the intention was right — any later change in the gait gates must be
attributable to the coupling, not to the rewrite. But bit-identity is
arithmetically unavailable here. What is available is agreement eleven orders
below anything the gates measure, and that is what the test asserts.

**#102 nearly became an overclaim.** The first version of that test checked one
step, found the two equal, and failed. Written the other way round it would have
"passed" and been read as evidence for bit-identity — which is false.

**What the cord now has that it did not:** a settable stride frequency and a
left/right asymmetry, which is how a symmetric pattern generator turns. The
load-feedback law is implemented, **switched off, and tagged INVENTED** — the
corpus names the law but gives no value for its gain, and neither source paper
has been read.

---

## Ledger — the dopamine offset resolved, and brain 3 finished (Session 7b–7c)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 103 | The dopamine discrepancy is a constant offset | **Refuted** | neither an offset nor a scaling — the **mechanism** was wrong |
| 104 | The published model modulates the striatal output slope | **Refuted** | it multiplies the **afferent input**: (1 + λ) to D1, (1 − λ) to D2, threshold **0.2**, quoted verbatim in Supplementary Methods §5 |
| 105 | Reading the files an archive contains is enough | **Refuted** | the builders every program includes are **absent**; the two present are the branch every program disables |
| 106 | Session 6e's "the whole model was wrong" was right | **Partly** | right about the structure. **Wrong about dopamine and the threshold, which the *first* implementation had right** |
| 107 | The published sweep is memoryless | **Refuted** | it is **hysteretic**; cold-starting each cell moves the table by ~1 percentage point |
| 108 | The corrected model reproduces the published table | **Confirmed** | mean error **0.103 pp**, all four onsets exact, peak at **0.22 = published**, axis fit **slope 1.000, intercept 0.000** |
| 109 | Attaching the integrated cord changes the gait | **Refuted** | every gate identical; hind duty **0.733418** both; speed differs by **3.2 × 10⁻¹⁴ m/s** |
| 110 | The parity harness is the official gate scorer | **Refuted** | it records at its own rate, so only the comparison is claimed, never the absolute count |
| 111 | isinstance is safe for path-loaded modules | **Refuted** | the same file under two names gives two classes, and a perfectly good cord was rejected |

### #104 → #106 — a correction that reversed a correction

Session 6e concluded "the whole model was the wrong one" and rewrote the
dopamine mechanism. It was **half right**, and the half it got wrong it got
wrong *away* from the truth: the original module had the dopamine form and the
striatal threshold **correct all along**.

The trap was specific. The archive ships two model builders and both select a
slope-and-pivot dopamine mechanism. Neither is used — every shipped program
disables that branch and includes a builder that **is not in the archive**. So
reading carefully what was present produced a faithful implementation of the
branch nothing runs.

**What settled it was the paper's prose, not its code**: dopamine is *"a
multiplicative factor in the equations, specifying afferent input to the
striatum"*. No pivot and no 0.15 appear anywhere in the paper.

### #108 — what the fix bought

| | before | after | published |
|---|---|---|---|
| Mean error across the table | 10.3 pp | **0.103 pp** | — |
| Peak clean selection at λ | 0.37 | **0.22** | 0.22 |
| Onsets (clean, distortion, multiple) | all late | **all exact** | — |
| Dopamine axis fit | slope 1.08 | **slope 1.000, intercept 0.000** | — |

A hundredfold improvement, and the offset carried as this module's one open item
is **gone**.

### Brain 3, finished

The cord drives the walker and **every gate is unchanged** — measured, not
assumed. The brainstem turns a chosen behaviour into a stride frequency and a
turn, and the urgency it uses is the gate value the decision was released with,
so a half-released flee drives the legs at 0.669 rather than 1.0. It carries the
thinnest evidence in the project: every direction published, every number
invented and declared. It has **no gait-selection channel**, because this
species does not change footfall pattern with speed.

---

## Ledger — scouting the eye (Session 8)

Five angles on what a leopard gecko's eye actually is, every load-bearing
number adversarially checked. The scouting found a **live defect** in work that
had been declared finished.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 112 | The food detector can see the food | **Refuted** | it tested for **green**; the prey that replaced the painted marker is **brown**. Returns exactly **0.000** at every illumination from 0.4× to 3.0× |
| 113 | The no-cheat world is in use | **Refuted** | referenced only by its own test and its generator. **No training or evidence run has ever loaded it** |
| 114 | The old detector at least worked for its own marker | **Refuted** | absolute channel thresholds, so the painted sphere also scored **0.0 at gain 0.4** — it failed on illumination change too |
| 115 | A colour detector can find the prey in this world | **Partly** | it can, but the margin is thin: the nearest confuser is **the gecko's own spots** at 0.145. A brown cricket on sand is genuinely camouflaged |
| 116 | The policy receives an image | **Refuted** | the encoder average-pools to **128 numbers over the whole visual field** — "how much", never "where". There is no retinotopic map at all |
| 117 | A rectilinear camera samples uniformly, as the animal does | **Refuted** | corners sample **4× finer than the centre** at 120° — an **inverse fovea**, worst exactly where the animal aims |
| 118 | The default build can resolve what the animal tracks | **Refuted** | the published 1.6° tracked dot spans **0.516 px** at fovy 120 — not resolvable |
| 119 | The eye's field of view is sourceable | **Refuted** | still NOT_IN_CORPUS for this species; the five corpus recommendations remain invented |
| 120 | Leopard geckos have a tapetum | **Refuted** | unverifiable from any primary source — every confident claim traced to a pet-care site. **Do not build one** |

### #112 — a detector that could not see its own food

Session 4c replaced a green sphere painted on after rendering with a real brown
prey geom. The detector that reads it kept looking for green. Prey visibility
was therefore **identically zero**, which means the gecko in the no-cheat world
could never have hunted.

It never ran, which is #113 and is its own finding: the world was declared
finished and then never adopted.

The fix removes the **class** of bug rather than the colour. The detector now
reads the prey's appearance **out of the model**, so the world and the detector
cannot state it separately and drift apart. Matching is on chromaticity rather
than absolute channel values, which also fixes #114.

### #116 → #118 — the eye is not where the effort went

The three findings compound. The camera cannot resolve the smallest feature the
animal is *measured* to track; it samples the periphery finer than the centre,
which is backwards; and whatever it does capture is averaged into a single
global number before the policy sees it. **There is currently no sense in which
the gecko knows where anything is.**

That is what brain 4 is for, and #119 and #120 mark where it must stay honest:
no published field of view exists for this species, and the tapetum everyone
"knows" geckos have could not be traced to a primary source at all.

---

## Ledger — the retina and the optokinetic reflex (Session 8b)

The eye now keeps position, and it is gated against the only quantitative
behavioural measurement that exists for this species' vision.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 121 | A gradient flow estimator on the cell grid is accurate | **Refuted** | **2.07x too large** at every velocity — the drum's stripes span four cells, where a central difference underestimates the gradient by a third |
| 122 | Estimating at pixel resolution fixes it | **Partly** | 2.07x to **1.15x**. The residual is the inverse fovea: pixels are not uniform in angle |
| 123 | Differencing against **azimuth** removes the bias | **Confirmed** | ratio **1.02** against truth across 10-40 deg/s |
| 124 | The published monocular asymmetry can be reproduced | **Confirmed** | naso-temporal gain **exactly 0.000** at all three velocities — a symmetric estimator cannot do this |
| 125 | The binocular-versus-monocular comparison can be run | **Refuted** | **the body has ONE camera.** There is no second eye to cover. Recorded as blocked, not as passed |
| 126 | The optokinetic gain should approach 1 | **Refuted** | published gain is 0.9 and **falls** to 0.7-0.8 by 40 deg/s. Perfect stabilisation would be over-reproduction |

### #121 to #123 — three attempts at one measurement

The flow estimator was wrong twice before it was right, and each error had a
different cause. Binning first destroyed the spatial sampling. Differencing
against the pixel index then inherited the camera's own non-uniformity — **the
inverse fovea showing up as a measurement error rather than as a picture**.
Only differencing against the actual angle of each column is unbiased.

### #124 — the discriminating result

Published: monocular naso-temporal stimulation elicits **no optokinetic
response at any velocity**, n = 4. Any front end that simply measures full-field
motion predicts a non-zero gain there, because the flow is present and just as
strong.

Here the silence is **predicted**, not written in: each hemifield is half-wave
rectified in its temporo-nasal direction, so backward flow contributes nothing
rather than contributing negatively. Measured: **0.000, 0.000, 0.000**.

The absolute gains are **fitted** and are therefore not evidence — the module
and the evidence file both say so, and the test gates only on the predicted
properties.

### #125 — a test the body cannot take

The published experiment covers one of two eyes. This body has a single central
camera, so binocular and monocular are the same condition and the difference is
identically zero. Recorded as **untestable**, not as passed. Adding a second eye
is a body change, and the body is validated 14/14, so it does not happen as a
side effect of a vision module.

---

## Ledger — the tectum, shipped NOT ACCEPTED (Session 8c)

The eye is complete and the prey-finding half **does not work**. It is committed
that way, with the evidence, exactly as the basal ganglia was in Session 6.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 127 | The tectum finds prey on synthetic stimuli | **Confirmed** | correct bearing at every position, silent on a still world, silent on pure self-motion |
| 128 | It finds prey in the **environment** | **Refuted** | fires on **99.3%** of frames and the bearing correlates **−0.03** with the prey's. Mean error **63°** |
| 129 | The failure is the map being too coarse | **Partly** | full pixel resolution reaches only **+0.10**. Resolution is part of the cause, not the whole |
| 130 | The prey is resolvable in the map at working range | **Refuted** | it covers **0.69 of one cell at 0.30 m** and 0.41 at 0.50 m |
| 131 | My self-motion control demonstrated what I claimed | **Refuted** | the **size test** was doing the work; field subtraction is worth a fourfold reduction, not elimination |
| 132 | The order of the two rejection tests is arbitrary | **Refuted** | size test *after* subtraction leaks **0.066**; *before*, **0.000** |

### #128 — a detector with a 99% hit rate and no information

This is the entry worth re-reading. The tectum reports a target on almost every
frame, so any summary that counted "did it see something" would call it a
success. The bearing it reports is uncorrelated with where the prey actually is.

**A hit rate is not a measurement.** The test is the correlation, and the
correlation is zero.

The cause is not mysterious: the animal's own body and the floor sliding past
generate more motion than a prey covering less than one cell. Raising the map to
full pixel resolution moves the correlation from −0.03 to +0.10, which says
resolution is a contributor and not the answer. **No fix is claimed.**

### #131 → #132 — two errors in my own controls

The first version of the self-motion control showed 0.000 with and without
field subtraction, and I nearly recorded that as "the subtraction works". It was
the size test rejecting the stimulus in both arms; the control tested nothing.
Isolating properly showed the subtraction is worth 1.000 → 0.266.

Then the two mechanisms turned out to **interact badly**: subtracting first
leaves a ragged residual whose surviving peaks look small and local, and it
slips past the size test. Asking "does this fill the field?" of the raw map
first fixes it.

**What works:** the gaze reflex, the retinotopic map, the published optokinetic
asymmetry. **What does not:** finding prey with a body attached. The eye is off
by default and a test keeps it off.

---

## My own errors — the meta-ledger

Failures of method, not of hypothesis. These are the ones worth re-reading.

| What I did | Cost | Rule it produced |
|---|---|---|
| Estimated training speed instead of measuring | quoted 45 min for a 96 min run | measure before quoting |
| Built a scoreboard with no stride-regularity rejection | crowned a checkpoint with CV 0.375 — rebuilt the exact Session 3g trap | reuse the official gate, never re-derive |
| Ranked step-0 as "best trained" | credited training with the base controller's own score | a setup check is not a result |
| Adjusted GPR constants three times by "does it look right" | two of three made it worse | rule 2 |
| Declared a blocker without verifying it | a day's delay; the parameters were freely available | search before declaring a wall |
| Took the **first** parameter table I found and called it published | shipped a regression that opened every gate at rest | one source is not a citation; check it against something it must independently predict |
| Let the environment's control rate set the solver's rate | a numerical instability I spent two sessions attributing to the model | a model and its integration are different things |
| Wrote a guard that pinned one publisher's name | it failed when the citation was **corrected** | guard the property, not the particular |
| Generated evidence from a throwaway script | it could not be regenerated when the parameters changed | evidence that cannot be re-run is an assertion |
| Recorded a second unverified blocker | the switching-bout definition was in the paper's main text all along | **twice now.** Recording a wall is not verifying one |
| Compared against a published number without checking what it counted | three sessions spent on a ratio with no published counterpart | read the definition before reproducing the value |
| Implemented a published quantity from its name | our distortion divides by the winner; the paper's divides by the total | a formula is not a word |
| Stored gate-space values in a pre-activation state variable | the comparison arm's accessor returned "nothing selected" one line after naming a winner | a round trip is not a round trip until you run it both ways |
| Scored one model against another model's published data | three sessions chasing a fault that was a category error | check the model matches before comparing its numbers |
| Rewrote a correct mechanism because the only files present said so | reversed a right answer into a wrong one for two sessions | when code and prose disagree, find which branch the archive actually builds |
| Assumed a published sweep was memoryless | a percentage point of error with no cause | ask what a harness does *between* samples, not only during them |
| Twice took a published formula from its prose | the distortion factor of two contradicts the code that made the table | when source code exists, it outranks the paper's own equation |
| Never opened the supplementary archive | the C++ AND the data were sitting there | look in the box before declaring it empty |
| Wrote a test asserting zero distortion at every dopamine level | the test **forbade the correct published behaviour** | tests encode expectations, and expectations can be wrong |
| Used 50 Hz white noise as fluctuating salience | measured seven behaviour switches per *second* | check the test before blaming the model |
| Claimed the cheat was removed | it was removed from the wrong environment | verify the claim where it matters |
| Shipped a video labelled "best checkpoint" | it was the *untrained* snapshot | name artifacts for what they are |

---

## Currently blocked

| What | Blocked on | Why it cannot be worked around |
|---|---|---|
| **Fear** | a chemosensory channel | Defence is **chemically gated**: smell 0.20, sight 0, touch 0. There is no odour field, and MuJoCo has no scent primitive. Any fear response now fires on the wrong sense. |
| **Thermostat** | a temperature field | Built, wired to the published 29.5–31.9 °C band, reports `thermostat_inert: True`. Zero mentions of temperature anywhere in the world. |
| **Hunting gates** | a strike behaviour | The walker cannot outrun prey and is not meant to. A 16–20 ms strike is shorter than one control step; scoring needs 500 Hz. |
| **Basal ganglia** | a constant dopamine offset | The saturation fault is **explained and fixed** — it was a model mismatch, not an error. `brain/prescott_bg.py` is the published model and reproduces akinesia exactly, peak clean selection to within 1.3 points, and winner efficiency to within 0.003. One residual: the whole curve sits **+0.10 higher in dopamine**. Every constant matches the authors' source, so it cannot be closed by tuning. |
| **Wiring the gecko in** | migrating onto the validated model | The gecko's own module is still the 2001 lineage. Now that a validated reference exists, the animal should be moved onto it — that is the next basal-ganglia step, and it is a migration rather than a research question. |
| **Brainstem** | a cord that can be driven | The spinal generator is a closed-form clock — `(t·freq + offset) % 1.0` — with no phase state and no drive input, so three of the brainstem's four declared output channels have nowhere to land. **The cord must be rewritten as integrated oscillators first.** |
| **Gait selection** | the animal itself | This species does not change footfall pattern with speed. The channel can be built but **has no acceptance test that is not invented**, and must be recorded as present-but-unvalidatable. |
| **Forefoot gate** | a wrist joint | Closed by decision, not failure. Both unmet checks are INVENTED targets. |
| **Two evidence files** | a body that no longer exists | `brain_legacy.json` and `brain_sealed.json` pin a hash **no file in the repo has**. Predates this work. |
| **Walking posture** | unassigned | Shoulder rides 0.125 SVL against a published 0.11. Static audit passes 14/14; the *walking* pose was never checked. Open. |

---

## What is not modelled, and is recorded rather than omitted

- **Digestion costs nothing.** Postprandial metabolism peaks at 3.7–7.3× resting
  for 62–170 h in another lizard genus.
- **Metabolism has no thermal dependence.** No metabolic Q₁₀ is published for this
  species. The corpus's 2.3 is a *sleep-period* Q₁₀ — transplanting it between
  unrelated processes would be inventing physiology.
- **The cold half of the thermostat has no anchor.** CTmax is published at
  41.07 °C. **CTmin is NOT IN CORPUS** for this species or any proxy.
- **Fatigue recovery is INVENTED.** The literature says how fast a lizard tires
  and nothing at all about how fast it recovers.
- **Cricket escape speed is DERIVED**, never cited as measured — a quotient of two
  published quantities the corpus never divides.
- Hydration, nutrient composition, gut passage time, field metabolic rate.
- **The GPR origin papers were never read.** Gurney, Prescott & Redgrave 2001
  parts I and II are paywalled. Every constant in the basal ganglia is
  **secondary-sourced** via Fox et al. 2009 and tagged as such in the module and
  in the evidence file. Six independent reimplementations agree with it,
  including the original group's own release — but that is corroboration, not
  the source.
- **The time constant τ is unresolved.** 40 ms (Fox), 25 ms (Girard), 10 ms
  (SpineML). It does not move the fixed point, so selection is unaffected — but
  it sets the absolute scale of switch counts, which is exactly the quantity
  that does not reproduce.
- **The thalamocortical extension is not GPR 2001** and has two incompatible
  versions in the literature. This project follows Girard's throughout and says
  so; mixing the lineages is how #64 happened.

---

## Next

1. **Brainstem + spinal CPG** — turning a chosen behaviour into locomotion.
2. **Close the basal ganglia** — reproduce the foraging task, or obtain the
   switching-bout definition, or demonstrate the task difference.
3. **Retina + tectum** — actually seeing prey. The recovered 29/30 vision brain
   reattaches here.
4. **Sleep, memory.**
5. **The 15-test battery, then publish** — with the not-real list at the front.

**Ceiling: R2-by-proxy.** Not a limit of effort or budget — no gecko connectome
and no gecko neural recordings exist for anyone. That is the frontier, not a
shortfall.

---

## Ledger — the eye was the wrong question (Sessions 8d–9)

Session 8d chased the cricket with better optics and failed. Session 9 stopped
building and went to the literature instead: six searches on what a real
leopard gecko does, what it senses, and how it understands its world. The
searches refuted more of this project's assumptions than any session so far,
including two I had stated to the user as fact.

**The headline, stated correctly on the second attempt.** My first version of
this line said no one has ever published a study of a leopard gecko hunting.
That is wrong, and a later search found the paper: **Delheusy, Brillet & Bels
1995**, *Amphibia-Reptilia* 16(2):185–201, n = 6 adult males, SVL 120 ± 4 mm,
filmed at 64 fps eating 22 mm *Gryllus bimaculatus*. It has a full kinematic
table.

What is true is narrower and still severe. **The capture itself is the one
phase that paper could not quantify** — only 5 lateral cycles from 2 animals
were usable, so capture is described qualitatively and excluded from every
table and statistic. At 64 fps the frame interval is 15.6 ms, which cannot
resolve a strike. So chewing, transport and licking are measured in this
species; **strike distance, strike speed, approach speed and capture success
are not**, and those are exactly the numbers a hunting controller needs. They
come instead from *Coleonyx variegatus*, a cousin in the same family, mostly
from **five juveniles**.

The correction matters more than the fact. I wrote "not one, nobody has filmed
it" from one agent's negative result, in a session whose entire subject was
this project repeating unchecked claims. The first agent had actually named the
paper and flagged that it could not retrieve the numbers; I turned "could not
retrieve" into "does not exist."

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 133 | Subtracting the surround stops the tectum hunting its own optic flow | **Refuted** | it subtracted a **scalar**, and `argmax(clip(x-c,0,None)) == argmax(x)` for any scalar c. Identical on **20000/20000** random maps — it had never changed a single reported bearing. Replaced with a spatially varying local surround, which moves the answer on **1735/2000** |
| 134 | A sharper eye finds the cricket | **Refuted** | four configurations, render 64→512 px and receptors 64→256. Correlation with true prey bearing ≈ **0** in all four. I had told the user resolution was the fix; it is not |
| 135 | There is prey motion for a motion detector to find | **Refuted** | `prey_total_travel_mm = 0.0` across 1800 frames. `FleeingPrey.step` only moves within 0.075 m while detection range is 0.30–0.50 m. **The prey is stationary** |
| 136 | The prey is in the camera's field of view | **Refuted** | present on **18.7%** of frames. Off the rendered image four frames in five |
| 137 | An open-source retina model can be adopted | **Refuted** | the only candidate, `openretina`, is a **mouse** retina (MIT code, CC-BY-SA weights) and retraining needs gecko recordings that do not exist for any reptile |
| 138 | Geckos head-bob to judge prey distance by motion parallax | **Refuted** | **no gecko study shows it**, and none shows it for prey in any lizard. It is a stated *hypothesis* in other lizards, never demonstrated. Every confident claim traced to pet-care sites. **I asserted this to the user as fact.** The one lizard-clade distance mechanism that IS established is chameleon **accommodation**, which is not motion-based |
| 139 | Stationary prey is undetectable by a visual predator | **Refuted** | mice capture **fresh-frozen crickets** as fast as live ones — 8.06 ± 1.08 s vs 7.6 ± 1.34 s, n=8, n.s. Motion is not required. The blindness is our architecture's, not nature's |
| 140 | A better tectum would find still prey | **Refuted** | in frogs the **pretectum** handles stationary objects and the tectum handles motion: pretectal lesions leave feeding and threat avoidance intact but the animal walks into a barrier it can see. A tectum-only model is **correctly** blind to a frozen cricket. This needs a second structure, not a better one |
| 141 | One prey channel is enough | **Refuted** | mouse superior colliculus splits it: **wide-field** (RF 700–900 deg², not direction-selective) triggers approach from >22 cm; **narrow-field** (<200 deg², direction-selective) supplies bearing during pursuit. Suppress the second and the animal reaches the prey and **stalls beside it**. Echoed in zebrafish posterior/anterior tectum |
| 142 | Prey selectivity is a conjunction of features | **Refuted** | the one worked-out vertebrate is the opposite. The toad tectum's default is *attack anything that moves*; selectivity is a **subtractive pretectal veto**. Lesion the pretectum and the toad attacks everything moving. Our tectum ANDs its criteria — architecturally backwards |
| 143 | Elevation in the visual field is a soft weight | **Refuted** | it is categorical. Identical looming disc: **75% escape overhead vs 53% and much slower at the side**. Identical small sweeping disc: freeze overhead, **80% approach** at the side. Floor-mounted looms produced **zero** escapes |
| 144 | The prey velocity band is a free parameter | **Refuted** | four independent lineages converge on the same order: toad behaviour peaks 30–60 °/s and dies above ~200; **pit viper tectum** (a squamate) is near-silent at 5 °/s and peaks ~50; mouse looming escape needs 35 °/s. This is the most robust cross-species number found |
| 145 | Our gecko basks | **Refuted** | it is **thigmothermic**. Body temperature tracks the **substrate** at r²=0.97 against air at 0.92, n=12. It takes heat by lying on warm ground, not from light. And preferred temperature is not a set point but a **diel ramp**, ~26 °C morning to ~30 °C at dusk — the rise is proposed as what triggers evening emergence |
| 146 | A visual threat channel models this animal's escape | **Refuted** | for **this species**, n=42: defensive reaction probability is **0.21 to snake smell and 0.07 to the sight of a snake**. Factorially: chemical χ²=8.098, p<0.0044; visual χ²<0.01, p>0.9. Verbatim: *"the initiation of costly defensive action remains strictly gated by chemoreception"* |
| 147 | Smell can localise prey | **Refuted as unsupported** | tongue-flicking is real and quantified for this species — **3.0/min baseline → 14.57/min to cricket chemicals**, n=7 — but the assay presents the swab **1 cm from the snout, in plain view**. It demonstrates *discrimination*, never *localisation*. Cooper flagged the missing experiment in 1998 and nobody has done it since |
| 148 | `cost_of_transport = 0.73` is a leopard gecko measurement | **Refuted** | it is *Teratoscincus przewalskii*, a different **family**. `docs/research/gecko_scorecard.md` had it right, tagged `[TP]` and defined as a proxy for energetics only; `config/proxies.yaml` — the file the code reads — said `species: E_macularius, confidence: verified`. Corrected. No cost of transport, sprint maximum, endurance or thermal performance curve has ever been measured in this species |
| 149 | 456 tests pass | **Refuted** | 8 `test_session2_gate.GateContracts` tests fail, and they fail in a **clean worktree at commit e4196d8** — they are not caused by session 8d's edits. See below |
| 150 | No feeding kinematics exist for this species | **Refuted — my own overstatement** | Delheusy, Brillet & Bels 1995, *Amphibia-Reptilia* 16(2):185–201, **n=6 adult males**, 64 fps, 22 mm crickets, full Table 1. I turned one agent's "could not retrieve the numbers" into "does not exist". What *is* absent is narrower: the **capture phase** was unquantifiable at 64 fps (5 usable cycles from 2 animals) and is excluded from the paper's tables |
| 151 | The gecko takes prey with its tongue | **Refuted** | **jaws alone** — the standard scleroglossan pattern, no tongue prehension. And the orienting rule is published: the head is aligned **transverse to the prey's long axis**. Capture is a single open–close cycle with no slow/fast subdivision: **~80 ms, peak gape ~37° at 47 ms**, head translating ~8 mm horizontally and ~28 mm vertically |
| 152 | Eyelid geckos cannot lick their own eyes | **Refuted** | Delheusy et al. 1995, verbatim: *"La langue se déplace hors de la cavité buccale et **atteint parfois l'oeil**"* — the tongue sometimes reaches the eye, during post-feeding licking and grooming. The "cannot" claim traces only to care sites. Separately: **labial-licking is a different behaviour** and the words *eye*, *ocular*, *eyelid* appear nowhere in the paper that defines it |
| 153 | This animal navigates by what things look like | **Refuted** | given a conflict between **geometry** and **visual features**, *E. macularius* preferentially used geometry (Kundey 2021, *Behav Processes* 188:104412). Its world model is shaped more by layout than by appearance — which is the opposite of what a vision-first build assumes |
| 154 | The circulating leopard-gecko strike numbers have some source | **Refuted, source pinned** | "486 m/s², 5.8 m/s, 35 cm" is Wainwright, Kraklau & Bennett 1991, *J Exp Biol* 159:109–133, **tongue projection in *Chamaeleo oustaleti***, n=3 sequences from 2 animals. A leopard gecko does not project its tongue 35 cm |
| 155 | Sleep is unmeasurable in this animal | **Refuted** | *E. macularius* is one of seven lizard species in Bergel et al. 2026, *Nat Neurosci* 29:543–550 — **sleep-dependent infraslow rhythms conserved across reptiles and mammals**, EOG under each eyelid. It **closes its eyelids to sleep**; the tokay cannot. Brain 5–8 has a starting point that did not exist when they were listed as ❌ |
| 156 | The gecko vibration band is press-release-only and unusable | **Refuted** | read from the paper itself. Han & Carr 2024, **n=31 tokay geckos, 38 single units**: saccule → *nucleus vestibularis ovalis* → torus semicircularis, a channel entirely separate from hearing. Best frequencies **50–200 Hz, mode ~100 Hz**; threshold **−53 to −23 dB re 1 m/s²**, mean −42.6; **phase locking VS 0.89**. Sound at 15 dB above the loudest vibration used evoked **nothing** — it is vibration, not sound |
| 157 | The ear covers low frequencies, so vibration adds little | **Refuted** | **no lizard auditory-nerve fibre has ever been recorded below ~200 Hz CF**, across four primary datasets, and the audiogram is a flat insensitive shelf from 100–500 Hz. The vibration channel covers **exactly the band hearing cannot reach**. They are not redundant; they are complementary |
| 158 | No lizard has been shown to find prey without seeing it | **Refuted** | *Scincus scincus*, buried in sand, detects crickets moving on the surface **up to ~15 cm away**, extracts **direction**, emerges and captures. It responds far less to **dead** insects at the same distance, so it is vibration and not smell, and it **discriminates prey speed** — faster prey, faster emergence. It also **plunges its head into the sand**, and experiments show this is what couples it to the substrate |
| 159 | No usable audiogram exists for this animal | **Refuted, with a substitution** | none is *accessible* for *E. macularius* — Werner et al. 1998 and 2008 hold it and both are hard-paywalled with no open copy anywhere. Two usable substitutes: eight gekkotan species best **22–72 dB SPL at 0.5–1.0 kHz** (Werner et al. 2008), and a **free, full** audiogram for *Coleonyx variegatus*, an actual eublepharid — best **~19 dB SPL at 500 Hz** at 21 °C (Wever et al. 1964, PMC300117) |
| 160 | One audiogram describes this animal | **Refuted** | gecko hearing is **temperature-dependent in both sensitivity and best frequency**: as the animal cools it hears less *and* its best frequency **shifts downward**. 249 sensitivity functions, 50 animals, 15–40 °C. A fixed audiogram is biologically wrong for an ectotherm, and this project's animal already has a body temperature |
| 161 | Vibration prey-detection is demonstrated in geckos | **Refuted** | it has **never been tested behaviourally in any gecko**. Han & Carr say so themselves — "no behavioral role for a dedicated vibration receptor has been demonstrated" — and their one mention of predator-prey sits bracketed with *wind and rain*. I first recorded this as a *mechanistic coincidence worth noticing* — cricket signals below 120 Hz, saccular channel 50–200 Hz, "the bands line up". **The next search refuted that too. See #162–164: on the numbers the gecko is not sensitive enough and is tuned to the wrong band.** The tympanic ear being ~55 dB down at 100 Hz stands |
| 162 | A gecko could feel a cricket walking | **Refuted on the numbers** | measured arthropod footsteps on sand are **4x10-4 to 1.5x10-3 m/s2** (Devetak et al. 2007, four species). The most sensitive gecko saccular cell ever recorded thresholds at **-53 dB re 1 m/s2 = 2.2x10-3 m/s2**, mean **7.4x10-3**. **The loudest measured footstep sits below the best single cell's threshold**, before any distance damping. A scorpion does this because its receptors trigger at **~1 nm (10 A)** - the gecko's need roughly **19 nm**. Twenty-fold short |
| 163 | The gecko's vibration band is the right band for sand | **Refuted** | sand is a lossy low-pass medium: Rayleigh waves at **40-50 m/s** dry, damped **0.26-2.61 dB/cm at 300 Hz**, worse as grains get finer. The band that actually *propagates* usefully is **200-500 Hz** (Brownell & van Hemmen 2001). The gecko's saccular channel is **50-200 Hz** - it sits **below** the window where sand carries prey information. And a cricket's own low-frequency signal peaks at **43 Hz**, below both |
| 164 | Detection range is a property of the animal | **Refuted** | it is a property of the **ground**. Antlion mean reaction distance is **3.3 cm in fine sand (<=0.23 mm grains) and 12.3 cm in coarse (1-1.54 mm)** - a near fourfold change from substrate alone, in the same species. Any vibration range this project ever quotes is meaningless without the grain size beside it |
| 165 | The privileged target channel is removed everywhere | **Refuted, third occurrence** | `GeckoBrainEnv` built the walker without naming `privileged_target`, so the walker's own `True` default won and nothing in the brain env could override it. Five numbers -- exact egocentric direction, distance and bearing to the goal, straight out of the physics engine, noiseless, at any range. **All 12 checkpoints in the repository record proprio_dim 92**, including the ones described as pure-vision. See #26 and #33 for the first two times this was declared removed |
| 166 | It is present but the policy ignores it | **Refuted, and this is the expensive one** | `tools/oracle_ablation.py` zeroes the five slots at inference, keeping the vector 92 long so the checkpoint still loads. The walker moves **further** with them zeroed (0.0857 -> 0.1103 m) and progress **toward the goal** collapses **0.0664 -> 0.0060 m, a 91% loss**. The legs are fine; the navigation was the oracle. Retiring it needs a retrained walker, so the flag is exposed and declared rather than flipped, and `info["walker_oracle"]` now reports it every step |
| 167 | The sessions 3-4 gait evidence is unverifiable | **REFUTED, and the claim was mine** | I reported that the recorded body hash `756765356417749c` matched no committed version and that the accepted walker's evidence therefore pointed at a body not in the repository. The physics was identical all along. The **comment-free ElementTree serialization** of the current body is `b178bf26b644b1d3`, which is exactly what `artifacts/evidence/morphology_reproducibility_line_endings_20260905.md` recorded for that file. Both committed versions in git history share that same physics hash. **A comment changed, not the animal.** The evidence was always valid |
| 168 | Eight gate contracts are failing | **Refuted -- it was one** | the body check raised **before** any other contract could reach its own assertions, so seven unrelated tests reported the wrong failure for four sessions. Fixing the one check turned all eight green. `common/body_identity.py` now compares what MuJoCo reads rather than what the bytes say, and the evidence writer records **both** fingerprints, because recording only the raw one is what made this unverifiable in the first place |
| 169 | Adding a strike makes hunting possible | **Refuted** | necessary, not sufficient. The prey bolted at a FIXED 0.075 m regardless of how it was approached, the strike must launch from 0.0203 m, and the prey escapes at 0.118 m/s against a 0.055 m/s walk. Measured over 60 s of pursuit: **closest approach 51.1 mm, zero strikes**. The gecko could not physically reach striking distance by any route |
| 170 | A stalk is just a slower walk | **Refuted** | it is only a behaviour if the PREY can tell the difference. The published ethogram names `walk slow motion`, "walking with a strongly reduced speed, mostly in context of prey capture" -- meaningless against a prey with a fixed flee radius. The flee radius now scales with approach speed: **75 mm at a charge, 12 mm at a creep**, which is inside strike range. Direction supported (flight-initiation distance rises with approach speed; crickets sense predators through cercal air-current receptors); magnitude INVENTED and the capture rate rests on it |
| 171 | The predator's own speed is what frightens the prey | **Refuted** | the predator point is the gecko's NOSE, which sways with the gait: measured **0.129 m/s of apparent approach during a creep that netted almost nothing**. Reading gait sway as a charge made stalking impossible. What a prey animal has to go on is how fast the threat is getting CLOSER, so closing speed is measured instead, floored at zero |
| 172 | Striking distance can be measured in three dimensions | **Refuted** | the nose sits **13.7 mm above** a prey that is 9 mm off the floor, so the vertical gap alone eats most of a 20.3 mm trigger. Closing it is what the strike is FOR: Delheusy et al. 1995 measured the head translating ~8 mm horizontally and **~28 mm vertically** in the 80 ms cycle. The trigger is horizontal |
| 173 | The prey can own capture once a strike exists | **Refuted** | proximity capture respawned the prey the instant it entered striking range, so the strike could never fire -- measured **0 strikes across 1050 frames with the prey placed in range on every single step**. Capture belongs to the strike or to proximity, never both |
| 174 | The frozen walker can deliver the animal to striking distance | **Refuted, and this is the live blocker** | it cannot. It arrives within **4 cm** of a goal and its nose trails its trunk by about **5 cm**, so "the animal arrived" and "the mouth is on the prey" are five centimetres apart. With prey motion, a working stalk, a collapsed flee radius and a target aimed past the animal, the best approach across 700 steps was **34.5 mm against the 20.3 mm needed**, oscillating rather than converging. The walker was trained to reach a goal, not to place a mouth on a 9 mm cricket |
| 175 | The strike itself works | **Confirmed** | with the prey placed in range -- which the approach cannot do -- 78 strikes, 63 caught, 14 missed: **81 % against the published 82.9 %**, unfitted. `artifacts/video/hunt_session9.mp4` part 2 |
| 176 | The Session 9 videos show a changed gait | **Refuted -- they show HALF SPEED** | every Session 9 video tool appends one frame per 0.02 s brain step and wrote them at **25 fps**. That plays 20.0 s of simulation over 40.0 s of video: **2.0x slow motion**, in all four clips. The Session 4 renders were **50 fps** and real-time, which is why they looked right. The "lag" the user saw was a container header. Gait, body, checkpoint and physics unchanged -- 481 tests, three independent body-hash confirmations, walker speed 0.054 vs 0.055 m/s. Fixed to 50 fps; `accepted_walk_replay.mp4` was already real-time because it samples the trace at the output rate rather than per step |
| 177 | The brain runs the accepted walker | **Refuted** | the 4/6 gate evidence was measured under `gait_profile="lab"` (`artifacts/evidence/lab_frozen/report.json`). `GeckoBrainEnv` never passes the parameter, so `GeckoWalkEnv`'s **`"legacy"` default wins**. Same body, same frozen checkpoint -- model sha256 `77b7a99d...` is byte-identical to the one the evidence used -- but a different CPG scaffolding underneath. Same shape of defect as #165: a default won because nobody named the parameter. **Everything built in Session 9 sits on the un-gated profile** |
| 178 | The Session 4 renders can be regenerated | **Refuted** | the script that made `renders/session4/*.mp4` was **never committed**. Only `utils/render_trace_video.py` is in the history, and it cannot produce them: it replays a trace against the archived BODY file, so it draws the default floor, has no prey, and disables shadows. The setup is reproducible; the file is not |
| 179 | The accepted walker can be re-run live | **~~Refuted~~ RETRACTED same session -- it was my harness** | loading the accepted checkpoint into a live `gait_profile="lab"` env makes it **fall after 1.2 s**, in both the body XML and the world XML, with and without a distant target. Cause located but not fixed: `lab_controller_snapshot` is **None** in the live env, so the lab base controller is not configured -- the evidence protocol carries a whole reward-calibration and controller block that a bare constructor does not reproduce. Found within the hour: `GeckoWalkEnv` defaults to **`control_mode="raw"`**, which builds **no CPG at all**, and the frozen policy is a RESIDUAL that rides on one. With `control_mode="cpg_residual"` the accepted walker runs the **full 20 s with zero falls**, 1.642 m of path at 0.082 m/s, and `lab_controller_snapshot` is populated. The walker was never broken; I gave it no scaffolding to stand on. Recorded rather than deleted, because I reported it to the user as a finding before checking my own harness |
| 180 | Running the brain on the gated profile costs something | **Refuted -- it costs almost nothing** | `gait_profile` is now a real parameter on `GeckoBrainEnv` and reaches the walker. Measured over 500 brain steps, same seed, same action: **legacy 1.041 m (0.1041 m/s), lab 1.021 m (0.1021 m/s), zero terminations either way** -- a 2 % difference. `lab_controller_snapshot` is populated under lab and None under legacy, which is the honest check that the profile actually took effect rather than merely being stored. The default stays `legacy` because every brain checkpoint was trained against it and flipping a default quietly is how this class of defect gets made; `info["gait_profile"]` and `info["gate_validated_profile"]` now report the truth every step |
| 181 | The camera shake is the ground sliding | **Refuted twice, and the second answer is the right one** | first I blamed the sliding ground and bolted the camera to the world, which stopped the shake and let the animal walk out of frame. Then I blamed the azimuth swinging with heading, which was real but not the whole of it. The remaining shake is the camera pointing **at the trunk**, and the trunk rises, falls and yaws once per stride -- so the whole world jitters at stride frequency. Following a smoothed path at fixed height instead: measured background jitter **1.538 -> 0.000**. Two further filming faults fell out of it: a 1 s smoothing constant left the camera ~10 cm behind a 0.0975 m/s animal and framed it from the rear, and a rear-quarter azimuth makes a sprawling lizard read as flat and splayed while it is fully upright -- up-vector 0.998, never below 0.99, trunk 19-27 mm off the floor for all 1000 frames |
| 182 | The push-then-drag is in the Session 4 walk too | **Refuted -- I filmed the wrong profile** | the user said their demo does not drag, and they were right. Front-foot duty factor against a 0.70 target: **lab FL 0.203 / FR 0.591** -- asymmetric by nearly threefold, one front foot loaded a fifth of the step and the other over half, which IS the limp. **legacy FL 0.483 / FR 0.500** -- balanced, and it reads as a walk. Every A/B render I made this session used `lab`; the Session 4 demo is `legacy`. So #9j's pixel-identical result stands as a statement about the CODE and answers a question nobody asked. **The uncomfortable part: `lab` is the gate-validated profile and it is the one that limps.** Whether the gate battery scored front-foot symmetry at all, or scored it and accepted the failure as one of its two unmet checks, is the next thing to establish -- looking better is not the same as being right |
| 183 | The limp is the profile | **Refuted -- it is the POLICY, and the ledger already said so** | #182 blamed `lab` versus `legacy`. Wrong again. With the residual switched OFF -- the hand-written base, no policy -- **both** profiles balance: lab FL 0.432 / FR 0.454, legacy FL 0.463 / FR 0.491. The asymmetry only appears when `models/v4_5b_speed_polish_1m` is loaded on top. **The accepted walker IS the open-loop base** -- #71 states it outright, #11 measured the trained residual at **3/6 against the base's 4/6**, and `lab_zero_residual/report.json` records controller "zero residual with contact reflex" in its own protocol field. I read all three this session and still loaded the checkpoint into every clip. So every video I sent was the REJECTED walker, and I explained its limp away three times -- as a camera artefact, a frame-rate artefact, and then as an under-actuated forelimb the user would have to accept. The forelimb IS under-actuated (#18) and that is real; it is not what made the animal limp on screen |
| 184 | The limp is the policy | **Partly -- it is the PAIRING, and #183 overstated it** | all four combinations measured, 800 steps each, front-foot duty against a 0.70 target: **lab + policy FL 0.203 / FR 0.591, gap 0.388 -- the only one that limps.** lab without policy 0.460/0.433. legacy with policy 0.483/0.500. legacy without policy 0.463/0.491. Three of four are balanced. So the trained residual is fine on `legacy` and wrong on `lab` -- a learned correction applied to a base it does not fit -- rather than a policy that is bad everywhere. #183's conclusion stands (the accepted walker is the zero-residual base, #71, and the trained residual measured 3/6 against it, #11); its stated MECHANISM was too broad, and the one combination I rendered all session happens to be the single broken pairing |
| 185 | Recording a broken configuration in the ledger prevents it | **Refuted -- it did not, for a whole session** | #71 and #11 both say the accepted walker is the zero-residual base, and `lab_zero_residual/report.json` states its controller in its own protocol field. All three were read this session, two were quoted back, and the checkpoint was still loaded into every render. Written evidence stops nothing on its own: the pairing was assembled out of **two independent defaults** -- `GeckoWalkEnv(gait_profile="legacy")` and a checkpoint path carrying no record of what it was trained against -- so no single line of code looked wrong. `common/walker_pairing.py` now warns at construction with the measured numbers and what to do instead. It **warns rather than raises**, because the broken configuration has to stay runnable or the evidence condemning it becomes unreproducible |
| 186 | The walker stops short because it walks badly | **Refuted -- it stops short because of what "arrived" MEANS** | `_target_egocentric` measured distance from `xpos[trunk]`, and `reached = dist < 0.04`. So the goal test never mentioned the mouth: the body centre gets within 4 cm, the episode succeeds, and the nose -- which trails the trunk by about 5 cm -- is still two strike-lengths away from prey that must be struck from **0.0203 m**. The policy is not failing at its job; its job was the wrong one. `approach_site="mouth"` measures from `nose_tip` instead. Additive: `"trunk"` stays the default because every checkpoint was trained on it, and the observation stays 92-D so none of them break |
| 187 | The elevation switch is a soft weight | **Refuted -- it is categorical** | the same stimulus means opposite things by height. Overhead looming disc: **75 % escape**, latency 0.45 s. Identical disc to the side: 53 %, latency 1.79 s. Small sweeping disc overhead: **freeze**. Same disc to the side: **80 % approach**. Floor-mounted looms produced **zero** escapes. Anatomically grounded -- medial colliculus carries the upper field to the escape pathways, lateral carries the lower field to the hunting pathway. Implemented as a hard switch at the horizon. **Mouse, and no reptile has ever been tested** |
| 188 | A single-frame difference can measure prey speed | **Refuted -- the band is finer than the instrument** | real prey moves **0.02-0.12 cells per frame**: at the registry's 0.047 m/s a cricket subtends 26.9 deg/s at 0.10 m and 5.4 at 0.50 m, needing **8 to 41 frames to cross one cell**. Differencing successive frames returned whatever the centroid noise was -- the same 0.3 px/frame stimulus read 110 deg/s and a 0.8 px/frame one read 0.0. Replaced with a windowed estimate that **refuses** rather than guesses below one cell of travel |
| 189 | The windowed estimate makes the velocity rule usable | **Refuted, and this is the honest limit** | measured over 120 frames of sustained motion: **9 deg/s reads 52 (484 % error), 27 reads 58 (114 %), 60 reads 62 (3 %), 150 reads 155 (3 %), 400 reads 414 (3 %)**. Accurate from ~60 deg/s upward and wrong below it -- and a cricket across the whole working range of 0.10-0.50 m sits at **5-27 deg/s**, entirely inside the unreliable band. **So the rule does not currently discriminate real prey.** It does no harm (the over-estimate lands near the peak, so prey is accepted rather than rejected) and it correctly rejects the genuinely too-fast and too-slow. Raising the resolvability floor from half a cell to a full cell moved the 9 deg/s error from 480 % to 484 %, so the fix is a finer map or a longer window, not a threshold |
| 190 | Adding a rule cannot break an existing passing test | **Refuted** | `test_it_reports_the_side_the_target_is_on` placed its synthetic cricket at rows 30:34 of a 64 px frame -- the exact vertical midline, which the new rule reads as the **horizon**. The target became ambiguous by design and was rejected. The vertical position was always incidental to a left-versus-right test, and it had never put the target where a cricket actually is, which is on the ground below the horizon. Moved to rows 44:48 and **the edit is documented in the test itself**, because changing a passing test to accommodate new code is exactly the move this project's rules exist to catch |
| 191 | The gecko basks under a light | **Refuted** | it is **thigmothermic**. Body temperature tracks the **substrate** at r2=0.97 against air at 0.92-0.93, n=12, and melanistic pigment has no effect on heating rate (r2=-0.08, P=0.68) -- so warming is by CONTACT, not radiation. The world therefore gets a warm **surface**, not a lamp, and the `bask` behaviour channel is named for a mechanism this animal does not use |
| 192 | The animal digs itself a burrow | **Refuted** | it uses voids that **already exist**. Retreat chambers traced through a demolished stone field wall were masonry artefacts -- verbatim, "the lizards apparently had done nothing in the setting of the site". So shelter is a raised slab with a crevice under it rather than diggable ground. GREY-FIELD: the source is a non-peer-reviewed field manuscript, and it is still the best description of wild habitat that exists for this species |
| 193 | An added world body must be mocap or it enters qpos | **Refuted -- that tests a proxy** | the guard refused any non-mocap addition, which would have forced shelter and a warm surface to be fake objects the animal could walk through. A **jointless static** body adds no degrees of freedom either: verified against this body, adding a jointless box leaves nq at 39 and nv at 38 unchanged while nbody goes 24 -> 25. The check now tests the thing it cares about -- nq and nv identical, and every added body mocap **or** jointless. A body carrying a joint is still refused |
| 194 | Safety beats warmth when the animal chooses shelter | **Refuted** | given a safe **cool** shelter against an exposed **heated** open area, geckos "selected heat sources over shelters" (n=8 juveniles). Corroborated in another genus: 9 *Gehyra* species, n=85, used a humid crevice for under 10 % of crevice time whenever the dry alternative was warmer. A measured conflict between two drives, which is what an action-selection model should have to resolve |
| 195 | Vision is this animal's route to danger | **Refuted** | antipredator reaction probability, **this species, n=42**: **0.21 to snake scent alone, 0.07 to the sight of a live snake**. Factorially only the chemical term is significant -- chemical chi2=8.098 p<0.0044, visual chi2<0.01 p>0.9, mechanosensory chi2<0.01 p>0.9. Authors verbatim: *the initiation of costly defensive action remains strictly gated by chemoreception*. A flee channel wired to the eye is close to modelling the wrong sense |
| 196 | Smell can be given a gradient | **Refuted -- it would be invention** | the assay behind every published number presents the swab **1 cm from the snout, in plain view**. It shows the animal can TELL cricket chemicals from a control -- 3.0 flicks/min against 14.57, n=7, a 4.9x rise. It shows **nothing about localisation**, and no gecko has ever been shown to find prey by smell alone. Cooper flagged the missing experiment in 1998 and nobody has run it since. `brain/vomeronasal.py` returns a scalar per odour and **no bearing**, and states the refusal in its output so a later reader finds an assertion rather than a missing key |
| 197 | This animal is simply nocturnal | **Partly -- crepuscular is the better reading** | the literature is split, the same species called nocturnal in one paper and crepuscular in another. The mechanistic result favours the second: **preferred body temperature rises through the light phase and peaks toward its end**, which Angilletta et al. 1999 propose is what initiates evening emergence. So arousal is modelled as already rising before dark rather than switching on at it |
| 198 | A sleep module can borrow the bearded dragon's cycle period | **Refuted -- and left as a gap** | *E. macularius* IS in the reptile sleep literature (Bergel et al. 2026, seven lizard species, **n=2** leopard geckos, EOG under each eyelid -- it closes them, a tokay cannot). That establishes the animal sleeps and its eye movements are measurable. It does **not** supply a cycle period. A well-known agamid figure exists and borrowing it would take one line, so `sleep_cycle_period_s` is recorded **null** and a test asserts it stays null. `brain/arousal.py` is therefore a **circadian gate, not sleep architecture** |
| 199 | The activity numbers are solid enough to build on | **Partly** | onset after dark is **81 min, n=1**, with **SD 89 and range 11-247** -- the standard deviation exceeds the mean. The evening peak window is n=18 but **captive**, from a welfare study that scored behaviour during the window it expected activity in, so it partly describes the experimenters. Both are used because they are all that exists, both are tagged `uncertain`, and **no field activity budget for a wild leopard gecko has ever been published** |
| 200 | The frozen walker cannot reach striking distance (#174) | **REFUTED -- it can, and #174 was measuring the wrong thing** | told that the goal is the MOUTH at the published 20.3 mm, the same frozen checkpoint arrives **75 times in 160 s -- one every 2.1 seconds**, mean closest 20.5 mm. #174 drew its conclusion from a scripted approach inside `GeckoBrainEnv`, not from the walker. **The walker was never the blocker**, and a 3 M-step retrain was launched to fix a problem that did not exist |
| 201 | Retraining on the mouth goal would improve it | **Refuted** | stopped at 2.14 M of 3 M steps. Closest mouth-to-goal **170.6 mm at 250 k and 167.9 mm at 750 k** against the old policy's 20.5 mm -- an order of magnitude worse. Episode length in training collapsed 1340 -> 12 steps while episode reward rose, which looks like reward hacking but is not: per-step reward also improved, −1.51 -> −0.32. The run was simply worse at the task and was killed rather than left to finish |
| 202 | The blocker is in the walker | **Refuted -- it is in the brain's targeting** | `GeckoBrainEnv._brain_action_to_target` places the walker's goal **0.05-0.80 m ahead in the BODY FRAME**, derived from the brain's action. It never sets the target **at the prey**. So the brain can only ask the walker to head roughly that way, and the last few centimetres -- the ones that decide a strike -- are exactly where that loses |
| 203 | Aiming the walker AT the prey fixes the approach | **Refuted -- it makes it strictly worse** | 5 seeds, 60 s each: aiming **past** the prey gives **29 strikes, 26 caught**, closest 10.2-18.6 mm. Aiming **at** it gives **0 strikes**, closest 23.4-29.4 mm. Arriving at a goal means stopping at it, and the mouth needs the animal to keep going. Kept as an option, defaulted off, measurement recorded. #202 was wrong about where the blocker was |
| 204 | The hunt composes end to end | **CONFIRMED** | moving prey, a stalk that collapses the flee radius, an approach to 10-19 mm against a 20.3 mm trigger, a strike, and a capture. **26 of 29 strikes land -- 89.7 % against a published 82.9 %**, within sampling noise at n=29. The first time in this project that the animal has caught anything |
| 205 | The walker or the targeting was blocking the hunt | **Refuted -- it was the PREY** | what unblocked it was prey motion and the speed-dependent flee radius, both added earlier this session. A stationary cricket that bolts at a fixed 75 mm regardless of approach cannot be hunted by anything. Three separate blockers were diagnosed -- the eye (#134), the walker (#174), the targeting (#202) -- and **all three were downstream of a prey model that made the task impossible** |

### #138 and #134 — the two I told the user were true

These are the entries that matter most, because both left this project as
statements of fact before they were checked.

I told the user a sharper eye would find the cricket. Four configurations later
the correlation was still zero. I told the user geckos head-bob to reveal
stationary prey by motion parallax. There is no gecko study, no lizard prey
study, and the confident version of the claim lives only on pet-care sites.

The second one is the worse error. It was not a bad inference from thin
evidence — it was **repeating a popular claim in the voice of a finding**, in a
project whose entire discipline is refusing to do that.

One half survives. The **pit viper** paper argues the mechanism directly: IR
tectal units are near-silent at 5 °/s, so a motionless snake cannot detect a
motionless object, and the authors conclude it must be revealed by **scanning
head movements**. Same clade as our animal, the authors' own conclusion. Self-
generated motion to reveal still objects stands; head-bobbing as a
*rangefinder* does not.

### #135 and #136 — the cricket was never findable

Two measurements settle why the tectum failed, and neither is about the eye.
The prey moves **0.0 mm**, and it is off the rendered image **81%** of the
time. A motion detector was pointed at a stationary object it could mostly not
see. Sessions 8b–8d spent their whole effort on the detector.

### #149 — a test report that was wrong

Sessions 8b and 8c both reported "456 tests pass, no expected failures". Eight
tests fail, and a clean worktree at that commit fails the same eight, so the
report was wrong when it was written rather than broken afterwards.

The cause underneath is worse than the count. **Every evidence file in sessions
3 and 4 records body hash `756765356417749c`.** `morphology/gecko_body_lab_v2.xml`
hashes to `db650f6bf55852cb` in LF form and `506712cb13cccfc3` in CRLF form,
and no historical committed version of the file produces the recorded hash.

So the accepted 4/6 walker's evidence was measured against a body that is not
in the repository, and the gate contracts have been saying so. This is
unresolved and it sits underneath the locomotion work, which everything else
stands on.

### What the literature changes about the plan

Five findings, all published, that redirect the build:

- **The animal does not chase.** Strike from ~2 cm at 0.851 m/s, walking at
  0.055 m/s, cricket escape 0.093–0.143 m/s. Pursuit is arithmetically
  impossible and the real animal does not attempt it. **There is no strike in
  the code**, so the hunting channel currently cannot succeed by any route.
- **Attack distance and strike speed scale together**, r=0.47, p=0.009 — a real
  control law, from real animals, ready to implement.
- **Two strikes, not one.** The published *E. macularius* ethogram (n=18)
  separates `snap` — "a snap can be unsuccessful" — from `bag jump`, "jump
  towards a prey in order to bag it", alongside `walk slow motion` "mostly in
  context of prey capture".
- **The tail is not decoration.** Lateral tail undulation drives pelvic
  rotation and therefore step length; restricting it reproduces the kinematic
  changes of autotomy. The body already has five tail yaw joints and an
  actuator pair, all currently unused.
- **Hearing degrades with cold** — sensitivity falls and best frequency drops
  as temperature drops, 249 functions across 50 animals. The only measured
  coupling between two systems this project already has.

### What cannot be built honestly

Recorded so the gaps stay visible rather than getting invented later:

- No CTmin for this species or any eublepharid. The cold side of the thermostat
  has no floor.
- No field study of wild *E. macularius* at all — no home range, no activity
  budget, no diet, no microhabitat selection. The richest habitat description
  in existence is a non-peer-reviewed field manuscript.
- No tightness-of-fit or crevice-preference experiment. The "tight hide" rule
  every care sheet states has **zero** primary support in this species.
- No prey-capture kinematics, foraging-mode metric, or approach speed for this
  species.
- No visual receptive field, size tuning, or velocity tuning for **any** lizard
  tectum. No prey-selective cell class has ever been described in a reptile.
- ~~No cricket substrate-vibration spectrum~~ - **found, and it closed the idea
  rather than opening it.** See #162-164. What is still NOT_FOUND is the
  vibration a cricket makes by *walking*: every measurement is of deliberate
  courtship signalling. The nearest usable figure is four unnamed arthropod
  species walking on sand (Devetak et al. 2007), and the identity of those four
  is behind a paywall.
- No calibrated vibration threshold has ever been published for a **whole
  lizard** - every reptile vibrogram in existence is a python.

---

## Scoreboard

| | |
|---|---|
| Hypotheses tested | **205** |
| Refuted | **165** |
| Confirmed | **27** |
| Partly | **12** |
| My own method errors | **35** |
| Tests passing | **524 of 524**, one skip |

**Seventy-seven per cent of everything tried was wrong.** That is what the map
is made of.

Session 9 is the sharpest entry in that number and the only one where the
refutations came from **reading rather than running**. Forty-three hypotheses,
forty-two refuted. Three of them — #134, #138 and #150 —
had already been stated to the user as fact before anyone checked. #150 was
stated *inside this session*, in the entry above, and corrected an hour later.

Two of the entries above retired a *failure* rather than a hypothesis. The
project had been reporting an inverted result for three sessions against a
published number that did not mean what it was being read to mean.

The one number worth watching: the count of *my own* method errors grew faster
this session than the hypothesis count. That is the healthier direction — a
method error found is a whole class of future failures closed, and four of the
thirteen were found by something other than me.
