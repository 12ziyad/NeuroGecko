# NeuroGecko — the map

**One document. Everything tried, everything that failed, why, and what fixed it.**
Past, present, future. Updated every session; nothing removed.

Last updated: Session 6f.

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
| **World** — no cheat, textured floor, narrowed camera, fleeing prey | ✅ done |
| **Brain 1/8 — hypothalamus** (hunger, energy, fatigue, thermostat) | ✅ done |
| **Brain 2/8 — basal ganglia** (action selection) | ✅ validated, migrated, and running live beside brain 1 |
| **Brain 3a/8 — spinal rhythm** (the leg oscillators) | 🟡 built, matches the clock, not yet driving the walker |
| Brain 3b–8 — brainstem, retina, tectum, sleep, memory | ❌ |
| **Proof** — 15-test battery | 🟡 1 run (A3, failed then fixed) |

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

### ← Where we are paused, right now

- **Done and pushed:** everything above, 404 tests green.
- **Running in the background:** a hunt for why our dopamine curve sits +0.10
  high. Not a blocker; nothing depends on it.
- **Half done:** the spinal rhythm is an oscillator with a command input and it
  matches the old clock to 3.3 × 10⁻¹¹ of a stride — but it is **not yet the
  thing driving the legs**.
- **Next up, not started:** wire the cord into the walking controller and re-run
  the six gait gates with the coupling on. Nothing is claimed about the gates
  until that runs. Then the brainstem.

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

## Scoreboard

| | |
|---|---|
| Hypotheses tested | **102** |
| Refuted | **76** |
| Confirmed | **21** |
| Partly | **4** |
| My own method errors | **22** |
| Tests passing | **404**, no expected failures |

**Seventy-three per cent of everything tried was wrong.** That is what the map
is made of.

Two of the entries above retired a *failure* rather than a hypothesis. The
project had been reporting an inverted result for three sessions against a
published number that did not mean what it was being read to mean.

The one number worth watching: the count of *my own* method errors grew faster
this session than the hypothesis count. That is the healthier direction — a
method error found is a whole class of future failures closed, and four of the
thirteen were found by something other than me.
