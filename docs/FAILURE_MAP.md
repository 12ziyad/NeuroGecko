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
| **Brain 6/8 — day/night clock** | ✅ **connected — it drives the `rest` channel** (#352). Built 9w, consumed by nothing until now (#262, #347). Sleep architecture still open — #198 |
| **Behaviour channels reachable** | 🟡 **5 of 6** — hunt, explore, flee, bask, rest. `groom` unreachable and honestly zero (#353), blocked on #354 |
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

### Where the work was paused before Session 13

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
- **The animal now runs its own brain (#273).** Hunger reaches the basal ganglia,
  the basal ganglia chooses, and the choice moves the body: 3 seeds, prey on
  screen 828/310/888 of 3000 steps, decision precision 67 % / 0.6 % / 100 %,
  closest approach 72 mm against a 20.3 mm trigger. Four of the five modules the
  audit found orphaned are wired (#274). It does not catch anything yet.
- **Brain 4 is half done, and session 12 changed what "the problem" means.** The
  session 9 explanation is **superseded** (#227), and so is session 11's
  replacement. Three framings have now been refuted in order:
    - "The prey is off-image 81 % of the time" — measured on a passive census
      with the prey stationary. Superseded by #227.
    - "The detector is the problem." Widening the temporal baseline did double
      the reported recall (#235), and that improvement is real, but **the recall
      it improved was measured wrong** — a report counted as a hit if the prey
      was anywhere in frame, not if the report pointed at it. Real recall during
      search is 7.5 %, not 27 % (#252).
    - "The animal faces its food and mis-sees it." With the oracle off it faces
      **away** from its food on 97 % of frames (#244), because the search
      heading is body-relative and the pattern could not turn around.
  **What is actually true, measured (#257):** walking, the motion map's peak has
  median 0.0696 with the prey on the image and 0.0872 with it off — the noise is
  larger than the signal, and no threshold, quorum, blob rule or persistence test
  can separate them. Standing still, the same two numbers are ****0.1499 against 0.0751**, and the best
  operating point moves from 57.2 % recall at 20.3 % false alarms to **92.7 % at
  48.5 %**. The eye works either way; it works *far better* for an animal that
  has stopped. (An earlier single-seed version of this claim said the noise
  exceeded the signal while walking — corrected in #259.)
- **The animal could not stop.** `engage` changed the walker's goal distance and
  nothing else; the CPG is a fixed-frequency oscillator and the body walked at
  35 mm/s whatever the brain asked (#253). Every "freeze and scan" behaviour
  written before session 12 was walking. `locomotor_drive` fixes it and is
  bit-identical at 1.0, so no gate moves — and it gives the project its first
  graded speed control (35 / 20 / 9 / 0 mm/s).
- **What is now known to be missing:** a strike. The animal walks at 0.055 m/s
  and the cricket escapes at 0.093–0.143 m/s, so pursuit cannot work and the
  real animal does not attempt it — it strikes from ~2 cm at 0.851 m/s. Until a
  strike exists the hunting channel cannot succeed by any route. Oracle-free,
  closest approach is now **30.5 mm** on the best seed against a 20.3 mm
  trigger, down from 326 mm, with a median of 60.6 mm over 8 seeds against
  152.8 mm for a bearing-scrambled control (#261). The last centimetre is the
  open problem, and it is the walker's, not the eye's.
- **Next decision, not yet taken:** chase the body-hash mismatch first, or
  build the world and the strike and come back to it.

---

### Phase 8 — the behaviour channels (Session 13)

| Session | What was done | Where it stopped |
|---|---|---|
| **13** | read the whole record first, then measured the four pending behaviour items before touching them | four of the record's own numbers did not reproduce |
| **13** | `rest` moved from fatigue to the day/night clock; brain 6 connected | **5 of 6 channels reachable**; the animal rests by day with its eyes shut |
| **13** | flee's unpublished 0.28 removed and its trigger unwired from touch | the prediction survived the correction rather than being rescued by it |
| **13** | groom's untagged 0.05 tonic removed | groom blocked on a contradiction inside this repository (#354) |
| **13** | **I truncated this file to 0 bytes and rebuilt it** | recovered and verified byte-for-byte in length (#359) |

### ← Where we are paused, right now

- **Five of six behaviour channels release**: hunt, explore, flee, bask and now
  `rest`. Measured on the accepted walker (`gait_profile="lab"`,
  `use_policy=False`, `info["accepted_walker"]` asserted), furnished world,
  seed 3, 600 autonomous steps per phase. In the light phase the animal covers
  **0.1 cm** with its lids shut at 70 degrees on every step; at its published
  activity peak it covers **33.5 cm** with them open on every step.
- **Brain 6 is wired to something for the first time** (#352). It was built in
  session 9w and consumed by nothing (#262); session 12 added a flag that made
  it look otherwise (#347). Arousal now drives `rest`, and that is the only
  consumer the sources speak to.
- **Four numbers the record carried to four decimal places did not reproduce**:
  the release floor (0.2008 against a measured 0.199878, #351), the rest-fatigue
  plateau (0.011, a typed literal, #350), the fatigue it stood for (an artefact
  of sampling speed over one control step rather than one stride, #349), and
  flee's 0.28 ceiling (a sum of two published numbers, added on a trigger wired
  to touch rather than sight, #355).
- **An untagged constant was hiding a degenerate fixed point** (#357). With an
  exactly empty salience vector the selector part-releases all six channels
  equally and `argmax` returns whichever is declared first; groom's 0.05 kept
  the vector from ever being empty, so the test asserting otherwise passed for
  the wrong reason.
- **`groom` is blocked, not solved** (#354). This repository asserts both that
  no primary source documents eye-licking in any eublepharid and that Delheusy
  et al. 1995 records the tongue reaching the eye in this species.
- **THIS FILE HAS NEVER BEEN COMMITTED PAST SESSION 11b.** `git HEAD` carries
  rows 1-226; rows 227-348 and every session-12 prose edit exist only in the
  working tree. That is how a truncation in this session (#359) came within one
  persisted tool capture of destroying 122 ledger rows. It was rebuilt and
  verified, but the exposure is still open until this is committed.
- **Not touched, and pre-dating this session**: the eight `GateContracts`
  failures (545 tests, 8 failures, 1 skip), the front-foot duty disagreement,
  and the body-hash question — whose failing test reports that the PHYSICS
  digest differs too, not only the file bytes, which #167 claimed was closed.
- **Next decision, not yet taken**: commit, then settle #354 from the sources,
  or take the light-phase `rest`/`bask` tie (#358) by giving the animal a
  retreat to rest in — which needs a source for when a leopard gecko retreats,
  and nobody has opened one.

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
| 206 | The front-foot pair is the right thing to measure | **Refuted -- the defect was in the HIND feet** | measuring all four with one consistent method: the configuration every Session 9 video used (`legacy` + policy) puts the front feet at 0.704/0.601 but the **hind feet at 0.435/0.367 against a published 0.765 -- barely half**. I had only ever measured the front pair, so the defect sat outside the metric I was checking. The user saw it on screen twice before it was measured |
| 207 | The accepted walker is only slightly better | **Refuted -- it is near-perfect and nothing else is close** | `lab` + **no policy**: **0.770 / 0.764 / 0.787 / 0.790**. All four feet even to within **0.006**, and all four near their published targets (fore 0.70, hind 0.765). Every other combination is visibly worse on at least one pair. This is the 4/6 walker, and it walks like one |
| 208 | The brain environment can run the accepted walker | **Refuted** | it defaults to `gait_profile="legacy"` and **always** loads the frozen residual. The accepted walker is `lab` + zero residual, and **nothing in `GeckoBrainEnv` could express that combination**. So every clip made from the brain env was filmed with a walker this project had already rejected -- not through a wrong argument, but because the right one was unreachable |
| 209 | The eye can tell prey-present from prey-absent | **Refuted -- the sharpest result in the vision work** | with prey in the world it fires on **72 %** of frames at mean salience **0.4161**. With **no prey in the world at all** it fires on **72 %** at **0.4034**. Separation d = **0.036**. The reported bearing correlates **+0.020** with the true one across 1500 of 1500 frames, mean error 48.7 deg, and **−0.034** within 15 cm. It is not finding prey badly; it is not finding prey |
| 210 | Salience as a ratio to the frame's own peak can work | **Refuted** | `local_peak / raw_peak` cannot score an empty world at zero, because the frame's own maximum is the denominator: when the whole field slides, both terms scale together and the quotient parks at a constant. Replaced with an **absolute** scale -- correct in principle, and it **did not fix detection**: d moved 0.036 -> −0.128, still noise. Recorded as attempted and insufficient rather than quietly kept |
| 211 | A local-surround comparison removes self-motion | **Refuted, and this is the root cause** | it assumes self-motion is spatially **uniform**, so that subtracting a neighbourhood mean cancels it. Optic flow is not uniform: a walking animal's field **expands from a focus of expansion**, and near ground slides faster than far ground, so local excess is large everywhere whether or not prey exists. The published route is to predict the flow field from the animal's **own motion** -- efference copy -- and subtract that instead. Mice hold prey in the retinal region with the **least** optic flow (Holmgren et al. 2021), which is the same principle used the other way round |
| 212 | The cricket is too small for this retina to see | **Refuted -- resolution was never the limit** | with the animal held STILL and prey wiggling ahead, peak salience by range: **0.357 at 0.04 m, 0.767 at 0.08, 0.675 at 0.12, 0.825 at 0.20, 0.673 at 0.30, 0.834 at 0.45**. It responds at every range tested, including 0.45 m where the prey covers **half a cell**. Three sessions of "the prey is smaller than one cell" were describing a real constraint that was not the operative one |
| 213 | An efference copy fixes prey detection | **Partly** | subtracting the flow the animal's own yaw and forward speed predict cuts false alarms from **72 % of frames to 3-7 %**, and flips the still-versus-moving effect to the right sign for the first time (**d = +0.181**, after −0.128 and −0.212). But detection during a real hunt is still weak in absolute terms: the same eye that peaks at 0.67-0.83 on a stationary animal averages **0.02** in a hunt, because prey is often out of view or not moving on a given frame. **The oracle stays until this closes** |
| 214 | The animal should look while it walks | **Refuted -- it must look while STILL** | measured across a real hunt: moving faster than 0.03 m/s the eye fires on **3 %** of frames at mean 0.0076; standing still it fires on **7 %** at **0.0223**. Walking is what blinds it, and the creep this project already implements -- move one step in four -- is exactly the freeze-and-look pattern a stalking predator uses. The detector should be **read during the pauses**, not continuously |
| 215 | Day, Crews & Wilczynski studied spatial learning in *E. macularius* | **Refuted -- and I asserted it twice** | **they never studied leopard geckos.** Their lizard output is three papers on two other families. I stated it as fact in an earlier session and then wrote it into the research brief as a known starting point, which could have propagated the error into the answer. The confusion is understandable -- David Crews published extensively on this species, on temperature-dependent sex determination -- and it is still a citation I invented the content of |
| 216 | Memory and learning are too thin to build honestly | **Refuted -- they are the buildable ones** | three papers on the TARGET species, 2022-2026, give real numbers. Learning: **hazard ratio 1.09 per trial**, ceiling **3.02x latency / 4.59x path** after 20 spaced trials, n=38-42. Memory: no significant loss at **2 months**, route precision degraded at **4 months** (path HR 0.49, p=0.0044) while the ability to reach the goal was not yet, and **indistinguishable from naive by 6-14 months**. The modules deferred for being unsupported turned out to be the supported ones |
| 217 | The prey detector is the well-evidenced module | **Refuted -- it is the unsupported one** | **no size tuning, speed tuning, contrast threshold, elevation tuning, detection latency or acuity has ever been measured for any gecko.** The detector must be assembled from a fish larva, a mouse and a toad. The one exception is the strike itself, which has numbers from a gecko in the same family. The urgent module is the invented one and the deferred modules were the evidenced ones -- exactly backwards from how this project has been treating them |
| 218 | Latency is the right way to score a hunt | **Refuted for this species** | **path length is significant across training and latency is not** -- F=24.157, p<0.0001 for path against F=0.326, p=0.568 for latency, n=42. Worse, geckos found a hidden goal FASTER in complete darkness than at the end of normal training, by swimming twice as fast along paths twice as long: **a fast undirected search solves a latency-scored task with no spatial memory at all**. Scoring the hunt by time-to-capture would reward exactly that |
| 219 | Removing the oracle will show whether the eye works | **Refuted -- not on its own** | when the oracle goes, capture will get worse for two different reasons at once and no single number can separate them. The spec's answer is instrumentation first: keep the oracle running as a **shadow signal** feeding only the logger, so "the eye cannot see the cricket" and "the eye sees it and the body cannot get there" stop producing the identical symptom |
| 220 | The eye's problem is that its bearing is wrong | **Refuted -- its problem is that it rarely reports at all** | first hunt scored with the oracle demoted to a shadow, 1500 steps: **not detected 1419, detected-bearing-wrong 37, detected-bearing-usable 44**. It reports something on **5.4 %** of steps, and when it does the bearing is usable **more often than not**. Three sessions of vision work assumed the bearing was the broken part. The broken part is the detection rate |
| 221 | Path efficiency in a real hunt is respectable | **Refuted** | **0.0146**. The animal covers roughly seventy times more ground than it closes. It is *approaching* on 47 % of steps by the published three-part criterion, so it is not idle -- it is pointed the right way and getting almost nowhere. This is the number the oracle removal has to be judged against, and it is the first time it has been measured at all |

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

## Ledger -- the report is generated, and the scoreboard had drifted (Session 11)

The user asked for the whole project as one document: what was wanted, what was
done wrong, the map, the failures, the rules, the plan, every proposition with
its timing, every published paper leaned on, what is pending, and what has to be
checked. Building it meant reading the repository with a parser instead of from
memory, and the parser disagreed with the file in four places.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 222 | The scoreboard in this file is the count of this file | **Refuted -- it had drifted** | the hand-kept tally read 180 refuted / 27 confirmed / 13 partly, which sums to **220** against **221** rows. Parsing the table gives **186 / 22 / 12** and one retracted (#179). Nobody miscounted once; the tally was updated by hand session after session while the table was updated by machine. The scoreboard is now regenerated by `tools/report_data.py` and the drifted numbers stay in the history above rather than being quietly overwritten |
| 223 | `git blame` dates a hypothesis honestly | **Partly** | it dates when a line was last **written**, not when the thing was tested. The living map was created in one commit that transcribed every hypothesis predating it, so 132 rows carry 2026-09-07 and the timeline chart shows a day on which 132 hypotheses were tested. False. Drawn separately, hatched and labelled, rather than smoothed or dropped -- the alternative was to guess the real dates, which is the defect this file exists to prevent |
| 224 | The citations harvested from the repository are a bibliography | **Refuted** | **212 author-year strings and 51 DOIs**, and **zero** of them checked against any database. This project invented a citation four rows ago (#215) and overstated the absence of a real one. A harvest says where the repository leans, not what exists. The report says so in a box above the table, and the companion brief makes verifying them task one |
| 225 | Lines of code written measures the work | **Refuted** | `git log --numstat` reports **19,460,993** lines added. **19,410,189** of them are recorded physics traces -- one episode file is 1.5 million lines. Actual source written: **50,804 added, 1,190 removed** across 107 commits. Counting recorded output as authorship would have made the churn chart a picture of how many episodes were logged |
| 226 | A regex can extract the author names from this repository's prose | **Partly** | the first pass returned 296 "works" including `Nature 2024`, `Jun 2026`, `SR et al. 1985` and `The 2024`. Venues, month stamps and bare initials all match the author-year shape. Filtering those and folding `X et al. Y` onto `X, Y & Z Y` gives 212. The filter list in `tools/report_data.py` is a record of what the regex actually got wrong, not a guess at what it might |

---

## Ledger — the eye, decomposed (Session 12)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 227 | The prey is off the image most of a hunt, so the detector is near its aperture ceiling | **Refuted — and it was the auditor's own prediction** | measured across 6 independent spawns x 400 steps of the #220 configuration: the prey is **on the rendered image 73.5 %** of steps, not 19 %. #136's 18.7 % was taken on the passive `tectum_fov_census` configuration with `prey_total_travel_mm = 0.0` (#135) and has never been re-measured since the prey was made to move. The external audit reasoned from that stale number and predicted the two-channel detector would be cancelled. **A perfect detector has ~11x of headroom.** Plan step 5 is vindicated, not cancelled, and the prediction that said otherwise was made by the same method the map exists to prevent |
| 228 | The detector's size and speed tuning is what rejects the prey | **Refuted — it is the motion floor, by an order of magnitude** | the tectum has recorded its own rejection reason in `self.last["rejected"]` since it was written and **nothing had ever read it**. On the frames where the prey IS on the image and the eye stays silent: **below motion floor 91.4 %**, above the horizon 7.9 %, too slow 0.4 %, too fast 0.3 %. Size and speed together account for **0.7 %**. Every tuning constant in the specified two-channel detector addresses the 0.7 % |
| 229 | The floor is set too high | **Refuted — the signal genuinely is not there** | `MOTION_FLOOR = 1e-4`, declared INVENTED and set from the noise floor of a still scene. The prey is ~5 px across and its image moves a median of **1.01 px per control step** — a **quarter of a 4x4-pixel cell**. A quarter-cell displacement of a 5-px object survives neither the mean-pool nor the 3-cell surround subtraction. This is ledger **#188** — "the band is finer than the instrument" — restated at the cell level and never carried into the eye |
| 230 | Differencing the prey map across more than one frame raises detection | **Confirmed, but only once the efference copy stops fighting it** | with the efference copy OFF, recall on frames where the prey is on the image: **window 1 = 65.2 %, window 3 = 73.9 %, window 6 = 86.8 %**, and precision rises with it (67.7 / 69.8 / 73.0 %). Bounded rather than tuned: control runs at 50 Hz and scotopic flicker fusion in two nocturnal geckos is "up to 18 flashes per sec" (Dodt & Jessen 1961, J Gen Physiol 44(6):1143-1158, *Hemidactylus turcicus* and *Tarentola mauritanica* — neither an eublepharid), so 50/18 = 2.8 frames. That bounds the ORDER, not the value |
| 231 | The efference copy expectation must be scaled by the window span | **Refuted — and this was my own change** | self-motion accumulates across the window exactly as prey motion does, so scaling looked obligatory. It is wrong, because the difference operator **saturates**: once displacement exceeds the feature size, `\|L(t) − L(t−N)\|` stops growing, for the background and for the prey alike. Scaling the expectation linearly therefore over-subtracts. Measured at window 3: **scaled recall 3.8 %, unscaled 15.4 %** — and unscaled is better on precision too (91.9 % against 90.5 %). The scaling cost more than the window bought, which is why the first sweep showed the window making detection monotonically worse |
| 232 | The efference copy is a net win at its shipped gains | **Refuted — it spends 58 points of recall to buy 18 of precision** | `flow_gain_yaw = 0.020` and `flow_gain_surge = 1.60`, both INVENTED, and their own docstring says they are "set so that the flow a walking animal generates is roughly cancelled" — tuned until the false alarms went away. They were tuned against **#213**, which measured only "false alarms from 72 % of frames to 3-7 %", because the project had no column saying whether the prey was on the image at all and therefore **could not see the cost side**. With the column: efference ON recall **7.6 %** precision 85.4 %; efference OFF recall **65.2 %** precision 67.7 % |
| 233 | #220's "when it speaks it is right more often than not" is a statement about the prey | **Partly — one report in seven is about nothing** | #220 counted 44 usable bearings against 37 wrong and could not ask whether either was about the prey. Joining the prey-on-image column: **14.7 % of every report the eye makes happens with the prey not on the image at all** (157 reports, 134 with prey on image). Not fatal to #220's reading, and not visible to it either |
| 234 | The operating point should be the one that maximises F1 | **Refuted — F1 picks the config that undoes #213** | the full sweep, 6 seeds x 400 steps. Highest F1 is window 6 at gain 0.20: recall 74.3 %, precision 75.0 %, **F1 0.746** — and it fires falsely on **18.2 % of all frames**. #213's accepted achievement was cutting false alarms from 72 % of frames to **3–7 %**, so the highest-F1 point spends the whole of that result. F1 is symmetric and this problem is not. The criterion used instead is the project's own prior accepted band, and the tool says in its own output that it draws the curve and does not choose the point |
| 235 | Widening the window costs something | **Refuted — at gain 1.00 it is free** | shipped `motion_window = 1`: recall 7.60 %, precision 85.4 %, false alarms 0.96 % of frames. `motion_window = 3`, **every gain untouched**: recall **15.37 %**, precision **91.86 %**, false alarms **1.00 %**. Recall doubles, precision rises, the false-alarm rate is unchanged within noise. A strict improvement on every measured axis with no constant tuned — the window is bounded by a published gekkotan flicker-fusion ceiling, not fitted. Shipped as the default. `gain = 0.50` reaches recall 29.3 % at 4.75 % false alarms, inside #213's band, and is left as a declared choice rather than a default because that one IS a trade |
| 236 | `Retina.reset()` forgets everything an episode should forget | **Refuted — and this was a bug I introduced** | the window ring was added to `__init__` and not to `reset()`, so luminance frames leaked across episode boundaries and the windowed difference compared a new episode's first frame against the previous episode's last. The eye then reported a **left** bearing for a right-hand target. Caught immediately by `test_it_reports_the_side_the_target_is_on`, an existing test that was not written for this. Recorded because the test earning its keep is the point, and because "measure before claiming, including your own tests" is rule 5 |

| 237 | Re-running a diagnostic tool is a read-only act | **Refuted — it silently overwrote recorded evidence** | `tools/tectum_fov_census.py` writes to a FIXED path with no run identity in it. Re-running it during this session replaced `artifacts/evidence/session8/tectum_fov_census_world.json` — a **12-seed, 1800-frame** record carrying the `prey_total_travel_mm: 0.0` that ledger **#135 cites** — with a 4-seed, 600-frame run in which the prey travelled 318.66 mm. Caught by `git status` and restored with `git checkout`, so nothing was lost, but nothing prevented it either. Same class as #222 (a scoreboard that drifted) and #223 (`git blame` dating a claim by when its line was last written): the record is only as good as what stops it being quietly rewritten. **Every evidence writer in `tools/` shares this shape** |

| 238 | A finer retinal grid would raise detection | **Refuted -- and the record had already refuted it** | I proposed this from `Eye.__init__`'s default of 16 cells. `GeckoBrainEnv` overrides it: the shipped eye runs at **64 cells, one per pixel, already the finest possible**. Forcing it coarser confirms the direction -- 16 cells 4.1 %, 32 cells 8.6 %, 64 cells 15.4 % recall -- so resolution DOES drive detection rate, which corrects **#212**'s "resolution was never the limit" (that was peak salience on a stationary animal, not detection during a hunt). But there is no headroom left. Recorded because re-proposing a refuted hypothesis from an unread default is the exact failure this map exists to stop, and it happened with the map open |
| 239 | "The prey is on the image 73.5 % of the time, so the field of view is not the problem" | **Partly -- it is conditional on the oracle** | that 73.5 % was measured while the scripted creep steered the animal at the prey every step, using the true bearing from the physics engine. The repo's own passive census records ~13 %. So the framing term is fine WHILE SOMETHING IS ALREADY AIMING THE HEAD, and the claim does not stand alone. The RECALL figures are unaffected -- they are conditioned on the prey being on the image however it got there -- but the ceiling claim is |
| 240 | The eye's problem is uniform across the approach | **Refuted -- it goes blind exactly where the strike fires** | prey elevation in camera coordinates falls as the animal closes: -4.2 deg at 0.30-0.50 m, -10.8 at 0.08-0.15, -20.2 at 0.04-0.08, **-35.1 inside 4 cm**. At fovy 70 the frame stops at -35, so the fraction of frames with the prey on the image collapses from **100 % at 4-8 cm to 44.8 % inside 4 cm**. The camera looks forward; the cricket is on the ground; closing the range pushes it out of the bottom of the picture |
| 241 | A gaze loop should relax to level when the target is lost | **Refuted -- it starves the loop it belongs to** | first version decayed the aim whenever the eye reported nothing. Inside 4 cm the eye reports on ~1 % of frames, so 99 % of steps un-aimed the head and the gaze never exceeded **9.1 deg** against a target at -35. A target lost LOW in the frame is evidence the head is aimed too high, not evidence there is no target. Holding the aim instead took on-image inside 4 cm from 44.8 % to **94.7 %** at gain 1.0 and **100 %** at gain 2.0 |
| 242 | The published elevation switch can be applied in camera coordinates | **Refuted once the head can move** | the rule rejects targets above the horizon. Pitching the head down to keep a cricket in view raises that cricket ABOVE the optical axis, so the eye's own aiming manoeuvre made its elevation rule reject the prey it had just aimed at -- "above the horizon" rejections persisted at every range while the animal was looking straight at its food. Part 10 had already said why: of the two published elevation figures, "Both are consequences of head posture, not a world-frame rule." Subtracting the animal's own gaze command recovered recall from 4.7 % to **8.6 %** and precision from 78.2 % to **86.6 %** |
| 243 | Fixing the eye unblocks the hunt | **REFUTED, and this is the result of the session** | stacked, 6 seeds x 400 steps: recall **7.6 % -> 20.8 %** (window 8), on-image inside 4 cm **44.8 % -> 100 %** (gaze 2.0). Both real, both measured, neither tuned to a published number. **Strikes: 0 before, 0 after, in every configuration.** Closest approach 20.8-22.5 mm against a 20.3 mm trigger -- the animal misses by under two millimetres whatever the eye does. A single strike observed at one window and one gain did NOT reproduce when stacked and was seed-level noise reported as a result. The eye was never the binding constraint. The binding constraint is the last millimetre, and it is the walker: one gait, one locked frequency, no drive input, aiming past the prey by design (#203) |

| 244 | A search pattern that alternates its heading covers ground | **Refuted -- it cannot turn around, and this is why the animal never found anything** | the walker's heading command is BODY RELATIVE (`_brain_action_to_target` rotates it by the trunk matrix), so alternating +35/-35 deg makes the animal wiggle down a straight line forever. Measured with the oracle off, 600 steps: the prey was **BEHIND the camera on 581 of them**, outside the horizontal field on 19, and on the image on **0**. Median bearing to the prey 82 deg off the optical axis; median range 361 mm. Not mis-seen -- faced away from. Every session spent on the retina was spent on the 3 % of frames where seeing was possible in principle |
| 245 | False alarms are scattered, so several agreeing frames identify a real target | **Refuted -- the noise is not scattered and the rate was 20x what the threshold assumed** | `PreyEvidence` was sized for a 1 % per-frame false-alarm rate measured under the oracle. Measured with the oracle off: **19.7 % per frame**, 118 false alarms and 0 true positives in 600 steps. At that rate a 25-frame window holds ~5 reports and three landing inside a 25 deg tolerance happens by chance nearly always -- the quorum committed on **545 of 600 steps, every one to noise**. The threshold had never been derived from the noise it existed to reject. Kept, with `from_noise_rate` sizing the quorum from a measured rate against a stated chance probability, which makes it falsifiable |
| 246 | Giving the animal a head to sweep is enough to make it see | **Refuted -- a smoothly sweeping head IS self-motion, and the eye reads it as the world** | the morphology has carried neck_yaw +-40 and head_yaw +-30 since session 1, actuated, commanded by nothing. Commanding a smooth sweep put prey on the image for the first time (**0 -> 127 of 1200**, 36 true positives against 0) and simultaneously drove false alarms to **27.2 % of frames while standing still**, because the camera rides on the head. Subtracting the commanded head rate in the efference copy fixed the false alarms (3.8 %) and destroyed the detector with them (recall **52.6 % -> 3.4 %**) -- over-subtraction, the same failure as **#231**. Replaced with SACCADE AND FIXATE: the head is either moving fast, with vision suppressed, or exactly still, so there is no smooth self-motion to subtract in the first place. Saccadic suppression is the published vertebrate behaviour this is named after; the three constants are INVENTED |
| 247 | The efference copy can be computed from the trunk | **Refuted -- the organ was mounted on the wrong body part** | the only inertial sensors on this animal were `gyro_trunk` / `vel_trunk`, above the hips, and the eye rides on a head that yaws +-70 deg relative to that trunk and bobs with every step of the gait. Measured walking straight: the head's own yaw rate is **3-4x the trunk's** (median 0.37 vs 0.13 deg/s; 95th percentile 2.75 vs 0.73). The prediction was systematically too small and the residual was read as the world moving -- 30.4 % of frames false-alarming with the animal standing still. The semicircular canals of every vertebrate sit in the SKULL beside the eye. Added `gyro_head` / `vel_head` on the site the camera already occupies: passive, no mass, no DOF, appended after the existing sensor block so no address any gate reads moves; walker observation stays 92-D. Not privileged information -- a vestibular system is what knowing your own head is turning is FOR. False alarms while fixating **30.4 % -> 1.6 %** |
| 248 | A real target persists at one bearing across a fixation; noise does not | **Refuted -- measured, the two are indistinguishable** | within one fixation, bearing spread with the prey on the image **66.6 deg**, with no prey present **60.5 deg**. The reports were not pointing at the cricket even when the cricket was there, which is why the spreads match. Persistence is not a discriminator for this detector |
| 249 | A quorum sized by a Poisson tail rejects chance agreement | **Refuted -- wrong null, the multiple-comparisons error** | `FixationEvidence.step` searches for the DENSEST cluster anywhere in the field; it does not test a band fixed in advance. Scoring that against a single-band tail understates the chance rate by roughly the number of bands (5.83 at +-6 deg over 70 deg), and a quorum "derived" for a 0.3 % chance rate fired on **15 of 24** fixations containing no prey. Corrected to `1 - (1 - p_band)^bands`, which raises the derived quorum from 4 to 5 at the same measured noise |
| 250 | The fixation dwell is a free parameter | **Confirmed as derivable, and it was set far too short** | evidence for a real target grows LINEARLY with dwell while the quorum needed to reject noise grows only as a Poisson tail, so a longer look is strictly better until the cost of looking elsewhere bites. At the measured rates: dwell 18 -> 34.2 % chance of detecting a look that contains prey; dwell 50 -> **90.8 %**; dwell 100 -> 99.5 %. Set to 50 steps, 1.0 s at 50 Hz, the knee of that curve |
| 251 | Freezing the head's yaw makes the fixation still | **Refuted -- the pitch loop was still running through every "still" look** | the gaze-pitch tracker is driven by the eye, so during a long fixation it chased its own false alarms and moved the head continuously. The animal's report rate went from 20 % to **49.7 %** of frames and the adaptive quorum followed it to 12, which no real target could reach. Disabling the pitch loop during search then took prey-on-screen from 383 frames to **0**, because the downward tilt is what puts the ground in the picture at all. A search fixation needs a HELD downward pitch, not a tracker and not level |
| 252 | "The eye detected the prey" means the eye fired while the prey was on the image | **Refuted -- and this inflates every recall number this project has published** | of 36 reports counted as true positives by that definition, only **8 were aimed within 10 deg of the prey's actual bearing**. The rest fired at something else while the cricket happened to be in frame. Real recall during search is **7.5 %, not 27 %**. The relative comparisons in #230, #235, #238 and #243 survive -- both arms used the same loose definition -- but the absolute recall figures do not, and `aimed_recall` replaces them |
| 253 | The brain can ask the body to stop | **Refuted -- there was no off switch, and the "stand still and look" behaviour was walking** | `engage` in the brain's action vector changes the walker's GOAL DISTANCE and nothing else, and the goal is re-placed a fixed span ahead of the animal every step -- a carrot on a stick. Measured: engage +1 and engage -1 both travelled **222.8 mm in 300 steps**, identical to one decimal. Pinning the goal to the animal's own position changed nothing: the CPG is a fixed-frequency oscillator and the body walks whatever the brain asks. Added `locomotor_drive`, blending the gait toward its own cycle-mean posture, which introduces no new number: **1.0 -> 222.8 mm (bit-identical, every gate untouched), 0.5 -> 120.7, 0.2 -> 10.6, 0.0 -> 0.8 mm**. This is also the project's first graded speed control: 35, 20, 9, 0 mm/s |
| 254 | The detection threshold is above the sensor's noise | **Refuted -- it sits 39x BELOW it, and this is the root of the false alarms** | with the camera held motionless (head angular speed 0.007 deg/s) and nothing in the scene moving, the median peak frame-to-frame pixel change is **0.0039** against a `MOTION_FLOOR` of **1e-4**, and the eye fires on **196 of 300 frames**. Every false-alarm number in this session -- 19.7 %, 27.2 %, 30.4 %, 46.6 %, 75.5 % -- is the detector reporting rendering noise. The floor was INVENTED and never checked against the noise it exists to reject |
| 255 | The cricket is too small to register in the motion map | **Refuted -- it registers strongly, and the peak is simply being stolen** | with the prey on the image, the prey's own cell is the map's GLOBAL ARGMAX on **35.7 %** of frames. Chance for a +-2 cell window on a 64x64 map is 0.6 %, so that is **60x above chance** and the signal is large. On the other 64 %, about three noise cells outrank it -- median 0.07 % of the map beats the prey cell. The detector's problem was never sensitivity |
| 256 | A cricket is a small blob and noise is scattered single cells, so component size discriminates | **Refuted -- the noise blobs are BIGGER** | connected component containing the peak: median 5.5 cells when the peak is the prey, **10.0 cells when it is not**, 75th percentile 10 against **61**. Requiring a minimum size keeps more noise than prey at every cut. Picking the strongest component of at most 15 cells instead of the global peak gave **15.7 % precision against a 16.1 % baseline** -- no effect. Magnitude, persistence, bearing spread and blob size have now all been tested and none of them separates signal from noise WHILE THE ANIMAL IS WALKING |
| 257 | Standing still is what makes the prey visible | **CONFIRMED, and it is the result of this session** | the same measurement, with the animal genuinely frozen (`locomotor_drive` 0.0, which did not exist until #253) and the prey within 180 mm, 12 trials: local-map peak median **0.1577 with the prey on the image against 0.0398 with it off**, a 4x separation, and the eye fires on **81.1 % against 51.8 %**. Walking, the same two medians are 0.0696 and 0.0872 -- the noise is LARGER than the signal and no threshold can separate them. This is why every discriminator in #245, #248, #252, #256 failed: they were all tested on a walking animal, and this animal could not stop. It also confirms the direction of **#214** and what Minton 1966 describes for wild *E. macularius* -- prey "slowly stalked or ambushed" |
| 258 | A body that is told to stop is still | **Refuted -- it sways for about seventy steps, and the eye reads the sway** | after `locomotor_drive` goes to 0, the head's angular speed decays 27.9 -> 17.5 -> 7.8 -> 3.5 -> 0.86 -> 0.13 -> 0.04 deg/s over the first 70 control steps, and the eye's firing rate only falls to its floor of 14-19 % once it gets there, having first RISEN to 78 % as the body rocks through the transient. A 50-step fixation was therefore spending all of itself reading its own wobble. With a 70-step settle added, the animal's report rate during a look fell from **50.2 % to 4.8 %** and the adaptive quorum from 12, which nothing could reach, to **6** |
| 259 | (correcting #257) Walking, the motion noise is larger than the signal | **Refuted -- that was one seed, and the multi-trial numbers say otherwise** | #257 quoted 0.0696 with the prey on the image against 0.0872 with it off, from a single seed-2 run. Over **14 trials** (`tools/motion_floor_derivation.py`): walking, median peak **0.0589 with the prey on the image against 0.0000 with it off** -- the signal is cleanly above the noise. The claim was a single-seed artefact reported as a general fact, which is rule 5 again and the second time this session. **What survives, and it is the part that mattered:** standing still raises the signal 2.5x, from 0.0589 to **0.1499**, and moves the best operating point from 57.2 % recall at 20.3 % false alarms (walking, floor 0.04) to **92.7 % recall at 48.5 %** (standing, floor 0.08), with peak separation rising from +37.0 to **+44.2**. Freezing before looking is still right; "the eye cannot work at all while walking" is not |
| 260 | (amending #247) The head's inertial sense should be a `<gyro>` sensor in the morphology | **Refuted -- it breaks the provenance guard, and it was never necessary** | adding `gyro_head` / `vel_head` to `morphology/gecko_world_v1.xml` worked and was reverted. `morphology/gecko_world_v1.xml` is GENERATED, and `tests/test_world.py::test_the_committed_world_matches_its_manifest` hashes it against its manifest -- the hand edit failed that test immediately, which is the guard doing its job. Putting the sensors in the SOURCE body instead would have changed `gecko_body_lab_v2.xml`'s hash, and that hash is referenced by the session 3-4 gate evidence this map already records an unresolved question about. `mujoco.mj_objectVelocity` on the EXISTING `head_site` returns the same six numbers a gyro mounted there would, from state the simulator already computes: head yaw rate 0.37 deg/s against the trunk's 0.13, identical to the sensor version. **Zero morphology change, and the finding in #247 stands unaltered.** 536 of 537 tests pass; the one failure was this guard, and it passes again |
| 261 | The animal is getting close to its food because it SEES it, not because it is wandering in a small arena | **Confirmed, with the caveat stated** | the control replaces every committed bearing with a uniform random one and changes nothing else -- same search, same freezes, same fixations, same decision rule, same seeds. 8 paired seeds x 3000 steps: **seeing got closer on 7 of 8**, median closest approach **60.6 mm against 152.8 mm**, mean improvement **87.7 mm**. Exact paired permutation test on the mean difference **p = 0.0234**; the sign test, which throws away magnitude, gives **p = 0.0703** and does not clear 5 %. So the effect is real and sizeable but rests on n = 8 and on the magnitudes rather than the count. The one seed that went the other way (3: 124.6 seeing against 87.5 blind) stays in the table. Evidence: `artifacts/evidence/session12/blind_control.json` |
| 262 | Brain 6/8 (the day/night clock) is done | **Overstated -- it is built and tested, and connected to nothing** | `brain/arousal.py` exists, has 13 tests, and tags its numbers correctly. `envs/gecko_brain_env.py` never imports it. The only place in the repository that constructs `Arousal()` is `tools/everything_video.py` -- a video. The animal has no clock. So "4 of 8 brain layers done" is really **3 connected, 1 built-but-disconnected, 1 half, 3 not started**. Found by an independent audit of the session, not by me |
| 263 | The eye now uses the derived operating point | **Overstated -- the derived numbers live in a tool, not in the library** | `motion_floor = 0.08` and `flow_gain_yaw = 0.006` (#254, #257, #259) are applied by `tools/oracle_free_hunt_video.py` and `tools/oracle_free_diagnosis.py` as overrides. `brain/tectum.py` still ships `MOTION_FLOOR = 1e-4` and `flow_gain_yaw = 0.020`. Deliberate, because changing library defaults moves every gate and test that exercises the eye -- but the consequence is that **the shipped eye still fires on renderer noise on two frames in three**. The standing-still operating point needs to become a first-class, named configuration of `Tectum`, not a tool-side patch. Open |
| 264 | Session 12's own provenance claims | **Four of four audited claims refuted or overstated, independently** | a 12-agent audit of this session's work, run because the user asked whether the head-turning was "real measurement from a real gecko". Findings: (a) **0 of 18** constants in `brain/search.py` come from any animal -- 16 INVENTED, 2 derived from the simulation or its own morphology; (b) the file's docstring "every constant is INVENTED and says so" was **false on its own terms** in both directions and is corrected; (c) "no gecko fixation duration has been published" was **wrong** -- the scorecard holds a published *E. macularius* fixation statistic (binocular fixation bouts toward a snake predator), which is antipredator staring, not search, and is now cited as such rather than denied; (d) the neck and head joint ranges (+-40, +-30) carry **no provenance tag anywhere**, the scorecard records no range-of-motion study for this species on any joint but the hip, so "seventy degrees of head yaw" is the sum of two untagged modelling choices; (e) the two vestibular/saccade claims are **true vertebrate biology with zero gecko-specific support** in the corpus, and the code now says exactly that; (f) Minton 1966 appears in two files and in neither the bibliography nor the registry. Evidence: workflow `wf_26b0fc1d-505`. **What survives:** the anatomy is right, the shape has one qualitative field sentence behind it, and every number is engineering that says so |
| 265 | Hooking the basal ganglia to the body would be theatre | **True when written, false now, and the wire is built** | `envs/gecko_brain_env.py` ran the selector and reported `behaviour_controls_nothing = True`, with the reason stated in the source: "Hooking a six-way chooser to a one-behaviour body would look like integration and mean nothing." The body can now do three distinguishable things -- search, chase, hold still -- the third only since `locomotor_drive` (#253). `brain/programs.py` is the seam, `behaviour_control=True` turns it on, and `autonomous_step()` is a SEPARATE entry point so `step(action)` is untouched for every gate, checkpoint and policy. `prey_visible` into the selector is now the eye's COMMITTED evidence rather than a raw salience, because hunt salience is hunger x prey_visible and feeding it a pixel count makes the animal lunge at renderer noise |
| 266 | Threat can be computed from vision | **Refuted for this species, and smell was already built** | `brain/vomeronasal.py` has existed, tested, since before this session, carrying the target-species factorial: antipredator reaction to snake SCENT 0.21 against SIGHT 0.07, chi2 = 8.098 p < 0.0044 for the chemical term and p > 0.9 for the visual, n = 42. The project's own report said in as many words that the module "is built and tested and no behaviour reads it." It is now read: `smell=True` routes `defensive_probability` into the threat signal the selector consumes |
| 267 | The world contains shelter, a warm surface and a threat | **Refuted -- the generator supports all three and the COMMITTED world has none of them** | `utils/build_world.py` can emit a shelter, a warm patch and a threat, and the status table at the top of this map lists them as done. `morphology/gecko_world_v1.xml`'s manifest records exactly one added body: **the prey**. Checked directly against the compiled model -- the only non-anatomical body present is `prey`. So the flee and bask channels have nowhere to go, `_predator_distance_m()` returns None on purpose, and the nose's predator term reads zero because there is nothing to smell rather than because the animal is safe |
| 268 | The animal gets hungry during an episode | **Refuted -- it has 3292 HOURS of reserve and the selector therefore releases nothing** | measured: over 600 autonomous steps the hypothalamus energy deficit reaches **0.0000** and the basal ganglia releases NO behaviour on any step. This is the real reason the selector was never connected: nothing would have come through the wire. `homeostasis_time_compression` has to reach about **20000** before any behaviour is released -- starving the animal over three days inside twelve seconds. Added `meals_owed_at_start` instead: a nocturnal forager emerging at dusk is some number of meals behind, which is the same statement made honestly. At 1.0 meals owed the animal explores and hunts; at 0.5 it still releases nothing |
| 269 | A flag describing the head is safe to read when the head is not being driven | **Refuted -- stale state latched the animal onto its own noise** | `MotorPrograms` asked the search pattern whether the head was still BEFORE running it. During a chase the search does not step, so the `looking` flag was left over from the previous fixation, and the animal accumulated evidence while walking. Measured: **1331 of 1500 steps chasing, prey on screen 0 of 1500**. Same class as **#236** |
| 270 | A committed animal can still choose to look around | **Refuted -- the salience arithmetic forbids it** | the explore channel is `0.35 * hunger * (1 - prey_visible)`, so the instant the eye commits, `prey_visible` goes to 1 and the drive to look around goes to **exactly zero**. A committed animal cannot elect to re-check; the pause has to be built into the chase itself. Added a stalk cycle -- walk a short leg, stand and look -- which is also the shape the eublepharid foraging record describes |
| 271 | "I see nothing" and "it is directly ahead" are distinguishable | **Refuted -- they were the same message, and it poisoned the whole loop** | `Eye.last["prey_bearing_deg"]` returned **0.0** when the tectum saw nothing, and 0.0 is a perfectly good bearing meaning straight ahead. Only `prey_elevation_deg` was None-gated. Every older consumer happened to guard with `food_visible_frac > 0` so it never bit; the new closed loop read the bearing without that guard AND carried a fallback to the previous bearing, so the accumulator received the same number every step, that number agreed with itself perfectly, and the animal committed on it. Measured: the eye fired on **27.1 %** of believed steps while the accumulator's own report-rate estimate read **1.0**, the adaptive quorum ran to 36, and it still committed **663 times with the prey off screen every time**. Fixed at the source -- silence now reports None -- and guarded at the consumer. **Decision precision went from 0 of 663 to 48 of 95 (50.5 %)** and prey-on-screen from 0 to 180 of 1500 in the same configuration |
| 272 | "~80 % of gaze stabilisation in geckos is head movement" | **Refuted -- wrong paper, wrong species, wrong quantity** | this sat in `docs/research/gecko_scorecard.md` attributed to Masseck, Roell & Hoffmann 2008 and was used there to justify the head-mounted camera. An independent sweep opened **Dieringer, Cochran & Precht 1983** and it measured ***Rana* (frog) and turtle** -- it never mentions a gecko or any squamate -- and its ~0.8 is an optokinetic head GAIN, not a head-versus-eye split. Masseck et al. measured HEAD movement only and justified ignoring the eyes by citing that same frog-and-turtle paper. `docs/EYE_SPEC.md` had already flagged the attribution as wrong and told readers not to tune on it; the scorecard had not been updated to match and still carried it as fact. The head/eye split in any gecko is **UNMEASURED**. The scorecard's page range for Masseck was also wrong -- 48:842-849, corrected to **48(6):765-772**, which `EYE_SPEC.md` and `best_achievable_brain.md` already had right |
| 273 | The animal can run its own brain end to end | **CONFIRMED -- first time, and it is the result of this turn** | hunger -> basal ganglia -> motor program -> body, with `env.food_xy` steering nothing and the animal not told it is hungry. 3 seeds x 3000 autonomous steps: prey on screen **828 / 310 / 888 of 3000**, decisions **144 / 158 / 96** of which right **97 / 1 / 96** -- precision **67 %, 0.6 %, 100 %** -- closest approach **72.0 / 216.7 / 154.7 mm**, time moving **26.1 / 25.6 / 27.5 %**. Seed 3 is a genuine failure and stays in the table. The animal spends roughly four fifths of its life in `explore` and one fifth in `hunt`, which it chose. **All 537 tests pass**, gates included, because `step(action)` was left alone and the loop lives in a separate `autonomous_step()` |
| 274 | Five brain modules were built, tested, and connected to nothing | **CONFIRMED, and four are now connected** | an independent audit found `gecko_selector`, `brainstem`, `spinal_cpg`, `vomeronasal` and `arousal` all complete, tested, and read by no behaviour -- the project's own generated report said of the nose, in as many words, that it "is built and tested and no behaviour reads it". Now connected: the **selector** drives motor programs (#265), the **brainstem** supplies the behaviour-to-effort map, the **nose** gates threat (#266), and the **clock** is constructible via `circadian=True`. Still connected to nothing: `spinal_cpg` as the walker's rhythm source, deliberately, because `attach_spinal_cord` would move the accepted 4/6 walker off the closed form it was gated on -- the cord is built and driven as the brainstem's output target only. `actor_critic`, `agwm`, `bc_actor`, `anatomy_graph` and `vision_encoder` are the training path and are not animal modules |
| 275 | The 8-frame temporal window is the eye's optimum | **Refuted for a standing animal -- the optimum was measured in the wrong regime** | #230 and #235 measured window 8 as optimal during an ORACLE-DRIVEN hunt, with the physics engine aiming the head and the animal walking the whole time. A/B in the closed autonomous loop, same seed, same everything else: **window 3 -> 828 frames with prey on screen, 97 of 144 decisions right, closest 72.0 mm; window 8 -> 317 frames, 0 of 176 right, closest 124.2 mm.** A longer baseline helps when the prey's image is sliding across the frame because the animal is walking at it, and hurts when the body is the only thing moving, because then it is integrating the animal's own sway. The autonomous default moves to 3. This does not overturn #235's measurement -- it bounds it to the condition it was taken in, which is the same error #239 recorded for the 73.5 % on-image figure |
| 276 | The strike can be de-oracled by estimating range from the prey's apparent size | **Refuted -- this eye reports no size to estimate from** | the last privileged channel inside the hunt is the strike trigger reading true range, and familiar-size is the obvious way to remove it: a gecko that eats crickets has a prior on cricket length, and angular size then inverts to distance. Measured first, and it does not work. Over an oracle-steered approach, on every frame where the eye actually found the prey, the connected component containing the peak is **1.0 cell at every range band** -- 0-60 mm, 60-120 mm and 120-250 mm all give a median blob of exactly one cell and an apparent width of 1.09 deg, which is simply the cell pitch at 64 cells over 70 deg. Correlation of log apparent width against log true range **r = +0.143**: no relationship, and the wrong sign. The surround subtraction that makes the detector work suppresses a blob's interior and leaves a single cell, so SIZE IS DESTROYED BY THE STAGE THAT MAKES THE TARGET DETECTABLE. De-oracling the strike therefore needs a second channel that keeps spatial extent, not a read of the existing map. Recorded before building anything |
| 277 | Masseck et al. 2008 can calibrate a target-orienting head turn | **Refuted -- it is a STABILISATION reflex and cannot set any voluntary parameter** | I claimed in conversation that the target-species head gains give "a real gain to check the orienting reflex against". The adversarial verification of that very paper, already in hand when I said it, states the opposite in capitals: it "MAY NOT be used to justify ANY voluntary or exploratory behaviour. This is a stabilising reflex to a rotating drum. It cannot set parameters for visual search, prey tracking, scanning saccades, head-cocking, or any self-initiated gaze shift", and the paper itself explicitly separates foveal prey tracking from gaze stabilisation. **This is the same error as #272** -- reaching for the nearest impressive citation and stretching it one category too far -- committed by me within hours of recording #272, with the refutation already written down. Caught by the user asking whether I had actually read it. **What Masseck CAN calibrate** is the stabilisation channel the pretectum already implements: whole-field optic flow, 20-40 deg/s, binocular head gain 0.9 / 0.85 / 0.75, monocular temporo-nasal 0.7 / 0.6 / 0.4, naso-temporal ZERO. **What is genuinely unmeasured in any gecko:** voluntary head-saccade amplitude, peak velocity, duration, latency, spontaneous saccade rate, head yaw range of motion, turning or head-sweep rate, and prey DETECTION distance. An orienting reflex built now has an INVENTED gain and must say so |
| 278 | Saccade duration is a free parameter to be invented | **Refuted -- it is a property of the actuator, and the shipped value was wrong** | measured step response of the neck (`neck_yaw` + `head_yaw` position servos) with the body frozen: a commanded step reaches 90 % in **17-18 control steps regardless of its size** -- 10 deg peaks at 35.7 deg/s, 22 deg at 79.5, 45 deg at 163.3, 65 deg at 236.3, and all four take the same time. It is a first-order servo with a fixed time constant, so duration is DERIVED and not chosen. `SearchPattern.SACCADE_STEPS` was **3**, meaning the eye was believed while the head was still moving for another fourteen steps of every scan |
| 279 | After a head saccade the eye is usable again immediately | **Refuted -- there is a measured 60-step blind window with a false-alarm BURST in the middle** | 14 trials, 22 deg head saccade, body already fully settled. The eye fires on ~0 % of frames for the first 24 steps -- the whole-field motion is rejected by the "fills the field" rule -- then **bursts to 78 %, 90 %, 96 %, 96 %** across steps 24-47 as that motion decays down through the detector's band, then falls away to its ~20 % floor by about step 60. The dangerous region is not during the movement, it is just after it. DERIVED, and it is the reason `OrientingReflex.BLIND_STEPS = 60` |
| 280 | The head should turn toward whatever the eye reports | **Refuted, three times, each with a different threshold** | built `OrientingReflex` with **zero invented constants** -- deadzone one retinal cell from the optics, gain 1.0 by the definition of orienting, travel from the morphology, duration from #278, blind window from #279. (a) Firing on any report above the one-cell deadzone: every sighting bought a saccade and 78 blind steps, and the animal reached **0 decisions on 3 seeds x 3000 steps**, having starved its own accumulator. (b) Trigger raised to 25 deg, DERIVED from the measured eccentricity curve (#281): decisions 1 / 19 / 0 against a baseline of 144 / 158 / 96. (c) Restricted to tracking an already-COMMITTED target: the reflex then fires 0-2 times per run and the animal is roughly back to baseline. **A reflex that acts on single frames inherits their noise** -- the lesson of #245 and #271 arriving in a third place. What survives: orienting is defensible only for something already decided to be real |
| 281 | The eye is better at the centre of its field than at the edge | **Partly -- it is FLAT out to 25 degrees and then falls off a cliff** | aimed recall against eccentricity over an oracle-steered approach: 44.5 % at 0-5 deg, 48.2 % at 5-10, 50.8 % at 10-15, 42.4 % at 15-20, 40.9 % at 20-25, then **30.9 % at 25-30 and 10.3 % at 30-36**. There is no fovea and no centre advantage -- which is what an afoveate gecko retina predicts -- so centring a target already inside 25 deg buys nothing and costs the 78-step blind window. This is what makes #280(b)'s trigger a derived number rather than a guess, and it is also why it still did not help |
| 282 | A number derived from the physics is safe to install wherever it seems to belong | **Refuted -- right number, wrong quantity, and it cost the run** | #278 measured the neck's true travel time, 17-18 steps, and I set `SearchPattern.SACCADE_STEPS` to 18 on that basis. That constant is not the movement's duration -- it is how long before the dwell timer starts -- and `SETTLE_STEPS` = 70 already waits out both the travel and the post-saccade blind window before the eye is believed. Raising it double-counted the wait and bought fewer fixations per rollout. Isolated A/B, 3 seeds x 3000 autonomous steps, everything else identical: **value 3 -> closest 34.9 / 147.3 / 156.2 mm with 177 of 224, 0 of 80 and 95 of 95 decisions right; value 18 -> 263.4 / 174.6 / 150.4 mm with 0 of 0, 0 of 93 and 96 of 96.** Reverted to 3, with the measurement kept in the docstring so the next reader does not re-derive the same mistake. #278's measurement stands; only its application was wrong |
| 283 | Orienting the head to a COMMITTED target helps | **Confirmed, and it is the best autonomous result the project has** | with `SACCADE_STEPS` back to 3 and the reflex restricted to tracking something the accumulator has already committed to (#280c), against the same three seeds with no reflex at all: **seed 2 closest 72.0 -> 34.9 mm and decisions right 97 of 144 -> 177 of 224 (79 %); seed 3 216.7 -> 147.3 mm; seed 4 unchanged at ~155 mm.** Prey on screen 828 -> 917 on seed 2. The animal is now **14.6 mm** from its own strike trigger of 20.3 mm on its best seed. Seed 3 still returns 0 correct decisions and stays in the table |
| 284 | The `rest` channel can win once fatigue has time to build | **Refuted -- the animal never tires, and that is correct physiology rather than a defect** | `rest` salience IS fatigue, and fatigue plateaus at **0.011** against an explore salience of 0.35, so the channel cannot compete. Raising `homeostasis_time_compression` does not help and the sweep shows why -- 1, 50, 100, 200 and 400 all end at 0.009-0.011 with an identical behaviour mix. Fatigue accrues as `effective / endurance` only while the animal is working and DECAYS as `exp(-effective / recovery)` otherwise, so compression scales both terms and the equilibrium does not move. Underneath that: the animal walks at **0.035 m/s**, below the sustainable speed at which `Physiology.endurance_s` returns infinity, so it is never working. A gecko strolling does not get tired. `rest`, `bask`, `flee` and `groom` therefore cannot be shown to do anything -- bask and flee additionally have nowhere to go (#267) -- and the six-channel selector is in practice a two-channel one. Not fixed, because the fix would be to make the animal tire when it should not |
| 285 | The last centimetre is fixed by measuring arrival from the mouth instead of the trunk | **Refuted -- it steers the body with the head** | `envs/gecko_brain_env.py` already had `approach_site="mouth"` + `reach_dist` = strike trigger, but only under `hunt_targeting` (the oracle), where the env's own comment reports the same frozen walker "arrives 75 times in 160 seconds -- mean closest 20.5 mm. It could always do this." Exposed as `mouth_goal=True` for the autonomous loop, with the 5 cm stop-distance floor dropped for mouth goals. Measured, 3 seeds x 3000 autonomous steps: **closest 34.9 / 147.3 / 156.2 mm without it, 167.3 / 235.2 / 236.0 mm with it** -- worse on every seed. Cause: `_target_egocentric` measures BEARING as well as range from the approach site, so with the nose as the reference the walker steers by the head's orientation, and the head is sweeping +-65 deg during a search. The arrival TEST should be the mouth; the steering reference must stay the trunk. Kept as a kwarg, default off, so the split can be built properly rather than re-discovered |
| 286 | A jaw can be added without breaking every trained walker | **Confirmed -- by putting it outside the policy's action space** | the walker's RL action dim IS `model.nu` (`action_space = Box(..., (nu,))`), so a 26th actuator would change the action shape every frozen checkpoint was trained on. The jaw actuator is declared in MuJoCo actuator **group 2**; `GeckoWalkEnv.nu_policy` counts group-0 actuators only, `action_space` is sized from it, a policy-sized action is padded to `nu` with zero residual for group 2, and `_apply_jaw` writes the gape after the CPG exactly as gaze is written. The residual controller zeroes `res_scale_vec` on group 2. The 92-D observation is built from fixed sensor-name lists and is unchanged. Structural counts: nq 39 -> 40, nu 25 -> 26 (25 policy + 1 jaw), nsensor 83 -> 85, nbody 24 -> 25. **On the old jawless body the patched code is bit-identical: 222.8 mm in 300 steps, same as before to one decimal.** On the new body: 230.1 mm, a 3 % change from the mass re-fit, which is what gate 2 exists to judge |
| 287 | An unactuated command leaves a joint at zero | **Refuted -- the residual controller fills every limited actuator with the MIDPOINT of its range** | with no gape commanded the mouth hung **20.1 deg open**, because `CPGResidualController.neutral = 0.5 * (lo + hi)` and the jaw's ctrlrange is 0..40 deg. `_apply_jaw` now writes the gape on every step, including 0.0, and reset closes the mouth. Measured after: closed 0.4 deg, commanded 37 -> 35.9 deg, closed again 0.4 deg. The published peak gape of ~37 deg (Delheusy, Brillet & Bels 1995, n = 6 adult *E. macularius*) is reached to within one degree by the servo |
| 288 | The animal has a mouth | **Confirmed -- lower jaw, teeth and tongue, added at the template and regenerated, and the certified body still passes 14/14** | added to the SOURCE template `morphology/gecko_body_r.xml` (not by hand to a generated file, which the manifest guard rejects, #260): a `jaw` body under `head` with a 0-40 deg hinge admitting the published 37 deg peak, a 0.6 g mandible capsule, a teeth strip and a tongue capsule under new materials so the generator's skin/eye rescaling loop leaves them alone, `mouth` and `tongue_tip` sites, jointpos/jointvel sensors, and a keyframe slot at qpos index 13. Regenerated `gecko_body_lab.xml` (its two CoM gates fail, and they failed identically BEFORE the jaw -- 0.762/0.609 baseline against 0.757/0.603 now; the jaw moved CoM toward the nose as predicted) and `gecko_body_lab_v2.xml` with `--fit-com`: **14/14 morphology gates pass, intact CoM 0.678112 unchanged to six decimals** -- the calibration absorbed the jaw. World rebuilt with the manifest's exact three edits. Geometry of the jaw is INVENTED; the gape range and cycle timing are PUBLISHED, target species |
| 289 | The pre-refactor controller fixture pins the actuator count | **Overstated -- it pins the walker's 25 commands, and the jaw is not one of them** | `tests/test_gait_config.py::test_pre_refactor_controller_fixture` compared the whole `base_ctrl` row against a recorded (2, 25) array and failed with shapes (2, 26) vs (2, 25); `eval/session2_controller.py` reruns that fixture before every gate-2 trial, so the jawed body could not even be gated. The 26th column is the jaw actuator (MuJoCo group 2), written by `GeckoWalkEnv._apply_jaw` and never by the controller. The test now compares the group-0 columns only, asserts there are exactly as many as the fixture has, and the recorded fixture passes **byte-for-byte** -- which is the proof that every command the walking policy was trained on is unchanged by the mouth. Recorded, not silent |
| 290 | The jaw changes how the animal walks | **Refuted -- same code, same gate, same three checks fail on both bodies, values within 1-2 %** | `eval/session2_controller.py` grew a `--xml` argument so the SAME code can score the pre-jaw v2 (recovered from git) as a control. Gate 2, registry `lab_base_parameters`, 20 s, zero residual: **pre-jaw** forward 0.0424 PASS, net path 0.825 PASS, hind swing 0.013/0.011 PASS, front stance 0.611/0.606 FAIL, hind duty 0.641/0.631 FAIL, limb phase 0.577/0.569 FAIL -- **jawed** 0.0427 / 0.820 / 0.012/0.008 / 0.610/0.612 / 0.652/0.640 / 0.583/0.579, identical pass/fail on every check. The 300-step travel on the old body with the patched code is 222.8 mm to one decimal (bit-identical); on the jawed body 230.1 mm. The mouth is not a walking change |
| 291 | The walking-evidence contract can be re-satisfied by re-measuring the base on the jawed body | **Refuted -- it cannot be satisfied today on ANY body, and the reason predates the jaw** | ten `GateContracts` tests fail with "Lab base evidence measured a different body", correctly: body identity is a PHYSICS fingerprint and a body with a jaw is a different animal. But the contract ALSO requires the recorded evidence to pass hind duty in **[0.73, 0.83]** (target 0.78 +- 0.05), and today's base measures **0.641 on the pre-jaw body and 0.652 on the jawed one** -- a 0.13 shortfall with the jaw absent. The accepted session-4 evidence (hind duty inside the band, body hash 7567...5c4b) was measured on a body this map already records as one "no committed version of gecko_body_lab_v2.xml produces", and the registry's own `lab_base_parameters` no longer reproduce its gait. `artifacts/evidence/body_identity_equivalences.json` forbids adding a pairing that cannot be cited, and a jaw is not a line-ending. So the ten tests stay red **for a stated reason**, and the decision is the user's: re-measure and re-accept the base at today's 3/6, restore the parameter set that produced 4/6, or bind walker training to the pre-jaw body while the hunting body carries the mouth. What is NOT open is whether the jaw caused it: #290 |
| 292 | A physical mouth catches prey at the published 82.9 % | **REFUTED AS BUILT -- 110 strikes, 0 hits, and this is the number the die was hiding** | `brain/strike.py` now carries the target-species capture profile (Delheusy, Brillet & Bels 1995, opened in full: ~80 ms cycle, ~37 deg peak gape at 47 ms, no slow-open phase, ~27 mm head drop, ~8 mm forward) and `_run_strike` runs it as a MOVEMENT: the jaw opens along the gape profile, the head pitches along the drop, and the prey counts as caught only if its centre passes inside the mouth volume while the jaws are open -- no probability. The statistical path is kept for jawless bodies so older evidence reproduces. Oracle-steered approach, 6 seeds x 1500 steps: the animal gets **13-19 mm** from the prey (inside the 20.3 mm trigger), launches **110 strikes, and lands 0**. The published 82.9 % was typed in (`strike_capture_success`, Coleonyx); the earned rate is 0.0 %. Diagnosis of WHERE the prey is at peak gape is running; one bug is already found and fixed -- the head drop was ACCUMULATED every step instead of set, pinning the head at its pitch stop (#296) |
| 293 | The head shape is wrong | **Refuted for proportions, and a real correction found for the mouth** | against the only published wild-male morphometrics for this species (two Nepalese males, SVL 109-119 mm): built head length **31.9 mm** (published 32-33), height **11.1** (11-12), eye diameter **6.6** (6-7), inner-eye gap **10.0** (10); width 19.6 against 21-22 at a smaller SVL, inside the head-width gate. The "wrong head" is a rendering impression of ellipsoids, not a proportion error. The MOUTH was wrong: Delheusy et al. give lower-jaw length tip-to-commissure **13 +/- 2 mm** (n = 5), and the built corner sat 4 mm behind the tip. Moved to the published value -- measured after regeneration **13.8 mm** -- body still 14/14 |
| 294 | Adding a jaw breaks the test suite | **Refuted -- 529 of 537 pass, and the 8 that fail are #291** | full suite on the jawed body after the version bump: the only failures are the eight `GateContracts` evidence tests, which fail for the documented, pre-existing reason (the accepted base evidence cannot be re-measured on any body today). Morphology audit, world manifest, realism recorder, reward calibration and the legacy controller fixtures all pass, the last byte-for-byte on the walker's 25 commands (#289) |
| 295 | The body does not curve when it walks | **Refuted -- it curves 15 deg; what is wrong is the TAIL'S PHASE, and it is measured against a published rule** | walking at full drive: `spine_lat_1/2/3` peak-to-peak **15.2 / 14.4 / 10.9 deg**, `tail_yaw_1..5` **7.3 / 7.8 / 7.3 / 6.7 / 6.2 deg**. The S-curve exists. But the tail bends the SAME way as the trunk (spine-tail correlation **+0.58**, tail lagging 0.68 s) with the shipped `tail_phase_lag` = 0.15. The published shape for this species (Jagnandan & Higham 2017, qualitative, target species) is a trunk standing wave that becomes a caudally TRAVELLING wave down the tail, with the tail base flexing toward the PROTRACTED hindlimb -- the opposite curve the user's video shows. And the built amplitude HALVES at half drive (9.1 / 3.7 deg) where the published tail-tip displacement DECREASES with speed. Two shape rules, both published, both contradicted; no absolute amplitude, frequency or phase number exists for any gecko, so the fix is a phase sweep against the rule, not a number from a paper |
| 296 | A cumulative profile can be added to the state each step | **Refuted -- it integrates twice** | `head_drop_profile(fraction)` returns the TOTAL drop so far; `_run_strike` added it to the pitch on every control step, so the head reached the 45 deg stop within a few steps of every launch. Fixed to SET the pitch from the profile. Same class as #236 and #269: state that is applied as if it were an increment |
| 297 | The physical strike fails because the geometry of the mouth volume is wrong | **Refuted -- three upstream defects, all measured at peak gape over 27 strikes** | (1) the jaw was **0.7 deg open** (max 11) at the moment the profile called for 37: an 80 ms strike is FOUR control steps and the shipped jaw servo takes SEVENTEEN to move (#278 measured the neck's, and the jaw used the same class); the mouth never opened. (2) Head pitch **45.0 deg**, the stop, on the median strike: the gaze loop had already pitched the head ~35 deg to look at a prey 13 mm below the nose, and the drop was added on top. (3) The prey sat **17 mm AHEAD of the nose** and nothing carried the mouth forward: the published capture drops ~27 mm from almost directly above the prey with only ~8 mm forward travel, and Coleonyx closes its 20 mm from a 0.85 m/s lunge. **This animal has no lunge.** It walks at 35 mm/s; closing 17 mm in 80 ms needs 0.21 m/s, six times its walk -- the ratio the registry's own strike note records. (1) and (2) are fixed (#298, #296); (3) is a missing body capability, a hindlimb push the cord does not have, and it is the next dependency for eating |
| 298 | The jaw actuator gain is a free choice | **Refuted -- it is DERIVED from the published cycle** | measured rise of the jaw servo to 90 % of the published 37 deg peak: kp x1 (0.06) **380 ms**, x5 120 ms, **x10 80 ms**, x20 80 ms, x40 60 ms. Delheusy, Brillet & Bels 1995 give the whole capture cycle as ~80 ms with peak gape at ~47 ms, so x10 (kp 0.6, kv 0.03) is the SMALLEST gain that meets the paper at 20 ms control resolution. Overshoot 6 deg inside a 40 deg hinge. Installed at the template and regenerated -- and then corrected once more, because the generator rescales every actuator gain by the mass ratio 0.038/0.0612 = 0.62, so a template kp of 0.6 arrived in the generated body as 0.37 and reached 90 % only at 120 ms. Template kp 1.0 (kv 0.05) puts the GENERATED jaw at the paper's 80 ms. A derivation has to be applied where it lands, not where it is written. Final, on the generated body with template kp 1.5 / kv 0.075: gape 4.4 / 12.8 / 21.3 / 28.1 / 32.8 / 35.6 deg at steps 1-6, **90 % of the published peak at step 5 (100 ms)** against the paper's ~80 ms -- one control step past it. Pushed no further: 20 ms is the resolution of the loop, and chasing a number below it is what rule 2 forbids |
| 299 | The tail's phase is right | **Refuted against a published rule, and the rule is met at lag 0.76** | Jagnandan & Higham 2017 (target species, qualitative): the tail base flexes toward the PROTRACTED hindlimb, i.e. phase 0 between tail-base peak and forward-foot peak. Measured on this body: shipped `tail_phase_lag` 0.15 -> **+0.40 stride** off the rule with the tail bending the SAME way as the trunk (corr +0.58); 0.35 -> -0.38; 0.50 -> -0.24 (corr -1.00, exactly opposite); 0.65 -> -0.10; **0.76 -> +0.02** (corr +0.07, a travelling wave, decoupled from the trunk); 0.80 -> +0.05. No absolute tail amplitude, frequency or phase has been published for any gecko, so this is a phase derived from a published SHAPE. `tail_phase_lag` is an entry in `lab_base_parameters` -- the accepted walker's gait -- so changing it re-opens the walking gate and the evidence contract (#291). Measured and recorded; **not installed**, because that is a decision about the accepted walker and not mine to take silently |
| 300 | A lever length can be read off the morphology by eye | **Refuted -- 45 mm assumed, 37 mm measured** | the strike converts the published ~27 mm head DROP (a translation) into the pitch that delivers it through the neck-pitch lever. The first version hard-coded 45 mm "from the morphology"; measuring neck-body origin to `nose_tip` in the stand pose gives **37.0 mm**, so the commanded pitch was 18 % too small for the drop it claimed. Now computed from the model at construction, so a regenerated body carries its own lever. Same lesson as #278/#282: measure the thing, and measure it where it is used |
| 301 | The animal can eat with its own mouth | **Confirmed once -- the first physical catch in the project, and 1.7 % is the honest rate** | with the derived jaw gain (#298), the drop delivered from level (#297), and the lever measured (#300): oracle-steered approach, 6 seeds x 1500 steps, **60 strikes, 1 hit, 1 eaten** -- seed 3, closest approach 12.6 mm, jaw closed on the cricket's centre inside the mouth volume. Earned capture rate **1.7 %**; the registry's typed-in Coleonyx figure is 82.9 %. The gap is #297(3): the published capture drops onto a prey almost directly below the head with a lunge that closes the last centimetres at 0.85 m/s, and this animal has no lunge -- it strikes from where its walk stopped, 13-19 mm short. The die is gone and the number is now a prediction the body has to earn |
| 302 | The pulsing under a gecko's chin is gills | **Refuted -- it is BREATHING, and no gecko's rate has ever been published** | geckos have no gills; the visible throat movement is the gular/buccal pump. What the session-12 sweep could open: Milsom 1984 (Gekkonidae, abstract only) -- geckos breathe in single breaths or short bursts separated by breath-HOLDS, and raise ventilation with temperature by shortening the holds while tidal volume and breath duration stay constant; Dial & Schwenk 1996 (*Coleonyx brevis*, **same family**, abstract only) -- "buccal pulsing" as an olfactory sniff preceding a defensive display; Owerkowicz 2001 (*Varanus*, full text) -- the only fully timed gular mechanics anywhere, up to 5 pumps per costal breath, peak pressure within 200 ms, glottis always closing before the nares open. **NOT FOUND for any gecko: breathing frequency, gular excursion, throat displacement.** Built anyway, because a still animal with a motionless throat reads as paused rather than alive -- `BREATH_HZ` 0.4 and a 5 deg excursion, both INVENTED and saying so at the constant |
| 303 | Installing the published tail phase costs the walk | **Refuted -- gate 2 is unchanged to the third decimal** | `tail_phase_lag` 0.15 -> **0.76** in `lab_base_parameters`, the value #299 derived from the published rule. Gate 2 on the same body, same 20 s protocol, against the pre-jaw control: forward **0.0428 vs 0.0424**, net path **0.820 vs 0.825**, front stance **0.615/0.605 vs 0.611/0.606**, hind duty **0.651/0.652 vs 0.641/0.631**, limb phase **0.577/0.579 vs 0.577/0.569**, hind swing load **0.009/0.010 vs 0.013/0.011** (better). Identical pass/fail on every check -- the same 3/6 that #290 and #291 record for this body with the OLD phase. The tail's phase is free at the gate, so the animal gets the curve the literature describes. The registry entry stays tagged INVENTED: the RULE is published, the number is derived from it by measurement here |
| 304 | Eyelids, a face and a fat tail can be added without disturbing the physics | **Confirmed -- 14/14 gates, 25 policy actuators unchanged** | added to the source template: jowls, brow ridges, a snout bridge, a rounded snout tip and a pale lip line (15 new visual geoms, material `skin2`/`pale` so the generator's head loop -- which rescales every `skin` geom and repositions every `eye` geom -- leaves them alone and the head-width gate is untouched); a bulbous visual overlay on tail1-4 (mass 0, no collision, so the tail-mass and CoM gates are untouched); a `eyelids` hinge carrying BOTH upper lids, which is what an eyelid gecko's lid drop looks like; and a `gular` throat body. Structural counts nq 40 -> 42, nu 26 -> 28 (**still 25 policy + now 3 brain**), nbody 25 -> 27, visual geoms 49 -> 64. Both lab bodies and the world regenerated; **v2 passes 14/14**. Two XML facts learned the hard way: a moving body needs a nonzero mass (the lids are 0.02 g), and the generator's parser rejects both a bare `&` and a `--` inside a comment where MuJoCo tolerates them |
| 305 | The lunge is the missing piece, and it must come from the legs | **Confirmed in direction, not yet in magnitude -- 1.7 % -> 6.5 %, and the speed is half the published** | `_apply_lunge` drives hip RETRACTION plus knee and ankle extension during a `bag_jump` strike, scaled by the strike's own velocity profile. The push is produced BY THE BODY through existing joints; an impulse on the trunk would have been a teleport with a physics-shaped name. Measured, 6 seeds x 1500 steps, oracle-steered approach: **31 strikes, 2 caught, 6.5 % earned** against 1.7 % without it, and closest approach improves to **6.5 mm** on the best seed (was 12.6). Peak nose speed during a strike **0.269-0.421 m/s, median 0.320** -- nine times the 0.035 m/s walk, and still **half** the same-family published 0.57-0.85 m/s (Vollin & Higham 2021, *Coleonyx*). `LUNGE_EXTEND` = 0.55 is INVENTED and is deliberately NOT tuned upward to close that gap: rule 2. What the shortfall says is that a bigger push needs the hindlimb geometry or the cord, not a larger number |
| 306 | A new body part can carry its own mass | **Refuted twice, and the second time cost five tests** | the eyelid and throat bodies move, so MuJoCo refuses them at zero mass. Giving them an explicit `<inertial>` compiled fine and broke five tests, because the generator's mass rescaling reads GEOM masses: 0.06 g outside that budget moved the total to 0.038060 (gate wants 0.038), the actuator scale to 0.620307 (pinned at 0.038/0.0612 = 0.620915) and the reward calibration's bodyweight fraction off in the seventh place. Fixed the way the jaw was: the mass comes from non-colliding geoms (contype/conaffinity 0) INSIDE the budget, and the 0.06 g is taken out of the head, so head + jaw + lids + throat = the 0.005 kg the head had alone. Template total back to **0.061200**, v2 to **0.038000**, and the suite back to 529/537 with only the #291 contract tests failing. Adding mass to this animal is only ever moving it |
| 307 | The animal looks wrong because it is made of primitives | **Partly -- half of it is that it has no PATTERN, which is independent of shape** | the species is called the LEOPARD gecko and this body has been solid `rgba="0.83 0.73 0.42 1"` since it was built: no spots, anywhere. Shape is what a mesh would fix; pattern is not, and it can be fixed now. `tools/make_skin_texture.py` generates a tiling 512 px skin -- two spot scales, granular speckle, a pale venter -- and a banded tail texture, wired through the `skin` and `skin2` materials. APPEARANCE ONLY: a material's texture is not a body, geom, site or actuator, so `assert_body_unchanged` does not see it, no mass or contact moves, and the animal's own `head_cam` looks at the world rather than at itself. The pattern is **INVENTED** -- read off photographs, since no quantitative colour-pattern morphometrics for this species were found -- and says so in the XML it is referenced from |
| 308 | A texture path is a cosmetic detail | **Refuted -- it broke the generator, the world builder and a test, each for the same reason** | MuJoCo resolves `texturedir` against the WORKING DIRECTORY when a model is compiled from a string and against the MODEL FILE when it is loaded from a path, so no single relative value satisfies both. Three call sites were relying on the second while doing the first: `utils/build_lab_morphology.py` compiles the candidate from a string (fixed by handing it the asset bytes via a new `_assets()` helper -- the honest fix, rather than bending paths until one case works), and both `utils/build_world.py` and `tests/test_world.py` wrote comparison copies to the REPOSITORY ROOT, where a model's relative `textures/` does not exist (fixed by writing them beside the source model, which is the only directory a model's relative assets resolve from). A copy of a model has to live where the original lives |
| 309 | A `type="2d"` texture shows on a body made of ellipsoids | **Refuted -- it is declared, loaded, and never drawn** | the skin texture was wired, the model compiled, the file loaded, and the animal rendered solid gold. MuJoCo maps a `2d` texture onto planes and boxes; every visual part of this animal is an ellipsoid or a capsule, which need `type="cube"`. Changed, and the spots appear. Recorded because the failure is SILENT -- no warning, no error, just an unspotted leopard gecko |
| 310 | A geom's material decides its colour | **Refuted -- the geom's own rgba wins, and a class default counts as the geom's own** | MuJoCo resolves a geom's colour from its `rgba` attribute in preference to its material's, and `<default class="visual">` in this body sets `rgba="0.78 0.72 0.45 1"`. **Every flat-material geom in this animal has therefore been drawing the class's cream since the body was first built** -- the eyes were never black, they were cream spheres, and the "two pale blobs" on the head that prompted this were the EYES. It stayed invisible because the whole animal was cream, and only became obvious once the skin got a texture (textures are multiplied by rgba rather than replaced by it, so textured geoms were unaffected). Fixed by giving the eye, pale, teeth, tongue, belly and spot geoms their material's colour explicitly -- 7 geoms. Found by colour-coding every head geom and re-rendering, after three wrong guesses at what the blobs were |
| 311 | Primitive shapes can be pushed into a convincing face | **Refuted, and the attempt is recorded rather than quietly reverted** | jowls, brow ridges, a snout bridge, a rounded tip and a lip line were added to widen the cranium and shape the snout. The jowls rendered as two cream spheres stuck to the neck and made the head lumpier, not more gecko-like; they were REMOVED rather than iterated on. What actually helped was not more geometry: it was the spots (#309), the eye colour (#310), and making the eyes matte and prominent. 62 visual geoms is where this approach stops paying. A real face needs a MESH -- and the search for one (#312) is the reason that is not a small ask |
| 312 | A mesh of this animal can be obtained rather than made | **Refuted for every candidate actually inspected, and the mesh was generated instead** | Print models (CGTrader, $14-$30) are solid watertight sculpts with no rig, sold for printing; the rigged candidates found were BLEND-only, or 5,092-poly generic "lizard" with a walk cycle that does not match this skeleton; the CC BY Sketchfab option is a stylised 84,479-tri sculpt; an AI sculpt (Meshy) came back at 1,958,254 faces. Every one of them arrives as ONE piece in SOMEBODY ELSE'S POSE, and this skeleton has 23 moving bodies -- so each would have to be cut into 23 pieces, re-posed, and re-seamed, with a licence attached. Generating the skin from the body's own primitives avoids all four costs and is owned outright. 5,520 triangles against the sculpt's 1.96 M |
| 313 | A body's position is a good enough station to sweep its skin from | **Refuted -- a body's position is its JOINT, not its middle** | The first generator placed one ring at each body's `xpos` and interpolated between them. The head's joint is at the back of the skull and its geoms run 32.6 mm forward of it, so the head got a 20 mm mesh that had already tapered to a tenth of its width by the eyes: the eyes, the lip lines and the whole lower jaw ended up OUTSIDE the skin. Fixed by intersecting every visual primitive with each station plane and fitting one ellipse to the union, so the skin is the shrink-wrap of the gated primitives rather than a guess between joints. Head mesh went from 20.0 mm to 32.5 mm long and 19.6 mm to 26.0 mm wide |
| 314 | A sweep can assume its own direction | **Refuted twice in one function, in opposite ways** | (a) The monotonic-ownership rule walks the chain nose-to-tail, but the sweep ran low-x to high-x, which is TAIL-to-nose: `max(prev_rank, rank)` then latched on the tail tip and awarded all 139 stations to `tail5`, producing one 5,520-triangle mesh for the last tail segment and nothing for the other eight bodies. (b) Reversing the sweep silently inverted every face, and an inward-facing normal renders as a flat dark silhouette. Both fixed by measurement rather than by assumption: the sweep now runs in CHAIN order explicitly, and `stitch` checks the sign of normal-dot-radial and flips the winding if it is negative |
| 315 | A texture that works on the primitives will work on the mesh | **Refuted -- and the two requirements are mutually exclusive** | #309 established that `type="2d"` does not map onto curved PRIMITIVES, so every texture in this body was declared `type="cube"`. But a cube texture is addressed by POSITION, and ignores texture coordinates entirely -- so the UVs the mesh generator writes were discarded and the skin rendered as untextured grey, which is what made the first meshed render look like a dark torpedo. Neither texture type serves both. Fixed by declaring both from the same PNG: the primitives keep `cube`, the meshes take a new `2d` twin |
| 316 | A class default is harmless on a textured geom | **Refuted -- SECOND appearance of #310, new symptom** | #310 found that a geom's own `rgba` beats its material's, and that `<default class="visual">` counts as the geom's own. It was closed by fixing the FLAT-material geoms, on the reading that textured geoms were unaffected. They are not: MuJoCo MULTIPLIES rgba into the texture. Skin texture (0.84, 0.73, 0.42) times the class default (0.78, 0.72, 0.45) is (0.65, 0.52, 0.19), which is the dark olive the first mesh render came out. Recorded as a second row because #310's closing statement was wrong, not because #310 was wrong. Every mesh geom now carries `rgba="1 1 1 1"` explicitly |
| 317 | Smoothing removes the segmentation from a chain of primitives | **Refuted -- smoothing rounds a valley, it cannot fill one** | Each axial primitive tapers to nothing at its own two ends, so where two meet tip-to-tip the union has a real waist: the tail measured 13.4, 11.5, 15.7 mm at 10 mm intervals, and an 11-station Hann smooth left 6 sign changes in the profile. The tail rendered as a stack of cones. Fixed with a rolling MAXIMUM of half a segment (6 stations) before the smoothing: it fills the valleys while leaving every peak at its measured value. 6 sign changes to 3, which is the minimum the silhouette allows (trunk, vent, tail peak). Two further faults surfaced underneath it -- the vent has a 6 mm stretch with NO primitive at all, and skipping those stations made the one-ring overlap between pieces 16 mm wide instead of 1.6 mm, so trunk and tail each poked through the other; and bridging that gap tip-to-tip gave a 9.8 mm waist between a 28 mm trunk and a 32 mm tail. The gap is now bridged from anchors 4 stations back in solid flesh, with a declared INVENTED 0.82 pinch, and is exempted from the rolling maximum so the anatomical waist survives while the artificial ones do not |
| 318 | The mesh and the skin are appearance-only, therefore behaviour-neutral | **Confirmed -- but measured, not assumed, because the assumption was not safe** | The animal's eye is a camera ON the animal, and `_policy_scene_option` in `legacy` mode DRAWS group 1, which is the group every new mesh geom and the skin sit in. The module's own docstring warns that sealed mode is the only one that masks them and that legacy "is not a guarantee that arbitrary appearance edits are distribution-neutral". So this was checked three ways rather than argued. (a) A lab body rebuilt from a mesh-free, skin-free copy of the same source has bit-identical `body_mass`, `body_inertia`, `body_ipos`, `jnt_range`, `dof_damping` and `actuator_gainprm` -- max abs difference exactly 0 on every array, total mass 0.038 kg both ways. (b) After 300 steps from the `stand` keyframe the two worlds' `qpos` are equal to the last bit. (c) The 64x64 `head_cam` image the retina actually consumes differs in **0 of 4096 pixels**, max channel difference 0. The mechanism, INFERRED and not separately tested: the camera sits inside the closed head surface, so the 3 mm of snout in front of it presents back faces and is culled. This row exists because the next session will see a poor hunt number next to a big appearance change and reach for the obvious explanation |
| 319 | A poor hunt result arriving next to a big change was caused by that change | **Refuted -- #318 is the evidence, and the numbers were already like that** | `tools/autonomous_gecko.py` on seed 2 gives 330 frames with prey on screen, 0 decisions and a 285.7 mm closest approach, against the 828 frames / 97 of 144 right / 72.0 mm recorded in that file's own comment. The obvious reading was that today's skin broke the eye. It cannot have: #318 shows the retina image is bit-identical and the trajectory is bit-identical, so the same seed would have produced the same 285.7 mm before any of it. Seed 2 at 3000 and at 3400 steps gives byte-identical results, so nothing here is stochastic either -- the animal simply walks away from the prey on that seed and never gets close enough for a 9 mm target to clear the motion floor. What IS open is which seed the file's comment describes: it is not stated in the comment, and it is not seed 2 |
| 320 | The autonomous hunt's recorded numbers describe how it behaves in general | **Refuted -- it varies from perfect to useless across seeds, and the recorded figure names no seed** | Seven seeds, 3000 steps, everything else at its default. seed 2: 330 frames on screen, 0 decisions, 285.7 mm. seed 3: 289, 276 decisions, **0 right**, 251.3 mm. seed 4: 595, 27, 27 right, 226.9 mm. seed 6: 890, 82, 79 right, 115.5 mm. seed 7: 245, 123, 30 right, 182.5 mm. seed 9: 391, 14, 12 right, 49.1 mm. seed 11: 907, 155, 155 right, 200.8 mm. Pooled: **303 of 677 decisions right, 45 %**, against the 97-of-144 (67 %) written in the tool's own comment. Per-seed precision is 0, 0, 100, 96, 24, 86, 100 % -- it is not a distribution with a mean worth quoting, it is bimodal, and a single seed was being reported as the result. Closest approach ranges 49 to 286 mm. Rule 6 applied to my own evidence: one seed is not a measurement |
| 321 | Rigid one-mesh-per-body is an adequate skin for an ANIMAL | **Refuted -- it is adequate for a robot, which is what it was borrowed from** | Every Menagerie model uses rigid per-body visual meshes, and every Menagerie model is a machine made of separate hard shells. Yaw this animal's neck and head 25 degrees and the two pieces rotate about different frames: a wedge opens in the neck, in precisely the movement this project has spent the most effort on. Replaced by one MuJoCo `<deformable><skin>` over the whole animal: 5,271 vertices, 9,480 triangles, 22 bones, 13,525 bone-vertex links, and 0 vertices needing the nearest-body fallback. Weights are exp(-(d/SIGMA)^2) to a point cloud sampled on each body's own primitives, with candidates restricted by region so a foot resting against the belly cannot be driven by the trunk. Written inline rather than as a `.skn` binary so the animal stays inside the one file whose SHA-256 the provenance chain hashes. DeepMind's own quadruped-with-a-tail does the same thing (`dm_control` dog `create_skin.py`), which is what prompted looking for it |
| 322 | A comment is safe text to write into a generated XML | **Refuted -- SECOND appearance, and this time it failed silently** | A bare `--` inside an XML comment is accepted by MuJoCo's reader and is a hard `ParseError` in ElementTree, which is what `build_lab_morphology` parses the source with. The source therefore COMPILED PERFECTLY and could not be built FROM: the generator threw, the previous lab body stayed on disk, and the world was rebuilt from it, so the skin was silently absent from the world while every check that looked at the source said it was there. Caught only because `nskin` was 0 in the world after a build that reported success. The generator now runs BOTH parsers before it keeps the file. The first appearance of this was a bare `&` in the same position |
| 323 | A part that fitted the rigid meshes still fits once they become a skin | **Refuted -- rigid and blended surfaces do not coincide, by construction** | The tooth rim was generated from the jaw's own rings and sat correctly against the rigid jaw mesh. Under the skin the same surface is blended between the jaw and the head, so it no longer passes where the rim does, and the rim juts out of the cheek as a pale wing. The pale throat ellipsoid failed the same way at the mouth corner. Both identified by colour-coding every geom on the head and re-rendering, after two wrong guesses -- the same method that settled #310, reached for sooner this time. The rim is retired; the throat is now a BONE of the skin, so the breathing shows as the skin itself swelling, which is what it should always have been |
| 324 | Drawing a few hundred spots over an image is cheap | **Refuted, and it is my own method error rather than a model one** | `_spots` evaluated every blob over the WHOLE image, nine times for wrapping. On the 512 x 512 tiles that is 116 blobs over 262 k pixels and nobody noticed. On the 512 x 2048 whole-body image it is 1,120 blobs over 1.05 M pixels nine times over: about 10^10 element operations, and the call did not return inside a two-minute timeout. A blob is zero outside its own radius, so it is now drawn into a window around its centre and the wrap offsets are only drawn where they actually overlap: the same image in **0.97 s**. Recorded because the failure looked like a hang rather than a mistake, and a hang invites waiting rather than reading |
| 325 | A measurement is safe to put on screen on its own | **Refuted -- a bare number under the word ACCEPTED reads as a pass** | The walking clip captioned itself "ACCEPTED WALKER" and "front duty 0.456/0.443". Both are true and together they are misleading: front-foot duty has a PUBLISHED target of 0.70 and this is one of the two gates of six that this walker fails. The canonical evidence clip prints the target next to the value for exactly this reason and I had dropped it while rewriting the caption. Re-rendered as "front duty 0.46/0.44 vs 0.70 target". Cross-checked against `tools/accepted_walker_clip.py` on the same controller: FL 0.447 / FR 0.444, 0.916 m in 20.0 s -- the showcase's 0.456 / 0.443 and 1.188 m in 26.0 s are the same walk, so the skin changed the gait in no respect, as #318 predicted it could not |
| 326 | #284's "two-channel in practice" is a tendency, not a hard fact | **Refuted -- it is exact, and re-measured on today's body** | 900 autonomous steps, seed 3, full loop on. Max GATE value over the whole run, per channel: hunt 1.0000, explore 1.0000, **flee 0.0000, bask 0.0000, rest 0.0000, groom 0.0000**. Steps won: explore 778, hunt 121, everything else 0. These are not small numbers, they are exactly zero for 900 consecutive steps, so four of the six channels are not losing a competition -- three of them are never entered into one. `flee` and `bask` because their inputs do not exist in the world (#267); `rest` because the animal walks at 0.035 m/s and is never working (#284, and that one is correct physiology, not a defect); `groom` has a 0.05 tonic salience and is fully inhibited by explore's 0.35 on every step. Prescott 2024's six-channel selector is running as a two-channel one, confirmed by measurement rather than inferred |
| 327 | The missing three world objects are a build task | **Confirmed for the world, refuted for the brain: the world was one command, the brain is not wired to read it** | `utils/build_world.py` already emits all three, with the natural history behind each already read and written into the file: SHELTER as a raised slab because this species does NOT dig (retreat chambers traced through a demolished stone wall were unmodified masonry voids); WARM PATCH as a SURFACE not a lamp, because body temperature tracks the substrate at r2 = 0.97 against air at 0.92 (n=12, Hastings et al. 2023); THREAT as a mocap body carrying its own caveat that a purely visual predator draws a defensive reaction on 0.07 of trials against 0.21 for scent (n=42). Built `morphology/gecko_world_furnished_v1.xml` in one command -- all four non-anatomical bodies present, skin intact, the animal verified identical to source by the build's own guard, and the committed world left untouched. But the brain still cannot see any of it: `_predator_distance_m()` returns None UNCONDITIONALLY, `brain_command` passes `shelter_bearing_deg=None` and `warm_bearing_deg=None` as literals, and no thermal term reads the patch. So the world is done and the three sensor paths are not. That is the next piece of work, and it is a wiring job, not a research one |
| 328 | The swept tube needs a taper to close its ends | **Refuted -- the measured envelope already closes, and tapering it again creased the snout** | The end rings were multiplied by a circular profile falling to 0.03. But the envelope is measured from the primitives, and the snout's own rounded tip and the tail's last segment shrink to nothing on their own, so the tube was being closed TWICE: twenty vertices piled up a fraction of a millimetre apart, which shades as a crease no matter how fine the mesh is, and that crease was the line running back from the nose. Replaced by a quarter-ellipse dome that grows OUTWARD off the last measured ring and ends on a single point, its length taken from that ring's own mean half-width so a blunt snout gets a blunt dome. Head mesh 32.5 to 34.7 mm long, and the crease is gone |
| 329 | A normal map can give the skin its tubercles without adding geometry | **Refuted -- MuJoCo 3.9 accepts the map and its renderer ignores it entirely** | `<material><layer texture="..." role="normal"/></material>` compiles without complaint and `ntex` counts both images, so the feature looks present. It does nothing: a flat slab lit from one side, rendered with and without a strong sinusoidal normal map, differs in **0 of 57600 pixels**, maximum channel difference **0**. Measured before building anything on it, which is the only reason the tubercles are not silently missing. Real geometry at tubercle scale needs roughly ten times the vertices, so they are baked into the colour instead, as a light side and a dark side per bump -- the pre-normal-map technique, chosen because the modern one is not actually available here |
| 330 | Stripping mesh geoms line by line is good enough | **Refuted -- it deletes whatever else shares the line, and the failure names the wrong thing** | All three generators compile the body with its mesh geoms removed, by dropping every LINE containing `type="mesh"`. When the wiring script wrote an eye sphere and its replacement mesh geom onto one line, that filter took both, and the eye generator reported `expected 2 eye geoms, found 0` -- a message that points at the eyes rather than at the stripper. Fixed in all three by removing mesh geoms as ELEMENTS with a regex over the geom tag, and by writing the two geoms on separate lines. Same class as #322: a text transform that is correct on the text it was written against and silently wrong on text that is merely formatted differently |
| 331 | A blanket rule for "meshes the skin supersedes" is safe | **Refuted -- it hid the eyes, and the animal rendered with no eyes at all** | The continuous skin replaced 23 rigid per-body meshes, so the wiring script draws every mesh geom transparent. Two new mesh geoms were then added that the skin does NOT supersede -- the eyeball and the eyelid, which the skin deliberately excludes because an eye is not skin and wrapping it would put a bulge where the pupil is. The blanket rule caught them both and the face came out blank. A rule written as "everything except the one exception I had in mind" acquires new members it was never checked against. Now excepted explicitly by name |
| 332 | Rebuilding the calibrated body is rebuilding the body | **Refuted twice, and the failure is invisible at the point it is made** | `build_lab_morphology.py --fit-com` writes the v2 body ONLY; v1 comes from the same script WITHOUT the flag. Rebuild v2 and forget v1 and the repository is left in a state that compiles, renders, and passes 14 of 14 morphology gates while v1 carries a stale source hash -- and three reproducibility tests fail with a message about a missing marker that names nothing relevant. This cost a debugging pass twice in one session. Closed with `tools/rebuild_body.py`, which runs the six steps in the one order that works and prints what each did |
| 333 | The four dead channels need new sensing built | **Refuted -- all three sensor paths existed as stubs and needed connecting, not writing** | `_predator_distance_m()` returned None unconditionally; `brain_command` passed `shelter_bearing_deg=None` and `warm_bearing_deg=None` as literals; `thermal_error_C` carried the comment "the world has no temperature field, so this reads exactly zero forever. It is wired so that adding a field switches it on". Connected: the threat's distance through the nose into flee, the ground's temperature under the animal into body temperature into the cold error into bask, and both bearings into the motor programs. Every one of them returns None when the body is absent, so the COMMITTED world is untouched -- re-measured on it afterwards and it gives the identical 121 hunt / 778 explore / four zeros. Result on the furnished world: **bask 1041 steps, and the animal walks 34.0 cm to the warm patch and warms from 25.2 to 29.7 C**, which is inside its published preferred band of 29.5-31.9 |
| 334 | A basking program is a program that walks toward warm ground | **Refuted -- it walked straight over the far edge and cooled down again** | Measured: the animal closed from 34.0 cm to 3.6 cm, warmed 25.2 to 28.9 C, then kept walking and cooled, cycling forever without ever reaching its band. This species takes heat by LYING on the ground -- thigmothermy, body tracking substrate at r2 = 0.97 against 0.92 for air, Hastings et al. 2023, n = 12 -- so stopping on it is not a convenience, it IS the behaviour. Added `on_warm_ground` to the motor program: drive goes to zero once the ground underfoot is at or above the preferred floor. Body temperature then reaches 30.0 C and the bask channel releases its grip |
| 335 | A channel with the largest salience is the channel that gets released | **Refuted -- the basal ganglia has a floor, and it is what actually caps this brain at four channels** | Bisected on one channel with everything else at zero: salience **0.2008** is the point at which the gate stops being exactly 0.0000. Below it nothing is released at all, however far ahead of the others it is. That makes two channels unreachable by any world: `groom` carries a 0.05 tonic salience and `rest` IS fatigue, which plateaus at 0.011 in an animal that strolls (#284). Both are an order of magnitude short. `flee` is the interesting case -- its salience IS the published probability of a defensive reaction in this species, 0.21 to scent alone and 0.28 with sight (n = 42) -- so it clears the bare threshold by **0.009** and is beaten by explore as soon as hunger passes about 0.2. Measured: with hunger 0.30 present, flee needs 0.2171 and has 0.2100, short by 0.0071. **Nothing was tuned.** The model's prediction is that a hungry gecko does not flee from something standing on it, and it is left standing as a prediction rather than corrected into agreement |
| 336 | Four of six channels is the ceiling, and it is the world's fault | **Partly -- two were the world's fault and are now fixed; the other two are the selector's** | `tools/four_brains_demo.py` runs two scenarios on the furnished world and releases **bask, explore, flee and hunt**: 4 of 6, up from 2. The two scenarios exist because of #335 rather than for presentation -- a hungry animal (0.90 meals owed) shows bask, explore and hunt, and only a fed one (0.15) can let flee through. `rest` and `groom` remain at exactly 0.0000 and cannot be raised without changing either a published constant or the basal-ganglia weights, and the weights were settled by a gating constant they must independently reproduce (Rule 6). Left alone and recorded |
| 337 | What I was about to model as gecko skin was right | **Refuted for most of it -- 26 of 36 claims died under adversarial verification** | A 42-agent pass researched six surface features and then tried to REFUTE every claim, defaulting to rejection for anything the verifier could not confirm from a source it had opened. **10 survived, 26 did not.** The survivors are specific and useful: the dorsum carries TWO element types, a pavement of small flattened scales with larger raised tubercles set into it, so a bump map modulating one tiling is the wrong structure (Russell et al. 2014, SEM, n = 5); the movable eyelid is real and the lids are THICK, needing volume rather than a painted ring (Boulenger 1885 and Smith 1935, independently, 50 years apart); there is NO spectacle, so the cornea is bare and wet and must be shaded as bare cornea, and the caseous plug clinicians see is routinely misdiagnosed as a retained one; the pupil is vertical; the tympanum is an open aperture on the side of the head with a visible membrane, photographed in this species; five digits per foot, no reduction. Also recorded: the dorsal tubercle-row count for *macularius* proper is in NO source that could be opened -- the nearest figure, 18 rows, is for the Rajasthan/Delhi population whose species status is itself disputed. Written to `docs/GECKO_SURFACE_ANATOMY.md` and NOT applied: the user stopped the appearance work before this returned, and a refuted majority is exactly the reason not to have built first and checked after |
| 338 | The brain demo was running on the accepted walker | **Refuted -- it was on the REJECTED one, and the file I was reading says so in its own comment** | `GeckoBrainEnv` defaults to `gait_profile="legacy"` with the frozen trained residual loaded on top, and `tools/four_brains_demo.py` passed neither argument, so it took both defaults. The accepted walker is `lab` with ZERO residual. The comment sitting twenty lines above the constructor states the measurement: lab + no policy gives foot duty 0.770 0.764 0.787 0.790, even to 0.006 against published targets of 0.70 fore and 0.765 hind; legacy + policy gives 0.704 0.601 **0.435 0.367**, hind feet at half. It also states why this keeps happening -- the defect is in the HIND feet, the front pair looks fine, and front duty was the only thing being checked. The user caught it. The env exposes `info["accepted_walker"]`, which was sitting there unused; the demo now asserts on it before filming and exits if it is False. Same error the canonical walker clip's docstring records being made repeatedly, made again |
| 339 | Once the sensors were wired, flee would release whenever a predator was on the animal | **Refuted -- the margin is 0.009 and ANY other salience closes it** | Re-measured on the correct walker. With the predator touching the animal, flee's salience sits at exactly **0.2100**, its published ceiling, for 181 consecutive steps, and the gate stays at **0.0000** throughout. The reason is in the salience vector at the closest approach: `hunt` was at **0.1504**, because the animal could also see a cricket. Flee clears the bare release threshold of 0.2008 by 0.009, so a competitor a seventh its size is enough to keep it shut. It releases at gate 0.938 the moment hunger is zero and hunt goes with it. **Nothing was tuned to get this**: the published 0.21 and the basal-ganglia threshold both stand, and the prediction that a gecko which can see food will not flee from something standing on it is left as a prediction. Final: `tools/four_brains_demo.py` releases **bask, explore, flee and hunt** on the accepted walker, 4 of 6, with `rest` and `groom` an order of magnitude below the threshold and unreachable |
| 340 | A swept oval can be shaped into a head | **Refuted -- the user's word for it was a missile, and that is what a sweep of an oval is** | An oval swept along a line can be fat here and thin there and that is its whole vocabulary: it cannot express a flat crown, a brow, a jaw angle, or a snout that narrows faster than it drops. Ten rounds of adjusting its numbers produced ten missiles. Replaced by a sculpted head: a SUPERELLIPSE cross-section with different exponents above and below the midline (flat on top, round underneath, which is what a skull is), and width and height on SEPARATE curves so the widest point can sit at the jaw angle behind the eyes while the snout narrows fast and stays tall. The profile is INVENTED from photographs and ANCHORED to the body -- its widest station is set to the widest half-width the sweep measured from the head's own primitives, so the gated head width is reproduced and only the shape between stations is styled. The first fifth blends from the measured neck ring so the join is invisible. From above it is now a triangle wide at the jaw and blunt at the nose |
| 341 | The body behind the head was fine and only the head needed sculpting | **Refuted -- same missile, same cause, and the pinch at the hips was a number I invented** | Everything from the neck back was still the sweep: no neck (the neck capsule is nearly as wide as the trunk), no belly (a capsule is one width all along), a 0.82 constriction at the vent that I invented in #317 and that the user correctly read as the animal being pinched behind its legs, and a tail that ended in a needle. All of it replaced by `tools/gecko_sculpt.py`: a neck that dips to 0.84 between skull and shoulder, a trunk widest at 0.45 of its length with a slightly flat back, a tail continuous with the hip (pinch deleted, VENT_PINCH now 1.0) that is fat through its first third, carries the real animal's transverse tubercle ridges as a 3.5 % radius wave at 4.6 mm, and tapers to a ROUNDED tip. Anchored at three measured stations: trunk maximum, tail maximum, and the head's occiput ring. Plus FIVE TOES per foot, built along the five digit capsules the body already defines (so nothing here places a toe) and slender rather than padded, because #337 verified this species has no adhesive pads; a knuckle sphere at each of the twelve limb articulations, because two tapered tubes meeting at an angle leave a wedge open and a twisted collar; a pale torus round each eye opening, which is the most recognisable feature of the animal's face; and a mouth line, nostrils and ear openings painted at the v positions the skin generator MEASURED off its own rings rather than guessed. Skin went from 9,019 to about 20,600 vertices. Gates 14/14 before and after |
| 342 | A knuckle sphere sized to the limb tube hides the joint | **Refuted on the first render -- it sat between tube ENDS that had tapered to 0.55 r, and looked like a ping-pong ball** | Sized at 1.06 of the tube's base radius, each knuckle was nearly twice the diameter of the two tube ends meeting inside it, and half of every sphere was painted with the pale BELLY colour because its texture coordinates wrapped the full 0..1 girth. Two causes, two fixes: the limb taper went from 0.45 to 0.25 so a bone no longer halves in radius over its length, and the knuckle went to 0.72 r; and every knuckle and toe now maps its u into the spotted dorsal band, 0.15-0.35, rather than the whole wrap. The eye rim was also glowing white for the same reason as #310 -- untextured, so its rgba IS its colour, and the blanket 1 1 1 1 rule caught it -- and now has its own pale value. Skin 23,751 vertices, 58,707 bone links, 0 unbound. Gates 14/14 |
| 343 | The body could be made to match the reference photographs without touching the physics | **Partly -- the appearance could; the blink and the tongue needed two joints and a slide, recorded as a body version bump** | Appearance, no physics: colour from the reference animal (yellow back, orange flanks, WHITE belly, white tail with orange near the vent, near-black spots -- the old buff was fairly called mud); ONE continuous skin per limb through the joints with a parallel-transport frame, replacing six tubes and six knuckle balls, which were the same mistake the trunk made before it got one skin; toes angled up 14 degrees and a tenth shorter because the flat digit capsules put their tips through the floor; the throat's skin weight tripled so the published breathing actually shows, and its amplitude raised 5 to 12 degrees (INVENTED, declared); the mouth floor painted dark pink on the mandible's upper face, which was first put on the same v as the snout and gave the animal a pink nose (#313's lesson again -- the jaw now lives on its own strip of texture at v 0.995). Body: the single `eyelids` body with a PITCH hinge swung both lids FORWARD over the front of the eye; a lid closes DOWN, which is a roll about the head's long axis, and a roll closes one side while opening the other -- so it became `eyelid_L` and `eyelid_R`, each with its own roll hinge and mirrored ranges. The tongue got a slide (0-12 mm) driven by two PUBLISHED numbers for this species and nothing else: post-feeding labial licking at 0.165 licks/s (Cooper, DePerno & Steele 1996, n = 16, already in the registry) and the tongue-flick rate the nose already computes. nq/nv/nu 42/41/28 to **44/43/30**, `nsite` delta 7 to 8, walker action dim still 25. Skin 21,663 vertices, 52,639 bone links, 0 unbound. Gates 14/14 |
| 344 | A mesh geom and a skin read the same texture the same way | **Refuted -- a mesh's texture rows count from the BOTTOM of the image, a skin's from the top** | The thick eyelid hood rendered pure white with texture coordinates (0.22, 0.10). Measured rather than guessed: `mesh_texcoordadr` was set, the material was the body's with its texture bound, and the PNG at (0.22, 0.10) is yellow (237, 200, 59) -- while at (0.22, 0.90) it is the white of the tail tip (228, 225, 218), which is exactly what the lid showed. The continuous skin, on the same image, draws the yellow head at v = 0 and the white tail at v = 1, so its rows count from the top; a mesh geom's count from the bottom. The lid now asks for 1 - v. The eyeball never revealed this because its image is a centred disc, symmetric under the flip. Anything else that draws a mesh from the body texture will need the same inversion |
| 345 | A body version bump is the count change and nothing else | **Refuted three times in one test run, and only one of the three was a count** | After #343 the suite showed 12 failures against the standing 8. (a) I had raised the instrumentation test's site delta from 7 to 8 for the second eyelid site -- wrong, because that delta counts the MEASUREMENT sites removed from the comparison model, and the eyelid sites are present on both sides of it; back to 7. (b) The generator's `actuator_scale` is 0.038 over the TEMPLATE mass and is pinned at 0.038/0.0612; the tongue body's 1e-5 kg had pushed the template to 0.06121 and moved the scale in the fourth decimal. Taken back out of the jaw's collision capsule, 0.0006 to 0.00059, exactly #306's move, and the template is 0.061200 again. (c) The realism recorder pins the hinge count: 35 since #304, now **36** -- one eyelid hinge became two; the tongue is a slide and is not counted. Every one of the three was a test that knew something about the body I had not thought to check before changing it. Final: **529 of 537**, the same eight `GateContracts` cases #291 documents |
| 346 | The spine should bend more while walking, and there is a published number for how much | **Confirmed for the number and the fit; refuted for making it the default -- it moves two of the walker's own selection criteria the wrong way** | A 24-agent pass (`docs/LATERAL_UNDULATION.md`, 18 numbers verified, 2 refuted) found a SAME-SPECIES measurement: Jagnandan & Higham 2017 (Sci Rep 7:10865, n = 10, SVL 104.6 mm, level trackway, walking, hind duty 0.78) -- pelvic girdle yaw excursion **~50 degrees per stride** with the intact tail, peaking AWAY from a hindlimb at 0.60-0.65 of that limb's stride and toward it at footfall. The accepted walker measured **24.4 degrees**, peaking at 0.83. The spine is one tendon across the three lateral joints, driven at `spine_amp` 0.30 of half-range: excursion is linear in it (24.3 / 36.4 / 48.9 / 61.8 / 74.6 degrees at 0.30 / 0.45 / 0.60 / 0.75 / 0.90), so **0.61** is DERIVED, and the timing was 0.21 of a cycle late, so `spine_phase` was exposed as a declared lab parameter at its existing default 0.0 (Session-3g move, behaviour unchanged) and set to **0.21**: measured 51.8 degrees peaking at 0.62. Then the gates, one declared change per trial as the protocol demands: baseline / amp only / fitted -- hind duty 0.636 -> 0.650 -> **0.675** (toward the required 0.73-0.83), speed 0.0427 -> 0.0427 -> **0.0369 m/s**, net-path 0.823 -> 0.712 -> **0.697**. The lab set was selected for > 0.04 m/s and > 0.8 net-path, and the published spine wave breaks both on this controller while helping the duty the contract is actually stuck on (#291). NOT made the default: the fitted pair is a declared parameter set, `--set spine_amp=0.61 --set spine_phase=0.21`, filmed as `walk_spine.mp4`, and the trade is recorded here rather than resolved by picking a smaller amplitude that matches nothing |
| 347 | Brain 6, the day/night clock, is connected now that the env has a `circadian` flag | **Refuted -- #262's verdict still stands, and the flag made it look fixed** | `GeckoBrainEnv` now takes `circadian=` and constructs `Arousal()` when it is True, and `self.clock` is reset alongside the other modules. But `clock.step` is **never called anywhere in the file**, and `self._arousal` is assigned exactly once, to the constant 1.0 at construction, then only ever READ -- to be copied into `info["arousal"]`. So switching the flag on builds the object, resets it, and reports a number the object never produced. #262 said "built, tested, and connected to nothing"; that is still true, except there is now a switch that makes it appear otherwise. Recorded rather than fixed, because fixing it means deciding what arousal should modulate, and nothing published says |
| 348 | `python -m common.morphology_audit` with no arguments audits the body that walks | **Refuted -- its default is the UNFITTED source and it reports 1 of 14** | `DEFAULT_XML` in `common/morphology_audit.py` is `morphology/gecko_body_r.xml`, the hand-written template, which is deliberately not to scale: 61.2 g against the 0.034-0.042 kg band, tail chain 1.038 of SVL against 0.69-0.77, hip height 0.196 against 0.14-0.16. Run that way it prints **1/14 morphology gates passed** and looks like the animal has collapsed. The body that actually walks is `morphology/gecko_body_lab_v2.xml`, the same source after the two fitting passes, and it prints **14/14** -- total mass 0.038, tail fraction 0.22, intact CoM 0.6781 inside [0.639, 0.679], hip 0.1489, shoulder 0.1093, femur/tibia 1.0199, hip pro/retraction 90 deg, inner-eye gap 0.010. Both numbers are true statements about different files. Found while assembling the handoff report, which called the audit the way its own docstring shows. Recorded rather than changed: the default is what the test suite passes in explicitly, and moving it would be a fix aimed at my reading of the output rather than at the code |

### What this changes in the plan

Plan step 5 builds a two-channel wide-field/narrow-field detector and tunes it
for size and speed. Size and speed reject **0.7 %** of visible prey. The motion
floor rejects **91.4 %**, and it does so because the signal arriving at it has
been attenuated twice — once by a one-frame baseline against a quarter-cell
displacement, and once by an efference copy tuned against a measurement that
could only see one side of its own trade-off.

Both are fixed without inventing a single gecko number: one is bounded by a
published gekkotan flicker-fusion ceiling, the other is a gain that was already
declared INVENTED and has simply never been scored on what it costs.

### What was built

`tools/report_data.py` extracts the ledger, the commits, the parameter registry
and the citation harvest into JSON, reading the registry through
`common/provenance.py` -- the same validating loader the simulation uses, so the
report cannot be built from a registry the code itself would reject.
`tools/report_charts.py` draws 21 figures from that JSON, with one rule: a number
measured in this repository is a bar, and a published number that has not been
reproduced here is a dashed marker or a shaded band, so a target cannot be
mistaken for a result. `tools/build_full_report.py` assembles a 62-page document.

Nothing in it is transcribed by hand, which is the point: hypothesis #222 is
what hand transcription does over 29 sessions.

---

## Ledger — rest is the clock, and four numbers that did not reproduce (Session 13)

Every row below was measured in this session before it was written, including
against my own solver. Four of them refute numbers the record had been carrying
to four decimal places.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 349 | `rest`'s salience is fatigue, and the 0.011 it carries is a measurement of the animal | **Refuted -- it is per-step velocity noise crossing a cliff, and #284's physiology was right for the wrong number** | `Physiology.endurance_s` returns **infinity at or below the 0.100 m/s aerobic ceiling** and a finite time above it, so it is a step and not a curve: 30 simulated minutes at 0.035 and at 0.055 m/s give fatigue **exactly 0.000000**, and at 0.101 m/s it saturates to **1.0**. There is no regime in which fatigue rests near 0.011 for a physiological reason. Measured over **5 seeds x 1500 autonomous steps** on the accepted walker (`gait_profile="lab"`, `use_policy=False`, `info["accepted_walker"]` asserted), furnished world: mean speed 0.0268-0.0311 m/s, fatigue equilibrium 0.00203-0.00254, and that equilibrium tracks the **fraction of control steps whose instantaneous trunk displacement crosses the ceiling** at **r = +0.9998**, against **r = +0.71** for the animal's actual mean speed. The falsifier stated in advance: re-integrating the SAME logged trace with speed averaged over one stride (42 control steps at the measured 1.1892 Hz) gives **0.00 % of strides over the ceiling and fatigue exactly 0.0**, against 4.20 % of control steps and 0.002177. `moving_speed` is one control step's displacement, so the number in the rest slot was a property of the sampling interval and not of the animal |
| 350 | The 0.011 rest-fatigue plateau is a figure some run produced | **Refuted -- it is a typed literal** | `tools/four_brains_demo.py` wrote `"rest_fatigue_plateau": 0.011` into the `measured_limits` block of `artifacts/evidence/session12/four_brains.json`, and nothing in the run computes it. No evidence file in `artifacts/evidence/session12/` carries a fatigue key at all. Corrected in place to the measured range |
| 351 | The basal-ganglia release threshold is 0.2008 | **Refuted -- it is 0.199878, and it is conditional on the rest of the vector** | Bisected on this model: with every other channel at zero the gate stops being exactly 0.0000 at **0.199878**; with groom's old 0.05 tonic in the vector -- which `salience_from_drives` supplied on every call -- it is **0.201091**. It is a hard step: gate 0.00000000 at 0.199878 and **0.9271** at 0.200000. The record's "flee clears the floor by 0.009" matches the WITH-groom figure (0.00891); the printed 0.2008 matches neither. It carries **no provenance tag anywhere**, and correctly so -- it is not a constant but an emergent property, conditional on the channel count, the tonic dopamine and the extended variant. It is quoted to four decimals in #335, #339, `tools/four_brains_demo.py`, the evidence JSON and three times in the generated report |
| 352 | Brain 6 can be connected without inventing a modulation for arousal | **Confirmed -- and #347 was right about every consumer except one** | #347 declined to fix the clock "because fixing it means deciding what arousal should modulate, and nothing published says". That holds for every candidate but `rest`: the only two published facts this species has about arousal -- activity onset 81 min after lights-out (Digirolamo 2024, n = 1, SD 89) and an evening peak window (Kronke & Xu 2023, n = 18, captive), both registered and both flagged uncertain -- are measurements of WHEN IT IS ACTIVE AND WHEN IT IS NOT, which is what rest is. `clock.step` is now called every control step on the same declared `homeostasis_time_compression`, `info["arousal"]` reports what the module produced instead of the constant 1.0, and rest's salience is `1 - arousal`. **The mapping is INVENTED and says so; it introduces no constant of its own.** Measured on the accepted walker, furnished world, seed 3, 600 autonomous steps each: at `time_of_day_h=2.0` arousal **0.000**, rest releases at gate **1.0000**, lids shut at the full 70 deg on **100 %** of steps, ground covered **0.1 cm**; at 16.0 h arousal **1.000**, rest gate **exactly 0.0000**, lids 0 deg on 100 % of steps, ground covered **33.5 cm** |
| 353 | Groom's 0.05 tonic salience is inert | **Refuted twice, and it was the least-tagged number in the animal** | It was a bare literal in `brain/gecko_selector.py` and `brain/basal_ganglia.py` with **no entry in `config/proxies.yaml` at any confidence**, no citation and no species -- the surrounding docstring declares the salience WEIGHTS invented, and 0.05 is not a weight on anything. (a) It raised the release floor for every OTHER channel from 0.199878 to 0.201091, narrowing flee's margin over the floor from **0.01012 to 0.00891** and the hunger it survives from **0.2400 to 0.2317**. (b) It was masking a degenerate fixed point -- see #357. Set to 0.0. This does not make groom reachable; it makes its unreachability honest |
| 354 | Grooming in this species has a published trigger the selector can use | **Could not verify -- this repository contradicts itself, and the item is blocked on it** | The generated report states that grooming here "has a real published trigger -- it is a shedding and eye-cleaning behaviour". The registry note on `post_feeding_labial_lick_rate_per_s` states the opposite in terms: *"NO primary source documents EYE-licking in this species or any eublepharid."* Ledger **#152** quotes Delheusy, Brillet & Bels 1995 -- *E. macularius*, opened in full, the same paper the strike is built on -- verbatim: *"La langue se deplace hors de la cavite buccale et **atteint parfois l'oeil**"*. Both statements are in this repository and they cannot both be right. Separately: the grooming that IS registered and published -- post-feeding labial licking, 0.165 licks/s, n = 16, Cooper, DePerno & Steele 1996 -- **already runs**, as a reflex on a 120 s window after eating in `envs/gecko_brain_env.py`'s tongue drive, entirely outside the selector, so the channel may be the wrong home for it. Left unresolved rather than filled |
| 355 | Flee's salience is the published defensive-reaction probability, 0.21 to scent and 0.28 with sight | **Refuted -- 0.28 is not published, and the term that produced it was wired to the wrong sense** | Only **0.21** (chemical) and **0.07** (visual) are registered, both Frydlova et al. 2026, n = 42. **0.28 is their sum**, computed in `brain/vomeronasal.py`, and it appears nowhere in `config/proxies.yaml` -- yet #335 and #339 print it as "0.28 with sight (n = 42)", a derived number wearing a published one's clothes in the two rows that carry this project's flagship prediction. Worse, the condition it was summed on was `predator_visible=bool(danger > 0.0)`, and `danger` is `0.65 * belly_contact + fallen` -- the animal's belly touching the ground, or it having fallen over. That is a **mechanosensory and postural** state rather than sight, and the same factorial measured the mechanosensory term at chi2 < 0.01, p > 0.9, with the authors' verbatim conclusion that neither vision nor touch independently elicited defensive behaviour. The animal also has no visual predator detection to gate on: prey-finding is NOT ACCEPTED and the threat body is not detected by the eye at all. Corrected to `predator_visible=False`; flee's ceiling is the one published number, 0.21. **It did not rescue flee** -- with #353's floor correction the margin is 0.0101 and it still loses to explore once hunger passes 0.24. The prediction stands, now on published numbers only |
| 356 | A resting gecko can be shown resting without inventing a behaviour | **Confirmed for the eye closure, with one invented conjunction declared** | PUBLISHED, this species: Bergel et al. 2026, Nat Neurosci 29:543-550 recorded sleep in **n = 2 E. macularius with EOG electrodes under each eyelid** -- possible only because eublepharids are the only geckos with movable lids and this one closes them, where a tokay cannot. The ANGLE reuses the already-declared invented `EYELID_CLOSED_DEG = 70.0` rather than adding a number; no lid excursion has ever been published for this species or any lizard. The CONJUNCTION -- that anything moving the legs opens the lids whatever the hour -- is INVENTED and declared at the call site. Measured: lids at 70 deg on 100 % of light-phase steps and 0 deg on 100 % of night steps |
| 357 | At exactly zero salience the selector selects nothing | **Refuted -- it selects `hunt`, and an untagged invented constant was hiding it** | With every salience at zero the extended model **settles** (`settled=True`, so it is not an oscillation) to a **uniform gate of 0.000402 on all six channels**, and `selected()`'s `argmax` on six identical numbers returns index 0, which is `hunt`. The animal "chose" to hunt because hunt is declared first in `CHANNELS`. The published classifier was never fooled -- `selection_class()` returned `"none"` throughout -- only this readout disagreed with it. It was invisible because groom's 0.05 tonic (#353) kept the vector from ever being exactly zero: any single channel at or above about 0.01 drives the network to the all-zero fixed point. So `test_a_contented_gecko_does_nothing_in_particular` was passing **because of an untagged constant**, which is precisely what its own docstring warns against. Fixed by using the authors' own published `PARTIAL = 0.05` -- the gate at which their classifier counts a channel as selected at all -- in place of an invented `1e-6` |
| 358 | Eye closure can be driven from the rest gate | **Refuted -- the gate flickers on a tie, and sleep is not a competition outcome** | Traced over 300 light-phase steps: rest releases **cleanly at gate 1.0 for about 18 steps**, then the animal lying still on cold ground drives `bask`'s salience up to 1.0 as well, and with **two channels tied at maximum salience** the model part-releases both at **0.2210 and 0.2209** -- the authors' "dual" outcome, settled on **100 %** of steps. `argmax` turns that tie into a coin flip (rest won 28 of 300, bask 272; selection class `part` on 591 of 600), so a lid driven from the rest gate **fluttered at a 16.7 deg mean** instead of closing. No source says an animal shuts its eyes because one channel out-competed another. Driven from the clock it shuts: 70 deg on 100 % of steps. Recorded separately, and not fixed: in the light phase the selector is **not cleanly selecting**, it is part-releasing two tied channels -- and what the animal DOES is the same either way, because both map to staying still |
| 359 | A destructive edit to this file is recoverable from the repository alone | **Refuted -- and it was my own error, in this session** | A script of mine opened `docs/FAILURE_MAP.md` for writing and only then hit a `UnicodeEncodeError`, leaving the file at **0 bytes**. `git HEAD` carries only Session 11b, so rows 227-348 and every session-12 prose edit were uncommitted and not in git. It was rebuilt from four sources: HEAD (exact, through 11b); a persisted `sed -n '1137,1340p'` capture taken earlier in this session (exact, lines 1137-1340); a verbatim copy of lines 25-235 read earlier in this session; and two separator lines fixed by the document's own invariant, that all 17 `## Ledger` headings are preceded by `---` and a blank. Verified before writing: **1340 lines**, **226384 bytes -- byte-count identical to the original**, and all **60** heading line numbers recorded from the original land exactly where they were. The lesson is the one this file keeps teaching: build the whole artefact, encode it, and only then open the destination. The standing risk it exposes is real and predates the accident -- **the map's last 122 ledger rows have never been committed** |

### What this changes, and what it does not

**Five of six channels are reachable**, up from four: `hunt`, `explore`, `flee`,
`bask` and now `rest`. `groom` remains unreachable and is now honestly zero
rather than propped at 0.05.

**Brain 6 is connected.** It was built in session 9w, consumed by nothing
(#262), and given a flag that made it look otherwise (#347). It now drives the
one thing the sources speak to.

**Nothing was tuned, and nothing was rescued by tuning.** The flee correction
removed an unpublished number and did not make flee release; the prediction that
a hungry gecko does not flee from something standing on it stands, and stands on
better evidence than it did.

**What is still open.** `groom` is blocked on the eye-licking contradiction
(#354). The light-phase tie between `rest` and `bask` (#358) is the world's
doing -- a real leopard gecko rests in a retreat and this one rests on cold
ground -- and sending it to a shelter at dawn would need a source nobody has
opened. The eight `GateContracts` failures are untouched and pre-date this
session; measured again here at **545 tests, 8 failures, 1 skip**.

---

## Ledger — the browser animal, and a skeleton that was not the animal (Session 13b–e)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 360 | A skeleton can be drawn for this animal without opening any osteology | **Refuted -- it was built, it looked the best thing on the page, and it was a lie about the animal** | Built `tools/export_web_skeleton.py`: skull with orbits, mandible, 8 cervical + 17 dorsal + 2 sacral + 28 caudal vertebrae with neural spines and transverse processes, ribs, both girdles, paired radius/ulna and tibia/fibula, three phalanges on each of twenty digits -- 8,943 vertices, 14,196 triangles. Only the LENGTHS were the animal's: each bone was fitted to the real body it belongs to, so the humerus was 11.9 mm because this animal's humerus body is. **Every shape and every count was mine.** No gecko osteology has ever been read into this corpus; the skull outline, the scapula profile and the vertebra form were a generic squamate reading, and the presacral count was chosen to look right -- Powell, Russell & Sutey 2018 would settle it for this species but only its abstract has been reached and it carries no count (a finder claimed one in the session-13 orienting pass and a verifier killed it for exactly that). The user asked the one question that mattered -- *is it based on our 48 solids?* -- and the answer was no. **Deleted.** What the lower panel shows now is the 48 collision solids themselves, rendered as bone: every shape in view is one MuJoCo integrates and one the 14 anatomical gates measure. It is less impressive and it is the animal. The generic version is recorded here rather than kept as an option, because a prettier picture that nothing downstream can check is the appearance form of a plausible number in published clothes |
| 361 | A signal drawn over the body helps you read what the brain is doing | **Partly -- the path is real, the first rendering of it drowned the thing it annotates** | When the selector releases a channel, a band runs snout to tail and the solids it passes are tinted. The PATH is real: selector -> `brain/brainstem.py` -> `brain/spinal_cpg.py` -> legs, head end to tail end, and the position of each solid along the body is recomputed every frame by projecting it onto the trunk's own forward axis so it holds however the animal turns. The TIMING is a drawing and says so, because no conduction velocity has ever been measured in this animal. First cut mixed 35 % of the channel colour into every solid and lit them at 0.85 emissive, which repainted the whole skeleton and hid the shape it was supposed to annotate -- the user's words were that it was too strong. Reduced to 10 % tint and 0.16 emissive over a band 1.5x narrower: a blink, not a wash |
| 362 | The walking was moving the animal forward | **Refuted -- I had pinned the ground to the animal, so it walked on a treadmill** | `RoomView.update` set `floor.position` to the trunk's own x and y every frame, so the sand travelled with the gecko and nothing ever passed it. On screen the legs cycled and the world stood still. The floor now re-centres only in whole texture tiles, which keeps the pattern fixed in world space, and ninety scattered stones give the eye a landmark to measure against. A `walked m` readout counts ground actually covered. Also removed: the four-walled enclosure I had injected into the model in the previous build, which nothing in this project ever asked for and which only caged the animal |

---

## Ledger — the limb had no joints in it (Session 13g)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 363 | The limbs look wrong because the skin is too coarse | **Refuted -- the resolution was fine and the PROFILE was monotonic, which is what makes a banana** | `gecko_sculpt.limb_skins` ran 26 stations through a Chaikin-smoothed path with five control radii -- `1.08, 1.05, 0.96, 0.80, 0.62` times each segment's own radius. Those descend monotonically, so **nothing narrows at an articulation**: there is no elbow, no knee, no wrist and no ankle anywhere in the surface, and a smooth curve through a monotonic taper is a banana by construction. Adding rings could never fix it, which is why #342 (knuckle spheres, refuted on the first render) and #341 attacked the wrong thing. The cross-section was also a perfect circle, and a sprawling lizard's limb is wider than it is deep. Replaced with seven control points -- the two segment MIDPOINTS are new -- carrying a muscle belly between two pinched joints (`1.06, 1.02, 1.22, 0.66, 0.98, 0.61, 0.55`), 34 stations, and every offset's world-vertical component squashed to `LIMB_FLATTEN = 0.74`. Rebuilt through `tools/rebuild_body.py` as the rules require: skin 21,663 -> **22,591 vertices**, 41,048 triangles, and the body still passes **14/14 morphology gates**. Physics proven untouched: the stripped web body regenerated from the new source runs 400 steps under identical controls against the full body at a maximum joint difference of **exactly 0.0**, and the browser port still conforms at 7.327e-15. **THE PROFILE NUMBERS ARE INVENTED AND SAY SO** in the source: no limb cross-section has ever been published for this species or any eublepharid, and the 42-agent surface pass (#337) killed 26 of 36 appearance claims. What is fixed is that a limb now has joints in it; what those joints look like is a shape choice that declares itself |

---

## Ledger — the path was being smoothed straight (Session 13h)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 364 | Putting joint waists in the radius profile (#363) is the whole limb fix | **Partly -- the profile was half of it; the PATH was still being smoothed straight** | After #363 the limb had a waist at the elbow and the wrist and still read as a bent tube. The second cause is `_chaikin(ctrl, 2)`: Chaikin subdivision CUTS CORNERS, and the elbow and the knee ARE corners -- the physics puts them there in `joints[]` and two passes rounded them into a single arc. Adding the two segment midpoints in #363 made it worse, because a longer control polygon is smoothed harder. Reduced to ONE pass, which still removes the hard kink without erasing the articulation. Two further shape corrections in the same pass: the foot was a rounded stump, so its flattening now ramps from 0.74 to 0.48 over the distal 30 % of the limb and its end dome went from three rings at 0.78/0.45/0.06 to four at 0.92/0.70/0.28/0.05 -- a gecko stands on a flat sole, not on the end of a rod; and the proximal radius went 1.06 -> 1.14 with the wrist waist tightened 0.92 -> 0.80, so mass sits where muscle is. Rebuilt through `tools/rebuild_body.py`: skin 22,591 -> **22,707 vertices**, 41,272 triangles, **14/14 morphology gates**, web body against full body over 400 steps under identical controls **max |qpos| exactly 0.0**, browser conformance unchanged at 7.327e-15. **EVERY NUMBER IN THIS PROFILE IS INVENTED AND SAYS SO** -- no limb cross-section, joint diameter or sole geometry has been published for this species or any eublepharid. What is now true is that the limb has corners where the animal has joints; what those corners look like is still a declared shape choice |

---

## Ledger — a negated heading, and a nerve instead of a wash (Session 13i)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 365 | Only two behaviours were reachable in the browser because only two are reachable at all | **Refuted -- a THIRD was reachable and my own arithmetic was steering the animal away from it** | The browser animal alternated `bask` and `rest` forever and never did anything else. The assumption was that this is simply what a gecko with no eye has left. It is not. `this.heading` was read as `atan2(xmat[10], xmat[9])`. MuJoCo's `xmat` is ROW-major, so the body's own forward axis in world coordinates is the first COLUMN -- `(xmat[9], xmat[12], xmat[15])` -- and the first ROW is the world x-axis expressed in body coordinates, which for a yaw is **the negative of the heading**. Every steering command therefore carried the wrong sign. Measured before the fix: released `bask`, computed a bearing of 0 deg to the warm patch, held a heading of 179 deg, and walked **7.9 m in the wrong direction** over 4,000 control steps while its body temperature sat at exactly 25.0 C and never moved. After the fix, same build, same seed: closes 0.346 m -> 0.153 m -> 0.084 m, reaches the patch, warms **25 -> 30 C**, `bask`'s salience collapses and **`explore` releases**. Python was never wrong here -- it takes yaw from the quaternion -- so the conformance suite could not catch it: `base_ctrl` was fed a faithful number computed from a corrupt one. The same wrong slice was also being used to project the solids snout-to-tail in the nerve view; that code fed the deleted band and is now gone rather than fixed |
| 366 | `explore` in the browser means walking in a straight line | **Refuted -- `explore` is not a gait, it is a PROGRAM, and the program stops and looks around** | `brain/programs.py` maps behaviours to motor programs and the browser was ignoring it: any of hunt/explore/bask advanced the same walker clock and the animal ploughed forward with its head locked to centre. Ported the mapper instead of improvising one. `explore` -> **search**, which is `brain/search.py`'s state machine: stop, saccade the head 22 deg at a time out to the neck's travel, fixate, come back, turn the body, walk a little, scan again. `bask` -> **warm**, which walks toward warm ground and then **lies down on it** -- walking once you are on it walks you off the other side, which #333 measured. `rest` and `groom` -> **still**. Ported honestly because the pattern reads only the animal's own trunk angle and nothing about a prey it cannot see. Every one of its ten constants is now read off the live Python class by `tools/export_web_brain.py` rather than typed into JavaScript, so the two cannot drift. `tools/conformance_web.py` gained four checks and runs both state machines side by side for **4,000 steps**: head yaw, heading command, moving flag and **the state string itself** all agree at **exactly 0.0**. Measured in the browser: 5,466 steps scanning, 1,934 turning, 600 travelling, head sweeping the full +-44 deg that `floor(65/22) = 2` saccades predicts, 10 body turns, 3.03 m covered. The head also actually moves now, spread across `neck_yaw` and `head_yaw` in proportion to their ranges -- the same routine `envs/gecko_walk_env.py::_apply_gaze_yaw` uses |
| 367 | A signal band drawn over the solids is the right way to show a decision reaching the legs | **Refuted twice, and the second refutation is the user's: it is not a band, it is a NERVE** | #361 dimmed the band; this deletes it. A motor command does not wash down a body, it travels along a fibre to one joint. Built 27 fibres, one per driven joint, each routed along **`model.body_parentid` -- the animal's own kinematic tree** -- out of the head, up to the lowest body the head and that joint share, then down into the limb, ending at the joint's true `xanchor`. A fibre to a hind ankle passes through the pelvis because the animal's body does; the fibre to the jaw is short because the jaw is next to the brain. Nothing is a straight line through the middle of the gecko. **WHEN an impulse launches is measured, not a timer**: `live.js` accumulates `|delta ctrl|` per actuator at control rate and the view converts it to impulses, so a still joint is **silent** and a hard-driven one crackles. **HOW FAST it travels is a drawing and says so** -- no conduction velocity has ever been measured in *Eublepharis macularius*, and `NERVE_TRAVEL_S` and `IMPULSES_PER_RAD` are labelled INVENTED in the source. First cut was invisible: the fibres run INSIDE the solids and depth testing buried them. Drawn as an overlay instead, which is what an anatomical figure does. Joint markers were also the loudest thing on screen and are now annotation: hub 2.6 mm -> 1.1 mm, axis stub 1.6 mm -> 0.55 mm radius and 26 mm -> 11 mm long |
| 368 | The browser world can leave out the warm patch, because the thermostat is just a number | **Refuted -- with nowhere warm to go, `bask` can never be satisfied and wins forever** | The browser relaxed body temperature toward a hard-coded 25 C with a hard-coded tau of 900, so `cold` never fell, `bask`'s salience never dropped and the animal had exactly one thing it could ever want. Three separate defects in one line: the substrate had no warm half, the time constant matched nothing in Python, and both numbers were typed rather than read. Now `AMBIENT_SUBSTRATE_C`, `THERMAL_TAU_S` and `THERMAL_TIME_COMPRESSION` come off `GeckoBrainEnv` itself, the 30 C surface comes from `warm_surface_temperature_C` (PUBLISHED, this species, Autumn & De Nardo 1995), and the patch is lifted out of the project's own furnished world, `morphology/gecko_habitat_v1.xml` -- **0.14 m across at (0.28, -0.2)** -- so the browser animal stands on the same ground the Python one does. It is a patch of FLOOR and not a lamp because this species is thigmothermic: body temperature tracked substrate at r2 = 0.97 against 0.92 for air, and melanistic pigment made no difference to heating rate (Hastings et al. 2023, n = 12). The bearing to it is handed to the brain the same way Python hands it over and carries the same label there: a stand-in for a spatial memory this project has not built |
| 369 | Playback and camera controls belong in the sidebar, next to the numbers | **Refuted by the user, and they were right: you change the camera while watching the animal** | Pause, orbit, the five camera angles and the two time-of-day jumps now sit on the render viewport itself, over the picture, with a one-line readout of clock, released behaviour and metres walked beside them; the nerve pane carries its own strip counting impulses in flight. The sidebar keeps only what you read rather than press. Also fixed on the way past: the page declared no charset, so every degree sign in it rendered as mojibake, and the skeleton pane framed the animal at 0.70 of the fit radius so the tail ran off the edge |

---

## Ledger — the eye, the cricket, and the edge of my own floor (Session 13j)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 370 | `hunt` cannot be made to release in a browser without handing the brain the cricket's position | **Refuted -- the eye was ported, and the animal finds the cricket in its own pixels or does not** | The shortcut was available and it is the one this ledger exists to refuse: give the selector `prey_visible = 1` and steer at a known point. Instead `brain/retina.py` and `brain/tectum.py` are now `site/eye.js` -- receptors with red dropped, temporal and windowed contrast, centre-surround, the efference copy, the whole-field-event veto, the published 5/45/200 deg/s velocity band and the below-the-horizon switch. A camera rides the animal's snout at the morphology's own `head_cam` offset (29.58 mm forward, 6.82 mm up), aimed by the head's real rotation matrix, rendered to an offscreen 128 px buffer every control step and low-passed to 64 receptors. `tools/conformance_web.py` pushes 140 identical frames through Python and through the port at BOTH resolutions and compares salience, bearing and elevation: **8.882e-16**, twice. One difference was found and copied rather than cleaned up -- Python stores elevation ROUNDED TO 2 DECIMALS in `Tectum.last` while returning salience and bearing raw, which showed up as a 2.633e-03 mismatch until the port rounded too. `FixationEvidence`, `OrientingReflex` and the whole of `MotorPrograms` went with it, so the chase is the real stalk: walk 40 steps, stop and look for 120, and only ever at a bearing the accumulator has already decided is one object rather than noise |
| 371 | The browser animal never gets near the cricket because the eye is too weak | **Refuted -- the eye was fine and the cricket was 65 cm away, where it is smaller than one pixel** | Closest approach over 2000 control steps was **64.8 cm** and the detector's hit rate against ground truth was 8.3 % with a median bearing error of 92 deg -- which is BEHIND the animal. The cause was not the detector. `Prey.reset` in the port placed the cricket in an INVENTED 0.22-0.40 m annulus and then clamped it at twice the arena radius from the animal, so its median range when the eye fired was **0.65 m**, where a 9 mm insect subtends 0.79 deg and cannot occupy a single pixel of a 64-pixel eye. There was nothing else it could have been firing on. Replaced with `envs/prey.py::_respawn` as actually written -- uniform over the arena DISC by area, never inside the flee radius -- and bounded by the arena radius itself rather than twice it. Closest approach 64.8 -> **19.4 cm**, and the first cricket was caught. The invented annulus is the error worth remembering: it was three lines that looked like a detail and it silently made the whole hunt impossible |
| 372 | The eye's false alarms come from unresolvable floor texture aliasing, so supersampling will fix them | **Refuted, measured -- it made the hit rate WORSE, 2.6 % to 0.0 %** | `brain/retina.py` names this failure in its own docstring and prescribes exactly this remedy, so it was the obvious first hypothesis. Raised the eye render from 64 to 128 with the 2x2 box downsample doing the low-pass, and coarsened the sand from 26,000 grains at 1.6 px to 7,000 at 3.4 px and lower contrast. Hit rate against ground truth went **2.6 % -> 0.0 %**. The change is KEPT, because rendering at the receptor resolution omits the optics entirely and that is wrong on its own terms, but the hypothesis it was made under is refuted and stays here. The real source was #373 |
| 373 | Something in the world is generating the false alarms and it cannot be identified without ground truth | **Confirmed, and it was the edge of my own floor** | Instrumented the tectum's peak cell. Over 900 control steps the peak landed on **cell row 8-9 on 212 of 212 firings** -- elevation 0 to -2 deg, the horizon -- while the cricket was in frame on **0 of 900**. The floor was a 6 m plane, so from an eye 3 cm up there is a hard bright-to-dark geometric edge about three metres away, and a straight edge sliding across the frame as the animal turns is the single strongest local motion in the scene. The animal was hunting the edge of my floor. The ground now runs 40 m and fades into the existing fog, so there is no edge to find. Closest approach **19.4 -> 6.3 cm**, commitments 2 -> 10 per 3000 steps, and crickets are now actually caught. Worth stating plainly: this was a defect in the SCENE, and the two hypotheses before it blamed the detector |
| 374 | With the eye in, the browser animal hunts about as well as the Python one | **Refuted, and the number is published here rather than smoothed over** | Measured on the shipping build over 3000 control steps with ground truth used only to SCORE the detector and never fed to it: the cricket was inside the 70 deg field on **215 frames**, the eye fired on **487**, and **18** of those firings happened while the cricket was actually in view. The detector fires at close to its background rate whether or not there is anything there, which is the same failure `brain/tectum.py` documents for the Python version -- 72 % of frames with prey, 72 % with no prey in the world at all -- and the velocity rule cannot help, because that module's own docstring measures it as unreliable below 60 deg/s while a cricket at the working range subtends 5-27. So the hunt works by persistence rather than by acuity: 10 commitments per 3000 steps, closest approach 6.3 cm against a 4.07 cm capture distance, and captures that happen but are **rare** -- 2 over roughly 8,000 driven steps. The site says the firing rate on its own face. Fixing this properly needs a finer retinotopic map or a longer integration window, which is a change to the MODEL and not to this port |
| 375 | The animal rests too much, so the arousal curve should be adjusted | **Refuted by the rules -- the curve is published, so the VIEWER's clock was changed instead** | Measured over 6.3 simulated days: 19,771 control steps resting against 5,721 exploring and 4,492 basking. That is a crepuscular animal on a published arousal curve and it is not going to be edited to make a nicer web page -- that is rule 2, tuning a model until it looks right. What is genuinely a viewing choice is how fast the day runs, so the day/night clock now advances 14x while the animal is asleep, the way a film cuts the night. The timestep, the walker, the thermostat, the eye and the cricket all run at exactly the rate they always did, and the animal sleeps for precisely as many of its own hours. Measured after: rest 589 of 3000 steps with **four channels releasing** -- hunt 1132, bask 744, explore 533. The readout says `clock x14 through the sleep` whenever it is doing it, because a speed-up nobody is told about is a lie about the animal |
| 376 | The basal ganglia can be shown without claiming anything false about where they are | **Yes, by separating the arrangement from the numbers and saying which is which** | Six nodes ride inside the head of the skeleton, one per channel, each in its channel's colour and lit by that channel's REAL gate -- the same converged fixed point the sidebar bars read. The released channel sits open and the other five sit dark, which is what a basal ganglia does: inhibit everything, release one thing. THE RING IS A DIAGRAM and the source says so; six nodes in a circle is a legible picture, not a claim about where a gecko's striatum sits relative to its pallidum. What is not a diagram is every number driving it. In the same pass the nerve view gained the cricket, drawn cold so it reads as a target rather than a second animal, and a ray from the head along the bearing the tectum is reporting -- dim for one unconfirmed report, solid once committed. When the animal is wrong that ray points at nothing, which is the useful case to be able to watch |

---

## Ledger — a declared oracle, and a lizard that bends (Session 13k)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 377 | An oracle can be added to the web render quietly, because it is only a rendering | **No -- it is added, and it is LABELLED in four places, which is the only version of this that is allowed** | The user asked for the page to guide the hunt so that a visitor sees a catch rather than an hour of failure, and that is a fair ask: #374 measured the honest detector firing at its background rate whether or not a cricket was there, which makes the page unwatchable. So guided mode exists. **What it does:** when the cricket is genuinely inside the animal's own 70 deg field and inside the arena, the bearing handed to the evidence accumulator is the true one instead of the tectum's guess. **What it does not do:** it does not see behind the animal, through its own body, or beyond its field; the accumulator still has to reach quorum, the orienting reflex still has its 78-step blind window, the stalk still stops to look, the cricket still flees, and the catch still has to land inside 4.07 cm. **Where it is not:** nowhere in `brain/`. Not one line of the Python model knows it exists, `tools/conformance_web.py` runs against the unguided path, and every number quoted anywhere in this repository was measured with it OFF. It is labelled on the render as a badge, in the sidebar as a switch, in an expandable note that gives the measured unguided numbers, and in the source. A switch one click away turns the real animal back on. An undeclared oracle would be the appearance form of a plausible number in published clothes, which is rule 1; a declared one with its cost printed next to it is a demonstration |
| 378 | A lizard turns its head to look at something, the way a person does | **Refuted by a photograph -- it bends its whole trunk first, and the render had it backwards** | The user put a picture of a tokay next to the render: the animal's pelvis stays planted, the trunk curves, and the neck carries the last of the angle. The render kept the trunk rigid and swung a long neck, which reads as a person. The morphology already had what was needed and nothing was using it -- `spine_bend` is a tendon over `spine_lat_1`, `spine_lat_2` and `spine_lat_3` at coefficients 1.0, 1.0 and 0.8, so a single command bends the trunk hardest at the front, which is the shape in the photograph. A postural layer now splits the commanded heading: **55 % into the trunk, the remainder into the neck, and the neck starts 14 steps after the trunk does**. The tail counter-bends. And the ORDER is enforced: while the trunk is still swinging into place the locomotor drive is held at zero, so the animal stops, bends, looks, and only then walks. **Every number in that layer is INVENTED and the source says so** -- no lateral trunk excursion, bend rate or bend-before-step latency has been published for this species or any eublepharid. The shape is from photographs; the timings are a controller. The walker is untouched: once the animal is moving, the gait owns the spine again, because that is the accepted walker and it is not something to improvise over |
| 379 | The mouth hanging half open was a rendering choice | **Refuted -- it was the morphology's neutral, and nothing had ever commanded the jaw** | `standing_ctrl` for the jaw actuator is 0.3491 rad against a range of 0 to 0.698, which is exactly half open, and no code in this project had ever written to that actuator. So the animal stood with its mouth ajar forever, in every render, in every video. It is now shut at rest, gapes to swallow, and takes an occasional gape while awake. The gape is a named category in the published ethogram for this species; its amplitude, its duration and its interval are not measured anywhere and all three are INVENTED here |
| 380 | The animal cannot close the last 3 cm on a cricket because the prey outruns it | **Partly -- the prey does outrun it, but the measurement was taken from the wrong point on the animal** | Closest approach sat at 7.1-7.8 cm against a 4.07 cm capture distance, run after run. The prey was being advanced against the HEAD BODY'S ORIGIN, which sits inside the skull about 3 cm behind the mouth -- so the animal was effectively three centimetres shorter than it is, and most of its own head lay inside a capture radius it could never reach. Measured from the snout tip instead, the same offset the eye camera uses out of the morphology's `head_cam`, closest approach went **7.8 -> 4.1 cm** and the first cricket went down: hunger 0.90 -> 0.45 in one step. The prey genuinely is faster -- 0.118 m/s against a walk near 0.05 -- and that part stands; what was wrong was where the animal's front end is |
| 381 | Creeping will collapse the cricket's flee radius and close the gap | **Refuted, measured -- 7.1 cm to 7.8 cm, no better and slightly worse** | `flee_radius_for` scales the prey's flight distance with how fast the threat is CLOSING, from 7.5 cm at a charge down to a 1.2 cm floor, and it exists precisely so that a stalk can work. The published behaviour that should exploit it is named in this species' own ethogram -- `walk slow motion`, "walking with a strongly reduced speed, mostly in context of prey capture". So the stride clock was slowed to 0.34 while closing on a committed target. It did not help: the prey's closing-speed estimate is an exponential average with a short time constant, so it recovers fully inside each 40-step move burst of the stalk and the cricket bolts anyway. **The change is KEPT** -- the behaviour is published, the animal visibly creeps, and it costs nothing -- but the hypothesis that it was the blocker is dead, and the blocker was #380 |
| 382 | Hunger only needs to fall when the animal eats | **Refuted -- it ate once and then wanted nothing for the rest of its life** | `explore`'s salience is `0.35 * hunger`, so it needs hunger above **0.571** to clear the release floor at all. One meal took hunger to 0.45 and the animal fell permanently into a world where only `rest` and `bask` could ever release, with a stretch of steps where NOTHING did. Hunger now recovers, and the rate is not invented: hunger in this model is a deficit measured in MEALS, the published mean inter-meal interval for this species is **2.33 days**, and `brain/hypothalamus.py` has the animal burn about 71 % of one meal's energy across it. So the browser reads 0.71 meals per 2.33 days off the registry. What is still INVENTED and still says so is how much of that deficit ONE CRICKET removes -- the project's energetics are not ported, and rather than half-port them the fraction is declared |
| 383 | The eye's view is dark because that is how a gecko sees | **Refuted -- it is darker than the animal, and two of the three reasons are mine** | The user asked the right question about the small black square on the render. Three things make it dark and only one is the gecko. **Mine:** the scene is lit for a night animal against a near-black sky, so the top half of the frame is background rather than anything seen. **The animal's:** the red channel is discarded before the brain sees anything, because this species' longest-wavelength pigment sits near 521 nm -- spectrally green -- so warm sand really does read dimmer to it than to us. **Missing, and it matters in the other direction:** the third opsin peaks near 364 nm and this renderer has no ultraviolet, and there is no low-light gain of any kind -- no tapetum is claimed, because no primary source confirms a gecko tapetum and every confident claim traced to a pet-care site. A nocturnal gecko's eye gathers far more light than this render does. So the square is **darker than the truth, not brighter**, and the page now says exactly that behind a question mark next to it, rather than letting a visitor read a rendering limitation as biology |

---

## Scoreboard

| | |
|---|---|
| Hypotheses tested | **317** |
| Refuted | **258** |
| Confirmed | **34** |
| Partly | **18** |
| Other | **7** |
| Tests passing | **529 of 537**, one skip |

**81 per cent of everything tried was wrong.**

THIS TABLE HAD DRIFTED AGAIN, AND THE LINE CLAIMING OTHERWISE WAS FALSE. It read
221 / 186 / 22 / 12 and "now regenerated by `tools/report_data.py`". That tool
regenerates `artifacts/report/ledger.json`; it does not write this markdown, so
the table stayed hand-kept and fell 96 rows behind. The figures above are counted
from `ledger.json` by verdict. This is #222 recurring in the very table that was
added to close #222, and the false claim is left visible here rather than
quietly deleted.

The test line is likewise measured, not assumed: 8 of the 537 fail, all of them
the `GateContracts` cases #291 documents as unrunnable on any body available
today (accepted base evidence cannot be re-measured because hind duty is 0.64
against a required 0.73-0.83). The previous "537 of 537" was written when that
was true and never revisited.

The old hand-kept figure was 77 %, from a tally that had drifted (#222).

That is what the map is made of.

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
