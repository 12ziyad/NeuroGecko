# NeuroGecko — EYE SPECIFICATION

Written against `C:\Users\ziyad\GeckoBrain`. Every number carries a tag. Where no number exists, the row says NOT_IN_CORPUS, because the gaps are the finding.

---

## 1. THE ANSWER

**Better than the green-pixel counter it has now — yes, and by a wide margin.** The policy does not currently receive an image: `C:\Users\ziyad\GeckoBrain\brain\vision_encoder.py:20` applies `nn.AdaptiveAvgPool2d((1,1))` after the third conv, so the 64×64×3 frame arrives as 128 numbers that are a global average over the whole visual field, with no retinotopic map (no "where", only "how much"). The separate prey signal is worse: `_food_visible_frac` (`envs/gecko_brain_env.py:87-93`) counts pixels where green exceeds red by 30, and the prey in `morphology/gecko_world_v1.xml:48` is brown (rgba 0.55 0.35 0.18) — I ran the mask across illumination gains 0.4 to 3.0 and it returns exactly 0.0 at every one, including saturation.

**Better than a real leopard gecko — no, and the question is partly unanswerable.** On the one axis that can be compared, the camera at fovy 70 / 64 px resolves 0.399 cycles per degree on the optical axis against a receptor-sampling ceiling of 2.0–3.1 cyc/deg derived from another gecko's optics, so there is roughly 4–8× of headroom before the model touches any published ceiling. But no grating acuity, contrast sensitivity function, flicker-fusion frequency or eye optics has ever been measured in *any* eublepharid, so "exceeding the animal" is undefined on most axes — what you *can* exceed is the handful of published behavioural results, and section 5 names which ones break when you do.

---

## 2. WHAT A LEOPARD GECKO'S EYE ACTUALLY IS

Only what survived verification. "Species" is the animal the number was taken from, not the animal we are building.

| Quantity | Value | Species / n | Provenance |
|---|---|---|---|
| Focal length (posterior nodal distance) | 3.5 ± 0.1 mm | *Tarentola chazaliae*, n=3 (NG2/3/6) | **PUBLISHED**, but *model-derived inside the source*: Gullstrand schematic eye with refractive index 1.58 back-solved to force emmetropia. Anyone using it inherits that assumption |
| Pupil diameter, fully dilated | 3.9 ± 0.2 mm | *T. chazaliae*, n=3 (NG6/7/8) | **PUBLISHED** (Roth+ 2009 Table 1) |
| f-number, dark-adapted | f/0.90 | *T. chazaliae* | **DERIVED** — 3.5/3.9, a ratio of two group means from two largely different animals; no error bar can be propagated |
| f-number, second family | f/1.08 (6.5 mm / 6.0 mm) | *Gekko gecko* | **DERIVED** (Citron & Pinto 1973, via Roth Table 1). Use the pair f/0.90–1.08 as a range, never either as a point value |
| Optical sensitivity Sw | 28 µm² sr (human 0.08, tokay 20) | *T. chazaliae* | **PUBLISHED**. The famous "350×" is this ratio — an optical quantity, *not* a ratio of threshold luminances |
| Cone outer segment | 37 ± 5 µm long × 10 ± 2 µm diameter | *T. chazaliae*, one further animal | **PUBLISHED** |
| Receptor-sampling ceiling | 2.04 cyc/deg (15 µm spacing) – 3.06 (10 µm) | derived from the two rows above | **DERIVED**. This is a *sampling* limit, not an optical one; the eye is multifocal with ~15 D between zones, so the true optical cutoff is lower and unmeasured |
| Visual opsin classes | Three: RH2, SWS1, LWS. RH1 (rod) and SWS2 (blue cone) absent | Gekkota; *E. macularius* by BLAST of a **draft** genome, never by eye transcriptome | **PUBLISHED** (gene-level, not expression) |
| λmax | 521 / 467 / 364 nm | *Gekko gecko* | **PUBLISHED**. Across nocturnal geckos the classes span 520–537 / 446–470 / 363–366 nm — use ranges |
| Photoreceptor morphology | Transmuted: rod-*shaped* cells carrying cone opsins. UV pigment in only ~6–20 % of the thin members of type C double **rods** | *Gekko*, *Hemidactylus*, *Teratoscincus* | **PUBLISHED**. "No rods" is true of the gene, false of the cells |
| Colour discrimination confirmed at | 0.002 cd/m² (dim moonlight), blue vs grey, brightness-matched | *T. chazaliae*, n=2 | **PUBLISHED**. A demonstration point, **a floor, not a threshold** — the authors say the animals might do better |
| Retinal specialisation | No fovea; ~1–2 photoreceptors per ganglion cell, uniform, no area centralis | *E. macularius* / nocturnal geckos | **PUBLISHED / likely** |
| Head optokinetic gain, binocular | 0.9 @ 20 °/s; 0.8 @ 30; CW 0.8 / CCW 0.7 @ 40 (direction difference significant only at 40, p=.004) | ***E. macularius*, n=4** | **PUBLISHED** — the only quantitative sensorimotor measurement that exists for this species |
| Head OKR, monocular | Temporo-nasal 0.7 / 0.6 / 0.4 (p=.025). **Naso-temporal: no OKR elicited at any velocity** | ***E. macularius*, n=4** | **PUBLISHED** |
| Smallest visual feature demonstrably tracked | 1.6° random dot | ***E. macularius*, n=4** | **PUBLISHED** — a tracked feature, **not** a measured acuity |
| Light levels the animal was tested at | 21,662 lx and 0.25 lx; gait invariant across both | ***E. macularius*** | **PUBLISHED** |
| Eye diameter / interorbital / snout-to-eye | 6–7 mm / 10 mm / 8 mm | *E. macularius*, n=2 (Rawat+ 2019) | **PUBLISHED** |
| Retinotectal termination | Contralateral tectal layers 8–14; ipsilateral "particularly layers 8 and 9"; 14-layer scheme, 1 deepest | *Gekko gecko*, n=12, silver degeneration | **PUBLISHED**. Ipsilateral component is species-labile across geckos and **unknown in Eublepharidae** |
| Grating acuity, any gecko | — | — | **NOT_IN_CORPUS** — never measured in any gecko species |
| Axial length, PND, pupil diameter, any eublepharid | — | — | **NOT_IN_CORPUS** — no eublepharid eye optics has ever been published |
| λmax, any eublepharid | — | — | **NOT_IN_CORPUS** |
| Critical flicker fusion, *E. macularius* | — | — | **NOT_IN_CORPUS**. The widely-repeated "18–20 Hz" is *Gekko*, *Hemidactylus*, *Tarentola*, all strictly nocturnal, and the corpus contradicts itself about it (`gecko_scorecard.md:115` excludes it; `no_gecko_and_appearance.md:369` asserts it) |
| Contrast sensitivity function, any gecko | — | — | **NOT_IN_CORPUS** |
| Total and binocular visual field, *E. macularius* | — | — | **NOT_IN_CORPUS**. Nearest: ~30° binocular in *Lygodactylus*, an unpublished observation quoted inside another paper's introduction — a diurnal arboreal genus |
| Retinal ganglion cell density map | — | — | **NOT_IN_CORPUS** for any gecko |
| Colour-vision *threshold* | — | — | **NOT_IN_CORPUS** for any gecko |
| Tapetum lucidum | — | — | **NOT_IN_CORPUS** — unverifiable from any primary source; every confident claim found was a pet-care site. Do not build one |
| Eye-vs-head share of gaze stabilisation | "~80 % head" is in the corpus but attributed to the wrong paper (Dieringer 1983, not Masseck 2008) | uncertain | **uncertain** — do not use as a tuning target until reread |

---

## 3. THE GAP — our camera against the animal

Two configurations exist and they are not the same eye. `envs/gecko_walk_env.py:34` sets `DEFAULT_XML = gecko_body_r.xml` (fovy 120), and **`gecko_world_v1.xml` (fovy 70, textured floor, real brown prey) is loaded by `tests/test_world.py` only** — no training or evidence run has ever used it.

| Quantity | The animal | Default build (fovy 120, 64 px) | World v1 (fovy 70, 64 px) | Verdict |
|---|---|---|---|---|
| Angular sampling, on axis | receptor spacing ⇒ ~0.33–0.49 °/receptor | 3.101 °/px | 1.254 °/px | 2.5–9× coarser |
| Nyquist limit, on axis | 2.04–3.06 cyc/deg (sampling ceiling) | 0.161 cyc/deg | 0.399 cyc/deg | **5–19× coarser** |
| Sampling uniformity | uniform, afoveate (published) | corner samples **4.00× finer** than centre | corner **1.49× finer** | **Wrong shape.** A rectilinear render is uniform in *pixels*, not in *angle* — it gives an **inverse fovea**, worst where the animal aims. The senses table's "uniform 64×64 — already correct" is wrong |
| The 1.6° tracked dot | tracked, n=4 | **0.516 px — not resolvable** | 1.276 px — resolvable | The default build fails the only published spatial capability of the target species |
| Field of view | NOT_IN_CORPUS (nearest ~30° binocular, wrong genus) | 120° | 70° | **Both INVENTED.** `gecko_scorecard.md:114` says outright that fovy 120 "has no biological basis". No published number can adjudicate |
| Eyes | two, laterally placed, converged forward | one midline camera, no stereo | same | INVENTED. Model eye spheres are cosmetic (mass 0, no collision) and sit 20.24 mm apart in the default body vs 16.6 mm in world v1, against a published 10 mm interorbital gap |
| Spatial information reaching the policy | retinotopic map ⇒ tectum | **1×1** — global average pool | **1×1** | The single most important fact. There is no map to build a tectum on. The only azimuth cue today is a zero-padding artefact (measured: 38–88× horizontal anisotropy in trained checkpoints, 0.4–1.6× at init) |
| Channels | UV / mid / long; **no red receptor** | human sRGB | same | Gives the policy an axis the animal has no photoreceptor for |
| Update rate | NOT_IN_CORPUS | 50 Hz (1 render / 25 substeps × 0.0008 s) | same | Cannot be judged. Do not decimate on a species-smuggled CFF |
| Sensitivity / adaptation | pupil range 100–150×, transition ~1 h | fixed exposure | fixed exposure | Adaptation time constant ≫ episode length, so fixed exposure is defensible; set once per episode from light level |
| Gaze stabilisation | head OKR gain 0.9 @ 20 °/s, NT gain zero | absent | absent | The largest missing published behaviour |

---

## 4. BUILD PLAN — BRAIN 4 (retina + tectum front end)

Numbered. Each step names its file, its acceptance test, and the published result the test compares against. Steps with no published counterpart say so instead of being given an invented one.

**4.1 — Freeze the camera geometry and make it a parameter.**
Build: expose `fovy` and `camera_width/height` as env parameters; point `DEFAULT_XML` at the world that has prey and texture; regenerate `gecko_world_v1.xml` with a corrected header note (its current note carries the refuted linear arithmetic — "0.85 px / 1.46 px" — which is 65 % and 14 % optimistic).
Files: `C:\Users\ziyad\GeckoBrain\envs\gecko_walk_env.py:34`, `C:\Users\ziyad\GeckoBrain\envs\gecko_brain_env.py:109-110`, `C:\Users\ziyad\GeckoBrain\utils\build_world.py:220`, `C:\Users\ziyad\GeckoBrain\morphology\gecko_world_v1.xml:8,89`.
Acceptance test: render a sphere at known off-axis angles in MuJoCo and recover on-axis sampling = 1.2537 ± 0.01 °/px at fovy 70 / 64 px; then require the **1.6° dot to span ≥ 1 px on the optical axis**.
Published result compared against: Masseck, Röll & Hoffmann 2008 — 1.6° random dot tracked, *E. macularius*, n=4. (The 1 px criterion itself is INVENTED; the 1.6° is not.)

**4.2 — Raise resolution to 128×128 at fovy 70.**
Build: 128 px input; retrain is forced, so bundle it with any other input-shape change.
Files: `envs\gecko_brain_env.py:109-110,225,412-421`, `brain\vision_encoder.py`, `brain\bc_actor.py:23-30` (hard-codes 64,64,3), the env's error string. `offheight="480"` in all five XMLs is sufficient for 128 and needs no change.
Acceptance test: **none published.** No grating acuity has ever been measured in any gecko, so no test can say 128 is right. What exists is a two-sided *constraint*: 128 at fovy 70 gives 0.798 cyc/deg, which stays below the *Podarcis muralis* figures (1.56 RGC-density, 2.05 optomotor) and below the 2.0–3.1 receptor ceiling. 256 at fovy 70 gives 1.595 cyc/deg and **breaches the *Podarcis* RGC figure** — that is the cap, and it is a lizard proxy, not a gecko measurement.
Published result compared against: the constraint only, not the value.

**4.3 — Give the encoder a retinotopic output.**
Build: delete `nn.AdaptiveAvgPool2d((1,1))` and emit an H×W feature map; the tectum consumes the map, the policy consumes a pooled summary plus a decoded bearing.
File: `C:\Users\ziyad\GeckoBrain\brain\vision_encoder.py:20`.
Acceptance test: a linear probe on the encoder output must recover prey azimuth sign and magnitude above a stated baseline, and must **not** recover it from the padding artefact alone (control: uniform-field stimulus with the target at mirrored positions).
Published result compared against: **none.** This is an engineering gate. Its biological warrant is qualitative — the retina projects topographically to tectal layers 8–14 (*Gekko gecko*, n=12) — not quantitative.

**4.4 — Remap the channels to the animal's receptor set.**
Build: drop the red channel; feed two channels mapped R→520–533 nm, G→452–470 nm; leave UV unmodelled and say so. Prey must be discriminable on the 520-vs-467 nm axis, not on human-green.
Files: `envs\gecko_brain_env.py` (render post-process), `brain\vision_encoder.py` (3→2 in-channels), `morphology\gecko_world_v1.xml:48`.
Acceptance test: **none.** MuJoCo's sRGB primaries are not gecko cone fundamentals; this is a mapping, not a match, and no behavioural colour result exists for *E. macularius*. Do not use 0.002 cd/m² as a threshold — it is a demonstration floor in *Tarentola*, n=2.
Published result compared against: only the gene complement — three classes, RH1 and SWS2 absent (Pinto+ 2019, *E. macularius* by draft-genome BLAST). That justifies *removing* the red channel; it validates nothing else.

**4.5 — Replace the prey-visibility signal.**
Build: `_food_visible_frac` currently gates brain 2's selector through a green mask that is identically zero for the brown prey. Replace with a small-target motion / contrast detector over the retinotopic map.
File: `C:\Users\ziyad\GeckoBrain\envs\gecko_brain_env.py:87-93,603-612`.
Acceptance test: with the prey in frame at 0.30 m the detector returns non-zero, and with it absent returns zero. Prey subtends 3.44° at 0.30 m = 2.74 px at fovy 70/64 and 5.48 px at 128 — so the detector must work at ~3 px.
Published result compared against: **none for the detector.** Prey radius 0.009 m is itself INVENTED (a sphere proxy for published 0.30–0.40 SVL prey). The one real anchor is the chemosensory range — published bioassays presented stimuli at 3 mm–1 cm from the snout — which says vision, not odour, must carry the approach.

**4.6 — Optic flow: keep the textured floor, verify it in the 64/128 px buffer.**
Build: floor `texrepeat 50` (2 cm squares) already exists in world v1 and is INVENTED; verify against aliasing by dumping actual policy-resolution renders, not the viewer.
Files: `morphology\gecko_world_v1.xml`, `utils\build_world.py`.
Acceptance test — **this is the strongest gate in the plan:** dropping illumination from 21,662 lx to 0.25 lx must leave speed, stride frequency and duty factor unchanged.
Published result compared against: Higham & Schmitz 2019 — the three nocturnal species including *E. macularius* maintained sprint performance at both light levels. If the trained policy slows in the dark, locomotion has become vision-dependent, which this animal is not.

**4.7 — Pretectal OKR loop driving neck and head yaw.**
Build: hemifield optic flow, **half-wave rectified per hemifield** so backward (naso-temporal) flow produces zero head-turn command; output to `neck_yaw` + `head_yaw`.
Files: new `brain\pretectum.py`; `envs\gecko_brain_env.py` action routing.
Acceptance test: in a simulated optokinetic drum, head angular velocity ÷ stimulus velocity = **0.9 at 20 °/s, 0.8 at 30 °/s, 0.7–0.8 at 40 °/s** binocular; monocular temporo-nasal 0.7 / 0.6 / 0.4; **monocular naso-temporal gain = 0 at all three velocities**. Test only at 20/30/40 °/s; refuse to extrapolate. Accept a 0.7–0.9 band rather than one-decimal point values — the source measured at 50 Hz video, 0.4°/frame at 20 °/s.
Published result compared against: Masseck, Röll & Hoffmann 2008, *Vision Research* 48:765–772, ***E. macularius*, n=4** — the target species itself. The NT-zero row is the discriminating half: it is what a symmetric flow estimator cannot produce.

**4.8 — Leave the render rate at 50 Hz.**
Build: no change.
Acceptance test: **none, deliberately.** No CFF has been measured in *E. macularius* from a readable primary source; the "18–20 Hz" figure is from three strictly nocturnal species and the corpus contradicts itself about it. CFF is a fusion threshold, not a frame rate — a Nyquist argument wants roughly *twice* the CFF, so decimating is plausibly backwards.

**4.9 — Register every INVENTED value.**
Build: `fovy` (70 and 120), 64/128 px, camera pose, prey radius, texture scale, detector thresholds all enter `C:\Users\ziyad\GeckoBrain\config\proxies.yaml` tagged INVENTED with species n/a. None of them is there today.
Acceptance test: a registry-completeness test — every constant read by the vision path resolves to a `proxies.yaml` entry.
Published result compared against: none; this is rule 1.

---

## 5. WHERE EXCEEDING THE ANIMAL WOULD BREAK SOMETHING

Named published results that stop reproducing.

1. **Making vision good enough that locomotion leans on it** breaks Higham & Schmitz 2019 (*E. macularius*, verified): speed, stride frequency and duty factor are invariant between 21,662 lx and 0.25 lx. A policy that walks better in the light has failed a published null result. Locomotion must ride on IMU, touch and the CPG.
2. **A symmetric, un-rectified optic-flow front end** breaks Masseck+ 2008: monocular naso-temporal stimulation elicited *no* optokinetic response at any velocity in *E. macularius*. Any full-field flow estimator predicts a non-zero NT gain. It also runs away in practice — forward walking drives the reflex.
3. **An OKR loop with gain ≈ 1.0** breaks the same paper the other way: the published binocular head gain is 0.9 at 20 °/s and *falls* to 0.7–0.8 by 40 °/s. Perfect stabilisation is over-reproduction, not success.
4. **Adding a fovea, centre-weighting, or a log-polar transform** breaks "strictly nocturnal geckos completely lack foveae" and the ~1–2 photoreceptors per ganglion cell, uniform across the retina. Note the current camera already violates this — in the *opposite* direction, an inverse fovea, 4.00× at fovy 120.
5. **Resolution above ~1.5 cyc/deg** (256×256 at fovy 70 = 1.595) puts a crepuscular afoveate gecko above the *diurnal* wall lizard *Podarcis muralis* at 1.56 cyc/deg RGC density — the wrong sign. Above 2.0–3.1 cyc/deg the eye's own receptors cannot sample it at all, and the real optical cutoff is lower still because the eye is multifocal with positive spherical aberration.
6. **Keeping the red channel** hands the policy an axis with no receptor behind it: the longest gecko pigment is 521 nm (called LWS, spectrally green), and SWS2 is gone. A prey-detection strategy that separates prey from floor on red-vs-green is solving a problem this animal cannot solve, and it will not transfer to any biologically-shaped front end.
7. **Modelling a tapetum or a low-light gain hack** rests on nothing: no primary source confirms a gecko tapetum, and Roth+ 2009 explain the whole 350× optical-sensitivity advantage with pupil, focal length and outer-segment size, invoking none.
8. **Treating 0.002 cd/m² as a ceiling** understates the animal by roughly 35×: the optics predict ~350× the human's sensitivity while the two behavioural luminances differ by only 10×, so the tested value sits well above whatever the real threshold is.

---

## 6. NEW LEDGER ENTRIES

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 112 | The camera is "uniform 64×64, no foveation — already correct" (senses table) | **Refuted** | a rectilinear render is uniform in pixels, not in angle: the corner samples **1.49× finer** than the axis at fovy 70 and **4.00× finer** at fovy 120. The build has an **inverse fovea** — coarsest exactly where the animal aims |
| 113 | The policy sees a 64×64 image | **Refuted** | `brain/vision_encoder.py:20` global-average-pools to (64,1,1) before the Linear. Spatial resolution reaching the policy is **1×1**; the 128 outputs are channel means. Confirmed independently by the recovered checkpoints, whose `extractor.vision.net.8.weight` is (128,64) |
| 114 | The default environment renders the fovy-70 world | **Refuted** | `envs/gecko_walk_env.py:34` `DEFAULT_XML = gecko_body_r.xml` (fovy 120); `gecko_world_v1.xml` appears in no non-test code path. Every archived evidence run used a fovy-120 body |
| 115 | The prey-visibility signal works with the brown prey | **Refuted** | `_food_visible_frac` requires g > r + 30; prey material is rgba 0.55 0.35 0.18, so r > g at every gain. **Measured 0.0 at gains 0.4, 0.7, 1.0, 1.4, 2.0, 3.0**, saturation included. The signal feeds brain 2's selector. Latent only because nothing outside `tests/test_world.py` loads that world |
| 116 | "Anything above ~256 is biologically unjustifiable" (`gecko_scorecard.md:113`) | **Partly** | right as a cap, wrong in its reason. 256 at fovy 70 = **1.595 cyc/deg**, still *below* the 2.0–3.1 receptor ceiling but *above* the *Podarcis muralis* 1.56 cyc/deg the same row says the gecko must sit under. The binding constraint is the lizard proxy, not the optics |
| 117 | A leopard gecko's eye optics are published somewhere | **Refuted** | no eublepharid PND, pupil diameter, axial length, acuity, CFF, contrast sensitivity or λmax exists in the corpus or on the web. Every optical number in this spec is *Tarentola* or *Gekko* |
| 118 | f/0.90 is a leopard gecko f-number | **Refuted** | *Tarentola chazaliae*: 3.5 mm from a Gullstrand model with n=1.58 back-solved to force emmetropia, 3.9 mm from a **different** three animals. Bracketed by *Gekko gecko* at f/1.08. Both bracketing species are **strictly nocturnal**; *E. macularius* is crepuscular, so f/0.90 is a brightest-case **lower bound** that over-estimates light gathering |
| 119 | 128×128 at fovy 70 stays under every published lizard ceiling | **Confirmed** | 0.798 cyc/deg on axis, vs *Podarcis* 1.56 (RGC) / 2.05 (optomotor) and a 2.04–3.06 receptor ceiling. The 1.6° dot spans 2.55 px |
| 120 | Ledger #28's evidence numbers are right | **Refuted** | the 1.6° dot spans **0.516 px** at fovy 120 (recorded 0.85) and **1.276 px** at fovy 70 (recorded 1.46). Verdict survives; both numbers in its evidence line are the linear-average error, 65 % and 14 % optimistic. Same error at `no_gecko_and_appearance.md:354,358` and `gecko_scorecard.md:113` |
| 121 | The render should be decimated to the gecko's 18–20 Hz flicker fusion | **Refuted** | every value behind 18–20 Hz is *Gekko*, *Hemidactylus* or *Tarentola*, all strictly nocturnal; `gecko_scorecard.md:115` excludes the figure as untraceable while `no_gecko_and_appearance.md:369` asserts it. CFF is a fusion threshold, not a frame rate — Nyquist wants ~2× CFF, so the argument is plausibly backwards. **NOT_IN_CORPUS for the target species** |
| 122 | "The gecko can be given a better eye than it has" | **Confirmed** | four independent axes have headroom before any published ceiling: 1×1 → retinotopic; 0.399 → 0.798 cyc/deg; three human channels → two gecko channels; absent → published OKR gains |
| 123 | "The gecko can be given a better eye than a leopard gecko" | **Partly** | on resolution the answer is yes and the ceiling is known within a factor of two. On acuity, contrast sensitivity, CFF, field of view and receptor spacing there is **no number to exceed** — the comparison is undefined, and exceeding the *behavioural* results breaks four of them (§5) |

---

## 7. WHAT IS STILL UNKNOWN

- **No gecko, of any species, has a published grating acuity.** Every resolution argument in this project is a proxy chain ending in a wall lizard or in cone spacing measured in a different family.
- **No eublepharid eye has ever been optically characterised** — no focal length, no pupil, no axial length. Every optics number here crosses a family boundary from a spectacled, eyelid-less, strictly nocturnal animal to an eyelidded crepuscular one, and the direction of bias (over-estimating light gathering) is known but not quantified.
- **Field of view is the weakest number in the build.** Both 70° and 120° are INVENTED. The only gecko figure available is ~30° binocular in *Lygodactylus*, an unpublished observation inside another paper's introduction, in a diurnal arboreal genus.
- **Flicker fusion in *E. macularius* is unmeasured**, and the corpus contradicts itself about the proxy value. 50 Hz stands because nothing published can move it.
- **Whether the gecko's eye moves independently of its head, and by how much**, is unresolved: the "~80 % of stabilisation is head movement" figure in the corpus is attributed to the wrong paper and has not been reread.
- **Whether geckos have a tapetum lucidum** could not be established from any primary source.
- **Whether the ipsilateral retinotectal projection exists in Eublepharidae** is unknown; it is present in one gekkonid and absent in four other lizards, and the literature calls it "unstable and mutable".
- **Abramjan+ 2020** (*Behavioural Processes* 173:104060) is the only published visual model of *E. macularius* colour vision and remains unopened; one secondary summary suggests a **four**-class receptor set, which would contradict the three-class gene complement. That is the single most valuable open check on section 2.
- Uncertain: whether a retinotopic front end trained with the privileged food vector still switched on will learn anything visual at all. Until `use_privileged=False` and the ground-truth `mouth_food_dist` reward are removed, every improvement in this specification is decoration.