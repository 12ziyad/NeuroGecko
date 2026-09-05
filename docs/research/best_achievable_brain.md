# NeuroGecko: is this the best brain in the world, and what is the best we can do?

*Lead computational neuroscientist's answer to the owner. Plain language; exact where it matters. Every claim carries a confidence tag: **[verified]** = read from the primary source or measured this session, **[likely]** = from a secondary source or a single abstract, **[uncertain]** = inference or unresolved discrepancy. Repository facts refer to `C:\Users\ziyad\GeckoBrain` (github.com/12ziyad/NeuroGecko, Apache-2.0, HEAD ddbb105) as read on 2026-09-05 (read-only).*

---

## The short answer

1. **No, the current brain is not top-level, and neither is the planned one as drawn.** Today's brain (`brain/vision_encoder.py` CNN -> `brain/actor_critic.py` -> 4-D command `[dir_x, dir_y, distance, engage]` consumed by `envs/gecko_brain_env.py::_brain_action_to_target`, plus the hand-invented `brain/drives.py::DriveState` ODE with hunger/energy/fear/curiosity/danger/target_interest, plus a hand-coded behaviour arbiter) is rung R0 on the ladder below. **[verified, repo]** The planned region-structured brain is rung R1 and can honestly be pushed to "R2-by-proxy". **[verified, survey synthesis]**
2. **"Function-shaped" is not the problem.** The world's top whole-animal models that actually behave in closed loop (NeuroMechFly v2, simZFish, the virtual rodent, even Eon's "whole-brain" fly) all bottom out in function-shaped modules and hand-mapped descending interfaces. What separates a top-venue brain from a hobby brain is not neuron count or spikes; it is (a) wiring that cites measured anatomy and (b) every module reproducing a named measurement. **[verified]**
3. **The ceiling for this animal is fixed by data that does not exist, not by compute.** There is no gecko connectome, no gecko neural recordings beyond 1983-85 nucleus-isthmi units in *Gekko gecko*, no gecko sleep EEG, no gecko tectal map, no lizard spinal-circuit data. Rungs R3 and above are physically closed. **[verified]**
4. **What is reachable is genuinely new, if the claims are worded carefully.** No reptile with a neural controller in the loop appears in the field's 2026 review of embodied brain models (reptile *musculoskeletal* simulations do exist, without brains); nobody has a data-constrained tectum hunting inside a whole-body model; and while several small agents already sleep from a linear "rest counter", none sleeps because of a Borbély-form two-process homeostat coupled to a lizard-derived ultradian rhythm. A region-structured gecko whose every connection cites a lizard tract-tracing paper, validated by a pre-registered behaviour + lesion + dynamics battery, would be the best whole-animal brain ever built for a vertebrate without a connectome. **[verified]**
5. **Cost:** roughly 8-12 weeks of one developer on the CPU laptop, zero dollars mandatory; the $500 AWS is best kept for PPO retrains with the camera, or for a MIMIC-MJX run if gecko video ever appears. **[likely]**

---

## 1. The honest ladder

Rungs are defined by **what data constrain the brain and what it is validated against**, not by how "neural" it looks. That is the criterion the field itself uses (Wang-Chen & Ramdya 2026 review, arXiv 2601.08056, v4 Jul 2026, journal version Curr Opin Neurobiol 100:103244; Beiran & Litwin-Kumar 2025, Nat Neurosci 28:2561). **[verified]**

| Rung | Definition | Who is there | Data that constrain it | Validation |
|---|---|---|---|---|
| **R0** | Hand-coded behaviour tree + hand-invented drive ODE + a learned black box | **NeuroGecko today** (`drives.py`, if/else arbiter, CNN -> 4-D command) | Body measurements only (the 276-item scorecard constrains the body, not the brain) | Task success |
| **R1** | Region-structured, neuro-inspired architecture with hand-set parameters, validated qualitatively | Prescott 2006 robot basal ganglia; NeuroMechFly v2 CPG + reflex layers (Nat Methods 2024); **NeuroGecko as planned** | Anatomy at the level of "these regions, these connections" | Behavioural switching, gait, qualitative phenomena |
| **R2** | Region-structured circuit whose modules are quantitatively fitted to a species' physiology/behaviour and checked against recordings; makes confirmed predictions | simZFish (Science Robotics 2025: retina -> DS cells -> pretectum -> hindbrain -> bout gate, fitted to Naumann-lab imaging, predicted lower-posterior visual field suffices, confirmed); Vishwanathan 2024 zebrafish brainstem wiring-diagram model | Imaging of the same species | Physiology + behaviour + a novel confirmed prediction |
| **R3** | Task/imitation-trained network on a measurement-based body whose *internal activity* predicts recordings of the same animal | Virtual rodent / MIMIC (Aldarondo et al., Nature 2024: 842 mocap clips, 7 rats; hidden units predict 732 DLS and 769 motor-cortex units better than kinematics); CMU intrinsic-goal zebrafish (arXiv 2506.00138: matches ~250k-cell whole-brain data at inter-animal reliability) | Behaviour for training; recordings for validation | Neural predictivity vs noise ceiling |
| **R4** | Connectome-constrained subsystem with fitted physiology | flyvis (Lappalainen et al., Nature 2024: 45,669 non-spiking units, 1.51 M connections, 734 free parameters, matched 26 studies) | EM connectome + task | Tuning curves across cell types |
| **R5** | Whole-brain connectome spiking model, open loop | Shiu et al., Nature 634:210 (2024): all 127,400 proofread neurons of FlyWire v630, LIF, one free parameter, >90% MN9 prediction accuracy; the 139,255 figure quoted by Berkeley press and Eon is the v783 connectome count (Dorkenwald 2024), one config change away in the MIT repo but not the published result; "50 M synapses" is the connectome's count, not a stated model count | Connectome + transmitter signs | Optogenetic/imaging confirmation |
| **R6** | Whole-brain connectome model embodied in closed loop | Eon Systems (Mar 2026: Shiu's LIF brain + NeuroMechFly body + flyvis front end + imitation-learned walking controllers; Eon itself calls it "an integration effort", not proof that structure suffices) and Tsinghua FlyGM (arXiv 2602.17997) | Connectome as architecture; physiology learned or unfitted | Behaviour only; **neural fidelity unproven** (the UW "digital sphinx" preprint, Brunton & Abe 2026: a *worm* connectome driving a fly body via RL also walks) |
| **R7** | Neuromechanical digital twin of a *particular* animal, continuously updated from its measurements | Nobody (definition from Wang-Chen & Ramdya 2026) | Everything, continuously | Everything |

Off the ladder: MICrONS/Wang 2025 mouse "digital twin" (an encoding model, no behaviour) and the Allen/Fugaku 10-million-neuron cortex (resting state, no body). Do not let "mouse digital twin" headlines set your expectations; neither is embodied. **[verified]**

**Where the gecko can go.** R1 -> R2-by-proxy: every module fitted to published physiology of the nearest species (zebrafish tectum, rat/gecko basal ganglia anatomy, salamander spinal CPG, Pogona sleep CPG, mammalian hypothalamus), and the whole animal validated against the gecko's own behaviour scorecard plus a lesion and dynamics battery. That is the architectural class of simZFish and NeuroMechFly v2, both top-venue papers. **[verified]** A "R3-lite" for the *locomotion layer only* opens if real gecko video ever exists: ~50 pose clips through MIMIC-MJX/track-mjx on one cloud GPU (BSD-3, converges under 1e9 steps on one A40) converts the CPG into a data-constrained controller. **[verified; cost estimate likely]**

**What the gecko can never reach, and should never claim.** R3 proper (internal activity validated against gecko recordings), R4-R6 (connectome), R7 (a specific measured individual). Beiran & Litwin-Kumar 2025 is the decisive theory: a connectome "often does not substantially constrain the dynamics" of a recurrent network; recordings from a subset of neurons *can* remove the degeneracy, with the number needed scaling with the dimensionality of the dynamics (random subsets suffice once they exceed it; theory-guided choice is an efficiency gain, and for high-dimensional chaotic networks the count grows with network size). The gecko will never have either a connectome or recordings, so any claim of neural fidelity is over-claiming. **[verified]**

---

## 2. What a gecko has that a made-up animal does not

Reptile neuroscience constrains the *architecture* of this brain far better than its *parameters*. That is still a real advantage: a made-up animal has neither. Below, the facts sorted by how species-correct they are, and what each one sets in the model.

### 2a. Species- or genus-correct (Eublepharis, Gekko, Paroedura)

| Fact | Source | Sets in model | Confidence |
|---|---|---|---|
| In geckos, **explicitly Gekko AND Eublepharis** (two families: Gekkonidae and Eublepharidae), the only basal-ganglia route to the tectum is via substantia nigra; the enkephalinergic pretectal relay of turtles/crocodiles/lacertids is "apparently" absent. This 1991 paper is the sole primary source and has not been replicated in 35 years; the abstract does not say whether Eublepharis received tracer injections or only enkephalin immunostaining | Medina & Smeets 1991, J Comp Neurol 308:614, PMID 1865018 | One BG->tectum gate (SNr), no separate pretectal BG channel; flag as resting on one paper | verified |
| *Gekko gecko* striatum afferents: dorsal cortex, DVR (three topographic zones), lateral amygdala, GP, VTA/SN, dorsal thalamus incl. **specific sensory nuclei topographically** and an intralaminar-like dorsomedial nucleus; accumbens afferents: cortex, diagonal band, VP, lateral preoptic, VTA, thalamus | Gonzalez et al. 1990, PMID 2257479 | Input channels of the dorsal and ventral BG loops | verified |
| *Gekko* striatum efferents: GP, entopeduncular nuclei, SNr; GP -> ventral thalamus; accumbens -> VP, lateral hypothalamus, VTA; VP -> ventromedial thalamus, hypothalamus, habenula, tegmentum | Russchen & Jonker 1988, PMID 3192764 | Output wiring of both loops; ventral loop reaches hypothalamus and VTA (motivation) | verified |
| *Gekko* dopamine: cells in VTA, SN, A8 and hypothalamus; **accumbens densest**, then striatum, amygdala, olfactory tubercle, septum, **DVR** (dense in geckos, weak in turtles/crocodiles); cortex sparse | Smeets 1986, PMID 3540035 | DA gain strongest on ventral loop; DA gain on the DVR object channel; none on dorsal cortex | verified |
| D2 activation does **not** inhibit ACh release/cAMP in *Gekko* striatum, unlike rat | Henselmans 1991-94, PMID 7916354 | Do not copy rat D2-cholinergic interactions; keep D2 as a simple gain | verified |
| *Gekko* tectum <-> nucleus isthmi pars magnocellularis reciprocal loop (direct and via n. profundus mesencephali); Imc units respond only to moving contrast targets, not touch/sound; topographic, binocular, 35% burst at RF entry/exit | Wang et al. 1983 (PMID 6194859), Wang et al. 1985 (PMID 3983614) | Tectum-Imc winner-take-all; the **only gecko midbrain electrophysiology that exists** | verified |
| *Gekko* retina -> contralateral tectal layers 8-14 (ipsilateral 8-9), plus LGN, pretectum, basal optic nucleus | Butler & Northcutt 1971 | Retinorecipient superficial tectal layer; separate pretectal/AOS motion channel | likely |
| Gecko *Paroedura picta*: lemnothalamic pathway -> dorsal cortex; collothalamic (tectofugal) pathway -> anterior DVR; DVR circuits convergent with, not homologous to, mammalian cortex | Rueda-Alaña et al., Science 2025 | Two pallial channels: LGN->dorsal cortex (context), tectum->rotundus->DVR (object) | verified |
| *Gekko* medial cortex: small-celled (Cxms) and large-celled (Cxml) parts forming a two-stage loop with septum and dorsal cortex | Hoogland & Vermeulen-VanderZee 1993, PMID 8514912 | Two-stage place-memory module | verified |
| **Leopard gecko** medial cortex = hippocampal homologue, two neuron types, adult neurogenesis: neuroblasts migrate ~30 days, survive >=140 days | McDonald & Vickaryous 2018, PMC6018638 | Slow structural plasticity timescale (weeks) for the memory module | verified |
| **Leopard gecko** Morris water maze: 56 naive animals started, 42 completed 20 weekly trials (~20 weeks; 14 excluded for passive floating, not for failing); learning shown at group level (path length and latency, trials 1-3 vs 18-20); no decline at a 2-month probe (3 trials; but not significantly above the naive start either); at 6 months (28 animals) performance back at naive baseline, relearned within 30 retraining trials while swimming slower (motivation confound). Use of both proximal and distal cues, individually variable, comes from the team's earlier study, not this one | Landová et al. 2025, Animals 15:2014, PMC12291964 (cue use: Landová et al. 2023, Acta Soc Zool Bohem 86:97-117) | Acquisition, retention and re-acquisition targets for the memory module | verified (2023 cue paper: likely) |
| **Leopard gecko** head-OKR (stabilising head movements): binocular gain symmetric; monocular response only to temporo-nasal motion. Shared by all afoveate geckos tested and by foveate *Lygodactylus*; only *Phelsuma* differs, so this is species-tested, not species-unique | Masseck, Röll & Hoffmann 2008, Vision Res 48:765 | Crossed pretectal/AOS wiring (an inference from the vertebrate literature, not a result of this paper); pass/fail test on head movement | verified |
| **Leopard gecko** EMG after autotomy of a tail that is ~25% of body mass (N=10, slow voluntary walking): caudofemoralis burst *amplitude* falls (peak 53.7->21.8 %max; stance RIA 74.9->30.9); gastrocnemius stance burst amplitude falls (38.5->15.4) while its late-swing burst is unchanged; burst timing, duration and shape unchanged for all muscles; puboischiotibialis unchanged despite reduced femur depression and knee angle; biceps and triceps brachii unchanged | Jagnandan & Higham 2018, J Exp Biol 221:jeb179564 | Pattern-formation layer amplitude targets; tail-autotomy perturbation prediction | verified |
| Across Gekkota brain volume scales with negative allometry (small geckos have relatively large brains); across 32 lizards most volumetric variation is in tectum and cerebellum | Lagorio et al. 2026; Platel 1976 | Give tectum/brainstem the largest compute share; pallium small and coarse | verified |

### 2b. Reptile proxies (Pogona, turtle, iguana, other lizards)

| Fact | Source | Sets in model | Confidence |
|---|---|---|---|
| Pogona SWS/REM alternate with a ~80 s period (abstract), 6-10 h/night, up to 350 cycles, ~50/50 duty; recorded at 27 °C (the same lab's 2026 paper states ~80-90 s at 27 °C); period shortens with temperature | Shein-Idelson et al., Science 2016; Albeck & Shein-Idelson 2026, Commun Biol | Ultradian CPG period anchor: 27 °C -> ~80-90 s (Pogona default, not a gecko claim) | verified (27 °C attribution: likely) |
| Fenk, Riquelme & Laurent 2024 report the same rhythm as a CPG upstream of the isthmus with period **133 +/- 11 s at ~21.5 °C** (their "room temperature"); phase-resettable by light, REM shortenable not lengthenable, monocular pulse resets only contralateral generator; the paper itself attributes the difference from 2016 to temperature | Nature 636:681 (2024) | Two coupled half-centre oscillators, one per side, with light PRC; second anchor 21.5 °C -> 133 s | verified |
| The same rhythm in *Laudakia vulgaris* across 17-35 °C: 48 s at 34 °C, 158 s at 21 °C, Arrhenius Q10 = 2.3; across species the period tracks body temperature (R² = 0.88) | Albeck et al. 2022, Commun Biol; Bergel et al. 2026, Nat Neurosci 29:543 | period(T) = 133 s x 2.3^(-(T-21.5)/10), which gives ~80-90 s at 27 °C and ~100 s at 25 °C, matching all four datasets | verified |
| Sharp-wave generator = claustrum homologue in anterior-medial DVR; lesion abolishes SWRs but not the rhythm; claustrum receives DC, DVR, thalamus, hypothalamus, VTA, SN, PAG, LC, raphe and projects to hippocampus, DC, aDVR, amygdala | Norimoto et al., Nature 2020 | A brain-state hub node separate from the rhythm generator | verified |
| REM interhemispheric competition (~20 ms lead switching each episode) lives in a GABAergic isthmic nucleus (bird Imc equivalent) | Fenk et al., Nature 2023 | The same isthmic WTA circuit serves attention and sleep | verified |
| 7 h sleep deprivation -> delta/beta rebound, period unchanged; corticotomy removes rebound but not baseline sleep | Hatori et al., PNAS 2025 | Homeostat drives amplitude, not period; cortex-dependent rebound | verified |
| Constant darkness keeps ~24 h sleep timing but breaks regular REM/SWS alternation; light restores it (do not take a period value from this paper; it mis-cites 2016 as "2-min cycles" and reports none at its 30 °C) | Yamaguchi et al., PNAS Nexus 2024 | Circadian gate + light gate on the ultradian CPG | verified |
| Tegu has two sleep states but **no regular periodicity** | Libourel et al., PLoS Biol 2018 | The rhythm is not universal among lizards; expose regularity as a parameter | verified |
| Turtle dorsal cortex: receptive fields cover the whole contralateral hemifield, no clear retinotopy, global scene analysis with position-specific adaptation | Fournier et al., Neuron 2018 | Dorsal-cortex module is a non-retinotopic global encoder, **not** a V1-like map | verified |
| Turtle 3-layer cortex: one pyramidal cell triggers reliable sequences over ~200 ms | Hemberger et al., Neuron 2019 | Replay-capable sparse recurrent sheet (optional) | verified |
| Iguana tectum: visual/somatic/auditory maps in register; superficial visual, deep multimodal; suppressive surrounds, strong habituation, velocity-dependent direction selectivity; most cells prefer stationary stimuli | Stein & Gaither 1981/1983 | Two-layer tectum with touch/IMU summed into the deep layer; built-in habituation | verified |
| Anolis motion threshold 0.22 deg, best at 1.5 Hz | Fleishman 1986 | Front-end sensitivity target | verified |
| Pogona saccades: 3-5 Hz active vs <1 Hz quiet; peak speed ~ 0.021 x amplitude + 0.083 deg/ms; 49% monocular | Leberstein et al., bioRxiv 2026 | Orienting controller main-sequence law | verified |
| Preoptic thermostat: heating preoptic area to 41 °C makes a lizard leave a hot chamber 1-2 °C sooner; cooling delays exit; body held 30-37 °C by shuttling | Hammel et al., Science 1967 (Tiliqua) | A comparator-style temperature drive (currently missing from NeuroGecko) | verified |
| Cold-sensitive preoptic units +23% firing for 30->20 °C | Liu et al. 2006 (Phrynocephalus) | Thermostat gain | verified |
| Medial/dorsal cortex lesions slow acquisition and abolish map strategy in whiptail lizards; turtles likewise | Day et al. 2001; López 2003; Rodríguez 2002 | Memory-module lesion prediction | verified |
| Anolis brain 44.6 mg, 4.79 M neurons; cerebellum 3-5% of mass | Storks et al. 2023 | Cerebellum is small; do not over-invest | verified |

### 2c. What does not exist (say so in the paper)

No gecko sleep electrophysiology; no gecko tectal single-unit map; no lizard tectal-lesion prey-capture study; no reptile hypothalamic feeding circuit; no lizard/gecko spinal CPG data (turtle scratch only); no published leopard-gecko brain mass; no leopard-gecko basal-ganglia tract tracing since 1991. **[verified: Europe PMC searches this session]** Every dynamical parameter in the brain will therefore be a transplant, and the honest formula is: *lizard architecture, proxy-species dynamics, gecko behaviour*.

---

## 3. The best achievable brain, module by module

Format for each: best model to borrow | species | code and licence | data that ground it | interface | rate or spiking | learning | feasibility for one developer.

### 3.1 Retina / early vision

- **Borrow:** an insect-style small-target-motion detector (ESTMD, Wiederman/Shoemaker/O'Carroll 2008) for the prey channel, the Kühn/Farrow 2025 twelve-channel ON/OFF transient/sustained bank with low-pass summation for wide-field prey/threat detection, and a looming channel. **[verified]**
- **Species:** dragonfly/hoverfly STMD (prey channel); mouse SC wide-field inputs (channel bank). Functionally identical to zebrafish small-dot RGC->tectum channels. Nocturnal gecko note: rod-like retina, slit pupil, scotopic cone pigments (Kojima 2021) — run on luminance, not colour. **[likely]**
- **Code/licence:** `MingshuoXu/Small-Target-Motion-Detectors` (Python/PyTorch, Apache-2.0, v2.3.0.8 Jun 2026) or ~150 lines of NumPy from the 2008 stage list; Kühn 2025 code on Zenodo (DOI in paper). flyvis (MIT) as an offline cross-check only. **[verified]**
- **Grounding:** STMD tuning curves; mouse WF neuron responses 15-120 deg/s. **[verified]**
- **Interface:** in = luminance from the head camera; out = three retinotopic drive maps: prey (small dark moving), threat (looming/large), whole-field motion per eye (for pretectum/OKR). **The camera must change first.** The current camera (`morphology/gecko_body_r.xml` line 72: `head_cam`, fovy 120 deg, mode fixed; `envs/gecko_brain_env.py` defaults `camera_width/height` = 64) is a rectilinear pinhole projection, so "120/64 = 1.9 deg per pixel" is wrong: at fovy 120 the centre pixel of a 64x64 frame subtends 3.1 deg and the edge pixel 0.8 deg, so a 5-deg prey at fixation covers only ~1.6 px (measured this session by rendering a sphere at known angles in MuJoCo 3.9.0; the repo's own `docs/research/no_gecko_and_appearance.md` makes the same linear-average error). At 128x128 and fovy 120 the centre pixel is 1.55 deg and a 5-deg prey covers ~3.2 px, still marginal against ESTMD's 1.4-deg optical scale; narrowing fovy to 90 deg at 128x128 gives ~0.9 deg per centre pixel and ~5.6 px per 5-deg prey at the cost of periphery. Recommendation: 128x128 with fovy exposed as a parameter (90-120 deg) and the trade-off recorded. Note the `camera_width/height` kwargs are passed by none of the 15 `GeckoBrainEnv(...)` call sites, `brain/bc_actor.py` hard-codes (64,64,3), `brain/vision_encoder.py` is the 64x64 CNN, and `<global offwidth="640" offheight="480"/>` caps offscreen height at 480 px — so a resolution change touches those files and forces a retrain. **[verified]**
- **Rate/spiking:** rate. Fixed filters.
- **Learning:** none. (If a learned front end is kept, call it "task-optimised", as Lappalainen 2024 does.)
- **Feasibility:** 1-2 days, CPU real-time.

### 3.2 Optic tectum (+ isthmi, pretectal release, escape integrator)

- **Borrow:** the **Zylbertal & Bianco 2023 recurrent LNP tectum** (eLife 12:e78381; "TectalLNP" is only the repo name): a recurrent sheet defined by seven fitted global parameters — six coupling-filter parameters (gain, spatial sigma and time constant for excitation and for inhibition: fast short-range excitation, tau 0.05 s, sigma 4.5 um; long-lasting suppression, tau 24.1 s, sigma 40 um) plus a baseline drive mu, with a fixed 0.01 cross-hemisphere scaling — fitted by evolutionary multi-objective optimisation to reproduce tectal bursting, response variability, spatially selective habituation, spontaneous hunt initiation and post-stimulus suppression. It is the most parsimonious published model fitted to zebrafish tectal spontaneous-activity statistics and linked to behaviour, not a field-anointed "best" (newer alternatives: Qian et al. 2025 connectome-constrained reservoir tectum, repo not reachable this session; Légaré et al. 2025 random-geometric model). **[verified]** Wrap it with: (i) Del Bene 2010 SIN-style large-stimulus inhibition and the Barker & Baier 2015 small->approach / large->avoid switch; (ii) Imc global inhibition (Wang 1983 *Gekko*; Fenk 2023) as one global unit per side with mutual competition; (iii) an SNr gate from the basal ganglia (Medina & Smeets 1991: the only BG->tectum route in Eublepharis, per one unreplicated 1991 paper); (iv) WTA/averaging softmax with a nucleus-isthmi gain that sustains a chosen target (Fernandes 2021; Henriques 2019); (v) a latched pretectal-style release gate for "engage" (Antinucci 2019); (vi) a **wide-field / narrow-field readout split** (Hoy, Bishop & Niell 2019, chemogenetic suppression in mouse): a pooled wide-field map (RFs 700-900 deg², 4-5 deg objects, speed-selective, projects to LP) required for rapid detection and *distant* approach initiation, and a fine narrow-field map (RFs <200 deg², superficial somata projecting to intermediate/deep SC and parabigeminal nucleus) that has sole authority over targeting accuracy and continuity once an approach is under way. This is a single dissociation on accuracy layered on a shared initiation deficit — suppressing *either* population delayed the first approach — so both channels must vote on initiation; only long-range initiation is wide-field-specific. Build the narrow-field channel as a small-RF positional-error signal, not a direction-selective motion detector: Krizan et al. 2024 (PNAS) confirmed narrow-field cells guide predation but showed removing retinal direction selectivity leaves hunting intact; (vii) an Evans 2018 threshold integrator on the threat map for escape. **[verified]**
- **Species:** larval zebrafish (dynamics), mouse (WF/NF, escape), *Gekko* and iguana (layering, Imc loop, registration of touch with vision).
- **Code/licence:** `azylbertal/TectalLNP` is MIT, pure MATLAB (built-ins plus `exprnd` from the Statistics Toolbox; it is adapted from, not dependent on, Pillow's GLMspiketools, which is also MIT). Port to NumPy with FFT convolutions, 2-4 days. Note the shipped `optimized_params.mat` uses unrounded values (sigma_E 4.54 um, sigma_I 39.7 um, tau_E 0.046 s, tau_I 24.1 s, g_E 6.46, g_I 0.021, mu 31.8; dt 50 ms; 14,597 neuron coordinates vs 14,733 quoted in the paper) and dense 14,597² single-precision weight matrices (~850 MB each) with an 18,000-step transient — sparsify by radius and shrink the sheet for a laptop. Nothing else has code; all wrappers are hand-written from the papers. **[verified]**
- **Grounding:** whole-tectum imaging and closed-loop hunting assays (Bianco); 1,759-cell tuning statistics (Förster 2020: 41.1% small-dot, 33.1% large-dot, 43.6% looming responders; posterior small/DS, anterior large); *Gekko* Imc RF-entry/exit bursts. **[verified]**
- **Interface:** in = the three retinal maps, foot/belly/snout touch and IMU (summed into the deep layer in register, as in iguana), SNr gate, DA/arousal gain; out = target azimuth/elevation and salience (steering), a latched `engage` release, an `escape` trigger with vigour, per-unit activity logs for validation. Replaces the CNN->4-D command. **[verified]**
- **Rate/spiking:** rate sheet (the LNP Poisson draw is optional and cheap; keep it for the bursting statistics test).
- **Learning:** none; habituation and refractoriness are dynamics, not weights.
- **Feasibility:** 2-3 weeks total for the tectal stack; >100 Hz at 128x128 on CPU once the sheet is sparsified. **[likely]**

### 3.3 Basal ganglia

- **Borrow:** the **GPR** selection/control model (Gurney, Prescott & Redgrave 2001) in its **extended** form — the Humphries & Gurney 2002 thalamocortical (VL/TRN) persistence loop, which is the version actually validated on robots (Girard 2003; Prescott 2006/2024) — split into a **dorsal loop** (which behaviour: hunt, flee, explore, bask/thermoregulate, rest, groom) and a **ventral/accumbens loop** (approach vs retreat vs rest motivation, heading) as in Girard 2005, with the two coupled trans-subthalamically so locomotion channels cannot be selected during a strike. **[verified]** Wire the channel inputs and outputs from the *Gekko* tract tracing: inputs from DVR object channel, dorsal-cortex context, sensory thalamus, amygdala-like fear node, hypothalamic drives; dorsal outputs via SNr to the tectum gate and via GP/EP -> ventral thalamus to the brainstem locomotor drive; ventral outputs via VP to lateral hypothalamus/VTA. DA from a VTA/SNc node = hypothalamic reward-prediction error, strongest on the ventral loop and on the DVR (Smeets 1986). Tonic DA lambda ~0.20 is the model's established baseline (D1 gain 1+lambda, D2 gain 1-lambda); tying it to arousal/energy is our design choice, not a finding of Prescott 2024, which mentions arousal only as a known influence on tonic dopamine. Prescott et al. 2024 sweep (18 lambda levels, 5 x 120 s robot trials each): at lambda 0.20-0.29, 89-95% of selection competitions are clean in the robot (73-81% disembodied), and all trials succeed at 0.20-0.28; at lambda <= 0.12 movement slows below 75% of intended vigour and the gripper fails to lift in most trials, with prolonged immobility only at <= 0.06 (14 s per trial at 0.06, 38 s at 0.03, vs ~2 s baseline); distortion (partial expression of losing channels) is visible from 0.31, outcomes are mixed at 0.31-0.34, 0.37 is clean, 0.40-0.43 fail in most trials with switching bouts about three times baseline (21.3 vs ~7; the winner-take-all variant gives 9.2 at the same levels), and 0.46 fails every trial. **[verified]**
- **Species:** rat (GPR parameters), *Gekko*/Eublepharis (wiring), lamprey-to-mammal conservation (Stephenson-Jones 2011; Grillner 2013) justifying the transplant. **[verified]**
- **Code/licence:** ModelDB 124111 (Girard et al. 2008: `CBGTC.py`, `basalganglia.py`, `thalamusFC.py`, NumPy-only, **Python 2**, includes a "GPR" option that is the Prescott-2006 updated GPR inside the thalamocortical loop, and a partial replication of the Prescott salience sweep: efficiency for channels 1-2 only) has **no licence** anywhere (ModelDB tree, GitHub mirror, author's own repo), so it is all-rights-reserved by default — do not vendor it into the Apache-2.0 repo; reimplement from the equations (~200-300 lines, 1-2 days) or email Girard. Prescott 2024 (Biomimetics 9:139, CC BY) ships 2003-era C++ in `biomimetics-09-00139-s001.zip` (README states CC BY, research use only; targets Webots 2.0 / Khepera I / X11 and pre-standard headers; one header the default build expects is missing, so build with `NEW_DA 1`; the shipped extended-model default dopamine is 0.3, set 0.20 to match the paper) — the standalone `bg/` + `test_bg/` parts are the portable core. Sheffield's ModelDB 83560/83562 originals are unlicensed MATLAB/Simulink. Nengo's `BasalGanglia`+`Thalamus` (GPL-2, copyleft) is a later spiking swap behind the same interface, never vendored. **[verified]**
- **Grounding:** GPR reproduces dopamine-dependent selection physiology and robot behaviour switching/disintegration; Girard 2003 metrics (bout duration, switching frequency, time in comfort zone) are the ablation against the current if/else arbiter. **[verified]**
- **Interface:** in = per-channel salience built sigma-pi style (Girard 2003 Table I) from the drive vector, tectal salience, DVR value, memory goal, plus persistence; out = per-channel disinhibition gate e_i = L(1 - y_i/c) in [0,1]; one hard-selected behaviour channel and a continuous heading bias. **[verified]**
- **Rate/spiking:** rate (~5N + 3N leaky integrators; microseconds per step).
- **Learning:** **yes, biologically** — three-factor cortico/tecto-striatal plasticity: eligibility trace (pre x post, tau ~1 s) gated by the dopamine RPE from the hypothalamus, D1 potentiated by bursts, D2 by dips (Gurney, Humphries & Redgrave 2015; Frémaux, Sprekeler & Gerstner 2013), with a Lindsey et al. 2025 efference copy of the selected action to both pathways so the rule learns correctly. OpAL* (MIT) is the algorithmic fallback. **[verified]**
- **Feasibility:** 3-5 days for the rate GPR + salience layer, 1-2 days for the dopamine sweep test, 2-4 days for the plasticity rule. Highest credibility per hour in the whole project. **[likely]**

### 3.4 Hypothalamus + drives

- **Borrow:** **Homeostatically Regulated RL** (Keramati & Gutkin, eLife 2014): internal state h in a homeostatic space with setpoints h*, convex drive D(h), reward = D(h_t) - D(h_{t+1}) minus small costs; reward maximisation is provably homeostatic regulation. **[verified]** Implement the embodied version of Yoshida et al. 2024 (PNAS Nexus; MuJoCo quadruped with interoception in the observation, per-step decay 0.00015, +0.1 per meal, posture 0.005 and action 0.0005 costs) and Yoshida's meta-HRRL (recurrent policy fed the drive history). Add: a **Hammel-style preoptic thermostat** (warm/cold-sensitive rectified errors around T_pref from the scorecard) — the only homeostatic drive with real reptile circuit data and currently absent from NeuroGecko; per-drive modular critics summed into BG salience (Dulberg et al., PNAS 2023, Apache-2.0); a Grimbly et al. 2026 (arXiv 2608.04232, SAB 2026) precision/attention weight per drive that scales tectal saliency; a 3CC-r muscle-fatigue state per actuator (MyoSuite constants F=0.0146, R=0.0022, r=10) as a minutes-scale drive; a hard death boundary; a fear/threat drive that is now *emergent* from the tectal escape integrator rather than hand-written. **[verified]**
- **Species:** mammalian hypothalamus-VTA (HRRL), lizards (thermostat: Tiliqua, Phrynocephalus, Dipsosaurus), *Gekko* for DA cells in periventricular/lateral hypothalamus. Hunger circuit is a mammalian proxy; say so. **[verified]**
- **Code/licence:** reimplement (HomeoRL repo has no licence; deeprl_gfn is MIT but pinned to mujoco-py/Ubuntu — port the equations only; Multiple-Selves Apache-2.0; attention-aif snapshot MIT). **[verified]**
- **Grounding:** HRRL reproduces anticipatory responding, satiation rise-fall, risk aversion, drive competition; Yoshida 2024 reproduces the Geometric-Framework nutrient rules; Hammel 1967 thermostat behaviour. **[verified]**
- **Interface:** in = body interoception (energy, temperature from a MuJoCo temperature field, hydration, fatigue, sleep pressure S, threat); out = scalar reward to the residual policy, per-drive deviations and precision weights to the BG and tectum, DA RPE to the VTA node, activity-level gain (Q10) to the CPG. Replaces `drives.py`. **[verified]**
- **Rate/spiking:** rate ODEs.
- **Learning:** the reward is normative (no learning); the per-drive critics are gradient-trained; an optional 2-layer interoceptive forward model for allostasis (Tschantz et al. 2022 idea) is gradient-trained.
- **Feasibility:** 3-5 days for the module; training cost is dominated by the body. Requires a temperature field in the world (the one real dependency). **[likely]**

### 3.5 Sleep / wake

- **Borrow:** Borbély two-process model (Process S: dH/dt = (mu-H)/chi_w awake, -H/chi_s asleep; human chi_w 18.2 h, chi_s 4.2 h as placeholders) driven through a Phillips-Robinson 2007 VLPO/monoaminergic flip-flop, with Process C from a light-entrained oscillator (Arcascope `circadian`, MIT, or a 2-variable Poincaré oscillator with light-pulse resetting) **inverted for a nocturnal animal**. Inside sleep, a **Fenk 2024 ultradian CPG**: two mirror-symmetric biphasic half-centre oscillators (one per side) with net excitatory coupling, REM as the active output and SW as default, phase-reset by light per eye, with **temperature in °C as the exposed parameter**: period(T) = 133 s x Q10^(-(T-21.5)/10), Q10 ~2.3 (Laudakia, Albeck 2022), anchored at 21.5 °C -> 133 s (Fenk 2024) and 27 °C -> ~80-90 s (Shein-Idelson 2016; Albeck & Shein-Idelson 2026), ~100 s at 25 °C (Bergel 2026). A separate **claustrum hub node** (Norimoto 2020) emits sharp-wave/replay events during SW and broadcasts neuromodulator gains (DA, 5-HT, NA, ACh in the Doya 2002 reading) to the pallial modules. Homeostat drives delta *amplitude*, not period (Hatori 2025). **[verified]**
- **Species:** human/rodent (two-process constants), Pogona/Laudakia (ultradian CPG, PRC, rebound, light gating, Q10), tegu (warning that regularity is not universal). No gecko data. **[verified]**
- **Code/licence:** equations public; ModelDB 247696 CellML (no licence); `Arcascope/circadian` MIT; `kerkphil/AnimalSleep` GPL-3 (only non-human two-process code; opportunity-cost idea). The Laurent-lab and Shein-Idelson-lab repos are analysis-only; the parametrisation above needs none of them. **[verified]**
- **Grounding:** forty years of sleep-timing data; Laurent-lab lesion, deprivation and light-pulse experiments give a full dynamical fingerprint (section 5). **Prior art to cite honestly:** several published agents already sleep because of a modelled homeostatic rest variable with *linear* dynamics — Maroto-Gómez et al. 2018 (Sensors 18:2691; Q-learning social robot with a Rest drive), Laurençon et al. 2024 (arXiv 2401.08999; continuous-space HRRL agent with a sleep-fatigue state that forces sleep above a threshold), Yoshida & Kuniyoshi 2025 (IEEE ICDL; homeostatic Crafter, whose energy/sleep mechanic is Crafter's own fatigue counter), and, pre-deep-RL, Grand's Creatures (1997). None uses a Borbély-form saturating two-process homeostat, none has a circadian gate, and none has an ultradian sleep-state CPG; Grimbly et al. 2026 has no sleep need in the published agent (only an unused rest channel in its repo). The defensible novelty is therefore "first embodied agent whose sleep timing comes from a two-process homeostat plus a lizard-derived ultradian CPG with a light phase-response, deprivation rebound and lesion signatures", not "first agent that sleeps homeostatically". **[verified]**
- **Interface:** in = scene light per eye, body temperature, S, threat; out = state {wake, SWS, REM}, phase, delta/beta proxies; effects = hard gate on brainstem descending drive, raised tectal thresholds, override by threat, learning updates and memory replay allowed only in SW. **[verified]**
- **Rate/spiking:** rate ODEs.
- **Learning:** none in the module; it schedules learning elsewhere (PPO/critic updates and memory consolidation in SW; Hobson & Friston 2012 complexity reduction as optional weight decay in REM).
- **Feasibility:** 1-2 days.

### 3.6 Brainstem command (MLR / reticulospinal / orienting)

- **Borrow:** a two-channel MLR drive (CnF-like speed/gait channel; PPN-like slow exploratory channel; Ryczko 2024) into bilateral reticulospinal populations with left/right asymmetry for steering, structured as in Ausborn et al. 2019 (CnF/PPN -> LPGi Glu-1 sets frequency, Glu-2 gates interlimb coupling) but reimplemented (GPL-3) and rescaled from mouse 2-12 Hz to gecko 1.0-1.2 Hz. Expose **hunting archetypes** (orient-turn, approach, strike; Lau, Fitzgerald & Bianco 2025) and an **orienting controller** producing head/eye saccades at 3-5 Hz when engaged with a main-sequence law (Alizadeh & Van Opstal 2022 spiking SC motor map as the design; Leberstein 2026 Pogona kinematics as the numbers). Note for the deep-SC motor readout: in freely moving mice deep SC activity corresponds to *head* movements rather than eye movements (Sharp et al. 2025), which suits a gecko that orients with its head. The salamander MLR data (Cabelguen 2003: drive level selects gait) is the tetrapod spec. **[verified]**
- **Species:** salamander (MLR->RS organisation), mouse (cell types, Ausborn), Pogona (saccade kinematics), primate (main sequence).
- **Code/licence:** `SimonDanner/CPGNetworkSimulator` GPL-3 (C++/pybind11) — do not vendor into the Apache-2.0 repo; reimplement ~20 ODEs. **[verified]**
- **Grounding:** Caggiano 2018 optogenetics reproduced by Ausborn 2019; salamander MLR microstimulation; Pogona saccade main sequence. **[verified]**
- **Interface:** in = BG gates (go/stop, selected channel), tectal steering vector and engage/escape, sleep gate, hypothalamic activity gain; out = d_limb, d_axial, L/R asymmetry, per-limb amplitude tweaks to the spinal oscillators, head-turn targets. Replaces the `distance`/`engage` scalars. **[verified]**
- **Rate/spiking:** rate.
- **Learning:** none.
- **Feasibility:** 2-4 days.

### 3.7 Spinal CPG

- **Borrow:** a **two-level, feedback-modulated oscillator cord** — (RG) four limb phase oscillators plus trunk and tail oscillators obeying the Tohoku load-feedback law phi_dot = omega - sigma * N * cos(phi) (Owaki & Ishiguro 2017) with Suzuki et al. 2021 leg<->trunk cross-coupling, omega and amplitude set by Ijspeert-2007-style saturating functions of the two MLR drives; (PF) the existing phase->joint synergy map (`envs/cpg_residual_controller.py::_limb_signals`) re-fitted to the Jagnandan & Higham 2018 EMG onsets/offsets so the 26 actuators are driven by muscle-like synergies (Rybak & McCrea 2006 RG/PF split). **Which regime the gecko sits in.** Rybak et al. 2024 (eLife 13:RP98841, cat, model calibrated to intact and spinalised treadmill data) conclude that at slow speeds (<0.4 m/s) the intact mammalian cord runs as a non-oscillating state machine whose phase transitions need sensory feedback *or external inputs*, switching with drive to a flexor-driven oscillatory regime and then a classical half-centre regime. The boundary is model-derived, claimed only for mammals, and in the authors' own reviewer response maps onto *cycle frequency* (~0.6-0.7 Hz), not belt speed. The gecko strides at 1.19 Hz, so by Rybak's own logic it is in the flexor-driven *oscillatory* regime despite its 0.11 m/s — the earlier argument that "0.11 m/s puts it in the state-machine regime" does not hold. The case for load feedback inside the rhythm generator therefore rests on the tetrapod evidence, not on Rybak: Harischandra 2011 (a lateral-sequence walk needs feedback), El Manira 2026 (speed-specific modules with proprioceptors inside the CPG), and the Tohoku/Twister robot results. The Tohoku oscillator is intrinsically rhythmic and feedback-entrained, which is exactly the flexor-driven-with-feedback picture; expose the feedback gain sigma so that sigma -> 0 (pure clock) and a large sigma (feedback-triggered) are both testable. Run **SalamandraSNN** (Pazzaglia et al. 2025, MIT, Brian2+MuJoCo) offline as the spiking reference to tune phase relationships against. **[verified]**
- **Species:** salamander (topology, drive maps), cat/mouse (RG/PF split, regime taxonomy), leopard gecko (EMG amplitude/timing, 1.1888 Hz, duty factors 0.62-0.70).
- **Code/licence:** NumPy from the papers (Tohoku equations fully specified); `farmsim/farms_network` Apache-2.0 optional; SalamandraSNN MIT (offline only; FARMS on Windows/MuJoCo 3.9 unverified). Rybak's `RybakLab/nsm` is GPL-3 Windows C++ with Visual Studio projects — its conditional-burster half-centre is a 2-4 ODE system you can rewrite in an afternoon rather than build. **[verified]**
- **Grounding:** Ijspeert 2007 made a confirmed physiological prediction; Twister robot 0.101 vs 0.044 m/s with vs without leg-trunk coupling; Rybak 2024 regimes and deletions. Gecko grounding stops at EMG and kinematics — no lizard spinal circuit exists. **[verified]**
- **Interface:** in = MLR drives, foot loads, stretch proxies, fatigue gains; out = 26 actuator targets. The PPO residual moves from "add to ctrl" to "modulate oscillator drive/amplitude/phase + small capped bypass" (CPG-RL 2022; Puppeteer & Marionette 2023; Yang 2025). **Any of these changes breaks the 1.1888 Hz-locked policy and requires one PPO retrain with frequency randomised over 1.0-1.2 Hz.** **[verified]**
- **Rate/spiking:** rate oscillators; SalamandraSNN spiking offline only.
- **Learning:** none inside the cord; the residual is gradient-trained (PPO).
- **Feasibility:** 1-2 weeks NumPy plus one 1-3 day CPU PPO retrain (as in the V4.2.x lineage). **[likely]**

### 3.8 Pallium / cortex

Small, coarse, three channels. **[verified: Platel 1976 and Storks 2023 justify keeping it small]**

- **Dorsal cortex (context):** a small CNN with global pooling and *no* retinotopic map (Fournier 2018 turtle), emitting a scene/context/novelty embedding; novelty replaces the hand-invented curiosity decay, ideally as a model-mismatch signal like 3M-Progress (Keller et al. 2025). Gradient-trained; label it task-optimised. Feeds the dorsal BG loop and the memory module. **[verified]**
- **DVR (object/value):** the tectofugal channel (tectum -> rotundus -> aDVR; Rueda-Alaña 2025; *Gekko* rotundus->striatum) producing object identity/valence with DA gain (Smeets 1986: DVR is DA-rich in geckos), feeding striatum and an amygdala-like fear node. Gradient-trained features; value learned by the BG three-factor rule. **[verified]**
- **Medial cortex (place memory):** a coarse allocentric map (successor-representation or a small grid of shelter/warm/food locations) updated from the dorsal-cortex embedding and IMU path integration, with Hebbian/SR updates and consolidation only in SW replay from the claustrum hub; tuned to the leopard gecko's own numbers: group-level acquisition over 20 spaced (weekly) trials, no measurable decline at 2 months, collapse to naive baseline by 6 months with re-acquisition inside 30 trials (Landová 2025), individually variable use of proximal and distal cues (Landová 2023), and a neurogenesis-scale slow structural timescale (weeks). Lesion prediction: falls back to beacon search (Day 2001). Output = goal vector to the dorsal BG loop, replacing the arbiter's implicit `distance` scalar. **[verified; 2023 cue paper likely]**
- **Code/licence:** none to borrow; 2-4 days each. Tosches lab ReptilePallium code is scRNA-seq analysis, not needed.
- **Rate/spiking:** rate.
- **Feasibility:** 1-2 weeks for the three; the most speculative part of the brain, build last.

### 3.9 Cerebellum

- **Role:** minor. The lizard cerebellum is 3-5% of brain mass (Storks 2023), and there are no gecko data to validate against. The one defensible role is an **adaptive feedforward correction** on the descending motor command — DeWolf et al. 2016's spiking cerebellum with the PES (local error-driven, LMS-like) rule, which is biologically plausible and stable-by-proof. **[verified]**
- **Borrow:** DeWolf 2016 architecture in rate form (or Nengo, GPL-2, if a spiking demo is wanted later); a Huebotter 2025-style forward model is the heavier alternative. **[verified]**
- **Interface:** in = efference copy of MLR drive and CPG phase, IMU error; out = small additive correction to the descending drive (this could absorb part of the PPO residual's job).
- **Rate/spiking:** rate. **Learning:** PES/LMS, local, plausible.
- **Feasibility:** 2-3 days; **optional**, low validation payoff. Build only after everything else passes its tests.

### 3.10 Wiring diagram (text)

```
                       LIGHT / TEMP / TIME                          BODY INTEROCEPTION
                              |                                    (energy, T_body, fatigue,
                              v                                     hydration, load)
                     [Circadian osc. C(t)]                                  |
                              |                                             v
                      [Sleep switch: S + PR flip-flop] <---- [HYPOTHALAMUS: drive vector h,
                              |                                D(h), thermostat, precision w_i,
                     [Ultradian CPG L/R (REM/SW), period(T)]   reward r = D(h_t)-D(h_t+1)]
                              |                                     |            |
                      [Claustrum hub: gains,                        |            v
                       replay trigger in SW] -------------------+   |     [VTA/SNc: DA = RPE, tonic lambda]
                              |                                 |   |            |
 HEAD CAMERA 128x128          |                                 |   |            |
 (fovy 90-120, pinhole)       v                                 v   v            v
 [RETINA: prey map, threat map, whole-field motion/eye]     [BASAL GANGLIA (extended GPR)]
      |            |                    |                    dorsal loop: DC ctx + DVR obj + thal + fear + drives
      |            |                    +--> [PRETECTUM/AOS] --> OKR/head stabilisation   -> SNr gate --> tectum
      v            v                                                                      -> GP/EP -> vThal -> brainstem
 [TECTUM sheet (Zylbertal-Bianco LNP) + SIN size filter] ventral loop: DVR value + drives -> VP -> LH/VTA (approach/retreat)
      |  <-> [Imc global inhibition, L/R competition]                ^          ^
      |  <-> [Nucleus isthmi gain: sustain pursuit]                  |          |
      |  <-- SNr gate                                                |   [MEDIAL CORTEX place map] <-- IMU odometry
      |--> [Pretectal release latch: ENGAGE]                         |          ^  (replay in SW)
      |--> [Escape integrator (mSC->dPAG-like)]: FLEE                |          |
      |--> steering (WF + NF both vote to initiate;                  |   [DORSAL CORTEX: global, non-retinotopic context/novelty]
      |     NF alone corrects pursuit) ---------------------+        |          ^
      +--> rotundus --> [DVR object/value, DA gain] ---------+-------+          |
      touch/IMU summed into deep layer                                          |
                                                                   LGN <---- retina (lemnothalamic)
                              |
                              v
 [BRAINSTEM: MLR CnF-like speed/gait + PPN-like explore -> bilateral RS; L/R asym = steer;
  hunting archetypes orient/approach/strike; orienting controller 3-5 Hz head saccades; sleep gate; fatigue gain]
                              |            ^
                              |            +---- [CEREBELLUM-like PES feedforward correction] (optional)
                              v
 [SPINAL CORD: RG = 4 limb + trunk + tail intrinsic phase oscillators with load-feedback entrainment,
  leg<->trunk coupling; PF = EMG-fitted synergy map -> 26 position actuators]  <-- foot load, stretch
                              |
                              v
                        MuJoCo gecko body
 [PPO residual (ANN, recurrent, sees h history): modulates RG drive/amplitude + small capped bypass]
```

---

## 4. Where this would sit in the world

**Blunt comparison.**

| | Virtual rodent / MIMIC (Nature 2024) | Fly: NeuroMechFly v2 / flyvis / Shiu / Eon | Zebrafish: simZFish / CMU agent | **NeuroGecko (ceiling)** |
|---|---|---|---|---|
| Body | 74 DOF mocap-derived | confocal-derived, 102 DOF | 6-7 segments | measurement-based, 26 actuators, tail, touch, IMU, camera — **at parity with flybody-class bodies** [verified] |
| Brain structure | monolithic MLP/LSTM | connectome (visual system, whole brain) or modular hand-built | hand-wired region chain (simZFish) or LSTM+world model | region-structured, every connection cited to lizard anatomy |
| Data constraining brain | 607 h mocap + ephys | EM connectome + transmitters | whole-brain imaging | proxy-species physiology + gecko behaviour scorecard |
| Ethological drives, sleep | none | none (Eon: sugar/water only) | intrinsic motivation, no sleep | homeostatic drives incl. thermostat, two-process sleep, ultradian CPG |
| Prey capture via a tectum | none | none | none in closed-loop whole-body models | **yes** |
| Neural validation | yes (same animal) | yes | yes | **no, impossible** |

**Claims this gecko brain could make that nobody has made** **[verified against the 2026 review and this survey; negatives are "not found", not proofs]**:
1. The first region-structured, closed-loop whole-animal brain for a **reptile with a neural controller in the loop**. No reptile appears in the 2026 review's ~50-entry table or reference list (it is a perspective, not a systematic survey). Reptile *musculoskeletal* simulations do exist (Wiseman 2021 crocodile hindlimb; Iijima, Blob & Hutchinson 2025 alligator; Clark et al. 2021 Draco glide) but none embeds a neural circuit, and the nearest non-mammal tetrapod in the review is a salamander spinal model (Pazzaglia 2025). So the claim is "first reptile", never "first sprawling-gait vertebrate" or "first reptile biomechanical model".
2. The first embodied whole-animal model with a **data-constrained tectal prey-capture circuit** (Zylbertal-Bianco LNP dynamics + isthmic selection) hunting in closed loop; the two 2025 frontier zebrafish agents split behaviour (Malik et al., no circuit) from neural fit (Keller et al., no tectum).
3. The first embodied agent whose **sleep timing is generated by a Borbély two-process homeostat and a lizard-derived ultradian CPG**, with light phase-response, deprivation rebound and lesion signatures — as distinct from the existing agents that sleep from a linear rest counter (Maroto-Gómez 2018; Laurençon 2024; Yoshida & Kuniyoshi 2025).
4. The first whole-animal model in which **basal-ganglia selection, hypothalamic drive-reduction reward, tectal orienting and a feedback-entrained spinal CPG** operate together, each borrowed from its best-validated source model and each passing a named published test.
5. Two gecko-tested pass/fail tests no other model has: monocular temporo-nasal-only head-OKR (Masseck 2008, measured in *E. macularius*; the pattern is shared by other afoveate geckos, so it separates the model from a Phelsuma-like gecko, not from all geckos) and the tail-autotomy EMG amplitude pattern (Jagnandan & Higham 2018) reproduced without retraining.

**Claims it cannot make** **[verified]**:
- "Reproduces gecko neural activity" — no recordings exist. Publish hidden-unit logs as *predictions* for the first lab to record.
- "Gecko parameters" — every dynamical constant is a transplant (zebrafish, mouse, rat, salamander, Pogona/Laudakia). Say "lizard architecture, proxy dynamics, gecko behaviour".
- "Digital twin" or "whole-brain emulation" — R7 language belongs to nobody yet, and R5/R6 need a connectome.
- "Spiking makes it more real" — see section 6.
- That Pogona's 80-133 s sleep rhythm applies to a nocturnal gecko — the tegu shows the rhythm is not even universal among lizards.
- "First agent that sleeps homeostatically" — see claim 3.

Where the paper would land: a credible eLife / PLoS Computational Biology / Science Robotics-class "behaviourally, dynamically and lesion-validated region model of a reptile", roughly where the zebrafish field stood in 2025. Not Nature — Nature-level whole-animal papers all had recordings or connectomes. **[likely]**

---

## 5. The validation battery

Reporting standard for every test (Feather et al. 2025 NeuroAI Turing test; Schrimpf 2020; Doerig 2023): **[verified]**
- Score = model-animal distance / animal-animal distance, using reported SDs and N to estimate the animal-animal ceiling.
- Pre-register the prediction table on OSF *before* running lesions.
- Every test has two rows: NeuroGecko-structured vs a matched-performance monolithic PPO+CNN baseline. A realism claim stands only where the structured brain matches and the baseline does not (Schaeffer 2022; Bowers 2023).
- Log hidden-unit activity with kinematics so an encoding-model comparison is possible later (Aldarondo logic).

### Tier A — body and reflex (species-specific data exist; ~1 week)

| Test | Against | Pass criterion | Proves | Confidence |
|---|---|---|---|---|
| A1 Gait: MLR drive sweep | Zaaf 2001; Fuller 2011 speed-frequency-duty relations | within reported SDs; frequency inside 1.0-1.2 Hz voluntary range | descending drive + CPG organised like the animal | verified |
| A2 EMG phase | Jagnandan & Higham 2018 onset/offset per muscle | PF synergy timings within measured phase windows | pattern-formation layer is muscle-realistic | verified |
| A3 Tail-autotomy perturbation, **no retraining** | Jagnandan & Higham 2018: caudofemoralis and gastrocnemius stance-burst *amplitude* down, gastrocnemius late-swing burst unchanged, all burst timings/durations unchanged, puboischiotibialis unchanged, biceps/triceps brachii unchanged; Jagnandan 2014/2017 posture | correct sign on amplitude for all five muscles and no timing shift | feedback-entrained cord generalises like the animal | verified |
| A4 Head-OKR | Masseck, Röll & Hoffmann 2008 (*E. macularius*): stabilising head movements | binocular symmetric gain; monocular temporo-nasal only | crossed pretectal/AOS wiring (inferred, not measured, in geckos) | verified |
| A5 Looming / multimodal predator cue | Evans 2018 threshold law; Frydlová 2026 gecko predator-cue response | escape probability/vigour rises with saliency; latency ordering | tectal threat channel + escape integrator | likely (Frydlová not read in full) |

### Tier B — prey capture: tectum + basal ganglia (~2 weeks)

| Test | Against | Pass criterion | Proves | Confidence |
|---|---|---|---|---|
| B1 Tectal tuning statistics (5 deg / 30 deg / looming / bars to the camera) | Förster 2020 (41.1/33.1/43.6% responders; posterior small+DS, anterior large); Bianco & Engert 2015 mixed selectivity | responder fractions and pole anisotropy within animal-animal spread; mixed size x speed x contrast selectivity present | tectal front end reproduces vertebrate tuning motif | verified |
| B2 Isthmi units | Wang 1985 (*Gekko*): RF entry/exit bursts, topographic binocular map, no touch/sound | qualitative match | the one gecko-recorded circuit is reproduced | verified |
| B3 Bursting/habituation dynamics | Zylbertal & Bianco 2023 (power-law burst sizes, spatially selective habituation, post-stimulus suppression) | same statistics after rescaling | dynamics faithful to the source model | verified |
| B4 Hunting sequence statistics | Mearns 2020 (abort within one bout on prey loss); Johnson 2020 (hunger-dependent bout selection); Bolton 2019 bearing-halving law; Probst 2023 tokay prey-size preference; Vollin 2021/2023 strike kinematics | stereotyped orient-approach-strike chains; abort latency; hunger shift; per-step bearing reduction with distance-scaled noise; size preference | tectum->BG->brainstem produce vertebrate-typical hunting | verified (gecko items likely) |
| B5 Pre-registered lesions | Gahtan 2005 (tectum off: no orienting, walking intact); Henriques 2019 (isthmi off: detect but cannot sustain pursuit); Hoy 2019 (WF channel off: slower detection, approaches start closer; NF channel off: inaccurate, discontinuous approaches; both delay first approach); Antinucci 2025 (pretectum contraversive vs tectum ipsiversive); Prescott 2024 (lambda <= 0.12: slowed movement; <= 0.06: immobility spells; 0.31-0.34: distortion; 0.40-0.43: majority failure with ~3x switching bouts; 0.46: total failure); Girard 2003 (equal saliences: WTA dithers, GPR persists) | correct sign for every lesion; GPR beats arbiter on bout duration/switching metrics | the architecture, not just the behaviour, is right | verified |

### Tier C — drives, memory, brain state (~2-3 weeks)

| Test | Against | Pass criterion | Proves | Confidence |
|---|---|---|---|---|
| C1 Homeostasis | Keramati & Gutkin 2014 predictions (anticipatory action, satiation rise-fall, drive competition, risk aversion); Yoshida 2024 intake rules; scorecard physiology (meal patterning, T_pref shuttling per Hammel) | hunger-dependent hunt/shelter switching; anticipatory basking before cooling; shuttling band ~ scorecard T range; drive-off lesion abolishes state dependence | hypothalamus is normative, not hand-tuned | verified |
| C2 Spatial memory | Landová 2025 (group-level acquisition over 20 weekly trials; no decline at 2 months; baseline at 6 months; re-acquisition within 30 trials; ~25% non-participation); Landová 2023 (proximal + distal cue use, individually variable); Day 2001 heated-rock with cortex lesion | acquisition curve inside animal spread; medial-cortex lesion -> slower learning and beacon strategy | memory module has gecko timescales and lizard lesion phenotype | verified (2023: likely) |
| C3 Individual differences | Fernández-Lázaro 2023; Sakai 2024 novel-object/boldness | parameter variation reproduces habituation/boldness distributions | drive parameters map to personality | likely |
| C4 Sleep fingerprint | Shein-Idelson 2016 / Fenk 2024 (period 133 s at 21.5 °C, ~80-90 s at 27 °C, ~50/50 duty, cycle lengthening ~10 s over the night, light PRC delay-to-advance mid-SW, REM shortenable not lengthenable, monocular contralateral reset); Albeck 2022 Q10 ~2.3; Bergel 2026 period-temperature law; Hatori 2025 (7 h SD -> delta rebound via SW amplitude, period unchanged; cortex lesion kills rebound); Fenk 2023 (isthmic lesion kills REM alternation); Norimoto 2020 (claustrum lesion kills SWRs not rhythm); Yamaguchi 2024 (DD keeps 24 h timing, breaks alternation) | every item qualitatively; period(T) and Q10 within Pogona/Laudakia spread | brain-state module carries the lizard dynamical fingerprint | verified |
| C5 Cross-species representational check (optional) | ZAPBench region subsets (Apache-2.0) via RSA | reported as supporting evidence only | | likely |

Total: ~6-8 weeks CPU, zero cloud spend. **[likely]**

---

## 6. Spiking and plasticity: the decision

**Principle.** Credibility in 2024-2026 comes from measured structure and validation, not from spikes: flyvis is non-spiking; Shiu's whole-fly-brain neuron is "elementary" LIF and its power comes from the connectome; the virtual rodent and flybody are plain ANNs; the eLife 2025 striatal plasticity paper is a rate model rated "fundamental". For a gecko with zero recordings, spike-timing detail is a set of free parameters nobody can check — exactly what reviewers mean by "spiking without constraints". **[verified]**

**Measured on the owner's laptop (i5-12450HX, 8 cores, re-run twice this session):** Nengo 4.1.0's single-threaded NumPy reference backend (no SciPy in the venv; installing SciPy made it ~30% *slower* on this model) runs 1,000 LIF neurons at 9.4-10.3x real time, 11,000 at 1.8-2.0x, 22,000 at 0.9-1.0x, and 11,000 with PES learning at 0.8-0.9x. A simplified PopSAN-style surrogate-gradient spiking actor for this body (T=5, 256-256, population size 10; simpler than the real PopSAN, so a *lower bound* on its cost) costs 12-15x the MLP per actor gradient update at B=2048 (93-116 ms vs 7-8 ms), ~6x at B=64, and 0.85-1.3 ms vs 0.06-0.11 ms per single inference. These are actor-only numbers, not full PPO iterations with environment stepping, and PopSAN's paper reports no CPU cost ratio (its headline is 140x lower energy on Loihi), so they are local measurements, not literature values. Zanatta 2024 shows spiking PPO walks worse on Ant-v4 unless heavily tuned; Proxy Target (NeurIPS 2025) and CaRe-BN (ICLR 2026) show parity is now reachable — at engineering cost, with no biology gained. **[verified]**

| Module | Rate or spiking | Learns? | Rule | Why |
|---|---|---|---|---|
| Retina | rate, fixed filters | no | — | tuning is the validation; no data to fit spikes to |
| Tectum | rate LNP sheet (Poisson draw optional) | no | dynamics-only habituation | the 7-parameter source model is itself LNP; keep its statistics |
| Basal ganglia | rate extended GPR | **yes, biologically** | three-factor dopamine-modulated eligibility rule (Gurney 2015 / Frémaux 2013) + Lindsey 2025 efference copy | cheap, matches published phenomena (~50-trial learning; DA-depletion slowing/freezing; excess switching); highest credibility per hour |
| Hypothalamus | rate ODE | reward normative; critics gradient | HRRL + PPO critics | reward is a theory, not a fit |
| Sleep/wake | rate ODE | no | — | schedules learning elsewhere (SW consolidation) |
| Brainstem | rate | no | — | |
| Spinal CPG | rate oscillators | no | — | validated by EMG/kinematics; a Nengo NEF spiking twin (200-500 neurons, ~1 week) only for a later neuromorphic paper |
| Pallium | rate | DC: gradient (task-optimised); DVR: gradient features + BG value; medial cortex: **Hebbian/SR, plausible**, consolidated in SW | | memory timescales are the gecko-validated part |
| Cerebellum (optional) | rate | **yes, plausible** | PES/LMS local error rule (DeWolf 2016) | stable, local, cheap |
| PPO residual | ANN, recurrent | gradient (PPO) | | stays ANN; call it task-optimised, as the field does |

**Do not adopt** Brian2/NEST/Lava/BindsNET for the runtime brain (no MuJoCo loop, supervised-only e-prop, archived and Python-3.10-only, AGPL). If any spiking framework is ever used, use **Nengo** (GPL-2, NumPy-only, built-in GPR basal ganglia and PES/BCM learning, documented MuJoCo and Loihi precedents; nengo-dl and nengo-ocl backends exist but are stale since 2023), behind the same interfaces and never vendored, as a deployment story for EBRAINS/SpiNNaker (free for basic research) rather than a biology claim. **[verified]**

**One-line verdict:** structure and validation are the science; plasticity in the basal ganglia (and the memory module) is the cheap, defensible biology; spikes are an optional deployment story for later.

---

## 7. Build order (no timelines)

Interface-first, test-gated: each module must reproduce its named measurement *before* it is wired in, and each step should leave the animal walking.

1. **Validation harness + monolithic baseline + logging.** Brain-Score-style registry of tests, metrics, ceilings; OSF pre-registration template; hidden-unit and kinematic logging. Everything after this is graded by it, and the baseline must be trained once at the start so it is matched, not post hoc. **Why first:** without it every later module is an unfalsifiable opinion.
2. **Hypothalamus.** Drive vector, HRRL reward, preoptic thermostat (needs a MuJoCo temperature field), fatigue, death boundary, per-drive critics, precision weights. Train with low-dimensional sensors and compressed metabolic time first. **Why second:** cheapest module, replaces `drives.py`, and every downstream module (BG salience, tectal gain, sleep, brainstem gain) consumes its outputs; fixing its interface early stops rework. Tests: C1.
3. **Basal ganglia (rate extended GPR, two loops), reimplemented from the equations.** Replaces the if/else arbiter behind the same behaviour-channel interface; DA sweep test against the corrected Prescott 2024 values; Girard metrics vs the old arbiter; then the three-factor learning rule. **Why third:** it needs drives as salience, and it defines the gates the brainstem and tectum will consume. Tests: B5 (BG rows).
4. **Brainstem drive + spinal CPG rebuild + camera change, with the single PPO retrain.** Two-channel MLR, RS, two-level feedback-entrained cord with exposed feedback gain, EMG-fitted PF, residual moved to modulation + capped bypass, frequency randomised 1.0-1.2 Hz; at the same time raise the camera to 128x128 (fovy exposed, 90-120 deg) and update `bc_actor.py`, `vision_encoder.py`, the env error string and `offheight`. **Why fourth and not earlier:** these are the changes that force a retrain, so do them once, after the command interface (BG gates, drive gains) is frozen. Run SalamandraSNN offline in parallel if time allows. Tests: A1-A3.
5. **Retina + tectum stack** (STMD/channel-bank front end; NumPy port of the Zylbertal-Bianco sheet, sparsified; Imc, SNr gate, isthmi gain, pretectal latch, WF/NF readout with shared initiation and NF-only correction, escape integrator; pretectal OKR channel). Replaces the CNN->4-D command. **Why fifth:** biggest scientific payoff, but it needs the brainstem orienting/steering interface, the BG gate and the new camera to exist. Tests: A4, A5, B1-B4, B5 (tectal rows).
6. **Sleep/wake + claustrum hub.** Two-process, flip-flop, circadian (nocturnal), ultradian L/R CPG with period(T), hub gains, SW-gated learning. **Why sixth:** cheap, but it gates the brainstem and schedules learning, so the modules it gates must be stable. Tests: C4.
7. **Pallium** (dorsal-cortex context/novelty, DVR object/value, medial-cortex place memory with SW replay). **Why last:** most speculative, least constrained, and it depends on every other module (tectum for the DVR channel, BG for value, sleep for replay, IMU odometry). Tests: C2, C3.
8. **Optional afterwards:** cerebellum-like PES correction; Nengo spiking CPG/BG twin for a neuromorphic demo; MIMIC-MJX locomotion imitation if gecko video ever exists (spend the $500 there).

Guardrails throughout: every parameter transplanted from another species is tagged with its source, species and recording temperature in code; the paper's methods table is generated from those tags.

---

## Sources

**Field position and theory**
- Wang-Chen S, Ramdya P (2026). The embodied brain: bridging the brain, body, and behavior with biorealistic neuromechanical models. arXiv 2601.08056 (v4); Curr Opin Neurobiol 100:103244. https://arxiv.org/abs/2601.08056
- Beiran M, Litwin-Kumar A (2025). Prediction of neural activity in connectome-constrained recurrent networks. Nat Neurosci 28:2561-2574. https://doi.org/10.1038/s41593-025-02080-4 (code: Zenodo 10.5281/zenodo.16618353, CC BY 4.0)
- Aldarondo D et al. (2024). A virtual rodent predicts the structure of neural activity across behaviours. Nature 632:594. MIMIC-MJX: arXiv 2511.20532.
- Lappalainen JK et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. Nature 634:1132. https://doi.org/10.1038/s41586-024-07939-3
- Shiu PK et al. (2024). A Drosophila computational brain model reveals sensorimotor processing. Nature 634:210-219. https://doi.org/10.1038/s41586-024-07763-9 (code: github.com/philshiu/Drosophila_brain_model, MIT)
- Dorkenwald S et al. (2024). Neuronal wiring diagram of an adult brain. Nature 634:124-138.
- Eon Systems (2026). https://eon.systems/ (posts of 7 and 10 Mar 2026)
- Wang-Chen S et al. (2024). NeuroMechFly v2. Nat Methods. simZFish: Science Robotics 2025. FlyGM: arXiv 2602.17997. CMU zebrafish agent: arXiv 2506.00138.
- Pazzaglia A et al. (2025). Salamander spiking neuromechanical model. PLoS Comput Biol 21:e1012101. https://doi.org/10.1371/journal.pcbi.1012101 (SalamandraSNN, MIT)
- Reptile musculoskeletal (no neural controller): Wiseman et al. 2021 J Anat https://doi.org/10.1111/joa.13431; Iijima, Blob & Hutchinson 2025 Sci Adv https://doi.org/10.1126/sciadv.adx3811; Clark, Clark & Higham 2021 ICB https://doi.org/10.1093/icb/icab073

**Gecko and reptile anatomy, physiology, behaviour**
- Medina L, Smeets WJAJ (1991). Comparative aspects of the basal ganglia-tectal pathways in reptiles. J Comp Neurol 308:614-629. PMID 1865018.
- Gonzalez A et al. (1990) PMID 2257479; Russchen FT, Jonker AJ (1988) PMID 3192764; Smeets WJAJ (1986) PMID 3540035; Henselmans JML et al. (1991-94) PMID 7916354; Hoogland PV, Vermeulen-VanderZee E (1993) PMID 8514912.
- Wang SR et al. (1983) PMID 6194859; Wang SR et al. (1985) PMID 3983614. Butler AB, Northcutt RG (1971).
- Rueda-Alaña E et al. (2025). Science (Paroedura picta pallial pathways).
- McDonald RP, Vickaryous MK (2018). PMC6018638.
- Landová E et al. (2025). Memory in leopard geckos in a Morris water maze task. Animals 15:2014. https://pmc.ncbi.nlm.nih.gov/articles/PMC12291964/ ; Landová E et al. (2023). Acta Soc Zool Bohem 86:97-117.
- Masseck O, Röll B, Hoffmann KP (2008). The optokinetic reaction in foveate and afoveate geckos. Vision Res 48:765-772. https://pubmed.ncbi.nlm.nih.gov/18234272/ ; Masseck & Hoffmann 2009, Ann NY Acad Sci 1164:430.
- Jagnandan K, Higham TE (2018). Neuromuscular control of locomotion is altered by tail autotomy in geckos. J Exp Biol 221:jeb179564. https://doi.org/10.1242/jeb.179564
- Lagorio et al. 2026; Platel 1976; Storks et al. 2023 (Anolis brain); Kojima 2021 (gecko photoreceptors); Zaaf 2001; Fuller 2011; Jagnandan 2014/2017; Probst 2023; Vollin 2021/2023; Frydlová 2026; Fernández-Lázaro 2023; Sakai 2024.
- Shein-Idelson M et al. (2016). Slow waves, sharp waves, ripples, and REM in sleeping dragons. Science 352:590-595.
- Fenk LA, Riquelme JL, Laurent G (2024). Central pattern generator control of a vertebrate ultradian sleep rhythm. Nature 636:681-689. https://doi.org/10.1038/s41586-024-08162-w
- Albeck N et al. (2022). Commun Biol. https://doi.org/10.1038/s42003-022-04261-4 ; Albeck N, Shein-Idelson M (2026). Commun Biol. https://doi.org/10.1038/s42003-026-10024-2
- Bergel A et al. (2026). Nat Neurosci 29:543-550. https://doi.org/10.1038/s41593-025-02159-y
- Norimoto H et al. (2020) Nature; Fenk LA et al. (2023) Nature; Hatori et al. (2025) PNAS https://doi.org/10.1073/pnas.2415929122 ; Yamaguchi et al. (2024) PNAS Nexus; Libourel PA et al. (2018) PLoS Biol.
- Fournier J et al. (2018) Neuron; Hemberger M et al. (2019) Neuron; Stein BE, Gaither NS (1981/1983); Fleishman LJ (1986); Leberstein et al. (2026) bioRxiv; Hammel HT et al. (1967) Science; Liu et al. (2006); Day LB et al. (2001); López JC (2003); Rodríguez F (2002).

**Tectum, superior colliculus, vision**
- Zylbertal A, Bianco IH (2023). Recurrent network interactions explain tectal response variability and experience-dependent behavior. eLife 12:e78381. https://elifesciences.org/articles/78381 (code: github.com/azylbertal/TectalLNP, MIT, MATLAB)
- Qian et al. (2025) bioRxiv 10.1101/2025.06.10.658856; Légaré et al. (2025) bioRxiv 10.1101/2025.08.08.669348.
- Hoy JL, Bishop HI, Niell CM (2019). Defined cell types in superior colliculus make distinct contributions to prey capture behavior in the mouse. Curr Biol 29:4130-4138. https://pmc.ncbi.nlm.nih.gov/articles/PMC6925587/
- Krizan J et al. (2024). Predation without direction selectivity. PNAS 121:e2317218121. https://doi.org/10.1073/pnas.2317218121
- Sharp SL et al. (2025). Neural dynamics in superior colliculus of freely moving mice. Cell Rep 44:116284.
- Förster D et al. (2020); Bianco IH, Engert F (2015); Del Bene F et al. (2010); Barker AJ, Baier H (2015); Fernandes AM et al. (2021); Henriques PM et al. (2019); Antinucci P et al. (2019, 2025); Evans DA et al. (2018); Gahtan E et al. (2005); Mearns DS et al. (2020); Johnson RE et al. (2020); Bolton AD et al. (2019); Lau, Fitzgerald & Bianco (2025).
- Wiederman SD, Shoemaker PA, O'Carroll DC (2008) ESTMD; Kühn/Farrow (2025); MingshuoXu/Small-Target-Motion-Detectors (Apache-2.0).

**Basal ganglia**
- Gurney K, Prescott TJ, Redgrave P (2001); Humphries MD, Gurney K (2002); Girard B et al. (2003, 2005); Prescott TJ et al. (2006). Neural Netw 19:31-61. PMID 16153803.
- Prescott TJ, Montes González FM, Gurney K, Humphries MD, Redgrave P (2024). Simulated dopamine modulation of a neurorobotic model of the basal ganglia. Biomimetics 9:139. https://pmc.ncbi.nlm.nih.gov/articles/PMC10967936/ (supplement: biomimetics-09-00139-s001.zip)
- Girard B et al. (2008). A contracting model of the basal ganglia. Neural Netw 21:628-641. ModelDB 124111, https://modeldb.science/showmodel?model=124111 (no licence).
- Gurney KN, Humphries MD, Redgrave P (2015); Frémaux N, Sprekeler H, Gerstner W (2013); Lindsey et al. (2025); Stephenson-Jones M et al. (2011); Grillner S (2013); OpAL*.

**Hypothalamus, drives, sleep models**
- Keramati M, Gutkin B (2014). eLife 3:e04811. Yoshida N et al. (2024) PNAS Nexus; Yoshida N, Kuniyoshi Y (2025) IEEE ICDL https://doi.org/10.1109/ICDL63968.2025.11204447 (homeostatic Crafter, MIT); Dulberg Z et al. (2023) PNAS; Grimbly S et al. (2026) arXiv 2608.04232; Tschantz A et al. (2022); MyoSuite 3CC-r constants.
- Laurençon H et al. (2024). Continuous time continuous space homeostatic RL. arXiv 2401.08999. Maroto-Gómez M et al. (2018). Sensors 18:2691. https://www.mdpi.com/1424-8220/18/8/2691 . Grand S et al. (1997) Creatures.
- Borbély AA (1982); Phillips AJK, Robinson PA (2007); Arcascope/circadian (MIT); kerkphil/AnimalSleep (GPL-3); Doya K (2002); Hobson JA, Friston KJ (2012).

**Brainstem and spinal cord**
- Rybak IA, Shevtsova NA, Markin SN, Prilutsky BI, Frigon A (2024). Operation regimes of spinal circuits controlling locomotion and the role of supraspinal drives and sensory feedback. eLife 13:RP98841. https://elifesciences.org/articles/98841 (code: github.com/RybakLab/nsm, GPL-3)
- Rybak IA et al. (2025) eLife 13:RP103504; Shevtsova NA et al. (2026) eLife RP107480.
- Rybak IA, McCrea DA (2006); Owaki D, Ishiguro A (2017); Suzuki S et al. (2021); Ijspeert AJ et al. (2007); Harischandra N et al. (2011); El Manira A (2026); Ausborn J et al. (2019); Caggiano V et al. (2018); Ryczko D (2024); Cabelguen JM et al. (2003); Alizadeh, Van Opstal (2022); SimonDanner/CPGNetworkSimulator (GPL-3); farmsim/farms_network (Apache-2.0); CPG-RL (2022); Puppeteer & Marionette (2023); Yang (2025).
- DeWolf T et al. (2016); Huebotter (2025).

**Spiking, benchmarks, validation standards**
- Nengo 4.1.0, https://pypi.org/project/nengo/ (GPL-2); nengo-dl, nengo-ocl.
- Tang G, Kumar N, Yoo R, Michmizos K (2020). PopSAN. CoRL 2020, arXiv 2010.09635; github.com/combra-lab/pop-spiking-deep-rl (MIT).
- Zanatta et al. (2024); Proxy Target (NeurIPS 2025); CaRe-BN (ICLR 2026).
- Feather J et al. (2025); Schrimpf M et al. (2020); Doerig A et al. (2023); Schaeffer R et al. (2022); Bowers JS et al. (2023); ZAPBench (Apache-2.0).
- Local benchmark scripts: `C:\Users\ziyad\AppData\Local\Temp\claude\C--Users-ziyad\4c01ec2a-ce6f-42df-b75f-e941604025e3\scratchpad\bench_nengo.py` (nengo_venv) and `bench_snn_torch.py` (run in `C:\Users\ziyad\GeckoBrain\.venv`).

**Repository and MuJoCo**
- github.com/12ziyad/NeuroGecko (Apache-2.0): `morphology/gecko_body_r.xml`, `envs/gecko_brain_env.py`, `envs/cpg_residual_controller.py`, `brain/drives.py`, `brain/vision_encoder.py`, `brain/bc_actor.py`, `brain/actor_critic.py`, `train/train_brain.py`, `docs/research/no_gecko_and_appearance.md`.
- MuJoCo 3.9.0 camera frustum: `src/engine/engine_vis_visualize.c` (mjv_cameraFrustum), `python/mujoco/rendering/classic/renderer.py`, github.com/google-deepmind/mujoco.