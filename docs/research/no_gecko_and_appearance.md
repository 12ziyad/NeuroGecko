# NeuroGecko Decision Document — Motion Without a Gecko, and What to Do About Skin

**Status:** architectural decision, supersedes the "kinematics-first / film a gecko" plan.
**Audience:** solo builder, CPU-only Windows laptop, $500 AWS, ~12 weeks remaining.
**Everything below is grounded in the research passes and checked against the primary sources. Proxies, inferences and my own estimates are marked inline.**

---

## Executive summary (read this if you read nothing else)

1. **The kinematics-first plan does not survive, and it did not deserve to.** Its two supposedly-downloadable pillars were checked and are false: XMAPortal's BROWN42 is Public Access "No" and BROWN38 is "Partial", both reporting *zero* public data collections; the Chong et al. 2022 "lizard videos" on Zenodo are 4,100 bytes of nine `.mat` files holding two scalars per species, for limb-reduced skinks swimming through sand. There is no public 3-D motion capture for any gecko on Earth. Filming your own was the only route that ever made that plan work, and you can't.

2. **A different strategy should now win: statistics-first.** The gait numbers you were going to extract from video already exist, measured, published, and mostly free, for *Eublepharis macularius* specifically. You do not need video and you do not need imitation. You need a gait-metrics logger, a reward that pins duty factor / footfall order / joint excursions to published values, and CMA-ES over your existing CPG parameters — **all of it at your currently locked 1.1888 Hz.** That is 3–6 weeks of work at $0 with no new dependency, no GPU, and no retraining of the walker from scratch.

3. **The goal has to change, because the premise behind it was wrong.** "Beat the 0.11 m/s locked-CPG walker" assumed you were far too slow. You are not. On the trackway data, 0.11 m/s is about **85% of the mean voluntary walking speed of a real leopard gecko** and sits comfortably inside the observed voluntary range. The defect is not speed. The defect is **stride length, limb excursion and posture** — at the correct cadence you are covering roughly 0.88 SVL per stride against a published 0.62–0.73 SVL, i.e. over-striding by something like 20–40%. Fixing that honestly will make the walk look far more like a gecko and may make it slightly *slower*. Say so up front rather than pretending speed is the metric.

4. **The frequency lock is load-bearing and must not be casually unlocked.** This was measured on your own checkpoint, not assumed. `rewards/gait_prior.py` hard-refuses any frequency outside 1.0–1.2 Hz; the residual's 92-D observation contains only `sin/cos(phase)` and no frequency or phase-rate channel; and rolled out deterministically, `models/v4_5b_speed_polish_1m/final.zip` loses **24% of its speed at 1.0 Hz, 43% at 0.9 Hz, 51% at 1.8 Hz and 83% at 2.4 Hz**, with returns collapsing by 69–98%. Unlocking the frequency is not a config change. It is a CPG rewrite, an observation-space change (which kills the saved checkpoint and its `vecnormalize.pkl`), a reward re-derivation, and a full retrain. **Plan around the lock, not through it.**

5. **On skin: the gecko's own appearance is cosmetic, near-zero effect on training, schedule it for week 11 and give it 3–8 hours.** The appearance of the *world and the prey* matters more — but not for the reason the earlier draft gave. The green colour mask is a logging diagnostic, not a reward term, and the food is not even a physical object in your MJCF. Recolouring the prey on its own changes nothing that trains. The real shortcuts are the privileged ground-truth food vector in the observation, the oracle-labelled supervision, and a reward computed entirely from ground-truth mouth-to-food distance. Fixing appearance without fixing those three is decoration.

---

# PART 1 — MOTION WITHOUT A GECKO

## 1. The full ranking table

Ranked roughly by (realism gained) ÷ (effort + cost). Days assume you, alone, on this laptop. Dollars are marginal spend beyond $0.

| # | Route | What it needs | Cost (days / $) | What it BUYS | What it CANNOT buy | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Reframe + gait-metrics logger** — score your current walk against published numbers before changing anything | Per-foot contact logging, joint-angle logging, stride/duty-factor/diagonality computation in MuJoCo | 0.5–1 d / $0 | Knowing *which* number is actually wrong. Likely finding: speed and stride frequency are already animal-plausible; the defect is stride length, limb excursion and posture | Any motion change by itself. It is instrumentation | **DO NOW — first** |
| 2 | **Jagnandan & Higham 2017, Sci Rep 7:10865 — Table 1** (CC BY 4.0, *your exact species*, n=10 animals of 36.3 ± 1.9 g and SVL 104.6 ± 2.1 mm — almost exactly your 36 g model) | 2–4 h to transcribe joint max/excursion pairs, duty factors, stride/step/stance | 0.5 d / $0 | Species-correct measured joint ranges (knee max 164.49 ± 0.93°, ankle max 139.65 ± 2.03°, femur depression 49.44 ± 2.49°, femur retraction 55.00 ± 2.18°), duty factors (HL 0.78, FL 0.70), stance times, stride lengths. Enough for an amplitude-matching reward *and* a validation harness | **Any time series.** No per-frame angles, no inter-limb phase beyond duty factor, no tail or spine trajectory, no GRF, no raw deposit anywhere. And its cells are not mutually consistent — see §3 | **DO NOW** |
| 3 | **McElroy et al. 2008, JEB 211:1029 — Table 1** (*E. macularius* measured directly, not taken from the literature — verified: the three literature-derived rows carry asterisks and this one does not) | Get the PDF — it is outside its embargo and free | 0.5 d / $0 | Hindlimb duty factor 72 ± 2.1% "walk" / 70 ± 0.7% "run"; limb phase 44 ± 1.1 / 43 ± 1.8 (near-trot, diagonal couplets, faintly lateral-sequence); speeds 0.24 and 0.29 m/s. This is your **footfall-order and duty-factor target** | Joint angles. Zero style information. And "walk"/"run" here are centre-of-mass *mechanics* classes, not gaits — for this species they are statistically identical (F₂,₇ = 0.36, P = 0.71), so do not build a reward that switches between them | **DO NOW** |
| 4 | **motion_imitation reference-motion FORMAT** (Apache-2.0) | Read the repo, copy the DeepMimic JSON schema | 1–2 d / $0 | A published file format for reference trajectories so you don't invent one; a portable imitation-reward structure | Any lizard data. It is a dog, parasagittal, no tail, no spine, 12 joints vs your 26 actuators | **DO NOW (format only)** |
| 5 | **Gait-timing / phase-clock reward, AT THE LOCKED FREQUENCY** (Walk-These-Ways swing-phase force penalty, Siekmann periodic reward, Kim barrier form) | Add a 3–4 term reward block; keep the CPG clock exactly as it is at 1.1888 Hz | 3–6 d + a warm-started retrain / $0 | Correct footfall order, correct duty factor, correct phase offsets, enforced with **no reference motion at all**. Precedent on a sprawling body: a salamander-like MuJoCo robot built this way went 38% further in 24 s than its Hildebrand-only baseline (150.4 vs 108.6 cm) | Joint angles, posture, sprawl, spine waveform. Note the papers you are borrowing from carry commanded frequency *inside* the command vector; you are deliberately taking the reward shape and leaving the frequency dimension out. Borrow the shape, never the training scale (150M–2.58B samples) | **DO NOW** |
| 6 | **CMA-ES fit of CPG parameters to published gait statistics** | 15–40 CPG scalars exposed; fitness = weighted squared distance from the published metric vector; head_cam **off** during optimisation | 7–14 d / $0 (maybe tens of $ AWS CPU) | The core of the new plan. Defensible claim: "statistically matched to published *E. macularius* gait metrics on N measures", every number citable to a primary source | Intra-stride waveform shape, GRF profiles, foot roll. You may **not** say "kinematically validated" | **DO NOW** |
| 7 | **"Pain" + cubic-effort reward terms** (Schumacher et al., iScience 2025, cheap tier) | Joint-limit penalty, GRF > 1.2 body-weight penalty, effort as a³ with an adaptive weight that ramps only after velocity is met | 3–5 d / $0 | Kills hyperextension and slam-down — the artefacts that read as "fake" instantly. Strongest published evidence of natural gait *with zero demonstrations* (43–73% of gait cycle inside human experimental SD bands) | The headline number came from a *musculoskeletal* model. **No gecko musculoskeletal model exists**; MyoSuite ships human models only. Do not attempt the expensive tier | **DO NOW (cheap tier only)** |
| 8 | **Nyakatura et al. 2019 Nature — Supplementary Data 1, file MOESM3 only** (skink, iguana, caiman, salamander) | Download MOESM3 xlsx from `static-content.springer.com`; 3–5 d to map onto your MJCF | 3–5 d / $0 | The **axial DOF the gecko paper does not report**: blue-tongued skink long-axis rotation 77.34° forelimb / 77.31° hindlimb, spine bending 20.75° / 19.41°, girdle counter-rotation, limb phase 0.44, duty factor 0.67, stride freq 1.36 Hz. Plus a genuine 21-point normalised stride cycle (salamander, in MOESM4) you can resample today | Lizard joint angles (only aggregate params per trial). Nothing at gecko body scale — skink ~300 g, caiman ~2 kg vs your 36 g. And three traps: MOESM4 is *Pleurodeles only*, the skink data is in MOESM3 alone; the LAR entries are per-trial excursion **ranges**, not instantaneous angles, with SDs of ~18.6°; and the paper's own gait-space convention uses **half** the spine-bending value | **DO NOW (for axial params, read §3d first)** |
| 9 | **Load-feedback inter-limb coupling** (Owaki/Ishiguro sprawling rule) | Replace the closed-form global clock with four independently integrated per-leg phases driven by local load + trunk torque | 3–6 weeks, not days / $0 | Genuinely the highest-credibility idea on the list. Built *for sprawling tetrapods*. Published: one parameter sweep produced spontaneous walk→trot→gallop with spine standing→travelling wave, matching *Dicamptodon* duty factor and diagonality, **no learning, no reference data** | It requires the frequency unlock (Owaki sweeps ω over ~0.95–4.14 Hz), a per-leg phase state your code does not have, a new observation layout, and a full retrain — and its emergent timing directly fights the prescriptive timing reward in item 5. **Probably does not fit in twelve weeks.** See §4b | **DEFER — scope it, don't start it** |
| 10 | **Symmetry + periodicity priors** (mirror-augment the PPO rollout buffer) | Hand-write the observation/action mirror permutation for 26 actuators + 5 tail segments | 1–2 d / $0 | Removes the limp/jitter that reads as "RL artefact" to a viewer. Essentially free polish | Cannot *guarantee* symmetry. And a real leopard gecko is **not** perfectly symmetric — it laterally undulates a heavy tail; over-enforcing symmetry suppresses exactly what makes it read as a gecko | **DO LATER** |
| 11 | **Energy / cost-of-transport as ONE weighted term** | Add `-τᵀq̇` at a small fixed weight | 0.5 d + retrain / $0 | Gait *selection* and speed-dependence; Froude scaling a biomechanist would check | Gait *form*. Two independent results say energy alone is wrong: viability (not CoT) drove gait transitions in Shafiee et al. 2024 — CoT actually *increased* after transitions; and Schumacher et al. found CoT/metabolic rewards produced kinematics *further* from real data | **DO LATER (as a term, never the objective)** |
| 12 | **2-D keypoints from YouTube gecko video** (SLEAP preferred over DeepLabCut on CPU; ~100–200 hand-labelled frames) | Source and crop clips (the slow part), label ~150 frames (~4 h), train a small CNN, write a foot-y-trace duty-factor script | 7–14 d / $0 | The one thing published stats don't give: **stride length vs speed**. This matters more than it used to: if *E. macularius* modulates speed by stride length rather than cadence, your locked 1.1888 Hz is defensible and the expensive frequency-unlock track can be dropped outright | Any 3-D. Foreshortening compresses out-of-plane limb sweep to near-zero, which is precisely the sprawled-gecko signal. No metric scale without a known length in frame. Reptiles are the *worst* class on Animal Kingdom's own pose benchmark (PCK@0.05 ~56 vs ~77 for birds) — expect to fine-tune, not zero-shot | **DO LATER (one weekend, as validation)** |
| 13 | **Fuller, Higham & Clark 2011, Zoology 114:104–112** (paywalled) | Email the corresponding authors, or pay | 1 d + 2–3 d to fold in / $0–60 | Footfall-instant posture anchors: hip height **36.0 ± 1.8% of total limb length** and 3-D knee **101.6 ± 1.8°** at footfall (vs *T. scincus* 48.7 ± 2.4% and 120.6 ± 3.9°) — and, behind the paywall, the actual femur-elevation numbers that justify the "hyper-sprawled" label | **Not what the earlier plan claimed.** The two headline figures are already free in the abstract, and "over a range of speeds" describes the sampling, not a speed-resolved result — the abstract reports no speed-dependence for either quantity. It is a two-point anchor at one instant, not a stride cycle, and it does not say whether ± is SD or s.e.m. Buy it for the femur-elevation numbers and the Methods, not for the two figures you already have | **DO LATER (buy for the femur data specifically)** |
| 14 | **Animal Kingdom reptile pose subset** | Microsoft Forms request; 1–2 weeks GPU fine-tuning | 7–14 d / $0 (+GPU) | Best available starting point for a *reptile-aware* 2-D detector; 23 keypoints map almost one-to-one onto your MJCF including three tail points | Motion. 2-D single frames scraped from documentaries — uncalibrated, unscaled, no stride continuity. "Reptile" includes snakes and turtles. 640×360 is low-res for distal joints. No licence stated | **DO LATER (only if route 12 needs a better detector)** |
| 15 | **Truebones Zoo — Komodo dragon / crocodile BVH** | Buy, open in Blender, retarget to 26 actuators + 5 tail segments, export DeepMimic JSON | 3–5 d / **$99 (pay-what-you-want)** | A smooth cyclic full-body sprawling trajectory with tail and spine, in a format DeepMimic/AMP eats directly | **Any claim to biological realism.** These are hand-keyed artist animations. A policy imitating them learns what an animator *believes* a lizard looks like. Licensing is contested (vendor says royalty-free; research mirrors say CC BY-NC-4.0; AnyTop withheld its copy pending "licensing clarification"). Body scale is a 70 kg monitor lizard — needs Froude rescaling, not copying | **SKIP** (do it only if you want a demo video and are willing to label it "artist animation") |
| 16 | **MoCapAnything V2** (arXiv 2604.28130) — category-agnostic monocular video → BVH for an arbitrary rig | **Day 1: upload one gecko clip to the free hosted Space, in a browser, no GPU.** Only if that works: local install, BVH→MJCF retargeter | 3–5 d timeboxed / **$0 to test**, ~$100 only if it survives | The one cheap lottery ticket worth holding, and cheaper than the earlier draft assumed. Code and V2 weights are **MIT**. The reference-species set is 72 species and is *not* large-bodied-only — it includes Ant, Cricket, Hamster, Rat, and crucially **"Comodoa" (Komodo dragon)**, a sprawling lizard, which is the correct topological match for a gecko. Body size in cm is nowhere stated as a constraint | No published lizard or gecko result exists anywhere — quality on a sprawled 20 cm animal is simply undocumented. Three licence carve-outs sit inside the MIT wrapper: RMBG-1.4 background removal is **CC non-commercial** (BRIA), and DINOv2-large + SAM 2.1 weights are Meta-licensed; fine for research, must be swapped before anything commercial. No CPU-only path is documented (training was 8×V100). And even a perfect BVH is kinematics — you still owe DeepMimic-style tracking RL on CPU | **DO NOW as a free browser test; timebox the local install to 3–5 days and $100** |
| 17 | **MIMIC-MJX** (MuJoCo-native imitation from 3-D keypoints) | 3-D keypoint trajectories as input — which is the whole problem | 14–21 d / $50–200 | The mature, MuJoCo-native, explicitly model-agnostic path from 3-D keypoints to a trained policy on *your* MJCF. Demonstrated on stick insect and worm, so not mammal-locked | It does **not** produce 3-D keypoints. It is the back half of a pipeline you are missing the front half of. Also MJX is GPU-oriented | **DO LATER — only if route 16 succeeds** |
| 18 | **Mimic2DM-style 2-D-only physics imitation** (reprojection-error reward) | Virtual camera in MuJoCo, project MJCF sites, reprojection reward, curriculum, camera intrinsics for an uncalibrated YouTube clip | 21–35 d / $0–300 | In principle, real joint-angle style from ordinary video | The authors are candid: single-view tracking "suffers from severe depth ambiguity resulting in artifacts like foot sliding", and their fix is aggregating **multiple views you cannot get**. The failure axis is limb abduction — exactly the axis that defines a sprawled gecko. Code release unconfirmed | **SKIP** |
| 19 | **RLWAV — video-classifier reward, no pose at all** | Port the idea to CPU MuJoCo; render the robot every 5 steps for the reward | 28–56 d / $300–500 | Proof that a purely perceptual reward crosses the animal→robot gap on real hardware | Fidelity. The authors' own words: the learned walk had "relatively straight legs and a high body posture — less stable than traditional walking poses". A 4-way action classifier can express "walk", never "walk like a leopard gecko". Needs 2,048 IsaacGym envs on an RTX 4090 | **SKIP** |
| 20 | **DeepLabCut SuperAnimal-Quadruped + FMPose3D** (monocular 2-D→3-D lift) | Fine-tune the 2-D detector on hand-labelled gecko frames; GPU | 14+ d / $100–300 AWS | In principle a per-frame 3-D skeleton | **No reptile in any training set of either stage** — verified. The 3-D lifter is trained on Animal3D: 3,379 images of 40 *mammal* species parameterised by SMAL, whose limbs are parasagittal by construction | **SKIP** |
| 21 | **SMAL family** (SMAL/SMALST/BARC/BITE/AniMer+/SAM 3D Animal) | — | 0 d / $0 wasted if skipped | Nothing | Hard structural mismatch, not a tuning problem. SMAL's shape space came from scans of lions, cats, tigers, dogs, horses, cows, foxes, deer, zebras, hippos. AniMer+ is explicitly scoped "Mammalia and Aves" | **SKIP** |
| 22 | **Template-free 4-D reconstruction** (BANMo / Lab4D / SLoMo) | 250–500 GPU-hours for your 4–8 min target | 42–84 d / $150–500+ per attempt | In principle everything: rig, joint angles over time, *and* an appearance model | Affordability. SLoMo's own paper: "eight hours on a computer with 8 NVIDIA GeForce RTX 3080 GPUs" per **minute** of video ≈ 64 GPU-hours/min. Your entire AWS budget on one uncertain attempt. Also uses a point-foot model — you lose the foot detail your 4 touch sensors are built around | **SKIP** |
| 23 | **XMAPortal / XROMM lizard studies** | — | 0 d / $0 | Nothing. Verified dead end | BROWN42 Public Access = **No**; BROWN38 = **Partial**; both report *0 Public Data Collections*. The only public reptile study is iguana *breathing* | **SKIP — verified** |
| 24 | **Chong et al. 2022 Zenodo "lizard videos"** | — | 0 d / $0 | Nothing | The videos do not exist. `animal experiments.zip` is 4,100 bytes: nine `.mat` files, 3,035 bytes total, two scalars per species. The 1.26 GB companion is *robot* footage | **SKIP — verified** |
| 25 | **Dryad/Zenodo gecko biomechanics deposits** | — | 1 h / $0 | Essentially nothing — duty factor and speed for a different, fast diurnal, pad-bearing genus (*Rhoptropus afer*) | No leopard-gecko biomechanics paper from any lab has ever deposited raw kinematics or video | **SKIP** |
| 26 | **Mammal pose benchmarks** (AP-10K, APT-36K, Animal-Pose, Animal3D, AcinoSet) | — | 0 d / $0 | Nothing for a gecko | No reptiles in any. Even the keypoint *schemas* are wrong: parasagittal topologies, no tail chain, no spine | **SKIP** |
| 27 | **Pleurobot / EPFL *Pleurodeles* cineradiography** | Email the lab | — | Would be the ideal AMP dataset (64 tracked skeletal points, 3-D X-ray) | No public release exists beyond the Orobates supplementary. Species is an amphibian with an even more sprawled, swimming-derived axial pattern | **SKIP** |
| 28 | **DUSt3R / MonST3R / VGGT 4-D geometry** | — | 0 d / $0 | Nothing | Output representation is fundamentally wrong: pointmaps, video depth and camera pose. No skeleton, no joint angles, no contacts | **SKIP** |
| 29 | **Mammal-mocap style priors** (dog mocap → retarget, AMP off a German Shepherd clip) | Retarget + DeepMimic/AMP training | 7–14 d / $0 (CC-BY-NC) | **Negative value** | Dogs are parasagittal; leopard geckos are sprawled with a laterally undulating heavy tail. A German-Shepherd style prior imports mammalian limb posture and *suppresses* the sprawl and axial undulation that define gecko gait | **ACTIVELY SKIP** |
| 30 | **SCONE / Hyfydy predictive-simulation toolchain** | Leave MuJoCo | 7 d eval / €995+/yr | A mature, validated optimisation loop and a credible objective formulation to *copy* | Human-only models, no quadruped, no reptile, **no MuJoCo support**. Hyfydy's cheapest non-commercial licence eats your entire budget | **SKIP (borrow the objective, not the tool)** |

---

## 2. The one recommended path, with first steps

**Name it "statistics-first, cadence-locked."** Keep the CPG. Keep the CPG *frequency*. Keep the PPO residual and its 92-D observation. Keep MuJoCo 3.9. Add measurement as a *reward* and as a *validation harness*, never as a trajectory to track.

The single biggest change from the earlier draft: **the frequency unlock has moved off the critical path.** It was priced at "3–6 days of plumbing". It is actually a CPG rewrite plus an observation-space change plus a reward re-derivation plus a from-scratch retrain, and the measured evidence (§4b) is that your existing walker degrades at *every* other frequency, including inside the band its own code declares legal. Everything below is designed to deliver most of the realism gain without touching it.

### Week 1 — Audit, instrument, and make appearance work safe (do not train anything)

1. **Build the gait-metrics logger.** Per-foot contact booleans from your 4 foot touch sensors; from those compute per-limb duty factor, stride period, stride frequency, footfall order, limb phase, and diagonality. Log all 26 joint angles and compute per-joint max and excursion over a stride. Log foot slip and body height. **This is the single most valuable artefact in the whole project** — it is simultaneously your reward's input and your paper's results table.
2. **Download and read the primary PDFs yourself.** Jagnandan & Higham 2017 (CC BY 4.0, open access), the 2014 companion (free author PDF), Foster & Higham 2012 (free author PDF, for the variable definitions), and McElroy 2008 (free, outside embargo). **Read the Table 1 captions and Methods, not just the tables** — §3 explains why that is not optional.
3. **Download the Orobates supplementary MOESM3** (not MOESM4 — that one is *Pleurodeles* only), plus `modernSpecies.csv` from the EPFL page. Cite, do not re-host: no licence statement exists on either, and Springer Nature reserves all rights. Commit a fetch script and a small table of derived constants instead of the files.
4. **Do the two-render split now** (Part 2 (a)) and set `group="4"` on the four `footzone_*` sites. Two hours, before any appearance work and before any vision training you intend to keep.
5. **Produce a one-page scorecard**: your current gait's 10–14 metrics beside the published ones, with disagreement between sources shown as a *band*, not a point.

### Week 2 — Gait-timing reward, frequency still locked

6. Move the CPG from "opaque oscillator inside the loop" to "commanded phase clock outside the policy" — but keep its frequency pinned at 1.1888 Hz. Expose duty factor and the three inter-limb phase offsets as the tunable parts.
7. Add the swing-phase force penalty: `r = Σ_foot [1 − C_foot^cmd(θ_cmd, t)] · exp(−|f_foot|² / σ)` at a small negative weight, plus a stance-phase foot-velocity penalty. Use the **relaxed logarithmic barrier** form so stance can drift within a band — your target duty factor of 0.70–0.78 is *high*, and papers tuned for 0.5–0.6 mammalian trots will over-penalise you.
8. Set the targets from the literature: **hindlimb duty factor 0.72–0.78, forelimb ~0.70, limb phase 0.43–0.44 (near-trot, diagonal couplets, faintly lateral-sequence).**
9. Retrain the residual — but **warm-start from the existing checkpoint**, because the observation space is unchanged. This is the whole point of leaving the frequency alone. Expect a temporary dip; expect it to recover in a fraction of the samples a from-scratch run would need.

   *One thing the audit turned up that you should know before you do this:* freezing or 90°/180°-shifting the two phase observation dimensions costs the current policy only ~16% of return and no measurable speed. The residual is very nearly an open-loop, cadence-calibrated feedforward that ignores its own clock. A swing-phase force penalty is therefore a genuine change in what it has to learn, not a cosmetic addition — budget the retrain properly.

### Weeks 3–4 — CMA-ES fit to the statistics vector

10. Expose 15–40 CPG scalars: per-joint amplitude and offset, inter-limb phase offsets, sprawl/abduction bias, spine standing-wave amplitude and wavelength, tail undulation gain, swing/stance asymmetry, PD gains. **Not frequency.**
11. Fitness = weighted squared distance from the published metric vector, plus penalties for foot slip and non-periodicity. **Encode at most one member of the {stride length, cadence, speed} triangle at a time** — §3 explains why they are mutually unsatisfiable in the source itself. My recommendation: fit stride length and duty factor, and use speed only as a range check.
12. **Disable head_cam rendering during optimisation** or it will dominate your step cost entirely.
13. Add the axial targets from Orobates MOESM3 that the gecko paper does not report: long-axis rotation ~77° (blue-tongued skink; **proxy — different species, ~8× the body mass**, and it is a per-trial excursion *range* with an SD of ~18.6°, so treat it as a loose envelope not a setpoint) and spine bending. On spine bending, **use ~10.4° forelimb / ~9.7° hindlimb, not 20.75° / 19.41°**, if you want to reproduce the paper's own gait-space convention — the published caption states that half the value is used, making it relative to the pectoral girdle rather than to an inertial frame. Feeding the full value in double-counts spine bending by 2×.
14. Run pycma over parallel rollouts on your CPU cores. MuJoCo is fast enough: a 27-DoF humanoid simulates ~4000× real-time on one thread, so a 10 s episode is a fraction of a second.

### Week 5 — Naturalness terms

15. Add joint-limit "pain", GRF > 1.2 body-weight penalty, cubic effort `a³` with an adaptive weight that only ramps once the velocity target is met, and some series compliance. 3–5 days, kills the artefacts that read as fake.

### Weeks 5–7 — The vision work (Part 2 (b))

16. W1–W4 in the Part 2 table, in that order. W1 is already done in week 1. W2 (floor texture) is the largest single realism gain in the project and costs 1–3 hours.

### One weekend, whenever it fits — the video sanity check

17. SLEAP (lighter than DeepLabCut on CPU) on 100–200 hand-labelled frames of YouTube pet-gecko footage. Precedent for the exact workload exists: a published lizard-locomotion study labelled ten frames per video with 21 midline points plus 4 per limb. **Purpose: measure stride length vs speed.** This is now the highest-leverage weekend on the list, because if the animal really modulates speed by stride length rather than cadence, you can delete the entire frequency-unlock track from the schedule and say why. Expect to discard 90%+ of clips. Measuring numbers off someone's video for private research is one thing; redistributing frames or a model containing them is another — get advice before publishing anything containing the footage.

### Day 1, free — the lottery ticket

18. Upload one gecko clip to the MoCapAnything V2 hosted Space in a browser, with **Comodoa (Komodo dragon)** selected as the reference species. Zero cost, zero install, no GPU. If the sprawling-gait rotations come back as garbage there, a local install will not save it and you have spent an afternoon. If they look plausible, *then* spend the 3–5 days and the ~$100, and MIMIC-MJX is the MuJoCo-native back half.

### Explicitly out of scope for these twelve weeks

19. The frequency unlock and the Owaki load-feedback rule. Scope them, cost them, write down what they would require (§4b), and do not start them. If the video sanity check says stride length carries the speed modulation, delete them.

---

## 3. Honest problems with the numbers you are about to optimise toward

**This section is the reason I told you to read the table captions yourself.**

### 3a. Jagnandan & Higham 2017 Table 1 does not close arithmetically — and the reason is printed in its own caption

From the "Original" (intact-tail) column: hindlimb **stride length 0.62 ± 0.15 SVL**, **step length 0.06 ± 0.00 SVL**, **stance 0.64 ± 0.03 s**, **duty factor 0.78 ± 0.01**, **speed 1.23 ± 0.16 SVL/s**. Animals: SVL 104.6 ± 2.1 mm.

- Stance 0.64 s ÷ DF 0.78 → stride period 0.82 s → **1.219 Hz**.
- Speed 1.23 SVL/s ÷ 1.219 Hz → stride length **1.01 SVL**.
- The table says **0.62 SVL**. Run it the other way: 0.62 × 1.219 = 0.756 SVL/s ≈ **0.079 m/s**, 38.6% under the stated mean speed.

That gap is real and it is systematic — every limb × treatment cell fails, by 34% to 62%. But **the earlier draft's explanation was wrong, and its fear was wrong.**

The explanation is in the caption and Methods: *"Means + residuals (± s.e.m.) for each variable are given… Asterisks indicate variables that had a significant relationship (α ≤ 0.10) with speed"*, and *"The effects of speed… were removed by regressing the variables against body speed."* Stride length, step length, stance time and duty factor are all asterisked — they are speed-detrended constructs. Speed is not asterisked — it is a raw mean. **No `speed = stride × frequency` identity is expected between them.** They are not from different trials; they are from different statistical treatments of the same trials.

The fear — that 0.62 might really be a step length, a different normalisation, or a different condition — is refuted. Step length is its own separate row in the same column (0.06 SVL), and the cited methods source defines both explicitly: stride length is the 2-D distance over a complete stride cycle, step length the distance travelled during stance, both standardised to SVL.

**Two consequences that do change what you build:**

- **The over-striding diagnostic does not invert.** 0.62 SVL is the best-corroborated cell in the block: the forelimb row in the same column gives 0.63, the restricted and autotomized hindlimb columns give 0.72 and 0.70, and the same lab's 2014 companion study of the same species gives ~0.73 hindlimb / ~0.70 forelimb. So **0.62–0.73 SVL is this lab's repeated measurement for walking *E. macularius***, and the diagnostic stands.
- **The genuinely broken cell is step length, not stride length.** By the paper's own definition, step length must equal duty factor × stride length = 0.78 × 0.62 = 0.48 SVL. The table prints 0.06 — off by roughly 8×. (0.06 m ÷ 104.6 mm = 0.57 SVL, so a metres-versus-SVL slip is the likely cause, but that is a guess.) There is no published erratum. **Do not use the step-length row for anything.** The second-most suspect cell is the 2017 stance time: 0.64 s hindlimb, against 0.41–0.43 s in the 2014 study of the same species at comparable speeds.

**Practical rule: never put more than one member of {stride length, stance time/cadence, speed} into a single fitness function.** They cannot all be satisfied because they are not all the same kind of number.

### 3b. The two "disagreeing" speed sources do not actually disagree

The earlier draft framed Jagnandan & Higham (≈0.129 m/s) against McElroy (0.24 / 0.29 m/s) as a 2× conflict to be resolved with a band. That framing is wrong. The distributions overlap and the studies measure different constructs.

| Source | Protocol | Speed |
|---|---|---|
| Jagnandan & Higham 2017 | 1.0 × 0.13 m level trackway, sandpaper, room ~30 °C, **no speed-eliciting stimulus described** (the tail was the experimental variable, so they could not pinch it) | Intact-tail mean **1.23 ± 0.16 SVL/s ≈ 0.129 m/s**; autotomized mean 1.85 SVL/s ≈ 0.194 m/s (treatment effect not significant, P = 0.060); **observed range 0.59–3.36 SVL/s ≈ 0.062–0.352 m/s** |
| McElroy et al. 2008 | 5.2 m racetrack toward a hide box, force plate at 3–3.6 m, body temp ~26–30 °C, **speeds deliberately induced** — *"gently pressing on the tail or hindlimb"* for fast, tapping/hand-waving for medium, unstimulated for slow | "Walk" **0.24 m/s** (0.20–0.27), "run" **0.29 m/s** (0.29–0.32) |
| Zaaf et al. 2001 | level | **0.2–1.1 m/s** — the full range walk through sprint, not a walking target |

McElroy's 0.24 and 0.29 m/s convert to ~2.3 and ~2.8 SVL/s, both **inside** Jagnandan & Higham's own observed 0.59–3.36 SVL/s range, in its upper third. The 2017 geckos reached ~0.35 m/s, faster than any McElroy value. There is nothing to reconcile: a 1 m runway with no prodding versus a 5.2 m runway with tail-pressing is enough on its own to move a mean by 2× within one species.

Three further cautions on the McElroy numbers: they rest on roughly 4 running and 6 walking strides; "walk" and "run" are centre-of-mass mechanics classes (phase shift 135–180° vs 0–45°), not gaits, and for this species they are statistically identical (F₂,₇ = 0.36, P = 0.71) — it is one high-duty-factor trot reported twice; and the printed run entry "0.29 (0.29–0.32)" has a mean equal to its own minimum, while the Discussion describes the same running band as 0.16–0.24 m/s. Treat 0.29 as unreliable.

**Write it as one consistent literature, not a conflict.** Level-trackway *E. macularius* occupies roughly **0.06–0.35 m/s**. The voluntary, unstimulated central tendency is **~0.13 m/s**. Stimulus-induced, mechanics-classified means land at **~0.22–0.29 m/s**. Label every figure by its construct.

### 3c. What that means for "beating 0.11 m/s" — the goal has to change

Your 0.11 m/s is **~85% of the mean voluntary walking speed** and sits near the middle of the observed voluntary distribution. Against the induced means it is 38–46%, but a human had to touch the animal to get those. There is no 10× gap. There is barely a gap.

Now do the stride arithmetic, which is where the actual defect lives. Your animal is 36 g; the 2017 animals were 36.3 ± 1.9 g at SVL 104.6 ± 2.1 mm, so assuming your MJCF's SVL is near 105 mm is fair (**check it — this is an assumption, not a measurement**). At 1.1888 Hz, 0.11 m/s means **0.0925 m per stride cycle ≈ 0.88 SVL**. Published: 0.62–0.73 SVL. **You are over-striding by roughly 21–43% at approximately the correct cadence.**

And here is the uncomfortable corollary you should state before a reviewer does: at 1.1888 Hz, published stride lengths imply **0.077–0.091 m/s** — *slower than you are now*. You cannot simultaneously match published stride length, hold the published cadence, and go faster. That is the same mutually-unsatisfiable triangle that lives inside Table 1 itself. Something has to give, and the honest resolution is that the paper's stride lengths are speed-detrended constructs while its speed is a raw mean.

**So: stop optimising for speed.** The claim worth earning is that the walk has the right support fraction, the right footfall order, the right per-stride ground coverage and the right limb excursion — at a cadence and speed already inside the animal's voluntary range. My estimate for where the recommended path lands is **0.11–0.14 m/s with substantially better gait form**, and I am marking that as an estimate, not a published number. If speed goes down slightly while stride length and excursion move onto the published values, that is a win, and the scorecard will show it as one.

### 3d. Three traps in the Orobates data

- **MOESM4 contains no skink data.** It is *Pleurodeles* validation data only. The skink long-axis rotation and spine bending figures live in **MOESM3** alone.
- **The spine-bending values are convention-dependent.** MOESM3 reports 20.75° / 19.41°; the paper's own gait-space uses **half** of that, relative to the pectoral girdle rather than an inertial frame. The EPFL `modernSpecies.csv` confirms this — its skink values are 10.36 / 9.86.
- **`modernSpecies.csv` and MOESM3 disagree on skink hindlimb long-axis rotation** by ~31° (45.82 vs 77.31). The CSV value is numerically identical to MOESM3's skink hindlimb *retraction* mean, so one of the two has a column mix-up; the same forelimb/hindlimb inconsistency appears for caiman. **Treat MOESM3 as authoritative and the CSV as plot geometry only.** They do not corroborate each other.

---

## 4. Does the kinematics-first plan survive? Direct answer.

### 4a. No, not as written

"Film a real gecko for 4–8 minutes of 3-D-tracked walking, retarget it, train DeepMimic then AMP" required an animal you don't have or a public dataset that does not exist. Every substitute was checked and every one fails on a different axis:

- Public gecko 3-D data: does not exist anywhere (verified across Zenodo API, Dryad, figshare, XMAPortal).
- Substitute-species 3-D data: exists (Orobates, Truebones) but is either aggregate parameters per trial, or an artist's animation, or a 2 kg animal.
- Video→3-D: every mainstream method is mammal-trained by construction (SMAL, SuperAnimal, Animal3D), or priced at ~64 GPU-hours per minute of footage (SLoMo), or has a named unfixed single-view failure on exactly your critical axis (Mimic2DM). MoCapAnything V2 is the one exception worth a free browser test, and it has never been shown on a lizard.

**A different strategy wins.** *Statistics-first, cadence-locked.* It replaces "match a trajectory" with "match a measured distribution", which is (a) achievable with data that is free and mostly CC BY, (b) about your exact species, (c) implementable in your locked stack on a CPU laptop, and (d) more defensible in a write-up than a retargeted animation or a hallucinated monocular lift would ever be.

**The claim you earn is: "statistically matched to published *Eublepharis macularius* gait metrics on N measures."** The claim you do **not** earn is "kinematically validated." Intra-stride waveform shape, ground-reaction-force profiles and foot-roll are out of reach without force plates or 3-D tracking, from *any* method on the list. Say so in the write-up before a reviewer says it for you.

### 4b. And the frequency unlock does not survive either — this was measured

The earlier plan assumed unlocking 1.1888 Hz was a 3–6 day plumbing job that left the trained residual intact. It doesn't. Five independent things break, three of them at code level and one measured on your own checkpoint.

1. **A hard guard forbids it.** `rewards/gait_prior.py` sets `frequency_range_hz = (1.0, 1.2)` and raises `ValueError("frequency_hz must stay in the real-video 1.0-1.2 Hz range.")`. Owaki's demonstrated range is roughly 0.95–4.14 Hz, so this is not a parameter change — the guard, the frozen dataclass and every caller must be rewritten.
2. **The residual cannot observe frequency, and adding it kills the checkpoint.** The observation is 92-D (32 qpos + 32 qvel + 3 + 3 + 4 feet + 2 belly + 3 up + 3 gyro + 3 vel + 5 task + **2 phase**). The only gait-clock information is `sin(φ)`/`cos(φ)`; there is no frequency or phase-rate channel. Adding one changes the input width, so `models/v4_5b_speed_polish_1m/final.zip` (policy and value nets both 256×92) and its `vecnormalize.pkl` cannot be loaded. Retrain, not patch.
3. **It was never trained at any other frequency.** `FREQ_HZ = 1.1888` is a module constant in `envs/cpg_residual_controller.py`; `GeckoWalkEnv` never overrides it; `train/train_walk_ppo.py` has no gait-frequency argument and no domain randomisation anywhere.
4. **Performance collapses off 1.1888 Hz — measured, deterministic rollouts, 2 seeds, 400 control steps:**

   | CPG frequency | Mean forward speed | vs baseline | Mean return |
   |---|---|---|---|
   | **1.1888 Hz (as trained)** | **0.1034 m/s** | — | **+876** |
   | 1.0 Hz (inside the code's own legal band) | 0.0785 | −24% | +271 (−69%) |
   | 0.9 Hz | 0.0589 | −43% | +18 (−98%) |
   | 1.4 Hz | 0.0943 | −9% | +397 (−55%) |
   | 1.8 Hz | 0.0502 | −51% | −323 (sign flip) |
   | 2.4 Hz | 0.0176 | −83% | −599 |

   Zero-action control at every frequency: −0.004 to +0.009 m/s. The open-loop CPG base does not walk at *any* frequency — all forward motion is the residual's, and it is the residual that degrades. Forward speed is a physical measurement independent of reward weights, so the 24–83% loss stands regardless of which reward config was used for that run.
5. **The reward is calibrated in absolute units to that cadence.** The training script's final block hard-codes `target_speed=0.125`, `speed_floor=0.085`, `slow_speed=0.112`, `speed_track_sigma=0.035` — m/s targets that only correspond to a sensible stride length at 1.1888 Hz. At 1.8 and 2.4 Hz the achieved speed falls below `speed_floor`, so the gait and speed terms sit in permanent penalty.

**And Owaki needs a structure the code does not have.** Owaki & Ishiguro's rule is `φ̇ᵢ = ω − σ·Nᵢ·cos(φᵢ)` — four *independently integrated* per-leg phases with no predefined coupling, in which ω is the gait-selecting bifurcation parameter. NeuroGecko computes `φ = (t · freq + off) % 1.0` in closed form from one global clock with constant per-limb offsets, and `gait_prior.py` derives all four target contacts from a single global cycle fraction. Replacing that breaks the single global `sin/cos` phase observation (one pair cannot represent four decoupled phases), the target-contact array, the contact-match term and the anti-phase check.

The literature agrees on what it would take: CPG-RL puts the CPG states including `θ̇` in *every* observation variant it tests, including the deliberately minimal one; Walk-These-Ways carries commanded stepping frequency inside the behaviour command vector (trained 1.5–4.0 Hz) and computes the reward's contact schedule *from* it; Shao et al.'s phase-only controller works because all gait periods are trained together in one policy; and CPG-Actor reports roughly 66× higher reward when CPG parameters are tuned jointly with the feedback network than when the CPG is held fixed. Residual-RL's founding paper never claims or tests invariance of the learned residual to changes in the hand-designed controller it corrects.

**There is also a design conflict nobody flagged:** Owaki's whole contribution is that timing *emerges* from local load feedback, while a prescriptive gait-timing reward scores the robot against an externally imposed contact schedule. Run both and the reward penalises exactly the load-driven phase modulation the Owaki term exists to produce. Pick one, or make the timing reward track the CPG's *realised* phase rather than a fixed clock.

**Honest sequencing, if you ever do it:** (a) rewrite the CPG from a closed-form time function into four integrated per-leg phase oscillators; (b) replace the 1.0–1.2 Hz guard and the global-clock contact schedule; (c) extend the observation with per-leg phase and phase-rate, accepting the 92-D checkpoint is dead; (d) re-derive the reward's absolute speed constants as functions of ω; (e) retrain from scratch with frequency randomisation. Steps (a)–(d) might fit in a week of plumbing. Step (e) is the schedule driver, and every published variable-frequency locomotion policy I could find was trained with massively parallel GPU simulation. **On a CPU laptop this does not fit in twelve weeks alongside everything else.** Defer it, and let the weekend video check tell you whether you need it at all.

---

## 5. Answer to the specific practical question

> *Is 2-D gait timing from ordinary video (which IS obtainable) plus published gait statistics enough to beat the current 0.11 m/s locked-CPG walker?*

**The published statistics do nearly all the work — but "beat 0.11 m/s" is the wrong target, and the honest answer is that you should stop aiming at it.**

0.11 m/s is already ~85% of a real leopard gecko's mean voluntary walking speed and sits inside its observed range. What the pair actually buys you is *form*:

- Correct footfall order (near-trot, diagonal couplets, limb phase 0.43–0.44)
- Correct duty factor (hindlimb 0.72–0.78, forelimb ~0.70 — an almost-continuously-supported walking trot, which is very specific and probably *not* what your current CPG produces)
- A stride length target of 0.62–0.73 SVL against your current ~0.88 SVL — the clearest single defect in the walk
- A defensible, correctly-labelled speed picture (voluntary ~0.13 m/s, elicited 0.22–0.29 m/s, full observed range 0.06–0.35 m/s) instead of an inflated sprint target
- Joint excursion envelopes for all 8 fore+hind DOF, plus footfall-instant posture anchors (hip height 36.0 ± 1.8% of limb length, 3-D knee 101.6 ± 1.8°)
- Spine and long-axis-rotation amplitudes (Orobates skink, proxy, halved for the SGS convention)
- Stride length as a function of speed (video only) — which also tells you whether the frequency lock is a real constraint or a non-issue

**The hard ceiling, stated plainly:** no 2-D-derivable signal will ever give you gecko-correct joint angles. *E. macularius* is "hyper-sprawled" — it holds the femur far more elevated than other studied lizards, and its hip sits at only ~36% of total limb length at footfall against ~49% for the comparison species. That elevation is an out-of-plane rotation. A single side-view camera compresses it to nearly zero. So plan on a walker that steps at the right times, in the right order, with the right support fraction, with the right ground coverage per stride, at a speed inside the animal's real voluntary range — **and still swings its limbs in a way a herpetologist would flag.** That is a large, defensible improvement. It is not imitation, and you should not describe it as imitation.

---

# PART 2 — SKIN, COLOUR, APPEARANCE

## The direct answer

**Does appearance affect training?** Split the question, because the two halves have opposite answers.

**(a) The gecko's own skin, colour and pattern: essentially no effect. Cosmetic. Week 11. 3–8 hours.**

**(b) The appearance of the world and the prey: it matters — but far less than the three ground-truth shortcuts sitting behind it. Fix the shortcuts first, or the appearance work is decoration. Weeks 5–7.**

---

## (a) The gecko's own appearance — cosmetic

### Why it cannot matter

Three independent mechanisms, all verified:

1. **MuJoCo documents visual assets as physics-inert.** `compiler/discardvisual` deletes every material, every texture and every `contype = conaffinity = 0` geom, and the docs state the compiled model "will have exactly the same dynamics as the original model." Your `class="visual"` geoms are already `group=1, contype=0, conaffinity=0, mass=0` — correct structure, already physics-free.
2. **Skins are defined as purely visual** — they do not collide, add mass, or appear in contact.
3. Nothing about the gecko's colour enters the observation, the reward, the CPG, or the PPO residual.

### The three narrow leaks — and they are about GEOMETRY and RENDER FLAGS, not texture

1. **Not the snout.** The earlier draft said a sliver of the snout enters the frame at `fovy=120`. It does not, and the arithmetic that suggested it was 1-D. The snout ellipsoid does overhang the camera by 1.16 mm in +x (x_max 0.03074 vs camera 0.02958), but the camera sits 1.87 mm *above* the ellipsoid's top and looks along body +X: the closest-to-visible surface point is **81.1° below the optical axis**, against a 60° half-angle. You would need `fovy ≈ 162°`. And it is moot anyway — the near clip plane is `stat.extent × visual.map.znear = 0.28 × 0.01 = 2.8 mm`, so the entire 1.16 mm protrusion is inside it. A segmentation render over 625 poses spanning the full head and neck joint ranges shows the snout, head and eye geoms in **zero** poses, 0 pixels at 64×64 and 0 at 480×480. (The head *collision* capsule does reach 10.4 mm ahead and the camera is inside it, but it is `class="collision"` → group 3, and the default `MjvOption` draws groups 0–2 only.)

   **What IS self-visible is the trunk, neck and forefeet.** The same segmentation sweep shows `trunk_anterior`, `neck` and `manus_L`/`manus_R` entering the frame once the head is pitched **down** *and* combined head+neck yaw exceeds roughly 50°: 0 px at 40°, 24 px at 50°, 409 px at 60°, up to **675 of 4096 pixels (16.5% of the observation)** at the joint limits. Pure yaw alone gives nothing; pure pitch alone gives nothing; it needs both — which is exactly the posture a foraging policy adopts when turning toward food. So the "no green anywhere on the animal" rule is genuinely mandatory, and the constraint belongs on the **trunk, neck and forefoot** geoms, not the head.

2. **Self-shadow.** `light/castshadow` defaults true and you set `shadowsize=2048`. The gecko casts a real shadow onto the ground in front of itself, inside the head_cam frame. Body *shape* drives that; skin texture does not.

3. **Green sites — not firing, but not for the reason anyone assumed.** `envs/gecko_brain_env.py:314` calls `update_scene(self.walk_env.data, camera="head_cam")` with **no `scene_option`**, so the policy's camera renders all default-visible geom groups *and all sites* — including four `footzone_*` sites at `rgba="0.2 0.8 0.3 0.16"` (group 0, and the default `sitegroup` is `[1,1,1,0,0,0]`, so they *are* drawn). The food detector is `mask = (g>120) & (g>r+30) & (g>b+30)`.

   Measured, not calculated: across **720 real rollout frames** (multiple control modes, multiple episodes, driven by the trained walker), the footzone sites changed **zero pixels** and tripped **zero mask pixels**. The food signal is not corrupted and prior vision results are not suspect on this ground.

   But the estimate of "g ≈ 102, about 20 units under threshold" was wrong, and wrong in the dangerous direction. Where a footzone site actually does render, the measured blended green averages **~130** with a maximum of 224, and **45% of covered pixels trip the full three-part mask.** The alpha-blend model was fine; the assumed background was too dark. There is **no colour headroom at all.** What is saving you is purely geometric: the camera is at the nose looking forward and the feet are behind and below it. In poses with the trunk held level, the closest a footzone came to the optical axis was **63.8°** — only ~3.8° outside the frustum edge — and because the square 64×64 frustum's diagonal half-angle is ~67.8°, sites *did* appear in image corners in 65 of 600 extreme-joint poses and tripped the mask in 15 of them. With trunk pitch and roll also perturbed, the worst frame produced **170 mask pixels = 0.0415 of the frame, 3.5× the food-visible saturation threshold** — a full false "food dead ahead" signal.

   **Do not bank a 4° margin.** Any future change to `fovy`, camera placement, head/neck range, or trunk pitch flips this silently. Also note `utils/dump_brain_camera.py` checks `g>150 & r<80 & b<80`, which is stricter than the production mask and would never have caught this class of leak.

### THE fix — 2 hours, and it is still the highest-value item in this document

Build **two `MjvOption` objects** and pass them to `update_scene`:

- **Policy camera:** `geomgroup[1] = 0` (visual body off), `sitegroup[*] = 0` (sites off), shadows off.
- **Third-person video camera:** everything on.

Same model, same process, two renders. This **permanently decouples the demo from the experiment**: after this, no amount of appearance work can perturb the policy's input distribution or invalidate a training run.

And take the two-second belt as well as the braces: give the four `footzone_*` sites `group="4"`. The default `sitegroup` renders groups 0–2 only, so they vanish from every render, while the touch sensors — which read site *geometry*, not colour — keep working unchanged.

Separately, `compiler/discardvisual="true"` gives a smaller, faster `mjModel` with identical dynamics for any **locomotion-only** stage (it is all-or-nothing and discards all textures, so not for the vision stage).

### The concrete MuJoCo pipeline for the gecko's skin — recommended version

**Cube-mapped spot texture on the existing primitives. No mesh. No Blender. Works on MuJoCo 3.9 today.**

MuJoCo 3.9's **default** texture type is `cube` (verified in `mjs_defaultTexture`, `src/user/user_init.c`), and cube mapping "shrink-wraps a texture cube over an object" by ray-casting from the object centre — **it requires no UV coordinates.** At source level, `settexture()` in `src/render/classic/render_gl3.c` enables `GL_TEXTURE_CUBE_MAP` with object-linear texgen and is called for every geom with a material *before* the geom-type switch, with no geom-type gate. Capsules and ellipsoids go down the identical path. The official `model/humanoid/humanoid.xml` in 3.9.0 does exactly this on capsule geoms, so the path is proven rather than theoretical. (3.9.0 also ships only the `classic` and `noop` renderers, so there is no alternate path where this might be unimplemented.)

1. Write ~40 lines of numpy/PIL that generate a square PNG: buff-yellow base, Poisson-disc-sampled dark ellipses for the adult spotted phase (or horizontal bands for juvenile). This is biologically defensible, not a cheat: adult leopard-gecko spots are formed by melanophores alone, while juvenile banding requires melanophore–iridophore interaction, and the bands break into spots within the first few months as iridophores disappear.
2. Add to `<asset>`: `<texture name="gecko" type="cube" file="gecko_skin.png"/>`, and point the existing (currently unused) `skin`/`spot` materials at it with `texuniform="true"` to normalise scale across differently-sized segments.
3. **Palette rule, non-negotiable: yellow / buff / brown / black only. No green, ever.** This is also what the biology says — melanophores are black/brown, xanthophores red/yellow, iridophores reflective. There is no green in a leopard gecko. And per the self-visibility finding above, any green on the trunk, neck or forefeet is one head-down turn away from being eaten by the food detector.

**Cost: 3–8 hours, $0, zero code changes to the learning loop, no retraining.**

Four honest caveats, none fatal:

- **It is four lines, not two.** `<texture>` plus `<material>` is two, but a texture only reaches the renderer through a material a geom references — so you also need `material="…"` on each gecko geom, or a `<default>` class carrying it.
- **Use a PNG file, not `builtin`.** There is no "spot" builtin: `builtin` is limited to `{none, gradient, checker, flat}`, and the only procedural speckle is `mark="random"`, whose dots are exactly **one pixel** — at a typical `width="512"` they render as sub-pixel aliasing noise, not spots. If you ever do use a procedural texture, note that `width` is **mandatory** (it defaults to 0 and the compiler throws "Invalid width of builtin texture"), and `height` is ignored for cube type and forced to `6 × width`.
- **Cube mapping is centre-projected, not conformal.** Capsules and ellipsoids are drawn as unit primitives then non-uniformly scaled, so with `texuniform="false"` (the default) round dots stretch into ovals along an elongated tail or limb. `texuniform="true"` multiplies by the geom size instead. The docs' prose for this attribute reads inverted relative to the code path, so treat it as a knob to try both ways rather than a spec, and budget one tuning pass. Each of your five tail capsules also gets its own shrink-wrapped copy, so patterns will not be continuous across segment joins. It looks right in motion at video resolution and wrong in a close-up still.
- **`gridsize` defaults to `1 1`,** repeating the same image on all six cube faces, so structured patterns show mirrored discontinuities across the projected cube edges. Irregular spots hide this well; anything regular will not.

If you want a photo-derived texture instead: Wikimedia Commons `Category:Eublepharis_macularius` has 114 files, licences are **per-file** (commonly CC BY-SA, which is share-alike and would infect your texture file). Most captive-gecko photos are selectively-bred morphs (Blizzard, Bell Albino, Lemon Frost), not wild type — filter deliberately. CC0 libraries like ambientCG are a dead end for this specifically; a search of its 2000+ materials returned nothing reptilian.

### What NOT to do for the gecko's skin

**Do not upgrade MuJoCo for nicer UV mapping.** Explicit texture coordinates on built-in primitives (Plane, Box, Sphere, Ellipsoid, Capsule, Cylinder) landed in **MuJoCo 3.12.0, released 20 August 2026** — three minor releases and about three months ahead of your locked 3.9.0 (27 May 2026). The changelog files it under **Breaking API changes**: 2-D textures on primitives now use canonical UV parameterisations rather than projecting onto the x,y plane, and finite planes re-anchor textures to the bottom-left corner instead of the centre, which phase-shifts your procedural checker floor — i.e. it perturbs **every head_cam pixel your CNN has ever seen**. 3.12.0 also removes `mjData.efm_L_*`, flips the `bvactive` default, changes mocap weld-group semantics, drops the custom binary texture format and redesigns `dcmotor` input semantics; 3.9.0 itself sits on a breaking contact margin/gap redesign. That is a real revalidation surface for a purely cosmetic gain, and a locked-stack violation. **Stay on 3.9 and use cube mapping, which needs none of it.**

**Do not sculpt in Blender.** The full pipeline (sculpt → retopo → UV unwrap → paint → per-body OBJ → `obj2mjcf`) is 2–4 weeks if you don't already sculpt. It buys zero reward, zero cm/s, and zero policy change. It costs measured throughput: on the Menagerie Cassie model, adding visual meshes moved a step from 0.0180 ms to 0.0211 ms (~17% slower) and roughly doubled `pos_collision` time **even with no mesh collisions**. And decisively: **you have no gecko and no scan**, so the mesh would be *artistic*, replacing a defensible measurement-based MJCF with a mixed-provenance one and undercutting the project's single strongest claim. (Also: STL cannot carry texture coordinates — the textured route is OBJ only; and MuJoCo re-centres mesh geoms at compile time so post-compile `pos`/`quat` do not equal what you wrote, a classic multi-day debugging trap.)

**Do not build a MuJoCo `<skin>`.** dm_control's virtual rodent does exactly this — one line, `<skin file="rodent_walker_skin.skn"/>` — and it is the right primitive for a continuous body with a long tail. But there is **no standard Blender→.skn exporter**; dm_control's was authored by a contributor and shipped as a binary blob. You'd be writing a converter. Strictly worse cost/benefit than the cube texture.

**Do not expect PBR.** MuJoCo's native renderer permanently cannot use normal or roughness maps: `material/metallic` and `material/roughness` were added explicitly as "visual properties which are ignored by the native renderer". If you want a genuinely beautiful hero video, export the trajectory (USD export exists in the Python bindings) and render it in Blender/Cycles once, in week 11, decoupled from the critical path. 1–3 days for one video, and you get subsurface scattering and soft shadows that MuJoCo structurally cannot do.

---

## (b) The world and the prey

This is where appearance stops being cosmetic — but the earlier draft misdiagnosed where the leverage is, so read this section before touching a texture.

### Problem 1: the shortcut is real, but the green mask is not it

The shortcut is not a colour-threshold food detector wired into the reward. Verified against the repo:

- **The 4-D command is the brain policy's action space** (`spaces.Box(-1, 1, shape=(4,))`), not something computed from pixels. The env's own reference producer, `oracle_action()`, is pure simulator state — it reads `self.food_xy` and the trunk rotation matrix and returns `[ego_x, ego_y, dist_cmd, engage_cmd]`. Zero pixels.
- **Ground-truth food is an explicit network input.** `_privileged_vector()` returns `[ego_x, ego_y, dist, cos(heading), sin(heading)]` from that same ground-truth pose, and it is a key of the observation Dict. Three of the five saved brain runs have `use_privileged: true`; the two that don't are smoke tests. `BrainBCActor` defaults to `use_privileged=True`.
- **All supervision traces to the oracle.** The BC collector's own comment reads `# label: always clean oracle for current state`, followed by `label = env.oracle_action()`. The DAgger collector does the same. The one genuinely image-only network — the visual distillation student, built with `use_privileged=False` and hard-asserting its privileged input is all zeros — is trained on the *privileged teacher's* actions, and that teacher was fit to oracle labels. The ground truth is baked into the supervision even where it is absent from the input.
- **The reward is 100% ground truth and contains no vision term at all.** `r_progress = 12.0 × (mouth_dist_before − mouth_dist_after)`, where both distances are `‖food_xy − site_xpos[nose_tip]‖`; `r_eat = 10.0` when `mouth_dist_after ≤ eat_radius`. No image term is ever added.
- **The green mask is telemetry.** `_food_visible_frac` is called *after* the reward components and its outputs go only into `info["food_visible_frac"]` / `info["food_visible_signal"]`. It feeds no reward, no loss, no network input. Deleting the function changes no gradient anywhere in the repo. The collectors even print "NOTE: food_visible uses green color proxy mask."
- **And the food is not a physical object.** There is no food geom or body anywhere in `morphology/gecko_body_r.xml`. Food is the numpy variable `self.food_xy` plus a green sphere (`rgba = 0.1 0.95 0.25 1.0`) injected into the render scene each frame. The image is a *derivative* of the ground-truth pose, not an independent observation.

**Consequences, and they are not small:**

- Recolouring the prey, texturing it, adding decoys, or deleting the green mask changes nothing that trains. Worse — since the sphere's colour is the *only* visual evidence of prey that ever reaches head_cam, desaturating it without other changes does not remove a shortcut, it blinds the camera.
- To make the prey redesign load-bearing you need three things together: **(i)** add a real geom/body to the MJCF so prey has physical existence, appearance and occlusion; **(ii)** set `use_privileged=False` and `privileged_target=0.0` so the ground-truth bearing/range vector leaves the observation (the `privileged_food_dropout_prob` / `privileged_food_scale` tapers already exist for this); **(iii)** replace the ground-truth `mouth_food_dist` progress reward and `ate` termination with something the agent could only obtain through its sensors.
- Only then is the "one saturated green sphere" critique the shortcut-learning story it sounds like. The canonical demonstration still applies once you get there: an RL agent silently latched onto the *Sonic* scoreboard, and blacking out that one artefact raised test reward ~10% (NatureCNN 1052→1141, IMPALA 1130→1250, between-run sd 40).

It is also **biologically wrong in a specific way** worth fixing at the same time: nocturnal geckos have three cone-derived pigments at λmax ≈ 364 nm (UV), 467 nm (blue), 521 nm (green). **There is no long-wavelength/red receptor.** "Green versus red" is not an axis a gecko can represent. Real *E. macularius* prey are crickets, mealworms, spiders — brown/tan, low-contrast, moving, at a working range of a few centimetres.

### Problem 2: the floor is untextured, so it emits literally zero optic flow

As the gecko walks, every floor pixel keeps the same value. The CNN gets no self-motion, no distance, no speed signal. The gecko-locomotion literature makes optic flow the central visual variable linking the photic environment to locomotor control: a diurnal gecko lost stride length and frequency and adopted a more sprawled posture in dim light, tied to maintaining optic flow, while nocturnal species held performance. **You have removed the cue the animal's own locomotion is regulated by.** This one is unaffected by the ground-truth issues above, costs 1–3 hours, and is the largest single vision gain available.

### Problem 3: the camera resolves worse than the real animal is proven to see

At `fovy=120` with 64 rows you get **1.875°/px**. The one behaviourally measured spatial feature *E. macularius* demonstrably tracks is a **1.6° random dot** (Masseck et al. 2008 optokinetic drum, n=4). At 1.875°/px that dot is **0.85 px — below your camera's resolution.** Narrowing to `fovy≈70` gives 1.094°/px and the dot becomes 1.46 px, resolvable, at **identical pixel and render cost.**

Do the prey arithmetic too: a 20 mm cricket at 30 cm subtends ~3.8° ≈ **2 pixels** at fovy 120. Your prey may currently be at or below the detection limit unless the sphere is large.

Two honest framings on this. First, `fovy` is a field-of-view parameter, not an acuity parameter — no acuity measurement calibrates it, and the 1.6° dot is a *demonstrated tracked feature*, not a measured grating acuity. Second, **no grating acuity has ever been measured for any gecko species.** The nearest measured squamates are the wall lizard *Podarcis muralis* at 2.05 cyc/deg (optomotor) and 1.56 cyc/deg (retinal ganglion cell density), and the sleepy lizard *Tiliqua rugosa* at an upper limit of 6.8 cyc/deg. Your 64×64 sensor resolves a Nyquist limit of ~0.27 cyc/deg at fovy 120 and ~0.46 cyc/deg at fovy 70 — **3× to 15× coarser than any measured lizard.** The camera, not the optics model, is the binding constraint, and no amount of blur or noise modelling changes that. (One gap worth a library check: Masseck et al. 2008 is paywalled, and its Methods may report a spatial-frequency threshold that would be the first real gecko acuity number.)

### The concrete pipeline for world and prey, ranked by realism-per-hour

| Step | What | Time | Why it matters |
|---|---|---|---|
| **W1** | **Split policy vs human render** (`MjvOption` per camera) + `group="4"` on the four footzone sites | 2 h | Prerequisite. Makes everything below safe and permanent, and removes the only live green-in-frame hazard |
| **W2** | **Texture floor + walls + skybox** — CC0 PNGs from ambientCG (CC0 1.0, no attribution required), coarse-grained so they don't alias at 64×64 | 1–3 h | Turns the camera from a blob-detector into a motion sensor. Largest single gain on the list, and entirely independent of the ground-truth issues. Verify by dumping actual 64×64 renders, not the viewer |
| **W3** | **Make prey a real perceptual problem** — this is three changes, not one: (i) add a real geom/body for the food in the MJCF instead of a render-time overlay; (ii) drop the privileged food vector (`use_privileged=False`, `privileged_target=0.0`) and stop labelling from `oracle_action()`; (iii) replace the ground-truth `mouth_food_dist` progress term and `ate` condition with sensor-derived equivalents. **Then** brown/tan low-contrast prey, 3–6 same-colour decoys, per-episode hue randomisation | 2–4 d, not hours | Without all three, appearance changes are decoration — the policy still gets bearing and range for free, is still trained on oracle labels, and is still rewarded on ground-truth distance. With them, this becomes the headline result of the vision phase. Expect it to get much harder — use a curriculum |
| **W4** | **Narrow `fovy` 120→70**, keep 64×64 | 10 min + retrain | Free. Fixes the sub-animal acuity error above. Costs peripheral field, so the agent must turn its head — arguably more realistic, but note that head-down-and-yawed is exactly the posture that puts the trunk and forefeet in frame |
| **W5** | **Drop the red channel** — feed 2 channels (G≈521 nm, B≈467 nm) | 1–2 h | 33% smaller CNN input (real CPU saving) and removes a channel the animal physically lacks. Honest caveat: MuJoCo's sRGB primaries are not gecko cone fundamentals — this is a mapping, not a match, and UV is simply unmodelled |
| **W6** | **Render vision at ~20 Hz with a temporal low-pass** instead of every physics step | 3–5 h | Potentially 5–25× fewer render calls — likely your biggest CPU win in the whole project (see the profiling note below). **And it is now properly grounded, not inferred.** Gecko critical flicker fusion has been measured: 20 Hz by ERG in *Gekko gecko*; up to 18 flashes/s scotopic in *Hemidactylus turcicus* and *Tarentola mauritanica*, rising to 25 and 50 flashes/s photopic; ~26.8 flashes/s by behavioural optomotor in *Sphaerodactylus inaguae* at 26.7 °C; up to ~50 in the diurnal *Phelsuma inunguis*. So a **dark-adapted nocturnal gecko sits at 18–20 Hz**, which is exactly the render rate you want. On the filter: a single-pole low-pass has f_c = 1/(2πτ), so τ = 50 ms → 3.2 Hz and τ = 100 ms → 1.6 Hz — both an order of magnitude below the measured fusion limit. **Use τ ≈ 25–50 ms**, and state which adaptation state you are modelling, because CFF in these animals rises with light level. Note none of these measurements is *E. macularius* |
| **W7** | **Domain randomisation** — lift robosuite's MIT-licensed `TextureModder` / `LightingModder` / `CameraModder` rather than writing your own; randomise at *episode reset*, not every step | 4–8 h | Tobin et al. trained a real-world detector accurate to 1.5 cm from *only* randomised non-realistic simulated images. Widen ranges gradually and measure — over-wide ranges destroy sample efficiency |
| **W8** | **Random-convolution augmentation** (~40 lines in the SB3 features extractor) | 3–6 h | Best generalisation-per-hour on the list and it runs on CPU: CoinRun unseen-level success 34.6% ± 4.5 → 76.7% ± 1.3. Apply to the **visual branch only**, never to proprioception/IMU |
| **W9** | **Held-out visual eval suite** — unseen textures, unseen hue, shifted camera pose/FOV, changed light, moving background | 4–8 h | Turns "realism" into a number. Keep eval conditions strictly disjoint from training randomisation ranges. Only meaningful *after* W3, since before it the policy is not solving a visual problem at all |
| **W10** | **Lighting**: low, warm, randomised key light; turn the default headlight down explicitly (it silently flattens the scene and wastes your texture work); shadows off for speed | 2–4 h | Crepuscular is the biologically correct default. Nocturnal helmet geckos discriminate colour at 0.002 cd/m² with an eye ~350× more sensitive than human cone vision. But **do not literally darken the render** — MuJoCo is not radiometric, and a dark buffer just wastes 8-bit dynamic range. Represent low light as *reduced contrast plus noise* |
| **W11** | *(optional)* **Optical blur + noise** — Gaussian σ≈0.8–1.0 px, slightly larger on blue than green, then zero-mean noise | 2–3 h | The nocturnal gecko eye is f/0.90 and **multifocal** (~15 dioptres between concentric zones) with positive spherical aberration. It is not a sharp pinhole camera. Also acts as free domain randomisation. σ and noise amplitude are knobs you are *choosing*, not fitting — no gecko contrast-sensitivity function or grating acuity has ever been measured |
| **W12** | *(optional)* **Two lateral 32×32 eye cameras** instead of one 64×64 | 4–8 h | *Half* the pixels, far more of the real visual field. This is what DeepMind's flybody does (32×32, fovy 150). But it is a breaking change to your CNN input shape — do it before you accumulate results, or not at all |

### Things NOT to do to the world

- **Do not model the slit pupil.** At the light levels a leopard gecko hunts in, the pupil is round and fully open; the four-pinhole vertical slit only appears in bright daylight, and the field transition takes about an hour at dawn/dusk. Dynamic aperture is wasted work.
- **Do not add a tapetum lucidum or a low-light "gain" hack.** I could not verify from any primary source that geckos have a tapetum. Roth et al. 2009 explain the gecko's entire 350× sensitivity advantage by pupil diameter, short focal length and huge cone outer segments, invoking no tapetum; Fleishman's 2024 lizard visual-ecology review never uses the word. Every confident claim I found was on a pet-care website. **Genuinely unresolved — don't build on it.**
- **Do not adopt Habitat / HSSD / ThreeDWorld / iGibson / Isaac.** Wrong domain (human-scale indoor rooms, not desert substrate at gecko eye height) and Isaac's stated minimum is an RTX 3070. Steal one idea instead: HSSD-200's own result is that agents trained on **122 high-fidelity scenes beat agents trained on 10,000 procedural ones** for zero-shot transfer. Build one good arena and randomise it hard.
- **Do not spend the $500 on GPU batch rendering, and do not trust the 9–43% figure as a reason not to bother with W6.** MuJoCo Playground does say that "physics, rendering, and inference together only comprise 9% and 43% of the Cartpole and Franka total training times… with most of the time spent updating the expensive CNN-based networks" — but that measurement is MJX physics in JAX plus the Madrona CUDA batch renderer plus a GPU CNN update, across thousands of lockstep environments on an RTX 4090, at 37,000–403,000 steps/s. In that regime rendering is nearly free because it is amortised across the batch and never leaves GPU memory. **None of those conditions holds on your laptop.** On CPU, MuJoCo offscreen rendering carries a per-call readback tax that is largely resolution-independent — a profiled report shows `mjr_readPixels` still taking ~30 ms at **64×64**, the same order as at 1920×1080. A per-call tax that ignores resolution is precisely what render decimation removes and what shrinking the camera would not. Note also that the paper's bucket lumps physics + rendering + inference together, so it never isolates rendering at all, and 43% is nearly half the budget even on its own terms. **Profile your own machine before spending a dollar or dismissing W6**: wrap `time.perf_counter()` accumulators around `mj_step`, around `mjr_render` + `mjr_readPixels`, and around the forward/backward pass, and print the split over 1000 steps. That five-minute measurement, not a 4090 paper, decides the render-rate question. (Madrona-MJX is deprecated in any case, and porting to MJX/Warp is a 1–3 week rewrite of the CPG, actuators, sensors and reward — not a rendering switch.)

---

## Cosmetic vs load-bearing — the summary table

| Item | Verdict | When | Cost |
|---|---|---|---|
| Gecko's spots, colour, skin texture | **Cosmetic** | Week 11 | 3–8 h, $0 |
| Gecko's silhouette / sculpted mesh | **Cosmetic, and actively harmful to your provenance claim** | Never | — |
| MuJoCo 3.9 → 3.12 upgrade for UV mapping | **Cosmetic, and a locked-stack risk with breaking changes** | Never | — |
| Beauty render in Blender for the demo video | **Cosmetic** | Week 11 | 1–3 d, $0 |
| Splitting policy vs human render (`MjvOption`) + `group="4"` on footzone sites | **Load-bearing** (removes the one live green-in-frame hazard; makes all cosmetic work permanently safe) | Week 1 | 2 h |
| No green anywhere on trunk, neck or forefeet | **Load-bearing** (those geoms genuinely enter head_cam at up to 16.5% of frame when the head is pitched down and yawed) | Week 11, enforced from day 1 | 0 h if you never use green |
| Floor / wall textures | **Load-bearing** (optic flow) | Weeks 5–7 | 1–3 h |
| Prey as a real geom + dropping the privileged food vector + replacing the ground-truth distance reward | **Load-bearing — this is the actual shortcut** | Weeks 5–7 | 2–4 d |
| Prey colour, contrast, motion, decoys, hue randomisation | **Cosmetic on its own; load-bearing only after the three changes above** | Weeks 5–7 | 2–5 h |
| `fovy` 120→70 | **Load-bearing** (fixes sub-animal acuity) | Weeks 5–7 | 10 min + retrain |
| Dropping the red channel | **Load-bearing-ish** (biologically correct, 33% CPU saving) | Weeks 5–7 | 1–2 h |
| Vision at 20 Hz + temporal filter (τ ≈ 25–50 ms) | **Load-bearing for your schedule** (probably the biggest CPU win — but profile first) | Weeks 5–7 | 3–5 h |
| Domain randomisation + random-conv | **Load-bearing** (robustness) | Weeks 5–7 | 7–14 h |
| Held-out visual eval suite | **Load-bearing** (turns realism into a number) | Weeks 5–7 | 4–8 h |
| Blur / noise / lighting realism | **Marginal** — nice, defensible, tuneable, unfittable | Weeks 5–7 if time | 4–7 h |

**The one-sentence version:** the risk you asked about (the gecko's own skin) is near-zero; the risk you didn't ask about is that your food-finding policy currently receives the food's true bearing and range as an input and is rewarded on its true distance, so no amount of prettier prey makes it a vision problem until those three wires are cut.

---

## Total marginal spend for everything recommended

| Item | $ |
|---|---|
| Everything in the recommended Part 1 path (weeks 1–5) | **$0** |
| Fuller/Higham/Clark 2011 PDF (optional — buy for the femur-elevation numbers, not the two abstract figures) | $0–60 |
| MoCapAnything day-1 test on the hosted Space | **$0** |
| MoCapAnything local install, only if the free test survives | ~$100 |
| Everything in Part 2 (both halves) | **$0** |
| CMA-ES parallelisation on AWS CPU (optional) | tens of $ |
| **Total** | **$0–200 of a $500 budget** |

The binding constraint is your twelve weeks, not your money — and the biggest schedule risk on the list is the frequency unlock, which is why it is out of scope. Spend the weeks on items 1, 2, 3, 5, 6 and 7, plus W1–W4 in the vision table, and you will have a walker whose duty factor, footfall order, stride length and posture are each traceable to a primary source, a scorecard that proves it, and a demo that looks like a gecko — without ever needing an animal.

---

## Sources

**Gecko and lizard locomotion**
- Jagnandan & Higham 2017, *Sci Rep* 7:10865 — https://www.nature.com/articles/s41598-017-11484-7/tables/1 and https://pmc.ncbi.nlm.nih.gov/articles/PMC5589804/
- Jagnandan, Russell & Higham 2014, *J Exp Biol* 217:3891 — https://biomechanics.ucr.edu/Jagnandan_etal_2014.pdf
- Foster & Higham 2012, *J Exp Biol* 215:2288 (stride/step-length definitions) — https://biomechanics.ucr.edu/Foster_Higham_2012.pdf
- McElroy, Hickey & Reilly 2008, *J Exp Biol* 211:1029 — https://journals.biologists.com/jeb/article/211/7/1029/18146/
- Fuller, Higham & Clark 2011, *Zoology* 114:104 — https://pubmed.ncbi.nlm.nih.gov/21392953/
- Nyakatura et al. 2019, *Nature* (Orobates) — https://www.nature.com/articles/s41586-018-0851-2 ; Supplementary Data 1 (skink) https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0851-2/MediaObjects/41586_2018_851_MOESM3_ESM.xlsx ; EPFL https://biorob2.epfl.ch/pages/Orobates_interactive/data/modernSpecies.csv

**Gecko vision**
- Meneghini & Hamasaki 1967, *Vision Res* 7:243 (*Gekko gecko* CFF 20 Hz) — https://doi.org/10.1016/0042-6989(67)90088-0 ; value reproduced open-access in Healy et al. 2013 — https://europepmc.org/articles/PMC3791410
- Dodt & Jessen 1961, *J Gen Physiol* 44:1143 — https://europepmc.org/articles/PMC2195136
- Arden & Tansley 1962, *J Gen Physiol* 45:1145 — https://europepmc.org/articles/PMC2195235
- Crozier & Wolf 1939, *J Gen Physiol* 22:555 — https://doi.org/10.1085/jgp.22.5.555
- Roth et al. 2009, *J Vis* 9(3):27 — https://doi.org/10.1167/9.3.27
- Masseck, Roll & Hoffmann 2008, *Vision Res* 48:765 — https://doi.org/10.1016/j.visres.2007.12.004
- New & Bull 2011, *J Comp Physiol A* (*Tiliqua* acuity) — https://doi.org/10.1007/s00359-011-0635-8 ; Kawamoto et al. 2025, *J Exp Biol* (*Podarcis* acuity) — https://doi.org/10.1242/jeb.249422

**MuJoCo**
- 3.9.0 XML reference, asset/texture — https://mujoco.readthedocs.io/en/3.9.0/XMLreference.html#asset-texture and https://github.com/google-deepmind/mujoco/blob/3.9.0/doc/XMLreference.rst
- Changelog (3.12.0 primitive texcoords, breaking) — https://mujoco.readthedocs.io/en/stable/changelog.html
- Visualization / rendering docs — https://mujoco.readthedocs.io/en/stable/programming/visualization.html
- Offscreen readback cost discussion — https://github.com/google-deepmind/mujoco/discussions/2222

**Learning and control**
- MuJoCo Playground — https://arxiv.org/html/2502.08844v1
- Owaki & Ishiguro 2017, *Sci Rep* 7:277 — https://doi.org/10.1038/s41598-017-00348-9 ; Aoi et al. 2017 — https://doi.org/10.3389/fnbot.2017.00039
- CPG-RL — https://arxiv.org/abs/2211.00458 ; Walk These Ways — https://arxiv.org/abs/2212.03238 ; Shao et al. — https://arxiv.org/abs/2201.00206 ; CPG-Actor — https://arxiv.org/abs/2102.12891
- Residual RL — https://arxiv.org/abs/1812.03201 ; zero-shot generalisation survey — https://arxiv.org/abs/2111.09794
- MoCapAnything V2 — https://arxiv.org/abs/2604.28130 ; licence https://raw.githubusercontent.com/animotionlab26/MocapAnything/main/LICENSE ; free hosted demo https://huggingface.co/spaces/kehong/MoCapAnythingV2 ; RMBG-1.4 non-commercial terms https://huggingface.co/briaai/RMBG-1.4

**Project**
- NeuroGecko — https://github.com/12ziyad/NeuroGecko (`envs/gecko_brain_env.py`, `envs/cpg_residual_controller.py`, `rewards/gait_prior.py`, `morphology/gecko_body_r.xml`, `train/train_walk_ppo.py`)