# After Phase 2: What Actually Makes NeuroGecko Walk Like a Gecko

**Audience:** you, solo, on a laptop plus one g5.xlarge.
**Scope:** what to do after Phase 2's statistics gates go green.
**Confidence tags:** `[verified]` = read from a primary source this session · `[likely]` = strong secondary evidence · `[uncertain]` = inference, arithmetic, or a claim I could not close · `[repo]` = measured or read in `C:\Users\ziyad\GeckoBrain` today.

---

## 0. Five things I found before writing this that change the plan

I read the repo and re-read the sources rather than taking the brief at face value. Five findings reorder everything below.

### 0.1 — The base is not a symmetric sinusoid, but it is spatially symmetric, and two of its four legs never clear the ground `[repo]`

`C:\Users\ziyad\GeckoBrain\envs\cpg_residual_controller.py:69-74`:

```python
def _limb_signals(phi, stance):
    if phi < stance:
        s = phi / stance
        return (1.0 - 2.0 * s), 0.0          # stance: fore-aft sweep +1 -> -1, no lift
    s = (phi - stance) / (1.0 - stance)
    return (-1.0 + 2.0 * s), math.sin(math.pi * s)   # swing: sweep back, half-sine lift
```

The fore-aft channel is a piecewise-linear **triangle**, not a sinusoid, and it does have a stance/swing timing split. So "the base is a symmetric sinusoid" is wrong about the *waveform*.

But it is right about the thing that determines locomotion. Measured fore-aft excursion is **1.998 in stance and 1.998 in swing — identical.** The duty factor changes only the *speed* of each sweep, never the *distance*. Net thrust requires either broken ground contact during swing or unequal fore-aft amplitude, and the base has neither. `[repo]`

Three further defects, all measured on zero-action rollouts:

**(a) `_limb_signals` is not what reaches the front limbs.** `base_ctrl` (lines 202–215) overrides it for FL/FR with a constant stance press — `front_stance_press` 0.40 (FL) and 0.50 (FR) — plus a swing bump `front_swing_lift` that peaks at 0.40. The swing peak never exceeds the stance press. Net front swing clearance: **FL 0.0°, FR −6.0°** (the FR elbow is *less* extended at mid-swing than in stance). The front limbs have no swing-clearance term at all. `[repo]`

**(b) The hind swing lift is inverted, not merely insufficient.** With the knee axis `(0 1 0)` and the tibia vector as built, a positive knee command pitches the pes toe-down harder than it raises the shank. A frozen-trunk kinematic sweep puts commanded mid-swing hind toe **5.4 mm below** its own commanded stance toe. Measured ground contact *during commanded swing*: HL 80–85%, HR 89%. The hind feet essentially never leave the ground, and they carry **more** normal load in swing (236/257 mN) than in stance (143/183 mN). `[repo]`

**(c) The front stance press levers the body up.** The commanded front stance pose demands 13.7 mm (FL) / 14.9 mm (FR) of ground penetration, so the servo lifts the whole front of the animal: trunk rides at 31.8 mm against 22.6 mm at the "stand" keyframe. Measured front contact fraction is 0.29–0.36 against a *commanded* 0.68/0.70. That is the mechanical origin of the nose-up pitch, and it is under-contact, not drag. `[repo]`

**(d) The fore/hind amplitude is symmetric and biology says it should not be.** `AMP["fa"] = 0.68` is applied identically to `hip_proret_*` and `shoulder_proret_*`, both with ctrlrange ±0.698 rad, giving **54.4° peak-to-peak for both**. Jagnandan & Higham 2017 Table 1 measures femur retraction 82.57 ± 2.29° against humerus retraction 44.41 ± 1.80° — a ratio of 0.54, against the sim's 1.00. `[repo, verified]`

**And the ~0 m/s is a heading failure, not a drag failure.** Over 20 s the zero-residual base covers **1.270 m of path** and only **0.152 m of net displacement** (8.4:1). Forward speed has sd 0.069 m/s and swings between −0.127 and +0.143 m/s, with a consistent lateral drift of about +0.005 m/s. If swing drag were cancelling the stance sweep, path distance would also be near zero. It is not. The base surges and circles; it does not stick. `[repo]`

So the earlier "the toe never clears the ground, so the return sweep drags and cancels the stance sweep" hypothesis is **wrong as a single mechanism**. The correct picture is four coupled defects: an inverted hind swing profile, a front girdle that over-presses in stance and never reaches in swing, a spatially symmetric fore-aft sweep that generates no net thrust, and no heading regulation. All four are amplitude/sign parameters inside `base_ctrl`.

**Critically, none of this is reachable by RL.** `lock_front_lift=True` with `front_lift_residual_scale=0.0` sets the residual scale on `elbow_L` and `elbow_R` to exactly **0.0**, and the knee residual is capped at 0.05. The policy has zero authority over front clearance and near-zero over hind clearance. This must be fixed in the base. `[repo]`

**The measurement is already on disk.** `artifacts\evidence\legacy_zero_residual\traces\episode_000.json` logs `foot_position_m` (1001×4×3), `foot_force_N` (1001×4) and `time_s`. Gate 2 is a few lines of analysis, not an hour of instrumentation. And use the **force** channel, not toe height: `foot_position_m` is the centre of the `footzone_*` box *site* (hind half-extents ~9.9 × 7.7 × 4.1 mm), not the collision pad (hind box half-size 4.5 × 3.2 × 1.6 mm), and MuJoCo's soft contact permits interpenetration while `margin` generates force before geometric touching. Site height reads 4.3 mm at moments when measured load reads 100%. `[repo, verified]`

### 0.2 — The residual is a steering controller, not the engine `[repo]`

Three corrections to the "the residual is tubed to 25% and still supplies 100% of forward motion" claim.

**It is a gain, not a tube.** `compute()` does `residual = action * res_scale_vec * half; ctrl = base + residual; ctrl = clip(ctrl, lo, hi)`. The only clip is on the *total* command against actuator ctrlrange. The bound `|residual| ≤ 0.25·half` holds only because `gecko_walk_env.py:310` clips the raw action to [−1, 1] with `action_scale=1.0`. And because the total-ctrl clip is applied *after* the base, the admissible residual set is **asymmetric** wherever the base already sits off-centre — which is everywhere the front press, spine amplitude and shoulder tuck act. "A symmetric tube is already implemented" is not what the code does.

**It is not global.** `GeckoWalkEnv` defaults `residual_scale=0.2`; 0.25 is what every logged run passes. Post-override, only **13 of 25** actuators keep 0.25 — hips, ankles, neck, head, tail_lift. Twelve are capped, and the floor is **0.00** (elbows), not 0.05. The module docstring says knee 0.08; the executed dict says 0.05. The docstring is stale.

**"100% of forward motion" is legacy-only, and only on one metric.**

| Configuration | base alone | with residual | base share |
|---|---|---|---|
| Legacy body + legacy timing, signed forward | −0.001373 m/s | 0.049517 m/s | ~0% |
| Legacy, **path** speed | 0.074722 m/s | 0.128484 m/s | 58.2% |
| Legacy, net displacement over 17 s | 0.1518 m | 1.0236 m | 14.8% |
| V2 body + lab timing, signed forward | 0.011736 m/s | 0.019909 m/s | **58.9%** |

Under the lab profile — the profile Phase 2 is moving to — the base supplies most of the forward speed. What the residual actually supplies is **heading**: net-to-path ratio goes from 11.9% (base) to 46.9% (trained). And the 13 actuators left at full 0.25 authority are exactly the set the file's own docstring labels "steering & balance — untouched."

So "the base is actively self-cancelling" is a legacy-only finding and it describes circling, not stalling. Under the lab profile the base is *merely weak*, which is the opposite architectural reading.

Caveat on all of this: `reset_noise = 0.0` and evaluation is deterministic, so the 20 baseline "episodes" have speed sd of 0.00000 — they are identical replays, not samples. Every ablation above is **n = 1 deterministic run**. The directions are not in doubt; the exact percentages are one operating point.

### 0.3 — The base gait is a *diagonal*-sequence singlefoot, and the 0.0117 m/s number belongs to a different run `[repo, verified]`

The constants are as quoted:

```python
STANCE = 0.62                                  # hind duty factor
FRONT_STANCE = {"FL": 0.68, "FR": 0.70}        # fore duty factor
PHASE = {"HL": 0.00, "FL": 0.25, "HR": 0.50, "FR": 0.75}
```

But the controller is **additive**, and deliberately so — the in-code comment says "Deliberately preserve the old ADDITIVE controller convention." Phase is `(t*freq + offset) % 1` with stance at `phi < stance`, so touchdown occurs at cycle fraction `(1 − offset) mod 1`, **not** at `offset`. Computed touchdowns: HL 0.00, FR 0.25, HR 0.50, FL 0.75. Footfall order **HL → FR → HR → FL**. Hildebrand limb phase = **0.75 on both sides**.

By McElroy et al. 2008's own cutoffs (lateral sequence < 50%, diagonal sequence > 50%), that is a **diagonal-sequence singlefoot**. The 0.25 in the earlier draft is the repo's `diagonal_pair_separation_cycle` field — a folded `min(p, 1−p)` quantity that is 0 for a trot and 0.50 for a Hildebrand-0.50 gait. It is not limb phase. Measured limb phase on the zero-residual base is HL→FL **0.7185** (resultant 0.836), confirming the additive prediction.

Consequences:

- The target (0.43–0.44) is itself a **lateral**-sequence gait, near-trot. "Lateral-sequence singlefoot" does not distinguish the two gaits; the legacy gait is the diagonal-sequence one.
- The gap is **0.315 cycle**, not 0.185.
- The **fore** duty factor (0.68/0.70) is already at target. Only the hind constant is off, and its measured contact duty with zero residual is 0.683/0.713 — inside the published envelope, not 0.62.
- **0.0117 m/s is not the legacy walker.** It is the *lab-profile, zero-residual* run (V2 body, touchdown delays HL 0.00 / FL 0.44 / HR 0.50 / FR 0.94, duty 0.765/0.700) — i.e. the Phase 2 target gait already being executed, badly. The legacy walker measures 0.128484 m/s path / 0.049517 m/s signed forward on the original body, 0.105698 / 0.033660 on V2.

One provenance note that matters for Phase 2: **0.765 is not a published number.** `config/proxies.yaml` calls it a "Chosen aggregate target; source-study values differ. Not a pooled statistical estimate." The published values are Jagnandan & Higham 2017 (hind **0.78 ± 0.01**, fore **0.70 ± 0.01**) and McElroy et al. 2008 (walk 0.72 ± 2.1, run 0.70 ± 0.7). Use 0.78/0.70 and say where each came from.

And the convention fix is not free: swapping legacy → lab timing under the *same frozen residual* dropped path speed 0.105698 → 0.064364 m/s and signed forward 0.033660 → 0.019909 m/s. Retiming the base requires a residual retrain.

### 0.4 — Compute dollars are trivial. Neither dollars nor developer attention is the binding constraint — a stable PPO recipe is `[repo, verified]`

The run numbers are exact. `aws_results\aws_10m_autostop.log` ends `fps 853 | iterations 2442 | time_elapsed 11719 | total_timesteps 10002432`, bracketed by 11,728 s of wall clock. g5.xlarge us-east-1 on-demand is **$1.006/hr** on two independent price mirrors, so 3.2553 h × $1.006 = **$3.28** of instance time.

Four corrections, in ascending order of how much they matter.

**The $440 is not the recorded budget.** `docs\BUILD_LOG.md` records that the user "directly authorized $60 maximum additional AWS cost," restated as "The $60 direct authorization remains the ceiling, not a target to spend." At $3.28/run that is **~18 runs**, not 100 — and less, because EC2 bills uptime, not `learn()` time, and the same log says "EC2 itself remains on and billing." A g5.xlarge left up for a day is $24.14.

**The GPU was never used.** Line 2 of the log reads `Using cpu device`. The A10G sat idle for the whole $3.28. The same run on a 4-vCPU c7i.xlarge ($0.1785/hr) is ~$0.58. So the arithmetic overpays 5.6× — but the converse bites harder: with 4 vCPUs and a CPU-bound env, throughput is pinned near 853 fps no matter the budget. **100 serial runs = 326 h = 13.6 days of wall clock.** And "Running On-Demand G and VT instances" defaults to a **0-vCPU quota**, so parallel g5 runs are quota-gated, not money-gated.

**The cited run is the project's documented negative result.** Eval reward peaked at **1128.76 ± 555.71 at 1.5 M steps**, then decayed: 1091 @ 2M, 845 @ 4.5M, 505 @ 8.5M, **422.97 @ 10M** — down 63% from peak. `approx_kl` went 0.0516 → 67.26 and `clip_fraction` 0.44 → 0.907. `docs\locomotion_v4_final_results.md` concludes: "Stop V4 locomotion training. Use V4.5B as final base. Future speed beyond 0.115 would require a new project-level morphology/actuator/controller redesign, not another PPO continuation." The accepted champion is a **1 M-step** run. So $0.49 of compute produced the best policy and $2.78 actively destroyed it, and buying 100 more of these buys 100 diverged runs.

**Therefore:** the two-stage plan in §3 costs about **$6.55** — comfortably inside the recorded $60 ceiling. Justify it on that, on wall-clock, and on early stopping. Do **not** justify it on an invented budget, and do not argue that "compute is cheap, so spend more PPO." The project's own record says more PPO made the walker slower. The right structural conclusion is **shorter runs with eval-triggered early stopping**, not longer or more numerous ones.

### 0.5 — Zaaf 2001 settles the frequency question, and the answer is bad for the lock `[verified]`

The earlier draft flagged a contradiction between "PNAS 2015 says stride length" and "Zaaf 2001 says frequency." **There is no contradiction — one of the two sources does not study this animal.**

Higham, Birn-Jeffery, Collins, Hulsey & Russell (2015) *PNAS* 112(3):809–814 is a comparative study of the southern-African *Pachydactylus* radiation — *Rhoptropus*, *Pachydactylus*, *Chondrodactylus*, *Colopus*, *Tarentola*. *Eublepharis macularius* is cited in passing and never measured. There is no PNAS paper on *Eublepharis* locomotion in any year. Its stride-length sentence concerns the level-to-incline transition, not level speed control. That survey line is **misattributed and should be deleted.**

Zaaf, Van Damme, Herrel & Aerts (2001) *J. Exp. Biol.* 204(7):1233–1246 is real, and the *E. macularius* data are in **Table 3**, not Table 2 — Table 2 is the *Gekko gecko* table. Anything citing "Zaaf Table 2" for leopard gecko is citing the tokay.

From Table 3 (N = 22 strides, 3 animals; SVL 12.81 ± 0.48 cm, mass 49.51 ± 9.25 g), with `log10(y) = a + b·log10(v)`:

- **Stride frequency:** r² = 0.97***, b = 0.757 ± 0.031 (hind)
- **Stride length:** r² = 0.71***, b = 0.241 ± 0.034 (hind)
- **Duty factor:** r² = 0.61***, b = −0.201 ± 0.036 (hind)
- Step length, sprawling, limb angles, relative phase: all n.s.

The paper's own worked example: 0.5 → 1.0 m/s is "an 18% increase in stride length and a 69% increase in stride frequency." The abstract: "Higher level or climbing speeds are realized mainly ... by increasing stride frequency. Stride lengths and duty factors vary with speed in the ground-dweller."

So **frequency is the primary channel (~76% of log-speed), stride length is a real secondary channel (~24%, highly significant), and duty factor is a third.** "Mainly frequency" is right; "frequency only" is wrong. The `[verified]` tag survives.

**What this does to the lock.** Solving the hind regression for f = 1.1888 Hz gives **v ≈ 0.104 m/s** — below the slowest measured level trial (0.24 m/s) and squarely inside the animal's *vertical-climbing* band (0.025–0.085 m/s). Held to the animal's own observed stride-length envelope (10.9–15.6 cm over its level range), a locked 1.1888 Hz spans only **v ≈ 0.13–0.19 m/s — a 1.43× range against the animal's 4.4×**. Hitting even 0.24 m/s at locked cadence needs a 20.2 cm stride against the animal's actual 10.9 cm; hitting 1.05 m/s needs 88 cm, about 7 SVL per stride, anatomically impossible.

**So the "load the variability onto stride length instead" fallback fails quantitatively.** The fatal branch is the supported one — *unless* NeuroGecko's target gait is deliberately a slow walk in the 0.10–0.19 m/s band, where 1.1888 Hz is a legitimate operating point and stride length covers about ±20% of speed.

Which forces a target correction: **Phase 2's 0.08–0.10 m/s target is below the animal's entire measured level range**, and below the slowest lizard in McElroy's 18-species dataset (0.16 m/s). The defensible target at locked cadence is **0.13–0.19 m/s**, and even that is at the bottom edge.

The recommendation this changes: the fix is *not* necessarily a from-scratch retrain. Because Table 3 says frequency and stride length co-vary as a fixed 0.757/0.241 log-log split, **one scalar speed command can drive both.** Add a scalar frequency-gain (phase-velocity scaling) input to the CPG's phase integrator, freeze the existing residual, sweep the scalar, and measure the achievable speed range. That is hours of work, not weeks, and it is the single cheapest experiment that settles the Phase 3 architecture question.

Evidence strength: Zaaf 2001 is the **only** primary measurement of *E. macularius* level speed modulation, on 3 individuals and 22 strides, and it is closed access (© The Company of Biologists; Unpaywall `oa_status: closed`). Cite the coefficients as facts with attribution; do not redistribute the table. Direction = high confidence. Coefficients as simulator ground truth = medium confidence. No 2024–2026 replication exists.

---

## 1. Why matching statistics is not enough

Phase 2 will fit 15–40 CPG scalars to published summary statistics with CMA-ES. Here is the mechanism-by-mechanism reason that produces a gait that scores well and still reads as animation.

### 1.1 There is a published precedent for exactly this failure

Herbert-Read, Romenskyy & Sumpter (*Biol. Lett.* 11:20150674, 2015) built a self-propelled-particle model whose **polarization and nearest-neighbour distance statistically matched** real Pacific blue-eye schools, then ran a public Turing test. Eighteen expert researchers distinguished real from simulated **on the first attempt**; 1,775 online players detected a difference far above chance (χ² = 367.7, p < 0.0001) and improved with repetition. Their stated conclusion is that matching summary statistics is not sufficient for biological realism. `[verified]`

This is your situation with the species and the statistics swapped.

### 1.2 The statistics are means; the tells are in the residual structure

**(a) The timing tell — but you are currently measuring it upside down.** `[repo, verified]`

The intuition ("a closed-form clock has no stride-to-stride structure") is right. Almost every stated fact about it was wrong, and the direction of the error is the opposite of what the draft assumed.

- **Measured `stride_period_cv` on your actual eval path is 0.357–0.592 — that is 36–59%, not 0.** Pooled over all 20 baseline traces: HL 35.5%, FL 58.7%, HR 48.4%, FR 48.2%. Observed contact-cycle rates are **3.15–4.70 Hz against a commanded 1.1888 Hz** — three to four detected events per CPG cycle, i.e. foot chatter surviving the 0.04 s debounce. Per-limb stride counts differ by 52–78 over the same 17 s window. The repo's own gross-mismatch guard did not fire because the traces carry no frequency field, so `commanded_frequency_hz` is null.
- **Even the bare oscillator does not give 0.** `_cpg_t` advances on a discrete grid; 1/1.1888 = 0.841184 s = 42.0592 control steps, not an integer, so wrap-to-wrap periods alternate. CV is **0.56%** on the 50 Hz control grid and 0.048% on the physics grid. Exactly 0 holds only for the analytic continuous-time period — which is the definition of the driver, not a measurement of the gait.
- **DFA α is not undefined.** In float64 the detrending residuals are ~1e-15, never exact zero, so DFA returns a finite meaningless number: **α ≈ 0.07–0.14** on the eval-path and grid series alike, manufactured entirely from quantisation noise. Reporting "undefined" would misdescribe your own output; reporting 0.11 without noticing why would be worse. (`nolds` is not installed in the project venv, so nothing would flag it.)
- **The discriminator itself was wrong.** An animal driven by a perfect fixed-frequency pacemaker does *not* produce CV 0. Vaz, Cortes, Gomes, Jordão & Stergiou (2024, *J. Biomech.* 165:111972; 14 young men, 10-min walks) measured uncued stride-time **CV 1.7 ± 0.6%, α 0.91 ± 0.16**, and under an isochronous metronome **CV 1.3 ± 0.3%, α 0.52 ± 0.14**. Hausdorff et al. (1996) established the same qualitatively: metronomic pacing abolishes the long-range correlations of free walking. So the honest signature of a locked clock is **α collapsing toward ~0.5 with CV still around 1%**, not CV → 0.
- **Resolution floor.** Trace `event_time_resolution_s` is 0.02 on a 0.841 s period — a 2.4% quantum. A biological ~1–3% CV is barely more than one quantum, so the current 50 Hz logging cannot resolve biological variability even if it were present, while quantisation alone injects 0.56%.
- **Data budget.** Ravi, Marmelat, Taylor, Newell, Stergiou & Singh (2020, *Front. Physiol.* 11:562) require "at least 500 to 600 strides" for DFA; Mylonas et al. (2026, *Ann. Biomed. Eng.* 54(6):1844) find DFA "rarely exceeded moderate reliability" below 100 strides and recommend the Hurst–Kolmogorov process instead. Your episodes cap at `max_steps=1000` = 20 s ≈ 24 cycles, and with `reset_noise=0.0` plus `deterministic=True` the episodes are **byte-identical** (`episode_000.json` and `episode_001.json` have equal `foot_force_N` arrays), so you cannot accumulate strides by running more seeds.

So the honest statement is: *the commanded period is constant by construction and shows only 0.05–0.56% quantisation jitter; DFA on it returns a degenerate α from that quantisation; and the current contact detector reports 36–59% cycle CV from foot chatter, which must be fixed before variability is measurable at all.* Right now the eval scores the gait as **~14× too irregular**, not too regular. Injecting noise into that would be injecting noise into a measurement artefact.

The lizards-are-more-variable point (Ross et al. 2013: bradymetabolic tetrapods show higher inter-cycle variation in stride duration than birds and mammals, across 55 species, all P ≤ 0.044) still stands as a *direction*, but see §6.1 — the gekkonid row is unverified.

**(b) The instant-actuator tell — the plant has no dynamics.** Your control loop is 50 Hz (0.0008 s × frame_skip 25) into 25 `<position>` servos with `timeconst` unset, i.e. no activation filter. The PPO residual can inject content up to 25 Hz Nyquist and the joints track it essentially instantly. MuJoCo's own muscle defaults are `timeconst = (0.01, 0.04)` s. `[verified]` High-frequency, zero-lag micro-corrections are the single most recognisable "this is RL" signature and no gait statistic sees them.

**(c) The force-ceiling tell — but the headroom is one order of magnitude, not two, and the joint was wrong.** `[repo, verified]`

Four corrections:

- **Wrong file.** `kp = 0.5278` / `forcerange = ±0.8693` / mass 38 g are `gecko_body_lab_v2.xml`, an opt-in candidate whose own header says it "has NOT passed gait validation or fresh residual training." The **live** model — `DEFAULT_XML` in `gecko_walk_env.py:34` and the default in `train_walk_ppo.py:210` — is `gecko_body_r.xml`: `kp = 0.85`, `forcerange = ±1.4`, total mass **61.2 g**. (v2 is a uniform 0.62092× rescale, so the 1.647 rad ratio is identical in both — that one figure survives.)
- **Wrong joint.** `hip_proret` has axis `(0 0 1)`, world-vertical. A vertical ground reaction force exerts essentially zero moment about a vertical axis: measured static hold is **0.00007 N·m** (R) / 0.00012 N·m (v2). Weight support is carried by `hip_sprawl` (axis `1 0 0`), at 0.0067 / 0.0022 N·m. `hip_proret` is the *propulsive* DOF — femur protraction/retraction, the caudofemoralis channel. "Single-limb support torque" is not a load it ever sees.
- **Wrong criterion.** Compiled `biasprm` is `[0, −kp, −kv]` with kv = 0.014 (R) / 0.0087 (v2). Force is `kp(ctrl − q) − kv·qvel`. Saturation is not "1.647 rad of tracking error"; it is that expression exceeding forcerange. Measured `|qvel|` reaches 5.48 rad/s.
- **Wrong margin.** Peak measured `hip_proret_L` force over CPG rollouts: **0.107 N·m (7.8% of ±1.4)** zero-residual, up to **0.316 N·m (22.6%)** adversarial on the live model; 0.104–0.216 N·m (12–25%) on v2. True margin is **4–13×, under one order of magnitude**, not the ~290× the 0.003 N·m benchmark implies. That benchmark also does not reproduce from its own inputs: 38 g at 20 mm gives 0.0075 N·m for full-body support or 0.0019 N·m for a quarter share; the true horizontal offset from hip anchor to ipsilateral foot is 27.5 mm, not 20.

What *is* true, and it is what motivates the recommendation: **zero saturated control steps across all 25 actuators in both models**, with worst-case utilisation 23–24% (elbow_R). Nothing overshoots-and-settles, nothing sags. But sizing the fix off 0.003 N·m would make the actuator ~30× too weak to execute its own CPG command and the gait would collapse. Size against the measured adversarial peak instead: roughly a **4× cut**, to ~0.33 N·m on the live model.

Structurally, ctrlrange — not body weight — is why nothing saturates today. Max achievable `|ctrl − q|` is 1.3963 rad (R) → 84.8% of forcerange, or 1.5708 rad (v2) → 95.4%. On v2 the position term alone reaches 95.4% at the joint corner, so 4.63 rad/s of opposing velocity would saturate it — and measured `|qvel|` exceeds that. So "no actuator can ever saturate" is not even categorically true on the candidate body.

**(d) The open-loop tell — nothing responds to anything.** Freezing the residual's phase observation costs only ~16% of return `[repo]`, and the base contributes almost no *heading*. Together those mean the residual is a cadence-calibrated feedforward tape with a steering bias. A feedforward tape has no perturbation response, no load-dependent stance duration, and no stride-to-stride difference — three things a viewer reads as "alive" and no summary statistic encodes.

Rybak et al. (*eLife* 2024) sharpen the biological indictment: below roughly 0.3–0.4 m/s the spinal locomotor network operates in a **state-machine regime that requires sensory feedback for phase transitions**; intrinsic rhythm generation dominates only at higher speeds. `[verified]` Every speed under discussion here — 0.09, 0.13, 0.19 m/s — sits deep inside the feedback-gated regime, so a purely central clock is the biologically wrong architecture at exactly this speed. Note this argument is *independent of* and *stronger than* the Schumacher citation discussed in §1.3.

**(e) The rigid-posture tell — and its mechanical origin is now known.** A constant nose-up pitch held rigidly through every stride is the visual signature of a per-timestep posture reward. FreeMusco's motivation is exactly this: "in real animal locomotion, attributes like velocity, posture, and up direction fluctuate rhythmically around a desired value. Enforcing frame-by-frame targets in such settings often leads to overly constrained and unnatural motion." `[verified]` But per §0.1(c), the pitch is not only a reward artefact — the front stance press mechanically levers the trunk from 22.6 mm to 31.8 mm. Fix the base before blaming the reward.

**(f) The right-marginals-wrong-joint-distribution tell.** CMA-ES fits per-joint excursions and duty factors — marginals. A gait can match every marginal while visiting joint *combinations* no gecko adopts. Kinematic-manifold occupancy catches this; nothing in Phase 2 does. And the current marginals are badly off: commanded hind knee excursion is 36.0° peak-to-peak (and **exactly 0° within stance**) against 98.83 ± 1.39° measured; ankle 13.5° against 85.19 ± 2.38°; elbow 24–30° against 92.49 ± 2.27°. A limb held at neutral knee and near-neutral ankle through the whole 62% stance produces no propulsion and no support. `[repo, verified]`

**(g) The wrong-forces-right-angles tell.** Kinematics do not determine kinetics. Geckos have a strongly asymmetric force pattern: forelegs **decelerate the CoM in fore-aft only**, hindlegs supply **all** fore-aft acceleration in the second half of step, and lateral forces always point toward the midline and exceed fore-aft in magnitude (Chen, Peattie, Autumn & Full, *JEB* 209:249, 2006). `[verified]` A residual that pushes all four legs backwards reproduces every joint angle and gets the forelimb sign backwards.

**(h) The energy tell — and my earlier prediction was backwards.** `[verified]`

Centre-of-mass energy recovery is a consequence of *when* force is applied relative to where the CoM is in its arc, not of any joint angle. McElroy, Hickey & Reilly (*JEB* 211:1029, 2008) classify by KE–PE phase shift: **135–180° = walking, 0–45° = running, and 46–134° is explicitly EQUIVOCAL** — a third outcome the earlier draft dropped. Recovery is *not* part of the criterion; it is a reported outcome, and the quoted ranges (18–47% walking, 1–18% running) are ranges of **species means for wide-foraging species only**, which overlap at 18%.

Most importantly: **your species is in that dataset.** McElroy Table 1, *Eublepharis macularius*:

| | duty factor | limb phase | phase shift | % recovery | speed (m/s) |
|---|---|---|---|---|---|
| Walk | 72 ± 2.1 | 44 ± 1.1 | **151 ± 6.6°** | **32 ± 4.3** | 0.24 (0.20–0.27) |
| Run | 70 ± 0.7 | 43 ± 1.8 | **8 ± 3.2°** | **16 ± 3.9** | 0.29 (0.29–0.32) |

So drop the Coleonyx extrapolation entirely. And note what the row says: **real leopard geckos use bouncing mechanics (8° phase shift) at 0.16–0.24 m/s with ~70% duty factor.** In this species low speed does *not* imply walking mechanics, and walk and run speeds differ by only 0.05 m/s.

**This reverses the earlier prediction.** "Landing in the 0–45° bouncing band at low speed is biologically impossible for a walking gecko" is wrong — it is exactly what this animal does. What you cannot conclude is anything at 0.09 m/s, which is **below every speed in the dataset** (slowest walk anywhere: 0.16 m/s). Run the measurement; do not pre-register a pass/fail threshold you cannot ground.

On Farley & Ko (*JEB* 200:2177, 1997): the abstract says "**The lizards** conserved as much as 51%" — the study pooled *Coleonyx variegatus* (~4.2 g; Autumn et al. 1997) **and** *Eumeces/Plestiodon skiltonianus*. The 51% cannot be attributed to Coleonyx, the full text is paywalled with no OA copy, and 51% is a maximum: Reilly, McElroy, Odum & Hornyak (2006) put lizards at **~25%** typically. Correct usage: "up to 51% recovery in walking lizards (*Coleonyx variegatus* and *Eumeces skiltonianus*)."

### 1.3 The frequency lock has a named cost, but the citation everyone reaches for does not establish it

Schumacher et al. (*iScience* 28(4):112203, 2025, CC BY) contains this sentence, verbatim:

> "The prescription of hip movement at a certain frequency (step clock), keeping certain joint angles in pre-specified positions or minimizing torso rotation helped to achieve stable gaits, but prevented effort minimization and did not lead to natural kinematics."

Five things the earlier draft got wrong about it. `[verified]`

1. **It is not a tested ablation.** The string "clock" occurs **exactly once** in the paper. There is no figure, no table, no supplementary item, no frequency value in Hz, and no implementation description. The paper's actual documented ablations are three — *no-adapt*, *no-effort*, *only-vel* — and the step clock is not among them. It is an anecdotal design note.
2. **The verdict is undifferentiated across three constraints.** Frequency prescription is bundled with fixing joint angles and penalising torso rotation, and one collective verdict is applied to all three. The paper never separates them. Fixing joint angles is a far more aggressive clamp and the more plausible culprit.
3. **The ellipsis hides a positive clause** — "helped to achieve stable gaits, but."
4. **The paper never mentions CPGs.** "CPG", "central pattern generator", "oscillator" and "residual" occur zero times, in both the published version and the arXiv preprint. Its step clock is a **reward-side** prescription competing directly with the effort term inside a single objective. A CPG in the *action* path does not create that competition. Citing this as literature naming NeuroGecko's mechanism as a naturalness blocker is an extrapolation across architectures.
5. **Species mismatch.** Four human bipedal musculoskeletal models at 1.2 m/s. No quadruped, no sprawling posture.

**Cite it as:** experienced practitioners in human musculoskeletal RL report, in a single unquantified remark outside their formal ablations, that prescribing hip frequency — together with fixing joint angles and penalising torso rotation — bought stability at the cost of effort minimisation and natural kinematics.

**The real argument against the lock is §0.5 and §1.2(d), and you now own both.** Zaaf's coefficients say the lock removes ~76% of this animal's speed-control channel and confines it to a 1.43× speed range. Rybak says a central clock is the wrong architecture below 0.3–0.4 m/s. Those are measured, on-species (or on-mechanism) arguments. You do not need the Schumacher quote to make them, and it is weaker than either.

---

## 2. The ranked table

Ranked by **realism gained per developer-day**. Effort is calendar days for you, solo. "Lock" = survives the 1.1888 Hz lock. "Retrain" = none / finetune (residual fine-tune) / from-scratch.

One standing caveat: several items below change the base that `v4_5b_speed_polish_1m` was trained against. That checkpoint's action semantics are tied to `residual_scale=0.25`, to the additive phase convention, and to a plant where nothing saturates. Anything touching those needs at minimum a fine-tune, and the controller's own docstring precedent required a fresh policy for smaller changes than several of these.

### Tier A — do these in Phase 2, before the fresh retrain

| # | Method | What it is / why it is more real | Days | Retrain | Lock | Metric that proves it worked | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Effort charged on total actuator power, not on the residual** | Compute the cubic/effort cost from `actuator_force × qvel` (use `\|τ·q̇\|`, not signed), never from the residual vector. If effort is charged on the residual, the cheapest policy is residual→0 — and under the legacy profile the base alone nets ~0 m/s forward while under the lab profile it loses all heading regulation (net/path 11.9%). Either way the adaptive-α ratchet slowly strangles the thing the residual is actually for. | 0.5 | finetune | yes | Mean `\|τ·q̇\|` and mean `\|residual\|` logged separately; they must not be anti-correlated as α rises | **DO IN PHASE 2** — bug fix, not an improvement |
| 2 | **Fix the contact/stride detector and raise trace fidelity** | Today `stride_period_cv` reads 36–59% from foot chatter at 3–5 detected cycles per commanded cycle, `commanded_frequency_hz` is null in traces so the mismatch guard never fires, and 50 Hz logging quantises the 0.841 s period at 2.4%. **Every timing, duty-factor and variability metric downstream is currently measuring an artefact.** Log commanded frequency, log at ≥200 Hz, and segment strides on debounced load with an entrainment assertion. | 1–2 | none | yes | Observed same-foot cycle rate = 1.1888 Hz ± tolerance; `stride_period_cv` falls to single digits; `gross_frequency_mismatch_warning` clears | **DO IN PHASE 2 FIRST** — nothing else is measurable until this passes |
| 3 | **Base-walk diagnosis, then base-walk fix** | Diagnosis is **~10 minutes on traces you already have** (`artifacts\evidence\legacy_zero_residual\traces\`): per-foot **load fraction during commanded swing**, phase-resolved, from `foot_force_N` — not toe height, which reads the box-site centre and lies. Then fix, in `base_ctrl`: hind lift sign/amplitude (currently drives the toe 5.4 mm *below* stance), front stance press (currently demands 14 mm penetration and levers the trunk up 40%), fore-aft spatial asymmetry (currently 1.998 in both phases), fore/hind amplitude ratio (currently 1.00, biology 0.54), and the phase convention (§0.3). | 0.1 to test, 5–8 to fix | none to test | yes | Swing-phase load fraction: hind from 80–89% → < 10%; front stance load fraction from 0.29–0.36 → ≥ 0.65. Zero-action signed forward ≥ **0.04 m/s** with net/path ratio ≥ 0.5. **This is the gate on everything downstream.** | **DO IN PHASE 2** |
| 4 | **Measurement battery** (§5) | Experimental-match, CoM recovery + KE–PE phase, per-limb GRF sign, variability structure, continuity cost, foot-slip, LDLJ. Converts "looks fake" into numbers. Depends on #2. | 3–5 | none | yes | The battery *is* the metric. Baseline the legacy walker first. | **DO IN PHASE 2** |
| 5 | **Activation lag on all 25 actuators** (`timeconst`, τ = 20–30 ms) | Confirmed available on the `<position>` shorthand since MuJoCo **3.1.6** (Jun 2024): `timeconst` > 0 switches the actuator to `filterexact`, with the compiler reproducing `gainprm=[kp,0,0]`, `biasprm=[0,−kp,−kv]` — **no `<general>` rewrite**. If the 25 actuators share a `<default>` class this is a one-line edit. At 1.1888 Hz: τ=0.02 → 8.5° lag / 0.989 gain; τ=0.03 → 12.6°; τ=0.05 → 20.5° / 0.937 gain. **Caveats:** `na` goes 0 → 25 and `d.act` resets to **0 rad, not the current pose**, so seed `act` at every reset or every episode opens with a sweep transient; `mjSTATE_FULLPHYSICS` grows, so switch snapshots to `mj_getState`; `actearly` is *not* on the shorthand, so if you want lag without the extra one-step control→acceleration delay you do need `<general>`; use the `implicitfast` or `implicit` integrator with kv. | 1.5–2.5 | finetune | partial | Joint-angle PSD above 5 Hz drops ~10×; RMS jerk and RMS action-rate fall sharply | **DO IN PHASE 2** (before the CMA-ES fit, so scalars are fitted through the filter) |
| 6 | **Make `forcerange` bind** | Instrument peak commanded torque per actuator, then cut ceilings to ~2× measured adversarial demand. On the live `gecko_body_r.xml` that means `hip_proret` from ±1.4 to roughly **±0.33 N·m — a ~4× cut**, not the ~290× a static-gravity benchmark implies. Do **not** size against 0.003 N·m: that is a moment about the wrong (vertical) axis and would leave the actuator ~30× too weak to execute its own CPG command. Force-limited actuators produce overshoot-and-settle at touchdown, sag under load, and speed/force trade-offs. | 1.5–2.5 | finetune | yes | Fraction of control steps within 1% of `forcerange`: a few percent in stance, ~0 in swing, instead of exactly 0 everywhere. Also widen `hip_proret` ctrlrange from ±40° toward ≥ 90° to admit the measured 82.57° femur excursion | **DO IN PHASE 2** |
| 7 | **Contact-quality objectives**: foot slip, GRF > 1.2 BW, touchdown velocity | Isaac Lab's `feet_slide` is two lines. These are functions of *measured* contact and body state, so an open-loop feedforward cannot satisfy them — the mismatch between a memorised foot placement and the body's actual state discharges as slip and impact, which is exactly what reads as "CGI." | 1 | finetune | yes | Slip distance per stance < ~0.02 SVL; peak vertical GRF < 1.2 BW; touchdown velocity falls | **DO IN PHASE 2** |
| 8 | **Periodic reward composition** (Siekmann) with **per-limb** swing ratios | Phase-indexed cost on foot force and foot velocity: penalise force during swing, velocity during stance. Set **r_hind = 0.22** (from Jagnandan & Higham's measured 0.78, not the repo's aggregate 0.765) and **r_fore = 0.30**, offsets from limb phase 0.43–0.44. Constrains the instantaneous force profile at every phase rather than the episode mean, and is a function of state, so open-loop is insufficient. This is also the most direct pressure on the §0.1 defects — it penalises exactly the hind swing load and front stance absence that the base currently produces. Compatible with a fixed period. | 2–3 | finetune | yes | Swing-phase force integral → 0; slip → 0; realised duty factors converge to 0.78/0.70 without a duty-factor reward term | **DO IN PHASE 2** |
| 9 | **Mirror symmetry loss (w = 4) + mirrored-tuple augmentation** | Adds `‖π(s) − M_a(π(M_s(s)))‖²` to the PPO loss; DUP appends mirrored trajectories. Asymmetry produces a limp — viewer-visible and invisible to pooled duty-factor means. On 4 vCPU where MuJoCo stepping dominates, DUP doubles the effective batch at zero simulation cost. **Critical detail:** `M_s` must advance the phase channel by half a cycle as well as swapping limbs, or the loss fights the CPG. Note the base is currently *asymmetric by construction* — front press 0.40 (FL) vs 0.50 (FR) — so fix that in #3 first or the loss will fight the base too. | 1.5–2 | finetune | yes | Phase-Portrait Index toward ~0.3; Robinson Symmetry Index on stance duration and actuation | **DO IN PHASE 2** |
| 10 | **Signal-dependent motor noise** | `u' = u + k·\|u\|·ε₁ + σ₀·ε₂`, k = 0.02–0.05. Force variability scales linearly with mean force in voluntary contraction (log-log slope 1.05 ± 0.48; ~0 under electrical stimulation, so it is neural). The gecko wobbles where it pushes hardest instead of chattering uniformly. Depends on #2 — at a 36–59% measurement noise floor this is invisible. | 1 | none | yes | Per-phase variance correlates with per-phase `\|command\|`; joint-excursion CV becomes non-zero *and measurable* | **DO IN PHASE 2, after #2** |
| 11 | **gSDE at train time — and know exactly what it does not buy you** | `use_sde=True`, `sde_sample_freq` a few tens of steps. Genuinely useful as *smooth exploration during training*. But the earlier framing of its eval behaviour was wrong in a way that matters. SB3's own docs (added in 2.7.1, Dec 2025) confirm `model.predict()` never resets the noise — but the result is **not** the deterministic/mean action. It returns `μ(s) + latent_sde(s) @ exploration_mat`: a fixed, repeatable, *deterministic map from observation to action* that still varies smoothly along the trajectory. And `load()` *does* call `reset_noise()` once after `set_parameters()`, so each process load gives a different fixed perturbation — run-to-run variation, zero within-run variation. **The decisive point for this project:** gSDE noise is a linear function of the latent features, so with a CPG-phase-dependent latent the perturbation is itself phase-locked and periodic. **It will not produce stride-to-stride variability**, and resampling faster than one 0.841 s gait period reintroduces the jerk gSDE exists to remove. `reset_noise()` also asserts if the checkpoint was trained with `use_sde=False`, and needs `env.num_envs` or all parallel envs share one matrix. | 0.5 | finetune | yes | Continuity cost `C = 100·E[((a_{t+1}−a_t)/Δa_max)²]` stays in the low single digits. Do **not** score it on joint CV | **DO IN PHASE 2** — but for exploration smoothness only, not as a variability source |

### Tier B — Phase 2.5, after the gates pass

| # | Method | What it is / why it is more real | Days | Retrain | Lock | Metric | Verdict |
|---|---|---|---|---|---|---|---|
| 12 | **Micro-relief terrain (1–10 mm) + substrate compliance/friction randomisation** | Not robustness — the mechanism that converts a feedforward tape into a closed-loop controller. On a flat plane, feedforward is globally optimal, so PPO has no gradient pressure to use proprioception. Fu et al.: energy minimisation on flat ground converged to unnatural gaits in **every** trial. Your 92-D observation is blind and proprioceptive, so relief forces foot-raising and stride-shortening to emerge, which is the measured lizard response (stride 141→122 mm, duty 27%→33%, clearance up, speed unchanged). | 3–5 | finetune | yes | **Re-run the phase-freeze ablation.** If the cost rises well above 16%, the residual has become closed-loop. That number is the best scalar proxy for "is this alive" | **DO IN PHASE 2.5 — highest-value single item in this tier** |
| 13 | **Frequency-gain sweep** (cheap Phase 3 gate, not a Phase 3 commitment) | Add a **scalar** phase-velocity scaling input to the CPG's phase integrator (`_cpg_t += control_dt * g`), freeze the existing residual, sweep g, measure achievable speed range and speed loss. Zaaf Table 3 says frequency and stride length co-vary as a fixed 0.757/0.241 log-log split, so one speed command can in principle drive both. This is the empirical resolution of whether the lock is a mild constraint or the project's ceiling — and it costs hours, not a retrain. | 0.5–1 | none | tests it | Achievable speed range at fixed residual vs the animal's 4.4×. Expect collapse outside ~0.13–0.19 m/s | **DO IN PHASE 2.5 EARLY — gates all of Phase 3** |
| 14 | **Force-level Tegotae** (Zamboni/Owaki), not phase-level | `F_a(φ,F) = −min(0, −σ·cos(φ)·F)` — load-proportional extension torque gated by the locked phase. Derived from the same Tegotae function as the famous phase rule but adds *force*, not phase, so the clock is untouched. Legs coordinate through the ground instead of through a script. | 2–4 | none | yes | Per-leg peak GRF SD across strides rises from ~0; diagonal-pair load-sharing ratio stops being constant; **assert stride period stays 0.8412 s** | **DO IN PHASE 2.5** |
| 15 | **Phase-dependent reflex gain scheduling (reflex reversal)** | Every reflex gain becomes a function of φ, sign-flipping across the stance/swing boundary with a raised-cosine crossfade over ~5% of cycle. Forssberg's stumbling corrective reaction: the same tactile stimulus activates flexors in swing and extensors in stance. Without this, half of every reflex response looks wrong — a foot flinching upward while bearing weight is the loudest robot tell. Also makes the phase channel load-bearing again. | 1–2 | none | yes | Scripted toe-tap at 8 phases: the hip/knee response must invert sign across the stance–swing boundary. An easy before/after image | **DO IN PHASE 2.5** |
| 16 | **Limb-load → spine coupling** (Suzuki et al., *Twister*) | Trunk bends toward the loaded foreleg: `θ̄_b = σ_LB(δ_i − δ_j) − ρτ_b`, first-order follower. Replaces Phase 2's *scripted* spine standing wave with a load-driven one. **Corrected justification — the 2.3× headline does not belong to this term.** The paper's ablation zeroes σ_LB **and** σ_BL simultaneously, so 0.1010 vs 0.0436 m/s is "bending spine vs effectively rigid spine," not "load coupling vs some other spine drive." Worse, on the intact robot an **open-loop trot beat it: 0.1031 m/s at CoT 4.59 against 0.1010 at CoT 4.91**, and the authors say so ("the opposite tendency was observed in the intact condition"). The demonstrated benefit is **fault tolerance** (0.0675/0.0670 vs 0.0605 m/s under leg failure, ~+12%). Also: Twister is 9 DoF, 2.1 kg, **no tail**, salamander/lizard-inspired — not the closest published morphology to a gecko. **What survives, and it is the useful half:** the leg-load→trunk pathway (`f_LB`) reads foot contact and writes only the trunk actuator, never touching `φ`, so it is genuinely drop-in with a locked clock. The reverse pathways (`f_BL`, `f_LL`) write to phase *velocity* and are incompatible. | 2–4 | finetune | yes (f_LB only) | Spine-bend phase vs hip retraction acquires per-stride spread; assert stride period unchanged. Score it as robustness/naturalness with **unknown speed effect** | **DO IN PHASE 2.5 — cheap and compatible, but not on a speed promise** |
| 17 | **FreeMusco temporally-averaged reward window** (T_P = 32 steps, γ = 0.99) | Score the *window mean* of velocity, up-direction and pose instead of per-frame values (keep per-step form for heading and the height floor). Directly targets rigid pitch: a per-step posture reward pins the trunk; a windowed one permits the rhythmic fluctuation real animals show while still holding the mean. Pairs with the mechanical fix in #3. | 1–2 | finetune | yes | Peak-to-peak trunk pitch/roll/yaw per stride **increases** toward lizard values while mean pitch falls | **DO IN PHASE 2.5** |
| 18 | **Trunk-attitude (vestibular) reflex from the existing IMU** | Pitch/roll error and rates drive bounded corrections on hip neutral angles and spine. Converts a static pitch bias into a regulated variable and adds the small continuous postural adjustments that make a slow-walking animal look alive. Nearly free — the sensors exist. Also the most direct attack on the base's heading failure (net/path 11.9%). | 2–3 | finetune | yes | Mean nose-up pitch → a few degrees; pitch RMS about the mean shows corrections rather than a flat offset; lateral-shove recovery time; net/path ratio rises | **DO IN PHASE 2.5** |
| 19 | **Sensorimotor delay: 20 ms observation ring buffer** (Python, not native) | A `deque` in `_get_obs` handing the policy obs[t−1]. The 92-D shape is unchanged, so the checkpoint survives. **Prefer the Python buffer to MuJoCo's native `nsample`/`delay`/`interp`** (real since 3.5.0, and you are on 3.9): native delay adds `mjData.history` to the physics state the moment `nsample > 0`, delays under one timestep are impossible, `nsample` must be ≥ `ceil(delay/dt)` or you get non-causal extrapolation, `interp` must be `zoh` for any discrete sensor, and MJX parity is doubtful. 20–40 ms is the defensible gecko range; at 841 ms stride that is 2.4–4.8% of a cycle, so expect it to be nearly free — add it because it is true, not because it will transform the video. | 1 | none | yes | Return vs delay curve (k = 1, 2) compared against the existing 16% phase-freeze cost | **DO IN PHASE 2.5** |
| 20 | **Nonlinear passive joint springs + tendon dead-band series compliance** | MuJoCo 3.7.0+ polynomial stiffness `f(x) = −(ax + bx² + cx³)` and tendon `springlength` two-value dead-band. Moves joint-limit "pain" out of the reward and into the physics, and gives swing a passive restoring torque. Series elasticity is the catapult effect and the touchdown-impact decoupler; MuJoCo `<muscle>` has **rigid tendons**, so passive spring tendons are the only route to it. | 3–5 | finetune | yes | Fraction of stride within 5° of any joint limit → ~0; elastic energy stored/returned per stride; actuator work per metre falls at constant speed | **DO IN PHASE 2.5** |
| 21 | **Pink-noise, mean-removed stride-period jitter** | Accumulating phase with per-stride `f_k = 1.1888·(1+δ_k)`, `{δ_k}` pink, z-scored, **exactly mean-removed**, clamped to ±5%. Implementation is genuinely compatible: `_cpg_t += control_dt` is incremental, so jittering the increment time-warps phase continuously with no discontinuity and preserves mean cadence; the policy observes only `[sin φ, cos φ]` with no frequency channel, so zero-mean jitter stays in-distribution. **But it is now clearly a Phase 2.5-late item, not an early one:** the target is CV ~1–3% with **α ≈ 0.5** (the metronome signature from Vaz 2024), not "CV 3–10%, α 0.7–1.0"; your measurement floor is currently 36–59% and your logging quantum is 2.4%; and DFA needs 500–600 strides (420–505 s of continuous walking) against a 20 s episode cap. | 2–3 + eval plumbing | finetune | yes | After #2 and a `max_steps` increase: stride-period CV* in the 1–3% band with α near 0.5. Report the fall/termination rate alongside | **DO IN PHASE 2.5, LAST** |
| 22 | **Cost-of-transport reward, multiplicative and ramped** | `R = R_motion × R_energy` with energy normalised by distance, weight ramped rather than set at target. Removes the "stand still to save energy" optimum, which under a locked cadence is otherwise dominant. Fu et al. showed torque, Δtorque, foot slip, joint-speed and action-regularisation penalties all decline as a *byproduct* — five fewer weights to tune. | 1 + sweeps | finetune | partial | CoT tracked across training; the byproduct set must fall together. Target ~1.5 J·kg⁻¹·m⁻¹ external mechanical work | **DO IN PHASE 2.5** |
| 23 | **Per-individual "personality" offsets** | Draw a small fixed asymmetry vector once per episode (limb phase offset, per-joint neutral offsets, tail bias) and hold it constant. Real animals are individually and stably asymmetric — which is why published tables report between-individual SEM much larger than within-animal scatter (see §6.1: 0.15 SVL on a 0.62 SVL mean is a **between-animal** SEM implying ~76% between-animal CV). Zero fixed asymmetry plus zero-mean noise looks like a machine with a fault injector. | 0.5–1 | none | yes | Left–right duty-factor difference non-zero and stable within episode, distributed across episodes. Report within- and between-episode variance separately | **DO IN PHASE 2.5** |
| 24 | **Intermittency: heavy-tailed bout structure on the stop/rest command** | Sample walk/pause durations log-normally rather than on a fixed schedule. Lizards are intermittent locomotors; uninterrupted metronomic progress reads as machine-like even when every stride is individually plausible. Operates at the timescale a human observer actually attends to. **Implement as an amplitude/phase-hold gate, never as a frequency change.** | 3–4 | finetune | partial | Pause fraction 10–55%, individual pauses 0.1–0.35 s, heavy-tailed rather than unimodal. Gate the speed reward on the stop command or the policy learns to ignore stops | **DO IN PHASE 2.5** |
| 25 | **Point-light 2AFC preference study + small expert pool** | 12–20 dots on joint centres, black background, "which looks more natural." Needs no gecko footage: run legacy vs Phase 2 vs Phase 2.5 as a preference matrix. Naive observers identify animals from *dynamic* point-light displays but not static ones, so it isolates motion from every rendering confound. Run 15–20 experts separately — the virtual-animal uncanny valley appears in expert naturalness rankings but not lay ones, and a more realistic gecko can be rated *less appealing* while being rated more natural. Ask naturalness and appeal separately. | 5 | none | yes | "Observers prefer Phase 2.5 over legacy at X% (n=40, p<...)" — a real, honest, publishable naturalness result | **DO IN PHASE 2.5** |

### Tier C — later (Phase 3+), or gated on a measurement

| # | Method | Why later | Days | Retrain | Lock | Verdict |
|---|---|---|---|---|---|---|
| 26 | **Break the frequency lock**: co-varying speed command, or full Owaki phase dynamics inside a cadence PLL | **Promoted in importance by §0.5.** Zaaf's coefficients say frequency carries ~76% of log-speed in this species and a locked 1.1888 Hz confines you to ~0.13–0.19 m/s against the animal's 0.24–1.05 m/s. Design it as Zaaf specifies: one speed command driving frequency and stride length on the measured 0.757/0.241 log-log split, with duty factor falling at b = −0.201. Gated on #13's sweep result. Consumes the checkpoint; the 92-D observation has no phase-rate channel. | 8–15 | from-scratch | **no** | **THE MAIN PHASE 3 ITEM — gated on #13 and on §5.8** |
| 27 | **Emulated antagonistic muscle layer** (Schumacher 2024) | Two software muscles per joint, lengths affine in joint angle, `F_L·F_V·act + F_P`, 10 ms activation filter, derived damping `β = k_damp/(4·a₁·f_max)`. No tendon routing, no wrap geoms, only ~30% slower than PD. On a real quadruped with a deliberately under-specified reward, the muscle agent **lifted its feet clear while the PD agent dragged them** — which is precisely the §0.1 hind-limb failure. Doubles `nu` to ~50 and kills the checkpoint. The locked cadence survives if the CPG emits the EMG-shaped excitation profiles Phase 2 already specifies. | 5–10 | from-scratch | partial | **LATER — the real prize; do it once Tier A/B has proved the body is right** |
| 28 | **Lipschitz-constrained policy** (λ_gp = 0.002) | Replaces smoothness reward terms with one differentiable constraint and one coefficient. But it damps ∂π/∂obs, and on a barely-closed-loop residual that risks pushing it *back* open-loop. | 1 | finetune | yes | **LATER — only after #12 has established closed-loop behaviour; re-measure the phase-freeze number after** |
| 29 | **Full MuJoCo `<muscle>` with tendon routing** | Posture-dependent moment arms are the strongest physically-grounded prior available when there is no mocap. Only route to scoring EMG correlation against Jagnandan & Higham 2018. But: 15–25 days of muscle-path design, no gecko musculotendon data exists, `actuator_lengthrange` is a compile-time optimisation that is the commonest silent failure, and rigid tendons mean you still need #20. | 15–25 | from-scratch | partial | **LATER — after #27 proves the payoff on this body** |
| 30 | **Gecko foot adhesion** (MuJoCo `adhesion` actuator / `geom/adhesion`) | MuJoCo's docs are written around this exact animal. Changes the stance→swing transition: toes engage and peel, late-stance GRF goes negative. But no published quantification of the perceived-realism payoff, and the passive form is always-on which is wrong for walking. | 2–3 | finetune | yes | **LATER** |
| 31 | **Constrained RL (CaT / IPO)** | Replaces a dozen reward weights with thresholds in physical units you can read off the biology papers. Real value, but it is infrastructure work on SB3's PPO and only pays off once weight tuning is genuinely the bottleneck. | 2–5 | finetune | yes | **LATER** |
| 32 | **MJX / MuJoCo Playground migration** | The throughput case is real and large: MuJoCo's own docs report 950K steps/s on an A100 at batch 8192 and 2.96M under MJX-Warp, against your 853 — you are getting roughly 0.1% of what this hardware class delivers. But the blockers are exactly as documented: MJX-JAX is ~10× slower than MuJoCo for a *single* scene and needs thousands of parallel scenes to pay off, convex meshes want ≲200 vertices, throughput degrades faster than MuJoCo as contacts multiply, and it is a different contact-solver path — so the SB3 residual would not transfer and it mandates the from-scratch retrain the plan is rationing. Also: §0.4 says the bottleneck is a stable recipe, not steps/s. | 10–20 | from-scratch | unknown | **SKIP for now — revisit only if #26 needs large parallel sweeps** |

### Tier D — skip, and why

| Method | Why skip |
|---|---|
| **Full dynamics randomisation** (link mass, motor gain, CoM offset, latency) | Exists to close a sim-to-real gap you do not have. It demonstrably trades optimality for conservatism and produces stiff, hedged gaits. Keep only friction and substrate compliance and a little observation noise. Randomising mass or CoM also conflicts with the morphology gates. |
| **`feet_air_time` reward** | Under a locked CPG the air time is already fixed by the phase schedule, so it adds no information — while being a documented hacking route toward stomping and tapping. Use air time as a metric only. |
| **Generic flat-orientation penalty on roll and yaw** | This is the term that will quietly delete the lateral spine wave you are adding on purpose. Penalise *pitch* deviation and vertical CoM oscillation; give roll and yaw a target **amplitude**, not a zero target. |
| **Delay-aware RL** (augmented state, ACDA, VDPO) | These exist to *cancel* delay, the opposite of the goal, and augmenting the state changes the observation dimension. |
| **Walk-These-Ways gait *conditioning*** (8-D command vector with commanded frequency) | Requires a frequency channel in the observation. Transplant the individual auxiliary terms (footswing-height tracking at the largest weight, body pitch/height tracking) at the locked frequency; reject the conditioning. *Revisit if #26 goes ahead* — at that point a frequency channel exists and the conditioning becomes available. |
| **Symmetry `NET` and `PHASE` variants** | `NET` changes the architecture and voids the checkpoint; `PHASE` enforces a predefined cycle timing that your CPG already fixes. |
| **Raibert-heuristic footswing term** | Derived for parasagittal limbs under the hips. On a sprawling gecko it prescribes the wrong touchdown geometry. |
| **Behavioural-repertoire embedding (MoSeq / MotionMapper)** | With one locked gait plus a stop command, the repertoire is nearly a single point. Revisit once there are turns, inclines and startles. |
| **Granular sand simulation** | DEM took roughly a *month* of compute for one second of legged locomotion on a five-million-grain bed. Use `solref`/`solimp` randomisation as the compliance proxy and do not claim it is sand. Also: *E. macularius* is a rock-and-clay animal that reportedly avoids sand-dominated substrate. |
| **Imitation learning workarounds** | Closed, and it would not have helped anyway: MuscleMimic (416 muscles, full mocap) reaches kinematic correlation r = 0.90 yet muscle-EMG correlations of only **0.2–0.6**. Strong kinematic imitation does not guarantee physiological faithfulness. |
| **Native MuJoCo actuator/sensor delay** (as opposed to a Python buffer) | Real since 3.5.0, but see #19: it grows the physics state, has a one-timestep floor, silently extrapolates non-causally if `nsample` is undersized, and has doubtful MJX parity. The Python ring buffer is version-agnostic and keeps your snapshots simple. Note separately that **MuJoCo removed native sensor noise in 3.1.4** — `mjModel.sensor_noise` is now storage only, so all observation noise must be injected in Python with your own seeded RNG. |

---

## 3. The architecture question

**Options:** (A) keep the additive residual, (B) switch to CPG-parameter modulation, (C) go two-level.

### Recommendation: **(C) two-level, staged — but do not spend the retrain until the base walks and holds a heading.**

Concretely: **CPG-RES**. Two policies, trained in two stages.

- **Stage 1** — a *modulation* policy whose action is CPG/pattern-formation setpoints: per-limb amplitude, per-limb phase-offset rate, per-limb swing clearance, step length, body height, spine and tail amplitude. **ω is frozen at 1.1888 Hz and is not in the action space** (with the §0.5 caveat that a single co-varying speed scalar is the first thing to reintroduce in Phase 3). Trained on flat ground for velocity tracking plus the Phase 2 statistics terms.
- **Stage 2** — freeze stage 1 completely; train a *small, tubed* additive joint residual on micro-relief terrain with the contact-quality and periodic-composition rewards.

In SB3 this is two sequential single-head PPO runs with the base frozen in run 2 — not a two-head network. That matters: stage 1 is small, fast, and cheap to iterate.

### Why not (A), keep the additive residual

Keeping it spends the retrain on the architecture that produced the current failure. But the argument is narrower than the earlier draft claimed, so state it accurately:

- Under the **lab** profile the base supplies 58.9% of signed forward speed. It is not powerless.
- What it does not supply is heading: net-to-path ratio 11.9% (base) vs 46.9% (trained). The residual is functioning as a **steering controller**, and the architecture already concentrates its authority there — the 13 actuators left at full 0.25 are hips, ankles, neck, head and tail_lift.
- Freezing the phase observation costs only 16%, so the residual is a near-feedforward tape.
- Off-cadence speed loss is 24–83%, so the tape is calibrated to a clock it does not read.

The structural argument still holds: in **every** published system that works this way — PMTG, CPG-RL, SPRINT, RuN — the base supplies the trajectory *shape* and the residual supplies stabilisation. SPRINT's ablation makes the division measurable: removing the residual leaves top speed **unchanged** (5.96 m/s either way) while naturalness FID degrades 0.80 → 2.33. The residual's job is naturalness and stabilisation; it is not supposed to be the steering system either.

### Why not (B), pure CPG-parameter modulation

Modulation alone throws the residual away, and by SPRINT's and RuN's measurements the residual is precisely what buys naturalness and contact-level stabilisation. It also removes the only channel that can respond to a contact event the base cannot model. And with ω locked you have already given up CPG-RL's main adaptation channel — every published CPG-RL result modulates ω over [0, 4.5] Hz, and **no paper I saw reports what freezing ω costs**. Giving up the residual *as well* stacks two unmeasured losses.

### Why (C) wins, and what it costs

Two-level gives you both properties at once. The gait's shape, timing and published statistics live in a structured, biologically-parameterised base that you can inspect and gate; the residual retains exactly the authority it is good at. The two-stage schedule is what stops the residual from taking over again: in stage 1 the residual does not exist, so the base *must* learn a gait that works alone; freezing it in stage 2 makes it impossible for the residual to later erode.

The uncomfortable cost is **not** the Schumacher quote — that citation does not carry the weight the earlier draft put on it (§1.3). The real cost is measured and on-species: Zaaf's regression says the lock removes ~76% of this animal's speed-control channel and confines the model to a 1.43× speed range against the animal's 4.4×, and Rybak says a central clock is the wrong architecture below 0.3–0.4 m/s. The two-level design does not fix that. It contains it — the clock constrains only the mean cadence, everything else becomes free and sensory-driven — and it leaves exactly one clean seam (a scalar phase-velocity gain) where Phase 3 can reintroduce the missing channel without redesigning the controller.

### The one thing that would change my answer

If the base-walk gate fails — if you cannot get the base above ~0.04 m/s signed forward with a net/path ratio above 0.5 in a week — then the two-level architecture has no base to be the top level of, and **do not spend the retrain.** Go back to `base_ctrl`. Spending a retrain on a base that cannot walk or steer is the one genuinely irreversible mistake available here.

### And a correction to the framing of the question

"Exactly one from-scratch retrain is affordable" is false as a *money* statement but true for a reason the earlier draft missed. A 10 M-step run is $3.28 of instance time and the recorded authorisation is **$60**, so about 18 runs — not 100, and not 134. But money was never the constraint. The constraints are: 4-vCPU serial wall clock (100 runs = 13.6 days), a default 0-vCPU quota on on-demand G instances, and above all the project's own documented finding that **more PPO made the walker slower** — eval reward peaked at 1.5 M steps and fell 63% by 10 M while `approx_kl` blew up to 67. The two-stage plan costs ~$6.55 and is the right shape for that constraint: two short runs with eval-triggered early stopping, not one long one.

---

## 4. Phase 2.5: the concrete plan

Every step has a gate. Do not advance past a red gate — go back and fix, or explicitly record the failure as a Phase 3 requirement.

### Step 0 — Fix the instrument, then freeze the scorecard (2–3 days, before the Phase 2 retrain)

First fix the contact/stride detector, log `commanded_frequency_hz` into traces, and raise trace logging above 50 Hz. Then write the evaluation battery spec (§5) as a JSON schema and commit it. **Pre-register it.** Extend `realism_metrics.py` — `TraceRecorder` and stride segmentation already exist, they are just measuring chatter.

> **Gate 0:** observed same-foot cycle rate equals 1.1888 Hz within tolerance (today it is 3.15–4.70 Hz); `stride_period_cv` falls from 36–59% to single digits; the battery runs end-to-end on the legacy walker and emits a complete scorecard with seeds. Numbers can be bad; the file must be complete and the timing must be real.

### Step 1 — Fix the effort target (0.5 day)

Charge effort on `|τ·q̇|` from `actuator_force × qvel`, never on the residual vector.

> **Gate 1:** in a short run with α ramping, mean total power and mean `|residual|` are **not** anti-correlated.

### Step 2 — Make the base walk *and steer* (5–8 days, no training)

Run the swing-load analysis on the existing traces (10 minutes). Then, in `base_ctrl`: fix the hind lift sign/amplitude so commanded mid-swing toe is above commanded stance toe; cut `front_stance_press` until the front stance pose no longer demands ground penetration, and add a real front swing-clearance term that exceeds it; break the fore-aft spatial symmetry; set the fore/hind amplitude ratio toward 0.54; set `STANCE` toward **0.78** hind (keep 0.70 fore, already correct); and fix the phase convention so realised limb phase is **0.435, not 0.75**. Regression-test after each change; sweep one at a time.

> **Gate 2 (the hard one):** zero-action rollout ≥ **0.04 m/s signed forward** with **net/path ≥ 0.5**; hind swing-phase load fraction < 10% (from 80–89%); front stance load fraction ≥ 0.65 (from 0.29–0.36); realised hind duty within 0.05 of 0.78; realised limb phase within 0.03 of 0.435. **If this fails, stop. Do not train.**

### Step 3 — Fix the plant (2.5–4 days, no training)

`timeconst = 0.02–0.03` on all 25 actuators via a `<default>` class, plus `act` seeding at reset and a switch to `mj_getState`/`mjSTATE_FULLPHYSICS` for snapshots. Instrument peak commanded torque, then cut `forcerange` to ~2× measured adversarial demand (≈ ±0.33 N·m on `hip_proret`, a ~4× cut) and widen `hip_proret` ctrlrange toward ≥ 90° total. Switch to the `implicitfast` integrator. Regression-test the morphology gates and the zero-action rollout after each change.

> **Gate 3:** morphology gates still pass; zero-action speed retained within 20% of Gate 2; joint-angle PSD above 5 Hz down by ~10×; no reset transient in the first 0.2 s of an episode.

### Step 4 — Phase 2 CMA-ES *through the new plant*

The 15–40 CPG scalars must be fitted with the activation filter and force ceilings in place, or you will fit them twice. Fit toward the **published** targets (hind duty 0.78 ± 0.01, fore 0.70 ± 0.01, limb phase 0.43–0.44), not the repo's 0.765 aggregate, and record which paper each target came from.

> **Gate 4:** the Phase 2 statistics gates, **plus** the experimental-match score and CoM recovery computed and recorded (they may be bad — record them).

### Step 5 — Stage-1 retrain: modulation policy, flat ground (the retrain)

Action = per-limb amplitude, phase-offset rate, swing clearance, step length, body height, spine/tail amplitude. ω frozen. Reward = velocity + Phase 2 statistics + contact quality + periodic reward composition with per-limb ratios (r_hind = 0.22, r_fore = 0.30) + mirror symmetry loss + windowed posture terms. `use_sde=True`. **Early-stop on eval reward**, checkpoint every 250 K steps, and do not run past the eval peak — the 10 M-step precedent lost 63% of return after 1.5 M.

> **Gate 5:** the modulation-only policy reaches **0.13–0.15 m/s by itself**, with no residual at all. (Target corrected upward from 0.08–0.10 per §0.5: 0.09 m/s is below the entire measured level range for this species and below the slowest lizard in McElroy's dataset; 0.13–0.19 m/s is the band a 1.1888 Hz cadence can honestly occupy.) This is the pass/fail before spending stage 2.

### Step 6 — Close the loop in the base (3–5 days, folded into stage 1's tail)

Force-level Tegotae. Phase-dependent reflex gain scheduling. Trunk-attitude reflex. Limb-load → spine coupling (`f_LB` only — the leg-load-to-trunk direction; do **not** import Twister's `f_BL` or `f_LL`, which write to phase velocity and break the lock). All added to the **control sum**, never to the 92-D observation.

> **Gate 6:** stride period still 0.8412 s to numerical precision; per-leg peak GRF SD across strides > 0; per-limb GRF sign structure correct (forelimb net braking, hindlimb net propulsive, all lateral forces medial).

### Step 7 — Stage-2 retrain: tubed residual on micro-relief

Freeze stage 1. Train the clipped residual on 1–10 mm relief with `solref` randomised in (0.02, 0.4) and friction randomised. Keep flat ground as curriculum level 0. Same early-stopping discipline.

> **Gate 7:** phase-freeze ablation now costs **> 40%** of return (up from 16%); residual RMS < ~30% of base joint excursion; flat-ground statistics still inside Phase 2 tolerances.

### Step 8 — The frequency-gain sweep (0.5–1 day)

Add a scalar phase-velocity gain, freeze the policy, sweep, and plot achievable speed against g.

> **Gate 8:** you now know, empirically, the width of the speed range the lock permits. If it is materially narrower than 0.13–0.19 m/s, Phase 3's first item is #26 and you should say so in writing.

### Step 9 — Add variability last (3–5 days + eval plumbing)

Raise `max_steps` so continuous walks reach ≥ 500 strides. Then pink mean-removed stride jitter, signal-dependent motor noise, personality offsets, afferent delay, heavy-tailed bout structure.

> **Gate 9:** stride-period CV* in the **1–3%** band with a floor and a ceiling; DFA α near **0.5** (the metronome signature, not 0.7–1.0 — a locked clock should not produce free-walking α, and claiming it would be dishonest); **fall/termination rate reported alongside.**

### Step 10 — Held-out prediction test (1 day) — do not skip this

Freeze the policy and CPG scalars. Then **lock the tail joints** and check whether hindlimb joints become **more flexed**, as Jagnandan & Higham 2017 measured in real tail-restricted leopard geckos. This condition is fully unfitted.

> **Gate 10:** the sign of the effect matches. An unfitted directional prediction is worth more than any statistic you fitted.

### Step 11 — Human study (5 days)

Point-light 2AFC preference matrix across legacy / Phase 2 / Phase 2.5, plus a 15–20 person expert pool run separately. Ask naturalness and appeal as separate questions.

**Total: roughly 6–8 weeks of solo work, two PPO runs with early stopping, and well under $20 of instance time.**

---

## 5. The evaluation battery that cannot be faked

Three tiers. The point of the split is that **Tier 1 is what Phase 2 optimises**, so Tier 1 passing tells you nothing about realism.

### Tier 1 — fakeable. Report, never present as evidence.

Gait-diagram stance overlap; duty factor and limb phase on a Hildebrand grid; per-joint DTW distance and correlation against published excursions; footfall raster. A CPG whose offsets were fitted to limb phase 0.43–0.44 scores near-perfectly on these **by construction**. They prove Phase 2 did what it was told.

One caution learned the hard way (§0.3): report **Hildebrand limb phase** with its convention stated, not the folded `diagonal_pair_separation_cycle`. The two differ by a factor that flips gait classification. Under McElroy's cutoffs, < 50% is lateral sequence, > 50% is diagonal sequence, 37.5–62.5% is trot, and 12.5–37.5% / 62.5–87.5% are singlefoot.

### Tier 2 — the real evaluation. None of it derivable from joint angles.

**5.1 CoM external work: percent recovery and KE–PE phase shift.**
Sum every contact force each timestep, divide by mass, integrate to CoM velocity with the zero-mean-over-stride constraint, build KE_fore-aft, KE_lateral, KE_vertical and PE_grav.

**Follow McElroy's method exactly or the number is meaningless:** phase shift is the **time difference between the minima of KE and GPE, relative to STEP duration, ×360°, normalised to 0–180°**. A *step* is a half cycle — footfall of the first limb in one couplet to footfall of the first limb in the opposite couplet. At 1.1888 Hz stride that is **0.4206 s, not 0.8412 s**. Using stride duration halves the result and turns a genuine 150° walk into a spurious 75° "equivocal." This is the single most likely way to fail the test for a bookkeeping reason.

**Targets (conspecific — use these, not Coleonyx):** *E. macularius*, McElroy et al. 2008 Table 1 — walk 151 ± 6.6° with 32 ± 4.3% recovery at 0.24 (0.20–0.27) m/s; run 8 ± 3.2° with 16 ± 3.9% recovery at 0.29 (0.29–0.32) m/s.

**Report three outcomes, not two:** 135–180° walking, 0–45° running, **46–134° equivocal** (McElroy's own explicit band). Recovery is a reported outcome, not a classification criterion, and the published ranges (18–47% walk / 1–18% run) are ranges of species means for wide-foraging species that overlap at 18%.

**Do not pre-register a pass/fail at 0.09 m/s.** That is below every speed in the reference dataset (slowest walk anywhere: 0.16 m/s), and real *E. macularius* uses bouncing mechanics at 0.16–0.24 m/s — so landing in the bouncing band at low speed is *normal for this species*, not a failure. Needs ≥ 8–10 clean consecutive strides; undefined during any aerial phase. Include all three KE components (Farley & Ko found lateral KE < 5% of total, but it is part of the definition).

**Why unfakeable:** recovery is a consequence of *when* force is applied relative to where the CoM is in its arc. A controller can hit every joint excursion and vault zero energy.

**5.2 Per-limb GRF sign structure.**
Per-foot GRF in normal / fore-aft / mediolateral, integrated per stance, plus the stride phase of each peak.
**Targets (sign structure only — the magnitudes are from a 2.5 g *Hemidactylus* trotting at 0.47 m/s and do NOT transfer):** forelimb net fore-aft impulse **negative** (braking only); hindlimb net fore-aft impulse **positive** and larger; **every** mediolateral force vector pointing medially; lateral mechanical power a large fraction of total.
**Why unfakeable:** a residual that pushes all four legs backwards reproduces the kinematics and gets the forelimb sign backwards.

**5.3 GRF and joint waveform shape via SPM1D, not peaks.**
Time-normalise stance to 0–100% and test the whole curve against published mean±SD bands with 1-D statistical parametric mapping. Discrete summary statistics are exactly what CMA-ES fits; SPM1D detects "peaks right, rise/fall shape wrong."
**Caveat:** needs published mean±SD *curves*, not table values. Jagnandan & Higham Table 1 gives excursions. Use SPM1D for policy-vs-policy comparison and be honest about the reference gap.

**5.4 Experimental-match score — and a warning about its calibration.**
Percentage of the gait cycle for which the cycle-averaged simulated trajectory lies inside ±1 SD of published data, computed jointly over joint angles **and** a force channel.

**The 0.73 "ceiling" does not survive scrutiny.** Its source (Schumacher et al., arXiv:2309.02976v2, Sep 2023) is an **unpublished preprint**, never peer-reviewed, whose metric has not been reused by any citing work. Six problems with using it as a target:

- The paper never calls it a ceiling. On the simplest model, a hand-tuned **reflex controller scored 0.68 against RL's 0.67** — a non-learning baseline matched it.
- **The complexity story is backwards.** The simplest body (9 DoF / 18 muscles) scored **0.67**, *lower* than the 16-DoF body's 0.73. The score is not monotone in complexity.
- The numbers are **best-of-10-seeds cherry-picked checkpoints**; the ±0.01 is rollout variance of one hand-picked policy, not seed variance.
- The metric "serves as an optimization metric for our cost terms" — the reward weights were tuned to maximise it. **In-sample.**
- **Engine confound:** 0.73 is Hyfydy, closed commercial software (€995/yr personal, no free licence). The only MuJoCo number in the table is **0.43**.
- Falisse et al. (2019, *J. R. Soc. Interface*) produced demonstration-free physiologically realistic walking on a **29-DoF / 92-muscle** model — more complex than anything in that table — which directly refutes "complex body ⇒ 0.43–0.50 is the best achievable."

**And there is no gecko reference band.** The metric's reference is Bovi et al. 2011 (*Gait & Posture* 33(1):6–13) — 40 healthy humans, normative mean±SD per % of cycle. No such corpus exists for *E. macularius*. Building one from the published kinematics is a **prerequisite deliverable**, and until it exists any absolute target (including "0.40–0.55 is strong") is an unanchored import across species, morphology, engine and DoF count.

**Use it as a policy-vs-policy comparator, not a headline score.** Report per-signal, never pooled (pooling hides which joint is wrong and is trivially inflated by adding wide-SD signals). Report cross-correlation at optimal lag alongside — the authors themselves warn that "relatively natural gaits can still achieve a low metric if the angles are slightly shifted," and a *locked* cadence differing from the reference recording's cadence loses score for phase reasons alone. And note the Jagnandan & Higham SDs are **between-individual**, which makes the band generous — declare it.

**5.5 Cost of transport.**
External mechanical work per metre against ~1.5 J·kg⁻¹·m⁻¹. Being 10× high means the residual is burning torque against itself; being 10× low usually means skating (cross-check foot slip). Report mechanical work as the primary number; if you convert to metabolic, state the assumed efficiency — 1.5 ml O₂·kg⁻¹·m⁻¹ ≈ 30 J·kg⁻¹·m⁻¹ metabolic against 1.5 J·kg⁻¹·m⁻¹ mechanical is ~5% whole-animal efficiency, and conflating the two is an easy error.

**5.6 Physical-plausibility trio (one hour).**
Foot-slip distance during stance, floor penetration depth, joint-velocity jitter. Catches gross artefacts free. Note the front stance press currently commands ~14 mm of penetration by design (§0.1c) — this test would have caught it.

### Tier 3 — cannot be faked at all, and these are where the lock bites

**5.7 Variability structure.**
`CV* = (1 + 1/(4n))·CV` on stride period, stride length and duty factor; DFA α on the stride-period series; lag-1 autocorrelation; continuity cost `C = 100·E[((a_{t+1}−a_t)/Δa_max)²]`.

**Prerequisite, not optional:** today this measures foot chatter, not gait. `stride_period_cv` reads 36–59% at 3–5 detected cycles per commanded cycle, `commanded_frequency_hz` is absent from traces so the mismatch guard is silent, and the 0.02 s event resolution is 2.4% of the stride period — so a biological 1–3% CV is barely one quantum. Fix the detector and raise logging above 50 Hz first, or every number here is noise.

**Data budgets differ sharply:** CV needs ~15 strides; DFA needs ≥ 500–600 strides (Ravi et al. 2020), which is 420–505 s of continuous walking against your current 20 s episode cap — and with `reset_noise=0.0` plus deterministic evaluation the episodes are byte-identical, so extra seeds add nothing. Mylonas et al. (2026) find DFA "rarely exceeded moderate reliability" below 100 strides and recommend the Hurst–Kolmogorov process instead. **Never report α from a short rollout, and consider not reporting α at all** — no reptile α exists to compare against.

**Targets, corrected:** a system driven by a fixed clock should land at **α ≈ 0.5 with CV ≈ 1–3%** — that is the measured signature of metronome pacing in humans (Vaz et al. 2024: CV 1.3 ± 0.3%, α 0.52 ± 0.14; against free walking's CV 1.7 ± 0.6%, α 0.91 ± 0.16). Chasing α in 0.7–1.0 under a locked clock would be claiming a property the architecture cannot honestly have. Continuity cost in the low single digits.

**The gate has a floor AND a ceiling.** Animals actively *minimise* inter-stride variability and change gait to escape high-variability states. Elevated stride-time CV in humans is a clinical marker — ataxia 5.65 ± 3.64%. Always report CV next to the fall/termination rate.

The one-hour claim was wrong: this needs a detector fix, a `max_steps` change, and a long rollout. Budget days, not hours — but it remains the cheapest measurement that proves in one number whether the timing is biological.

**5.8 Duty-factor and stride-frequency vs dimensionless speed.**
Normalise as `û = v/√(g·h)` with h = **extended hindlimb length**, not standing hip height. Sweep speed and plot duty factor, relative stride length and relative stride frequency against û.

**The biology side of this question is now settled**, so run the sim side against a known answer rather than as an open question. Zaaf 2001 Table 3, *E. macularius*, level: `log10(stride frequency) = 0.819 + 0.757·log10(v)` (r² = 0.97); `log10(stride length) = −0.812 + 0.241·log10(v)` (r² = 0.71); duty factor slope **−0.201** (r² = 0.61). Frequency carries ~76% of log-speed; stride length ~24%; duty factor falls with speed. All three are significant.

**This test will fail by construction:** with a locked cadence the frequency slope is exactly zero and 100% of speed change must be carried by stride length. **Run it anyway**, and report the achievable range against the animal's: 1.43× (yours, held to the animal's stride-length envelope) versus 4.4× (Zaaf's measured level range, 0.24–1.05 m/s). That ratio is the single strongest argument in the repo for whether Phase 3 must unlock frequency.

**5.9 Perturbation battery.**
Calibrated impulses at known stride phases (ladder: 0.05, 0.1, 0.2, 0.4 BW·s — pick it once and never change it), a single unexpected step-height change, a friction patch, a tail mass perturbation. Measure recovery time, max lateral CoM excursion, heading error, whether recovery is a step-placement adjustment or a whole-body flail, whether the gait re-entrains, and fall rate vs impulse. Plus distance walked on unseen ±5° slopes.
**This cannot be gamed by any amount of statistics matching**, because summary statistics are defined on unperturbed steady locomotion. **Prediction:** near-zero rejection today.

### Reporting discipline

Never report one number. Measured correlations of individual objective motion metrics against human opinion run from **0.144 to 0.436**, while a seven-feature linear combination reaches **0.961**. Report the full vector plus one weighted composite, publish the weights, and keep **verification** (timestep and solver convergence, energy conservation, contact-parameter sensitivity, seed variance) reported separately from **validation** (comparison to biology). Hold out at least one joint or one speed from any weight-fitting.

And one more: **report which XML and which gait profile every number came from.** Half the errors this review found were legacy-vs-lab or `gecko_body_r`-vs-`lab_v2` confusions in numbers that were individually correct.

---

## 6. What still will not be right

Blunt, in descending order of how much it matters.

**6.1 There is no stride-to-stride variability data for *Eublepharis macularius*, and there may never be.**
Jagnandan & Higham 2017 recorded 3–5 forelimb and hind limb strides per individual **per tail treatment** from ten animals (36.3 ± 1.9 g, SVL 104.6 ± 2.1 mm) and states verbatim: "Averages of each kinematic variable for each individual per tail treatment were used for all statistical analyses." Table 1 is captioned "**Means + residuals (±s.e.m.)**," so the hind-limb original-tail values 0.62 ± 0.15 SVL and 0.78 ± 0.01 are **speed-residual-adjusted, between-individual standard errors over n = 10** — two transformations removed from stride-to-stride dispersion. (Forelimb: 0.63 ± 0.15 and 0.70 ± 0.01 — say which limb, or the duty factor is wrong.)

The naive-misuse arithmetic is worth putting in writing: 0.15/0.62 read as a CV gives 24%; converting the SEM to a between-individual SD (0.15 × √10 = 0.47) gives a **~76% between-animal CV**. For duty factor, 0.01 × √10 = 0.032, a ~4% between-individual CV, not the 1.3% a naive reading suggests. Neither is a stride-to-stride number.

The table is also not row-wise self-consistent: hind stance 0.64 s at duty 0.78 implies 1.22 Hz; fore 0.54 s at 0.70 implies 1.30 Hz; speed 1.23 SVL/s ÷ stride 0.62 SVL implies 1.98 Hz. Step length is tabulated at 0.04–0.06 SVL against a stride length of 0.62 SVL, which looks like a units mislabel. **Cells cannot be algebraically combined**, let alone mined for within-individual CV. (The implied ~1.22 Hz does loosely bracket the locked 1.1888 Hz — treat that as a happy coincidence, not a derivation.)

The paper has no data availability statement and no deposited stride table, and reports no SD or CV anywhere. So the within-individual component is not recoverable from the publication, and every CV number you set will be an *inference from lizards generally*.

The one thing that could change this is the Ross et al. 2013 Dryad deposit (doi 10.5061/dryad.dn822, CC0, `finallocomotiondata.xls`, 2.45 MB), backing Ross et al. 2013 *Evolution* 67(4):1209–1217 — whose stated subject is variance in locomotor cycle periods, exactly the quantity a locked clock pins. **The gekkonid row is unverified:** both Dryad file endpoints returned 401/403 to unauthenticated fetch this session, and the abstract names taxa only generically ("lizards, alligators, turtles, salamanders"). Download it by hand and grep for Gekkonidae / *Eublepharis* / *Hemidactylus* / *Gekko* before settling any band. **Do not assume a gecko is present.** If there is no gecko row, say so in the write-up.

**6.2 There is no gecko musculotendon architecture data.**
No PCSA, no fibre lengths, no pennation, no moment arms, no tendon slack lengths for any gecko limb muscle. If you go to muscle actuation (Tier C #27/#29), every parameter is a guess. The DeepMind/RVC canine model is the warning: deriving peak force from PCSA **failed** to track reference motion; MuJoCo's automatic `scale/actuator_acc0` route worked. MuJoCo's own advice is that the muscle **operating range** matters more than the FLV curve shape "and in many cases this parameter is unknown."

**6.3 The frequency lock removes this animal's primary speed channel. This is now settled, not open.**
The apparent contradiction in the source material was an artefact: the "PNAS 2015 = stride length" source studies African padless geckos, not *Eublepharis*, and its stride-length sentence is about inclines, not level speed control. Delete it.

Zaaf et al. 2001 Table 3 (**Table 3**, not Table 2 — Table 2 is the tokay) gives the answer: frequency carries b = 0.757 of a total 0.998, stride length b = 0.241, duty factor b = −0.201, all significant. A locked 1.1888 Hz corresponds to v ≈ 0.104 m/s on that regression — below the entire measured level range (0.24–1.05 m/s) and inside the animal's *climbing* band. Held to the animal's own stride-length envelope the lock permits **0.13–0.19 m/s, a 1.43× range against the animal's 4.4×**.

Two honest caveats. The evidence is **n = 3 animals, 22 strides**, and it is the only primary measurement that exists; no 2024–2026 work re-measures it. And the paper is closed access (© The Company of Biologists) — cite the coefficients with attribution, do not redistribute the table.

Nothing in Phase 2 or 2.5 fixes this. It is a Phase 3 architecture decision, §5.8 is how you measure it, and Tier B #13 is the cheap gate.

**6.4 There is no established ceiling, and the number people quote for one does not support it.**
The "best demonstration-free controller reaches 0.73" framing does not survive: it comes from an unpublished 2023 preprint, the paper never calls it a ceiling, a hand-tuned reflex controller matched it on the same body, the score is non-monotone in complexity (0.67 on the *simplest* model), it is a best-of-10-seeds checkpoint scored on the same metric the reward weights were tuned against, and 0.73 is on closed commercial software while the only MuJoCo number is 0.43. Meanwhile Falisse et al. (2019) produced demonstration-free realistic walking on a 29-DoF/92-muscle model.

What is true and useful: **there is no published normative reference band for gecko gait, so there is no scale on which to be scored.** Building that band is a prerequisite, and once built, the score's magnitude will depend on the band's own SD width — so cross-project comparisons are meaningless. Report per-signal, report lag-corrected, compare your policies to each other, and do not promise a number.

**6.5 Imitation would not have rescued physiological realism anyway.**
MuscleMimic (2026, 416 muscles, full mocap, GPU) reaches mean joint-kinematics correlation r = 0.90 and yet muscle-EMG correlations spanning only **0.2–0.6** across independently trained policies. The missing gecko mocap was never the path to physiological faithfulness. The closed door was not the important one.

**6.6 Only two muscles have EMG timings, and they are hindlimb.**
Caudofemoralis and gastrocnemius (Jagnandan & Higham 2018, *JEB* 221:jeb179564). Nothing for the forelimb, nothing axial, nothing for the tail. So the EMG-shaped pattern layer is a two-muscle stencil generalised across 25 actuators, and the EMG-correlation realism axis can only ever be scored on those two. Worth noting what those two say about the current base: caudofemoralis fires at or just before footfall and "persisted throughout the entire stance phase," and gastrocnemius "remained active throughout much of stance" — against a base that commands **zero knee excursion within stance**.

**6.7 No 3-D CoM mechanics for this species at your speed.**
McElroy's Table 1 *does* contain *E. macularius*, which is better than the earlier draft assumed — but its walk and run rows sit at 0.24 and 0.29 m/s, and the slowest animal anywhere in that 18-species dataset walks at 0.16 m/s. Everything below that is extrapolation. Farley & Ko's 51% is a pooled maximum across two species (*Coleonyx variegatus*, ~4.2 g, and *Eumeces skiltonianus*), not a Coleonyx value, and typical lizard recovery is nearer 25%.

**6.8 The perceptual claim is weaker than the measurement claim.**
The direct evidence that added variability makes *simulated animal* motion look more animal-like is essentially one qualitative 1995 graphics paper. The modern character-animation perception literature is dominated by human faces and gesture. **Claim the statistics, not the aesthetics** — and if you run the 2AFC study, report it as a preference between your own policies, not as "indistinguishable from a real gecko," which you have no stimulus to test.

**6.9 The live model is not the validated model.**
`gecko_body_lab_v2.xml` passes the 14 morphology gates at 38 g, but the default in `gecko_walk_env.py`, `train_walk_ppo.py` and `morphology_audit.py` is still `gecko_body_r.xml` at **61.2 g** — about 1.6× too heavy for its length, which inflates every ground reaction force, joint torque and `forcerange` by the same factor. v2's own header says it "has NOT passed gait validation or fresh residual training." Until the default is switched and a policy is trained on it, every dynamics number in this document is a `gecko_body_r` number. Say which body each result came from, every time.

**6.10 Everything downstream of walking.**
Adhesion and setal mechanics; thermal dependence (a nocturnal ectotherm's speed and reflex latency are temperature-dependent, and your model has no temperature); vision and the head camera; prey capture, tail autotomy dynamics, defensive posture, breathing; the sit-and-wait behavioural ecology that means a real leopard gecko spends most of its time not walking at all. Phase 2.5 makes the walk look like a gecko walking. It does not make the animal look like a gecko.

**6.11 One honest structural caveat on the whole plan.**
Nearly every published gain, threshold and coefficient cited here comes from kilogram-scale robots, human models, or cat/fly preparations. Twister is 2.1 kg with no tail. Schumacher's models are human bipeds at 1.2 m/s. The DFA and CV targets are human treadmill data. Your animal is 38 g (or 61 g, see 6.9) at 0.13 m/s with a 0.73-SVL tail and a sprawling posture. **Take the structure and the sign; never the magnitude.** Every gain in Tier A and B needs refitting at gecko scale, and the effort of that refitting is not in the day estimates.

---

## Appendix: file paths for the work above

- **Live body** (the default, 61.2 g): `C:\Users\ziyad\GeckoBrain\morphology\gecko_body_r.xml` — `hip_proret` kp 0.85, forcerange ±1.4, ctrlrange ±40°
- **Candidate body** (38 g, opt-in, not gait-validated): `C:\Users\ziyad\GeckoBrain\morphology\gecko_body_lab_v2.xml` (actuators ~365–393)
- Base CPG and residual: `C:\Users\ziyad\GeckoBrain\envs\cpg_residual_controller.py` — `FREQ_HZ` at 33/39; `AMP`/`STANCE`/`FRONT_STANCE`/`PHASE` at 39–44; `RESIDUAL_OVERRIDES` at 59–67; `_limb_signals` at 69–74; `residual_scale`/`front_lift_residual_scale`/`lock_front_lift` at 81–82; front-lift zeroing at 150–153; foot phase at 165–169; `commanded_contacts` at 177; front overrides in `base_ctrl` at 202–215; spine/tail at 226–231; `compute` at 236–241
- Observation (92-D), step metrics, defaults: `C:\Users\ziyad\GeckoBrain\envs\gecko_walk_env.py` — `DEFAULT_XML` at 34, `residual_scale` default 0.2 at 71, `max_steps` at 68, controller wiring at 140–141, `_obs` at 174/209, `_step_metrics` at 202, phase channels at 221, action clip at 310, `_cpg_t` advance at 318
- Phase observation construction: `C:\Users\ziyad\GeckoBrain\rewards\gait_prior.py:112-114`
- Gait profiles and the additive/subtractive convention warning: `C:\Users\ziyad\GeckoBrain\common\gait_config.py`
- Targets and provenance notes: `C:\Users\ziyad\GeckoBrain\config\proxies.yaml` (`hind_duty_factor` 0.765 "Chosen aggregate target"; `limb_phase_range`; `lab_gait_touchdown_delays`; `legacy_residual_scale` "not biological")
- Metrics harness to extend: `C:\Users\ziyad\GeckoBrain\realism_metrics.py` — `complete_strides` at 99, `analyze_trace` at 133, contact/duty at 198–233, `stride_period_cv` at 224, diagonal-pair separation at 247–251, zero-residual flag at 464, deterministic predict at 586
- Evidence and traces: `C:\Users\ziyad\GeckoBrain\artifacts\evidence\baseline\`, `...\legacy_zero_residual\`, `...\lab_zero_residual\` (each with `report.json` and `traces\episode_*.json`)
- Decision records: `C:\Users\ziyad\GeckoBrain\docs\BUILD_LOG.md` (the $60 authorisation; the dynamic candidate checks table), `C:\Users\ziyad\GeckoBrain\docs\locomotion_v4_final_results.md` ("Stop V4 locomotion training"), `C:\Users\ziyad\GeckoBrain\docs\research\gecko_scorecard.md`, `C:\Users\ziyad\GeckoBrain\docs\BLOCKED.md`
- Compute baseline: `C:\Users\ziyad\GeckoBrain\aws_results\aws_10m_autostop.log` ("Using cpu device"; eval peak at 1.5 M; `approx_kl` 67.26 at 10 M)
- Licence: `C:\Users\ziyad\GeckoBrain\LICENSE` (Apache 2.0)

---

## Sources

**Gecko and lizard biomechanics**

- Zaaf A., Van Damme R., Herrel A., Aerts P. (2001). Spatio-temporal gait characteristics of level and vertical locomotion in a ground-dwelling and a climbing gecko. *J. Exp. Biol.* 204(7):1233–1246. doi:10.1242/jeb.204.7.1233 (PMID 11249834). **Closed access.** *E. macularius* data in Table 3.
- Jagnandan K., Higham T.E. (2017). Lateral movements of a massive tail influence gecko locomotion: an integrative study comparing tail restriction and autotomy. *Sci. Rep.* 7:10865. doi:10.1038/s41598-017-11484-7 (PMC5589804, CC BY 4.0).
- Jagnandan K., Higham T.E. (2018). Neuromuscular control of locomotion is altered by tail autotomy in geckos. *J. Exp. Biol.* 221(18):jeb179564. doi:10.1242/jeb.179564.
- Jagnandan K., Russell A.P., Higham T.E. (2014). Tail autotomy and subsequent regeneration alter the mechanics of locomotion in lizards. *J. Exp. Biol.* 217:3891–3897. doi:10.1242/jeb.110916 (PMID 25267844).
- McElroy E.J., Hickey K.L., Reilly S.M. (2008). The correlated evolution of biomechanics, gait and foraging mode in lizards. *J. Exp. Biol.* 211(7):1029–1040. doi:10.1242/jeb.015503 (PMID 18344476). *E. macularius* in Table 1.
- Farley C.T., Ko T.C. (1997). Mechanics of locomotion in lizards. *J. Exp. Biol.* 200(16):2177–2188. doi:10.1242/jeb.200.16.2177 (PMID 9286099). **Paywalled.**
- Reilly S.M., McElroy E.J., Odum R.A., Hornyak V.A. (2006). Tuataras and salamanders show that walking and running mechanics are ancient features of tetrapod locomotion. *Proc. R. Soc. B* 273:1563–1568 (PMC1560307).
- Autumn K., Farley C.T., Emshwiller M., Full R.J. (1997). Low cost of locomotion in the banded gecko. *Physiol. Zool.* 70(6):660–669. doi:10.1086/515880 (PMID 9361140).
- Chen J.J., Peattie A.M., Autumn K., Full R.J. (2006). Differential leg function in a sprawled-posture quadrupedal trotter. *J. Exp. Biol.* 209:249–259.
- Higham T.E., Birn-Jeffery A.V., Collins C.E., Hulsey C.D., Russell A.P. (2015). Adaptive simplification and the evolution of gecko locomotion. *PNAS* 112(3):809–814. doi:10.1073/pnas.1418979112 (PMC4311805). **Does not study *Eublepharis*.**
- Fuller P.O., Higham T.E., Clark A.J. (2011). Posture, speed, and habitat structure: three-dimensional hindlimb kinematics of two species of padless geckos. *Zoology* 114(2):104–112 (PMID 21392953).
- Cartmill M., Lemelin P., Schmitt D. (2002). Support polygons and symmetrical gaits in mammals. *Zool. J. Linn. Soc.* 136:401–420. doi:10.1046/j.1096-3642.2002.00038.x
- Ross C.F. et al. (2013). The evolution of locomotor rhythmicity in tetrapods. *Evolution* 67(4):1209–1217. doi:10.1111/evo.12015 (PMID 23550769). Data: Dryad doi:10.5061/dryad.dn822 (CC0). **Gekkonid row unverified.**

**Variability, rhythm and gait analysis**

- Vaz J.R., Cortes N., Gomes J., Jordão H., Stergiou N. (2024). *J. Biomech.* 165:111972. doi:10.1016/j.jbiomech.2024.111972 (PMC11034849).
- Hausdorff J.M., Purdon P.L., Peng C.-K., Ladin Z., Wei J.Y., Goldberger A.L. (1996). *J. Appl. Physiol.* 80(5):1448–1457. doi:10.1152/jappl.1996.80.5.1448 (PMID 8727526).
- Hausdorff J.M. et al. (1995). Is walking a random walk? *J. Appl. Physiol.* (PMID 7713836).
- Ravi D.K., Marmelat V., Taylor W.R., Newell K.M., Stergiou N., Singh N.B. (2020). *Front. Physiol.* 11:562. doi:10.3389/fphys.2020.00562 (CC BY).
- Mylonas V., Wiles T., Kim H., Stergiou N., Likens A.D. (2026). *Ann. Biomed. Eng.* 54(6):1844–1858. doi:10.1007/s10439-026-04011-1 (PMID 41686388).
- Di Bacco V., Gage W. (2024). *Sensors* (PMID 39598953).
- Bovi G., Rabuffetti M., Mazzoleni P., Ferrarin M. (2011). A multiple-task gait analysis approach: kinematic, kinetic and EMG reference data. *Gait Posture* 33(1):6–13. doi:10.1016/j.gaitpost.2010.08.009

**Control, RL and simulation**

- Schumacher P., Geijtenbeek T., Caggiano V., Kumar V., Schmitt S., Martius G., Haeufle D.F.B. (2025). Emergence of natural and robust bipedal walking by learning from biologically plausible objectives. *iScience* 28(4):112203. doi:10.1016/j.isci.2025.112203 (PMC12002607, CC BY).
- Schumacher P. et al. (2023). Natural and Robust Walking using RL without Demonstrations in High-Dimensional Musculoskeletal Models. arXiv:2309.02976v2. **Unpublished preprint.**
- Falisse A. et al. (2019). Rapid predictive simulations with complex musculoskeletal models. *J. R. Soc. Interface*. doi:10.1098/rsif.2019.0402 (PMC6731507).
- Suzuki S., Kano T., Ijspeert A.J., Ishiguro A. (2021). Sprawling Quadruped Robot Driven by Decentralized Control With Cross-Coupled Sensory Feedback Between Legs and Trunk. *Front. Neurorobot.* 14:607455. doi:10.3389/fnbot.2020.607455 (PMC7820706, CC BY).
- Rybak I.A. et al. (2024). *eLife* — speed-dependent regimes of spinal locomotor circuits.
- Herbert-Read J.E., Romenskyy M., Sumpter D.J.T. (2015). A Turing test for collective motion. *Biol. Lett.* 11:20150674.
- Johannink T. et al. (2018). Residual Reinforcement Learning for Robot Control. arXiv:1812.03201.
- Raffin A., Kober J., Stulp F. (2022). Smooth Exploration for Robotic Reinforcement Learning. *PMLR* 164:1634–1644 (arXiv:2005.05719).
- Kim & Lee (2025). FreeMusco. arXiv:2511.14205.
- Bellegarda G. et al. CPG-RL. arXiv:2211.00458. · AllGaits. arXiv:2411.04787. · Gait in Eight. arXiv:2503.08375.
- Zakka K. et al. (2025). MuJoCo Playground. arXiv:2502.08844 (Apache-2.0).

**Software documentation**

- MuJoCo XML reference — `<position>` `timeconst`: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position (added in 3.1.6, Jun 2024)
- MuJoCo changelog — 3.5.0 actuator/sensor delays and `mjData.history`; 3.1.4 removal of native sensor noise: https://mujoco.readthedocs.io/en/stable/changelog.html
- MuJoCo modeling — Delays section: https://mujoco.readthedocs.io/en/stable/modeling.html
- MuJoCo computation — actuation model and contact: https://mujoco.readthedocs.io/en/stable/computation/index.html
- MuJoCo MJX feature parity and throughput: https://mujoco.readthedocs.io/en/stable/mjx.html
- Stable-Baselines3 — gSDE inference note (added 2.7.1, Dec 2025): https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html and `.../sac.html`; source at https://github.com/DLR-RM/stable-baselines3 (MIT)
- MyoSuite — MyoLeg specification: https://myosuite.readthedocs.io/en/latest/suite.html (Apache-2.0)
- AWS EC2 instance quotas (G/VT default 0 vCPUs): https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html
- g5.xlarge / c7i.xlarge on-demand pricing: https://instances.vantage.sh/aws/ec2/g5.xlarge, https://aws-pricing.com/c7i.xlarge.html