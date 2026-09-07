# NeuroGecko — build plan for module 3, and the basal-ganglia verdict

Everything below that says "measured" was run this session. Scripts are in `C:\Users\ziyad\AppData\Local\Temp\claude\C--Users-ziyad\121b3aeb-0730-4e14-a147-5e6b46ea2c93\scratchpad\` (`freqsweep.py`, `lab_zero2.py`, `fig5_pilot4.py`).

---

## 1. VERDICT ON THE PLAN

**"Brainstem + spinal CPG next" is right. The order inside it is backwards, and the skeptic is half right about why.** The skeptic's structural point holds and I confirmed it by reading the code: the cord is not an oscillator. `envs/cpg_residual_controller.py:296` computes `(t*freq + offset) % 1.0` — one global clock, closed form, no integrated phase state, no coupling, no drive variable. Its only runtime inputs are `base_ctrl(t, front_contact, heading_error)`. The brainstem's declared output is "d_limb, d_axial, L/R asymmetry, per-limb amplitude tweaks" (`best_achievable_brain.md:156`); of those, exactly one — L/R asymmetry — has anywhere to land today, via `heading_error`. Build the drive first and three of its four channels write to nothing. **The skeptic's headline claim is false, though, and it matters.** The claim that the CPG contributes about 2 % of the walking rests on a zero-action measurement I reproduced exactly (+0.00228 ± 0.00065 m/s, 5 seeds, reset noise 0.02) — but that measurement uses the default body `gecko_body_r.xml` and the *legacy* gait profile. The accepted walker is neither. The committed gate evidence (`artifacts/evidence/session3/gate2_all_limbs.json`) records `xml_sha256` = `gecko_body_lab_v2.xml`, `gait_profile: lab`, `controller: zero residual with contact reflex` — the accepted 4/6 walker **is** the open-loop CPG base, and it produces 19 complete strides at a measured 1.1892 Hz against a commanded 1.1888, hind duty factor 0.733, path speed 0.0553 m/s. On that configuration I measured 0.0493 m/s path speed at zero action. The cord walks. What 0.00228 m/s measures is *net* displacement of a walker that curves, on a configuration nobody accepted. So: keep module 3, invert its two halves — write the cord as four integrated per-leg oscillators with a drive input, then hang the drive on it — and close the basal ganglia first, because it is cheap now and it is the brainstem's declared input.

---

## 2. THE BASAL GANGLIA BLOCKER

**Verdict: (a) for the sub-blocker, (b) for the module. The definition is obtained; the test that was blocking is void; a better published test is now in hand and the module fails it.**

What was tried and what each route returned:

- **The recorded blocker** (`docs/BLOCKED.md`, Session 6b) says the switching-bout definition was in supplementary materials "which PMC lists but which were not retrievable." **Refuted.** The definition is in the *main text*, Section 3.1.3, freely readable: an uninterrupted run of time steps sharing the same winner with e_w ≥ 0.05 is one bout. The supplementary archive was also obtained and is on disk (`scratchpad/suppl/s001/…`), including the authors' C++ and two data workbooks.
- **The comparison the blocker rests on is undefined, not failed.** The published 21.3-against-7 is a count of bouts over "the first avoidance sequence and first foraging sequence of each trial", and 7 is the *task-topology minimum* (2 avoidance sub-behaviours + 5 foraging), not a rate. The authors say in the same sentence that they preferred this to "counting bouts (or switches) within a fixed time interval" — which is exactly what `tools/dopamine_sweep.py:48` does (`TRIAL_S = 120.0`). Disembodied, the denominator does not exist. **0.55× is not an inverted reproduction; it is a quantity with no published counterpart.**
- **The right test exists and is disembodied.** The supplementary workbook *Study 1 Data (Figures 5, 6, 7).xlsx* (CC BY) holds the non-embodied model's own selection statistics at **61 dopamine levels**: percentage of competitions ending in no / partial / clean / distorted / multiple selection, plus mean efficiency and mean distortion. Two channels, salience grid, no robot, no task, no bouts. That is a like-for-like acceptance test for a like-for-like model.
- **Pilot run against it (mine, 25×25 salience grid on [0.04, 1.00], settled to fixed point, 2 channels).** Ours over published:

| λ | none | partial | clean | distorted | multiple | mean eff | mean distortion |
|---|---|---|---|---|---|---|---|
| 0.20 | 5.12 / **4.32** | 46.88 / **16.72** | 47.36 / **78.61** | 0.64 / **0.34** | 0 / **0** | 0.751 / **0.954** | 0.079 / **0.046** |
| 0.29 | 4.16 / **4.17** | 16.64 / **0.01** | 54.08 / **73.16** | 19.52 / **6.74** | 5.60 / **15.93** | 0.894 / **1.000** | 0.219 / **0.104** |
| 0.43 | 2.72 / **3.34** | 6.72 / **0.10** | 45.12 / **59.75** | 19.52 / **11.48** | 25.92 / **25.33** | 0.940 / **0.999** | 0.400 / **0.162** |

  Every ordering reproduces, and the onset of distortion lands on the published step (first non-zero at λ = 0.20, ours 0.64 % against 0.34 %). But partial selection is 17–47 points too high, clean is 15–31 too low, and mean winner efficiency never reaches 1.0. That is one fault with one signature: **our winner does not saturate**. It is a diagnosis, not a tuning target — rule 2 applies, and if it cannot be closed from a paper it becomes a ledger row.
- **A separate defect found while doing this:** the paper's distortion is Eq. (3), `d_w = 2(Σe − e_w)/Σe`. `brain/basal_ganglia.py:422` computes `(Σe − e_w)/e_w`. Different quantity, so the recorded "distortion rises with lambda" reproduction was never against the published measure.
- **Do not wire the module yet regardless.** In `winner_take_all=True` mode `gates()` returns all zeros and `selected()` returns `None` one line after `step()` names a winner (`basal_ganglia.py:315` writes gate-space values into a pre-activation state variable). The accessor the brainstem is specified to consume is the broken, untested one.

---

## 3. BUILD PLAN

**Step 1 — Replace the basal-ganglia acceptance test.**
Build `tools/selection_sweep.py`; vendor the published targets as `artifacts/published/prescott2024_figure5.csv` with CC BY attribution. Two channels, salience grid, run to the fixed point (the paper runs its model to convergence every robot step, so τ cannot enter this test).
*Acceptance test:* reproduce the class orderings and the three published onsets — first non-zero distortion, first non-zero multiple selection, the λ of peak clean selection — across all 61 levels, and record the percentages without gating on them until the grid protocol is matched.
*Compares against:* Prescott et al. 2024, Biomimetics 9(3):139, Figure 5 / Study 1 workbook.
*Retire* the fixed-window switch count in `tools/dopamine_sweep.py`; keep the file and its evidence, add a second ledger row rather than editing.

**Step 2 — Implement the paper's distortion.**
`brain/basal_ganglia.py`: add `distortion_dw()` = `2*(sum(e) - max(e))/sum(e)`; keep the existing measure under its own name.
*Acceptance test:* mean d_w per λ against the workbook's `dis_mn` column (0.000 at λ ≤ 0.14 rising to 0.217 at λ = 0.60).

**Step 3 — Fix the winner-take-all accessor.**
`brain/basal_ganglia.py:315`, plus a test that calls `gates()` rather than passing gates in.
*Acceptance test:* `gates()` equals `step()`'s return in both modes. No published comparator — this is a code-correctness test and it is named as one.

**Step 4 — Rebuild the frequency measurement as a tool.**
`tools/frequency_sweep.py`. The §4b table in `no_gecko_and_appearance.md` is the single most load-bearing number for module 3 and no script in the repository can regenerate it.
*Acceptance test:* reproduce that table, or record that it does not reproduce. **No published comparator — this is a reproducibility step, not a biology one**, and the plan says so rather than pretending otherwise.

**Step 5 — The cord: four integrated phase oscillators.**
New `brain/spinal_cpg.py`, consumed by the controller; do not edit the closed-form path out. Per-leg φ̇ᵢ = ω − σ·Nᵢ·cos(φᵢ), σ exposed, σ = 0 reducing to a pure clock.
*Acceptance test A (regression):* at σ = 0 and ω = 1.1888 Hz the actuator trace is bit-identical to today's `base_ctrl`. This project has done exactly this before — hypothesis 24 kept the intact animal bit-identical to twelve decimals.
*Acceptance test B (published):* with load feedback on, the gate battery must still give limb phase 0.435 ± 0.03 and hind duty factor 0.78. *Compares against:* McElroy et al. 2008 limb phase 44 ± 1.1 % walking / 43 ± 1.8 % running, corroborated independently by Usherwood & Self Davies 2017 (median 43, SD 3.5); Jagnandan & Higham 2017 hind duty 0.78 ± 0.01, n = 10.

**Step 6 — Pattern formation from the published EMG.**
`common/emg_pattern.py` already holds *Eublepharis macularius* hindlimb bursts and is imported by nothing but tests. Drive the hindlimb synergies through it.
*Acceptance test:* activation onsets, durations and peaks inside the registered burst windows, and hypothesis 24 still reproduces (tail block collapses hindlimb excursion by −20.7 % / −16.5 %).
*Compares against:* Jagnandan & Higham 2018, J Exp Biol 221:jeb179564, Table 3, n = 10.

**Step 7 — The descending drive.**
New `brain/brainstem.py`: one drive scalar d → ω through a saturating map with a lower and an upper threshold, plus a left/right asymmetry channel writing to the existing `heading_error` path.
*Acceptance test:* (i) raising d raises stride frequency monotonically inside the band; (ii) a symmetric drive produces symmetric bilateral output; (iii) turning is produced only by the asymmetry channel.
*Compares against:* Cabelguen et al. 2003 (salamander, direction only — see the provenance table); Ryczko et al. 2016 (salamander, 53 ± 1 % / 47 ± 1 %, n = 3); Ijspeert et al. 2007 (model, turning by left/right drive asymmetry only), corroborated in the target family by Wang et al. midbrain stimulation in *Gekko gecko* eliciting left and right turns.
**The gait-selection channel of the brainstem has no acceptance test and must not get one by invention.** This species does not change footfall pattern with speed: Zaaf 2001 reports relative phase not significant across 0.2–1.1 m/s, and McElroy measures 44 ± 1.1 % walking against 43 ± 1.8 % running. Build the channel, wire it, leave it at a fixed value, and record it as present-but-unvalidatable.

**Step 8 — The frequency lock (only if steps 5–7 need it).**
The lock lives in three places: `common/gait_config.py:15` and its `__post_init__` guard, `rewards/gait_prior.py:31/76`, `envs/cpg_residual_controller.py:40`. Unlocking kills the 92-D checkpoint (one global sin/cos phase pair cannot carry four decoupled phases) and forces a retrain.
**This step has no acceptance test short of a retrain, and the plan says so.** Do not start it until step 4 has produced a re-runnable cost.

---

## 4. PROVENANCE TABLE

Only numbers the plan above actually uses.

| Value | Species / system | Source | Tag |
|---|---|---|---|
| Selection-class percentages at 61 λ (e.g. λ 0.20: none 4.323, partial 16.724, clean 78.610, distorted 0.343, multiple 0) | rate model, n/a | Prescott et al. 2024, Biomimetics 9(3):139, Fig 5 / Study 1 workbook, CC BY | PUBLISHED |
| Mean distortion per λ, 0.000 → 0.217 | rate model, n/a | same workbook, `dis_mn` | PUBLISHED |
| Selection labels: full ≥ 0.95, partial 0.05–0.95, unselected < 0.05 | rate model, n/a | Prescott 2024 §3.1.3, verbatim | PUBLISHED |
| d_w = 2(Σeᵢ − e_w)/Σeᵢ | rate model, n/a | Prescott 2024 Eq. (3) | PUBLISHED |
| Bout = uninterrupted run of one winner with e_w ≥ 0.05 | rate model, n/a | Prescott 2024 §3.1.3 | PUBLISHED |
| Gating constant c = 0.169 | rate model, n/a | Prescott 2024 §3.1.2 | PUBLISHED — but **NOT_IN_CORPUS**: absent from `docs/research/` and from `config/proxies.yaml`. Fix: register it with the section reference. |
| GPR weights 0.9 / 0.3 | rat model | Gurney/Prescott/Redgrave 2001 via Fox et al. 2009 | PUBLISHED, secondary. Corroborated by the resting output 0.16953 reproducing published c. |
| τ = 40 ms | rat model | Fox et al. 2009, verbatim, open access | PUBLISHED, **contested** (25 ms Girard variant; 10 in the ABRG SpineML release, which ships a striatal-threshold sign error). Does not enter step 1, which runs to convergence. |
| Limb phase 43–44 % of stride | *Eublepharis macularius* | McElroy et al. 2008 Table 1; Usherwood & Self Davies 2017 Table 1 | PUBLISHED, two independent labs |
| Hind duty 0.78 ± 0.01 (n = 10), fore 0.70 ± 0.01 | *E. macularius* | Jagnandan & Higham 2017 Table 1 | PUBLISHED |
| Hindlimb EMG bursts (caudofemoralis onset −0.0168, duration 0.8145, peak 0.1401, amplitude 0.5373; two gastrocnemius bursts) | *E. macularius*, n = 10 | Jagnandan & Higham 2018 Table 3; registered as `emg_hindlimb_preautotomy` | PUBLISHED |
| Tail-block hindlimb collapse −21 % / −17 % | *E. macularius* | already reproduced, hypothesis 24 | PUBLISHED |
| MLR drive raises locomotor frequency monotonically | *Notophthalmus viridescens*, n = 11 (both modes in 7 of 11) | Cabelguen et al. 2003 | PUBLISHED **direction only**. The absolute values are not transferable: that preparation steps at 0.08–0.16 Hz, an order of magnitude below the gecko's stride. |
| MLR → reticulospinal drive 53 ± 1 % ipsi / 47 ± 1 % contra, n = 3 | *N. viridescens* | Ryczko et al. 2016, CC BY | PUBLISHED. Note: this argues the MLR is *not* where left/right asymmetry arises, so it cannot ground `drive_L ≠ drive_R`. |
| Drive thresholds d_low = 1, d_high(limb) = 3, d_high(body) = 5 | model + robot, arbitrary drive units | Ijspeert et al. 2007 | PUBLISHED, but a **model**, not an animal, and single-source; the offered corroboration is inside the same paper. |
| Mouse model oscillation floor ~2 Hz; walk 2–4, trot 4–10.5, gallop 9–11, bound 11–12 Hz | mouse | Danner et al. 2017 (the model Ausborn 2019 defers to) | PUBLISHED. Consequence: the gecko's 1.19 Hz is **below** the mouse model's floor, so the whole range must be rescaled before anything is copied. |
| Midbrain tegmentum near midline elicits walking and left/right turns | *Gekko gecko* | Wang, Guo, Sun & Dai, Springer SCI 192 ch. 9, doi 10.1007/978-3-642-00264-9_9 | PUBLISHED, **abstract only** — paywalled, and the two venues disagree on n implanted (10 vs 20). **NOT_IN_CORPUS.** Fix: obtain the chapter; it is the only MLR work in the target family. |
| Stride frequency 2.03 ± 0.18 Hz (SEM, n = 4) at 0.18 m/s | *E. macularius* | Fuller, Higham & Clark 2011, open access | PUBLISHED. **NOT IN THE REGISTRY** — `config/proxies.yaml` has 116 entries and no published stride frequency. Fix: one row. |
| log₁₀(f) = 0.819 + 0.757·log₁₀(v), r² = 0.97, N = 22 strides, 3 animals | *E. macularius* | Zaaf et al. 2001 Table 3, transcribed at `beyond_statistics_realism.md:116-137` | **Provenance unresolved.** The same corpus says three times (`gecko_scorecard.md:60, 312, 537`) that this paper is closed-access and only its abstract is readable. Fix: obtain the PDF (Antwerp repository or the Herrel publications page) or record the session that produced the transcription. Until then it must not be cited as PUBLISHED. |
| 1.1888 Hz locked gait frequency | n/a | `proxies.yaml:735`, "legacy checkpoint-compatible controller frequency" | **INVENTED.** It cannot become published. It can only be replaced, and only by breaking the lock (step 8). |
| Frequency band 1.0–1.2 Hz | n/a | `gait_prior.py:31` | **INVENTED** — ±10 % around the line above. It excludes the corpus's own derived 1.22 Hz walk value and is far below the only published absolute figure, 2.03 Hz. |
| σ, the load-feedback gain in φ̇ᵢ = ω − σ·Nᵢ·cos(φᵢ) | robot | Owaki & Ishiguro 2017, Sci Rep 7:277, CC BY | **NOT_IN_CORPUS.** The equation is named in the corpus; no value for σ, ω or N is anywhere in the repository, and neither Owaki 2017 nor Suzuki 2021 has been opened. Fix: both are free; open them before step 5. |
| Zero-action net forward speed 0.00228 ± 0.00065 m/s (legacy profile, `gecko_body_r.xml`, 5 seeds) | n/a | measured this session | DERIVED |
| Accepted walker open-loop path speed 0.0553 m/s, 19 strides at 1.1892 Hz, duty 0.733 | n/a | `artifacts/evidence/session3/gate2_all_limbs.json`; corroborated at 0.0493 m/s this session | DERIVED |
| Pilot selection percentages in section 2 | n/a | measured this session, 25×25 grid | DERIVED |

---

## 5. NEW LEDGER ENTRIES

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 71 | The open-loop CPG contributes about 2 % of the walking; the residual does the rest | **Refuted** | reproduced the +0.00228 ± 0.00065 m/s zero-action figure exactly, then found it is the *legacy* profile on `gecko_body_r.xml`. The accepted walker (lab profile, `gecko_body_lab_v2.xml`) is the open-loop base and walks: 19 strides, 1.1892 Hz measured against 1.1888 commanded, duty 0.733, path 0.0553 m/s |
| 72 | The brainstem's declared output has somewhere to land in the current cord | **Refuted** | `base_ctrl(t, front_contact, heading_error)`; φ is closed-form; no ω, no drive, no per-leg phase state. One of four declared drive channels is consumable |
| 73 | The switching-bout definition needs the supplementary materials | **Refuted** | it is in the main text, §3.1.3, freely readable. Second occurrence of the #51 → #58 pattern: a wall recorded without being verified |
| 74 | The 0.55× switching ratio is a failed reproduction | **Refuted** | the published quantity is bouts over the first avoidance plus first foraging sequence against a task-topology minimum of 7; the authors explicitly declined fixed-interval counting, which is what `dopamine_sweep.py:48` does. Disembodied, the denominator does not exist |
| 75 | A published *disembodied* acceptance test for the basal ganglia exists | **Confirmed** | Prescott 2024 Study 1 workbook: 61 dopamine levels × five selection classes plus mean efficiency and mean distortion, CC BY, obtained |
| 76 | The module reproduces that table | **Partly** | all orderings reproduce and distortion onset lands on the published step (λ 0.20, ours 0.64 % against 0.34 %); partial selection is 17–47 points high, clean 15–31 low, mean winner efficiency never reaches 1.0. One signature: the winner does not saturate |
| 77 | `brain/basal_ganglia.py` computes the paper's distortion | **Refuted** | module uses (Σe − e_w)/e_w; Eq. (3) is 2(Σe − e_w)/Σe |
| 78 | `config/proxies.yaml` holds a published stride frequency | **Refuted** | 116 entries; the only frequency is `lab_gait_frequency_hz` = 1.1888, species INVENTED |
| 79 | This species changes gait with speed, so the brainstem's second channel has a target | **Refuted** | Zaaf relative phase n.s. over 0.2–1.1 m/s; McElroy 44 ± 1.1 % walk against 43 ± 1.8 % run |
| 80 | The published EMG pattern is wired into the controller | **Refuted** | `common/emg_pattern.py` is imported by tests and nothing else |
| 81 | MLR stimulation has never been done in a gecko | **Refuted** | Wang, Guo, Sun & Dai elicited walking, phonation and left/right turns from the midbrain tegmentum of *Gekko gecko*; abstract confirmed independently, chapter paywalled |
| 82 | My settle loop converged (my own method error) | **Refuted** | breaking on two equal gate vectors fired during the clipped-at-zero transient and reported 100 % no-selection at every λ. Fifth entry in the meta-ledger class "the failure was in my test" |
| 83 | My first walker measurement used the accepted configuration (my own method error) | **Refuted** | it used `gecko_body_r.xml` and the legacy profile; the accepted body is identified only by the `xml_sha256` in the gate evidence file |

---

## 6. WHAT IS STILL UNKNOWN

- **Why our basal ganglia's winner does not saturate.** Uncertain. The signature is single-cause and clean, but the cause is not identified, and it must not be found by adjusting constants.
- **The exact protocol behind Figure 5.** The paper says 50 × 100 × 100 runs; the workbook has 61 dopamine columns and 10 210 grid rows. The salience bounds, the settle criterion and the channel count for that figure are not fully recoverable from the text. Until they are, absolute percentages are context, not a gate.
- **σ, ω and N in the Owaki law.** Not in the repository, not in the corpus, and neither source paper has been opened. Both are free. Step 5 cannot start honestly without them.
- **Every gecko MLR parameter.** The one paper in the right family gives locus and elicited modes; currents, pulse widths and frequencies are not in the abstract and the chapter is paywalled. Everything quantitative in the drive is salamander or mouse.
- **The Zaaf coefficients.** The corpus prints them in one file and declares the paper unreadable in another. One of those is wrong and I could not determine which.
- **Whether breaking the frequency lock is affordable.** The only evidence is a table that no script in the repository can regenerate.
- **Whether the cord rewrite keeps 4/6 gates.** Unknown until built. The σ = 0 bit-identity test protects the starting point; it says nothing about what happens when σ > 0.
- **τ.** Still three values. Irrelevant to the Figure 5 test, which runs to convergence; relevant to anything embodied, because our module relaxes for the caller's interval and Prescott's does not. That is an architectural difference, not a parameter one.
- **No gecko spinal circuit exists.** The one gecko spinal-lesion study (Wang, Wang & Dai 2022, CC BY) is behavioural: no stride frequencies, no duty factors, no phase values. The ceiling here is unchanged.

Sources consulted outside the repository this session: [Wang et al., *Locomotion Elicited by Electrical Stimulation in the Midbrain of the Lizard Gekko gecko*](https://link.springer.com/chapter/10.1007/978-3-642-00264-9_9); [Wang, Wang & Dai 2022, *The Neural Control Mechanisms of Gekkonid Adhesion Locomotion*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9332208/).