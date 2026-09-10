# Analysis brief — hand this to another AI together with `NeuroGecko_Complete_Record.pdf`

Copy everything below the line into a fresh conversation, and attach the PDF.

---

## Who you are for this task

You are a research assistant with web access, working for someone who is
building a biologically-grounded simulation of a leopard gecko (*Eublepharis
macularius*) in the MuJoCo physics engine. The attached PDF is the complete
internal record of that project: 221 numbered hypotheses with verdicts, 143
registered parameters with provenance tags, 212 harvested citations, 107
commits, and a written specification for the parts that have not been built.

**Do not be agreeable about this document.** It was written by the system that
built the project, and it has a documented history of asserting things it had
not checked — including inventing a citation outright (ledger row #215) and
stating that no published footage of gecko prey capture existed when it does.
Your job is not to summarise it. Your job is to find where it is still wrong.

Take your time. This is not a task to answer in one pass. Read the whole PDF
first, then search, then read the actual sources, then come back and revise.
Prefer reading one paper properly over skimming ten.

---

## The single most important rule of this project

Every number in the simulation is tagged **PUBLISHED**, **DERIVED** or
**INVENTED**, and says which. The project's stated position is that an invented
number labelled as invented is fine, and a plausible number wearing a published
number's clothes is the worst possible defect. Everything you do should serve
that rule: move numbers from *invented* to *published* where a real measurement
exists, and move them from *published* to *invented* where the citation does not
support what is claimed.

A second rule matters for how you answer: **the project never accepts time
estimates.** Do not give any. Give orderings and dependencies.

---

## Task 1 — Verify the bibliography. Do this first.

Part 14 of the PDF lists 212 author-year strings and 51 DOIs harvested from the
repository's own text. The document states plainly that **none of them has been
checked against a bibliographic database.** Check them.

For each entry, especially the ones the project leans on most (Jagnandan 2014
and 2017, Fuller 2011, McElroy 2008, Delheusy 1995, Vollin 2021, Kröhnke 2023,
Bergel 2026, Frýdlová 2026, Masseck 2008, Rawat 2019, Zaaf 2001, Autumn 1994
and 1997, Ewert 1969, Han & Carr 2024, Prescott 2024, Girard 2005, Hoy 2019,
Yilmaz & Meister 2013, Whitford 2022, Wainwright 1991, Werner 1998/2005/2008,
Wever 1964):

1. **Does it exist?** Search Google Scholar, PubMed, Crossref (`api.crossref.org`),
   Semantic Scholar, and the publisher directly. Resolve every DOI at
   `https://doi.org/<the DOI>`.
2. **Is the species right?** This project has already been caught attributing
   leopard-gecko work to authors who studied other families. For every citation
   used to justify a parameter tagged `E_macularius`, confirm the paper's actual
   study species.
3. **Does the paper say what the parameter claims?** Where you can reach the
   full text or a preprint, check the specific number. The project's registry
   entries carry a `source` field with journal, volume and pages — verify those
   too.
4. **Flag every entry you cannot resolve.** "Could not retrieve" is a valid and
   useful answer. Do not fill a gap with something plausible.

Report as a table: `citation | exists? | species studied | number checked | verdict`
where verdict is one of CONFIRMED / WRONG SPECIES / MISQUOTED / NOT FOUND / PAYWALLED.

---

## Task 2 — Attack the named gaps

Part 15 lists quantities the project says have never been measured and has
therefore refused to invent. For each, search hard before accepting the gap.
Search in English and, where relevant, in the languages of the herpetological
literature (German, Czech, Russian, Chinese) and in the grey literature: theses,
zoo and husbandry research, conference abstracts, preprints on bioRxiv.

The gaps, in the order they would unblock the most work:

1. **Visual prey detection in any gecko.** Receptive field size, size tuning,
   speed tuning, contrast threshold, detection latency, acuity in cycles per
   degree. The project claims *nothing* exists for any gecko and is currently
   importing all of it from a mouse superior colliculus, a zebrafish larva and a
   toad. Look for optic tectum recordings in any gekkotan, any nocturnal lizard,
   any squamate.
2. **Whether this species hunts visually at all.** The only quantitative sensory
   study on *E. macularius* is about predator detection and found chemical cues
   dominant. If a study exists showing visually guided prey capture in this
   species or a close relative, it changes the whole architecture.
3. **Anything between 2 cm and 30 cm of a gecko prey-capture sequence.** The
   project has good numbers for the final lunge (from *Coleonyx*, same family)
   and nothing at all for the approach.
4. **Cost of transport, sprint maximum, endurance, thermal performance curve,
   CTmin** for *E. macularius* or any eublepharid.
5. **Any field study of wild leopard geckos** — home range, activity budget,
   diet, microhabitat. The project believes none exists.
6. **A fitted habituation curve or time constant for any lizard or any reptile.**
   The document calls this the single biggest gap in the learning module.
7. **Memory decay at hours, days or weeks in any lizard.** Everything published
   is at two months or beyond, and a hunt lasts seconds.
8. **A false-positive rate for any biological prey detector** — toad, mouse,
   fish larva, lizard. The document says no such measurement exists anywhere.
9. **Cricket walking-vibration spectrum on sand**, for a substrate-vibration
   sense.
10. **A sleep-cycle period for *E. macularius* specifically.** The project has a
    published bracket of 84–134.5 s across lizard species and refuses to pick a
    number inside it. The source paper reportedly gives per-species values only
    as a violin plot with raw data on request — see whether the raw data,
    supplementary material or a later paper prints it.

For each: either give the measurement with its citation and sample size, or
state clearly that you searched and found nothing, and say where you searched.

---

## Task 3 — Adjudicate the contested claims

The document flags several places where the evidence disagrees with itself.
Read the primary sources and take a position.

- **The two reptile hippocampal-homologue lesion studies contradict each other.**
  Turtles lost place learning and kept cue learning; whiptail lizards never used
  a place strategy at all. Neither animal is a gecko. Which, if either, should a
  gecko model follow, and on what grounds other than convenience?
- **The Ewert prey-recognition response law.** The document says it is three
  separate univariate relations (velocity, contrast, size, each measured with
  the others held constant) and *not* one joint surface, and that the
  worm/anti-worm tuning curves are unreachable. Confirm or correct this from
  Ewert's actual papers, and find the numeric curves if they are reachable.
- **The whiptail position-over-feature reversal advantage.** The document says
  this claim failed verification — that the advantage held during acquisition
  only, not reversal. Check it.
- **The nightly retreat loop.** The document says the claim that leopard geckos
  leave a shared retreat at sunset and return before dawn traces to a hobbyist
  magazine article recirculated through Wikipedia and care sheets, with no
  sample size, method or peer review. Trace it yourself and say where it
  actually comes from.
- **The "tight hide" rule.** Every care sheet states geckos prefer a snug
  crevice. The document says there is zero primary support in this species. Find
  the primary source or confirm its absence.
- **Whether the 82.9 % strike success rate transfers.** It comes from *Coleonyx*,
  a different genus in the same family. The simulation reproduces 89.7 %.
  Is that a validation or a coincidence?

---

## Task 4 — Check the arithmetic that was DERIVED

These numbers are not printed in any paper. The project computed them and says
so. Recompute them from the published points and say whether the derivation
holds.

- **Memory decay constants τ ≈ 3.79 months (route precision) and τ ≈ 4.32
  months (goal-finding).** Derived by interpolating between two published
  retention points and assuming an exponential form that was never tested, and
  by treating a hazard ratio as a linear index of memory strength. Is that
  defensible? If not, what should it be?
- **The learning curve shape.** The per-trial hazard ratio (1.09) and both
  ceilings (3.02× latency, 4.59× path) are published. The trajectory between
  them is drawn as an exponential and is invented. Does any published data
  constrain the shape?
- **The approach speed floor, 0.018 m/s.** Scaled from a mouse figure of
  0.05 m/s against a 0.164 m/s locomotor speed, applied to this animal's
  0.055 m/s walk. Is proportional scaling right here?
- **The angular size at which vision starts guiding approach, 6–8 degrees.**
  Derived geometrically from a 2 cm cricket at 15–20 cm. The document calls it
  the most load-bearing and weakest number in the whole prey rule.

---

## Task 5 — Challenge the architecture, not just the numbers

Read Part 2 (how one step runs), Part 9 (the eye) and Part 11 (the plan), then
answer these as an outside reader who is allowed to say the plan is wrong.

- The blocker is that the eye reports something on **5.4 % of steps** during a
  hunt. The proposed fix is a two-channel wide-field/narrow-field detector
  borrowed from mouse superior colliculus. **Is that the right fix for a low
  detection rate, or does a low detection rate point at something else** — the
  field of view, the head not being pointed at anything, the prey being out of
  frame, the sampling rate?
- The animal is nocturnal, afoveate, with a large pupil and short focal length.
  Every imported tuning constant comes from diurnal, foveate animals. **How big
  is that correction, and does anyone know?**
- The walking rhythm is hand-written and the trained residual is switched off
  because it was measured as worse. **Is a hand-written CPG a defensible
  scientific choice, or a convenience?** What would it take to show it is right?
- The document says only about one third of leopard geckos orient by the
  relationship between cues, and the rest may be running a non-cognitive motor
  search. **Should a single simulated animal model the majority strategy or the
  interesting minority one?**
- The learning numbers come from an aversive forced-swim escape task, and the
  simulation is about hunting. **Is there any principled way to carry a learning
  rate across motivational systems, or should the module say it cannot?**

---

## Task 6 — Find what the project has not thought of

Search for recent work (2023 onward, and check for anything in 2026) on:

- Reptile or lizard neuroscience relevant to any of the eight modules.
- Simulated animals built the same way — physics-engine models validated against
  published biology rather than trained to a reward. Note what they did about
  the same problems (provenance, gaps, oracle removal, validation).
- MuJoCo or physics-simulation work on sprawling-posture locomotion.
- Anything on *Eublepharis macularius* at all. It is a common laboratory and pet
  species and the project may simply have missed papers.

---

## How to report back

Write one document with these sections, in this order:

1. **Verdicts on the bibliography** — the table from Task 1, with counts:
   how many confirmed, wrong species, misquoted, not found.
2. **Gaps closed** — every quantity from Task 2 you found a real measurement
   for, with citation, sample size, study species, and the exact value.
3. **Gaps confirmed** — every one you searched for and could not find, with
   where you searched. This is as valuable as the previous section.
4. **Contested claims, adjudicated** — Task 3, with your position and the
   primary source you based it on.
5. **Derivations checked** — Task 4, with your recomputation.
6. **Architecture challenges** — Task 5. Be direct. If the plan is aimed at the
   wrong problem, say so and say what the right one is.
7. **What the project missed** — Task 6.
8. **The ten things you would change first**, in order, with the dependency for
   each. No time estimates.

For every claim you make, give the source. Mark anything you could not verify as
unverified. **If you are not sure, say you are not sure** — this project would
rather have a recorded gap than a confident guess, and it has the ledger to
prove why.

---

## What not to do

- Do not accept a number because the PDF states it confidently. The PDF is the
  output of the system being audited.
- Do not fill a gap with a value from a related species without saying so, and
  without naming the species and the distance between it and *Eublepharis
  macularius*.
- Do not cite a paper you have not opened. The project has already been burned
  by exactly that.
- Do not reproduce long passages from any paper. Summarise, cite, and quote at
  most a short phrase.
- Do not give schedules, timelines or effort estimates for anything.
