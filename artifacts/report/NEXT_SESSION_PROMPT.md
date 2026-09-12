# NeuroGecko — the brief for the next session

Paste this whole file as the first message of the new chat, together with
`NeuroGecko_Complete_Record.pdf`.

---

**READ EVERYTHING FIRST. DO NOT BUILD ANYTHING YET.**

You are continuing NeuroGecko, a biologically-grounded leopard gecko
(*Eublepharis macularius*) simulated in MuJoCo at `C:\Users\ziyad\GeckoBrain`.
The attached PDF is the complete record: 348 numbered hypotheses, 152
parameters, 217 works cited, the whole body, the whole brain, and everything
that has been refuted along the way.

## 1. Go and read the research. All of it. Actually open it.

Before you touch one line of this project, go and read every research paper,
every agent finding and every document listed below. Not the summaries — the
sources.

1. `docs/FAILURE_MAP.md` — all 348 numbered hypotheses, in order, including the
   284 refuted ones. The refuted rows are the map of where not to look and are
   the most valuable thing in the repository. Read the verdicts, not just the
   claims.
2. `docs/GECKO_SURFACE_ANATOMY.md` — the 42-agent adversarial research pass on
   this animal's surface. All 10 surviving claims, all 26 refuted ones, and all
   6 could-not-verify sections. The refuted half is the half that matters.
3. `docs/LATERAL_UNDULATION.md` — the 24-agent pass on how far a walking
   lizard's spine actually bends. All 18 surviving numbers with their species
   distance stated, the 2 refuted, the 4 gaps.
4. `config/proxies.yaml` — all 152 parameters with their provenance tags. Know
   which are PUBLISHED, which are DERIVED and which are INVENTED before you
   reason about any of them.
5. The bibliography in the PDF — 217 works, 51 with DOIs. Open the ones your
   task depends on. **Do not cite a paper you have not opened.**
6. `docs/BUILD_LOG.md`, `docs/DECISIONS.md`, `docs/BLOCKED.md` and
   `docs/EYE_SPEC.md` — the long-form context behind the ledger rows.
7. `CLAUDE.md` — the standing rules. They are not advisory.

Understand this animal **deeply** before you act: its 38.000 g and every
proportion in Parts 19–21 of the PDF, what each of its 38 joints can do, which
of its 14 morphology gates are load-bearing, which of its 8 brains exist, which
of its 6 behaviours it can actually reach, and which of its numbers are
inventions wearing a published number's clothes. Every single thing this gecko
needs, you should be able to state from the sources, not from memory.

## 2. Then come back knowing it, and say where we are resuming

**We continue from fixing the four pending behaviour items** (Part 26 of the
PDF):

1. `rest` is structurally unreachable — its salience IS fatigue, which plateaus
   at **0.011** in an animal that strolls, against a basal-ganglia release
   threshold measured at **0.2008**. Short by a factor of eighteen.
2. `groom` is structurally unreachable — a tonic salience of **0.05** against
   the same 0.2008 floor. Grooming in this species has a real published trigger
   (shedding, eye-cleaning), so the likeliest correct fix is that the salience
   is the wrong *quantity*, not the wrong size.
3. `flee` releases, but only when the animal is not hungry — it clears the floor
   by **0.009**; with hunger 0.30 present it needs 0.2171 and has 0.2100. Its
   salience IS the published probability of a defensive reaction in this species
   (0.21 to scent, 0.28 with sight, n = 42). The model's standing prediction is
   that a starving gecko does not flee from something standing on it.
4. Brain 6, the day/night clock, is a switch wired to nothing — `clock.step` is
   never called and `_arousal` is the constant 1.0, while `info["arousal"]`
   reports a number no module produced.

That is the resumption point. Do not start somewhere else because something
else looked easier.

## 3. Ask before you start

**When you have finished reading, report back with what you found, then ASK THE
USER FOR PERMISSION TO START.** Do not begin editing, fitting, rebuilding or
filming until the user has said yes. State clearly which of the four items you
propose to take first and why, and what evidence would settle it.

## The rules that are not negotiable while you work

- Every number is PUBLISHED, DERIVED or INVENTED and says which, in
  `config/proxies.yaml`. A plausible number wearing a published number's clothes
  is the worst defect available.
- **Never tune until the answer matches.** If the model needs its constants
  adjusted to reproduce a published result, the constants must come from a paper.
- The reward is what the model is told to want; the gates are what the animal
  does. Only the second counts.
- A failed prediction stays in the table. Adding a knob per failing joint is
  fitting the harness to the answer.
- Measure before claiming — including your own tests and your own solver.
- One source is not a citation.
- Do not cite a paper you have not opened. Do not fill a gap with something
  plausible. Mark anything you could not verify as unverified. Do not fill a gap
  with a value from a related species without naming the species and the
  distance.
- `docs/FAILURE_MAP.md` is updated before every commit, confirmed or refuted.
  **Nothing is ever deleted from it.** A refuted hypothesis later found true gets
  a second row, never an edit. Never renumber.
- Answering the user: short, plain language, jargon defined inline, and **never
  give a time estimate**.
- Use the accepted walker — `gait_profile="lab"`, `use_policy=False` — and assert
  on `info["accepted_walker"]` before filming anything. The other walker is the
  rejected one and is the default in `GeckoBrainEnv`.
- Rebuild the body only through `tools/rebuild_body.py`, never step by step by
  hand.
- Audit the body with `python -m common.morphology_audit --xml
  morphology/gecko_body_lab_v2.xml`. With no `--xml` it audits the unfitted
  source template and reports 1/14.
