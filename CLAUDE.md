# NeuroGecko — rules for every session

## The standing rule

**`docs/FAILURE_MAP.md` is the living map of this project.** Read it first, in
every session, before touching anything. Update it before every commit that
tests a hypothesis — confirmed or refuted. It carries past, present and future:
project state, every failure with its cause and its fix, every current blocker.

**Nothing is ever deleted from it.** A hypothesis refuted and later found true
gets a second row, not an edit. The refuted entries are the map of where not to
look; they are the most valuable thing in this repository.

Numbered hypotheses continue from the last entry. Never renumber.

## The five rules the ledger produced

1. Every number is PUBLISHED, DERIVED or INVENTED — and says which, in
   `config/proxies.yaml`. A plausible number wearing a published number's
   clothes is the worst defect available.
2. Never tune until the answer matches. If a model needs its constants adjusted
   to reproduce a published result, the constants must come from a paper.
3. The reward is what the model is told to want; the gates are what the animal
   does. Only the second counts.
4. A failed prediction stays in the table. Adding a knob per failing joint is
   fitting the harness to the answer.
5. Measure before claiming — including your own tests and your own solver.
   Recorded: five failures were in the test, not the model; one was in the
   integration step, not the model.
6. One source is not a citation. Check a parameter set against something it
   must independently predict — the basal-ganglia weights were settled by the
   gating constant they have to reproduce, not by a vote among sources.

## Answering the user

Short. Plain language. Define jargon inline. **Never give time estimates.**
