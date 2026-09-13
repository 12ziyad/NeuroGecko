# NeuroGecko

### ▶ [Watch it live — neurogecko.pages.dev](https://neurogecko.pages.dev)

The whole animal runs in your browser: MuJoCo physics at 500 Hz, a basal-ganglia
selector choosing between six behaviours, a retina and optic tectum looking for a
cricket in its own rendered pixels, and 27 nerves carrying each command from the
brain to the joint it drives. No video, no playback — it is solved while you watch.

A biologically grounded virtual leopard gecko (*Eublepharis macularius*) built from
published measurements, with a ledger of every hypothesis tested.

| | |
|---|---|
| **Live** | https://neurogecko.pages.dev |
| **Browser port vs the Python model** | walker `5.6e-16`, eye `8.9e-16`, gates / clock / search **exactly 0** |
| **Anatomy** | 14 / 14 gates against the literature |
| **Gait** | 3 / 6 of Gate 2. The three it fails are listed below, not hidden |
| **Ledger** | [`docs/FAILURE_MAP.md`](docs/FAILURE_MAP.md) — 427 hypotheses, 83 % of them refuted |

Every number in this project is **published**, **derived** or **invented**, and says
which, next to itself, in [`config/proxies.yaml`](config/proxies.yaml). A plausible
number wearing a published number's clothes is the worst defect available.

## The machine learning lost

This is a simulated animal, not a trained policy. Three PPO runs, about 9 million
steps, produced a residual controller that scored **3 of 6 gates against the
hand-written spinal oscillator's 4** ([ledger #11](docs/FAILURE_MAP.md)). The
walker that ships runs with the residual switched **off**.

That result is kept, with the checkpoints, because a negative result with evidence
is worth more than a positive one without.

```text
shipped:   ctrl(t) = CPG_base(t) + 0
available: ctrl(t) = CPG_base(t) + PPO_residual(obs)   # measured worse, kept anyway
```

## Reptile keepers found bugs the test suite could not

The model was posted to r/leopardgeckos. People who keep these animals found, in
minutes, three defects that 545 automated tests and 14 anatomy gates had passed
over — and in every case the part existed and was **wired the wrong way round**:

| What they said | What it turned out to be |
|---|---|
| *"the hind legs are the wrong shape"* | The knee bowed **backward**, like a second elbow. Every tetrapod's knee points forward. It had been wrong since the first build |
| *"too stiff… it slithers"* | The trunk bent into an **S with the kink mid-body**, which keeps head and hips aligned so the animal reads as rigid. The published standing wave has its nodes on the **girdles**, making the trunk bow as one arc |
| *"the legs are too stiff and curved incorrectly"* | The knee sat at **159.9°** — 98.5 % of full extension. Published stance range for a sprawling lizard is **69–96°**. It had never been inside that band |

One keeper filmed her gecko on request from three angles, including from
underneath through glass with a US nickel in shot for scale. Those clips produced
the first measurements this project has ever taken from a live animal: her trunk
swings **52°** through a stride where the model managed **2.3°**.

## What is in here

- a measurement-backed MuJoCo body — 30 actuators, 44 DoF, built by generator from
  a provenance registry, never by hand
- a CPG spinal oscillator driving the gait, with an optional learned residual
- an extended Prescott basal-ganglia model reproducing its published gating table
- a retina and optic tectum that find prey in the animal's own rendered view
- a circadian clock, a hypothalamus, thermoregulation on a warm surface
- a browser port checked against the Python to `5.6e-16`

## What it still gets wrong

Stated here because the ledger's first rule is that a failed prediction stays in
the table:

- **hind duty factor 0.567** against a published 0.78
- **limb phase 0.591** against a published 0.435
- **contact entrainment fails** — the footfall pattern is irregular enough that
  stride detection cannot certify the timing numbers
- **body bend 10.3°** against 52° measured from a real gecko
- `groom` is a channel that never fires, and says so rather than being faked

## Running it

```bash
python -m utils.build_lab_morphology --fit-com     # build the body
python tools/rebuild_body.py                       # meshes, skin, worlds, in order
python realism_metrics.py --xml morphology/gecko_body_lab_v2.xml \
       --zero-residual --gait-profile lab --hind-stance-compensation \
       --episodes 1 --output report.json           # measure the gait
python tools/conformance_web.py                    # browser port vs Python
python -m unittest discover -s tests -q            # 537 of 545; the 8 are documented
```

## The rules this project runs on

1. Every number is published, derived or invented — and says which.
2. Never tune until the answer matches.
3. The reward is what the model is told to want; the gates are what the animal
   does. Only the second counts.
4. A failed prediction stays in the table.
5. Measure before claiming — including your own tests and your own solver.
6. One source is not a citation.

Each was learned by getting it wrong. They are the real output of the ledger.

## Status

Active research prototype. Apache-2.0.
