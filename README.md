# NeuroGecko

### ▶ [Watch it live — neurogecko.pages.dev](https://neurogecko.pages.dev)

The whole animal runs in your browser: MuJoCo physics at 500 Hz, a basal-ganglia
selector choosing between six behaviours, a retina and optic tectum looking for a
cricket in its own rendered pixels, and 27 nerves carrying each command from the
brain to the joint it drives. No video, no playback — it is solved while you watch.

A biologically grounded virtual leopard gecko (*Eublepharis macularius*) built from
published measurements, with neural control and a ledger of every hypothesis tested.

| | |
|---|---|
| **Live** | https://neurogecko.pages.dev |
| **Browser port vs the Python model** | walker `7.3e-15`, eye `8.9e-16`, gates / clock / search **exactly 0** |
| **Anatomy** | 14 / 14 gates against the literature |
| **Ledger** | [`docs/FAILURE_MAP.md`](docs/FAILURE_MAP.md) — every hypothesis, refutations kept |

Every number in this project is **published**, **derived** or **invented**, and says
which, next to itself, in the source. A plausible number wearing a published
number's clothes is the worst defect available.

## Overview

NeuroGecko is a MuJoCo + PPO research prototype for training a virtual leopard gecko body with biologically grounded gait control.

The project combines:

- a measurement-backed MuJoCo gecko body
- Stable-Baselines3 PPO
- a CPG-residual controller
- contact-based gait evaluation
- front/hind limb participation metrics

## Core control idea

```text
final_ctrl = CPG_base(t) + PPO_residual(action)




PPO learns residual control for target movement, balance, and stability.

## Current system

- 25 actuators
- observation shape: 92
- action shape: 25
- calibrated contact threshold: 0.0564
- CPG-residual locomotion controller
- FL/FR lift residual locking for gait timing protection

## Status

Active research prototype.
