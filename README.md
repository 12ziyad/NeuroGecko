# NeuroGecko

A biologically grounded virtual leopard gecko trained with neural control and biomechanical measurements.

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
