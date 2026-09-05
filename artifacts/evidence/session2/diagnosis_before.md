# Session 2 — base diagnosis before controller edits

Before-change diagnosis from original saved traces; no new simulation, fitting or training, and no biological validation claimed.

Loads below are scalar MuJoCo touch readings, not world-vertical forces. No toe/site-height contact test is used. [MuJoCo sensor definition](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-touch).

All trajectories are n = 1 deterministic runs. Dispersion within a trace is not variation between independent runs or animals.

## legacy_zero_residual

Original trace SHA256: `f0e0a91836995fe56cbbf401c3a4d85112b1bc1b6cbb5dcac2f679b1efbd335c`.

XML SHA256: `92f48b3e30a7d42d65c78fa3f084dfcfd101ddb107f3a07611af389e24f89020`; profile: `historical legacy reference (trace omitted profile)`.

Window: 3.000–20.000 s; 50.0 Hz; 850 post-settle endpoints.

Command timing: INFERRED from explicit historical reference and repeated control_dt clock; sample i is endpoint of command interval i-1; reset sample excluded.

Path 1.270272 m; net 0.151754 m; net/path 0.1195; path speed 0.074722 m/s; signed body-forward mean -0.001308 m/s.

Loaded means touch > 0.056400000 N. Raw / debounced fractions are shown separately; debounce removes bounded contact AND flight intervals shorter than 40 ms.

| Foot | Loaded during commanded stance, raw / debounce | Loaded during commanded swing, raw / debounce | Mean force stance / swing (mN) | Debounced overall contact fraction | Contact cycles/s | Period CV | Entrainment check |
|---|---:|---:|---:|---:|---:|---:|---|
| HL | 0.744 / 0.771 | 0.604 / 0.589 | 187.042 / 185.221 | 0.701 | 3.415 | 42.208% | False |
| FL | 0.354 / 0.361 | 0.243 / 0.207 | 98.498 / 85.667 | 0.311 | 3.983 | 54.938% | False |
| HR | 0.726 / 0.783 | 0.666 / 0.716 | 181.225 / 198.058 | 0.758 | 3.529 | 44.533% | False |
| FR | 0.333 / 0.225 | 0.316 / 0.312 | 102.423 / 89.195 | 0.251 | 3.892 | 49.449% | False |

The JSON includes every local-phase bin, counts, raw/debounced cycles, command provenance, and conditional limb-phase statistics. Contact-cycle values are not promoted to verified biological strides; a failed entrainment check remains a failure.

## lab_zero_residual

Original trace SHA256: `a475108fa29482f23868eb08a3927e73db30e30a13d26d14bad64ff8aba40e63`.

XML SHA256: `cfc485e6fab752dc4ab25d25beebf549fded5fb241b3da6c828df55156b9c22d`; profile: `lab`.

Window: 3.000–20.000 s; 50.0 Hz; 850 post-settle endpoints.

Command timing: samples.gait_target_time_s: lab Session1 recorded executed held interval start.

Path 0.797192 m; net 0.166812 m; net/path 0.2093; path speed 0.046894 m/s; signed body-forward mean 0.011755 m/s.

Loaded means touch > 0.035019608 N. Raw / debounced fractions are shown separately; debounce removes bounded contact AND flight intervals shorter than 40 ms.

| Foot | Loaded during commanded stance, raw / debounce | Loaded during commanded swing, raw / debounce | Mean force stance / swing (mN) | Debounced overall contact fraction | Contact cycles/s | Period CV | Entrainment check |
|---|---:|---:|---:|---:|---:|---:|---|
| HL | 0.446 / 0.422 | 0.475 / 0.449 | 71.564 / 106.721 | 0.428 | 4.066 | 42.156% | False |
| FL | 0.483 / 0.523 | 0.360 / 0.348 | 97.013 / 82.189 | 0.472 | 4.038 | 31.139% | False |
| HR | 0.508 / 0.494 | 0.687 / 0.753 | 84.681 / 121.365 | 0.554 | 3.321 | 39.563% | False |
| FR | 0.556 / 0.517 | 0.137 / 0.090 | 98.593 / 39.351 | 0.388 | 3.265 | 42.180% | False |

The JSON includes every local-phase bin, counts, raw/debounced cycles, command provenance, and conditional limb-phase statistics. Contact-cycle values are not promoted to verified biological strides; a failed entrainment check remains a failure.

