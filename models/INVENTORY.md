# Recovered model inventory

Status: recovered from the existing AWS project and verified on this laptop.

Generated UTC: 2026-09-04T23:40:39.112920+00:00
Source checkout: `ddbb105ff40afa7736916750243e282bafd22b98` (weights are separately hashed below).
Files: **55**; uncompressed bytes: **125669326**.

Recovery location: `models_recovered/recovery-20260904T233438Z/models/`.
The original local `models/` and original AWS weights were not overwritten.

## Integrity

Archive: `models.tar.gz` (112548021 bytes).

SHA256: `dd93767f726dc5e254003b1072da5acebe0e92261e293d4766c794df08db302a`.

The archive checksum matched AWS. All individual extracted-file hashes matched
the independently downloaded AWS `files.sha256` manifest.

## Interpretation

### New safety-test run (separate from the 55-file recovery)

`recovery_pipeline_smoke` is a **256-step plumbing test, not a new trained walker**.
It ran on the existing AWS A10G with four subprocess environments. Before
learning and at each checkpoint, the laptop downloaded and verified the paired
model, normalization state and full training configuration before acknowledging.
The default production checkpoint interval is 100,000 aggregate steps; this
tiny test deliberately used 128 to exercise the path.

Local bundles: `models_recovered/build-20260904T233438Z/recovery_pipeline_smoke/`.
Remote bundles: `build_sessions/build-20260904T233438Z/models/recovery_pipeline_smoke/checkpoints/`.

| Bundle | Manifest SHA256 |
|---|---|
| step-0 | `e7408a4a3292929b612c1e823e91b69b20d4c6e8e8faa959f879f0776563fea2` |
| step-128 | `137bd57a1f154d82bf6126f8a62f1840ae73b81273317fac6b4a477ed308ae59` |
| step-256 | `aae77df24accfeae6547f0138d03fe6f89984e27a262f0bd3ca431d88b1abc72` |
| step-256-final | `61573d05e28b2763d37b23dd7b7c3a8eef03ffee429c4f4dbb6caa5321234103` |

Final paired model SHA256: `776ba320f6e45cf09baa24025e5ab8e066e5671418b74f1a7986c22535f3e98e`.
Final normalizer SHA256: `5b31f236234a5bce71851db4e471761bb3ae24da8f658a7b641f8677355592ba`.
Full config SHA256: `27ba4a22878f423b335095f0dcd7765fed3ecc3343d37505ae331c1b13a6fa62`.
All three exact files are included in each downloaded bundle. The final pair
successfully resumed locally from 256 to 320 steps with 10 additional optimizer
updates; original files remained unchanged. See
`artifacts/evidence/checkpoint_offhost_resume.json` for that verification.

### Recovered lineage

- `brain_v1_patch38c_visual_dagger_80k_seed0`: recovered visual student / named champion.
- `brain_v1_patch37b_dagger_200k_seed1`: recovered privileged teacher.
- `v4_5b_speed_polish_1m`: frozen low-level walker and normalization state.
- Other run names and configurations document lineage; names alone do not prove performance.
- Historical eating counts are task-specific outcomes, not a newly reproduced result or biological validation.
- No checkpoint was unpickled to produce this inventory.

## Files

| Relative path | Bytes | SHA256 |
|---|---:|---|
| `brain/brain_v1_camera_food_dropout_300k/final.zip` | 2551853 | `5a8c0c46e239c191e41838c1aec7c10b781bb77d59d8baa6db4939f337aa59e8` |
| `brain/brain_v1_camera_food_dropout_300k/train_config.json` | 1138 | `d33724bde1da1e3f3116c9fd8b06bba1c7d8c9f381583d7bc7efa60f69370cdc` |
| `brain/brain_v1_curriculum_100k/final.zip` | 2552031 | `2b777ed3f6b5f93d53084921604c6e4c74c8a3dba7cbf1952d3e74150e86a635` |
| `brain/brain_v1_curriculum_100k/train_config.json` | 799 | `973172d2ac472a4a508d831aa95b2d4d3d470fea058885a61ddcc21f56074ca6` |
| `brain/brain_v1_patch32_forward60_50k/final.zip` | 2552281 | `baac58f16c1364bd4c3c3b8701c372747e0918201e755c102d29aee6e5b1bb36` |
| `brain/brain_v1_patch32_forward60_50k/train_config.json` | 1372 | `29d8358e9fc89a9bdddd75e080a6b24751b30fb06b7868f86d8edd4870cddf1d` |
| `brain/brain_v1_patch33_vislock_forward60_50k/final.zip` | 2552305 | `3b0847fb216ea7bca1e79f2860c4b38183f4d200c0f1c1dfcebed6a5a4062b36` |
| `brain/brain_v1_patch33_vislock_forward60_50k/train_config.json` | 1408 | `e0403383a49ba20824fed787825603ba0fc8ffdd01b947ccdfb0608328eba46d` |
| `brain/brain_v1_patch34_visualdelta_forward60_50k/checkpoints/ckpt_375208_steps.zip` | 2552302 | `eebf5fc68fd622c1ec23febf0cd366762608c64e793e77d2112d69577a86f120` |
| `brain/brain_v1_patch34_visualdelta_forward60_50k/checkpoints/ckpt_400208_steps.zip` | 2551936 | `fa8b63befe3c885d58f3a612f27a667c9a935778ca915f4a4de5abae66871c34` |
| `brain/brain_v1_patch34_visualdelta_forward60_50k/final.zip` | 2552306 | `7dbcd45878f3678896fbea7b2a0e78fbf476455a993783d9bd06bbf7bd788bcc` |
| `brain/brain_v1_patch34_visualdelta_forward60_50k/train_config.json` | 1443 | `cf16009b1e7975dfd1512f67573f59c0823f3f447ac9bf95b8963e8b1e90a9b1` |
| `brain/brain_v1_patch35_bigcue_curric_50k/checkpoints/ckpt_375208_steps.zip` | 2552278 | `af4dc30c9c9dca15454dff3d3a0196344951b6c4c8c8c5e46317b90bcc84da45` |
| `brain/brain_v1_patch35_bigcue_curric_50k/checkpoints/ckpt_400208_steps.zip` | 2552304 | `ca9b1c46ff570ef769e5c88ee39fc06900a8df7281a0d7d3df3df8697cc149d8` |
| `brain/brain_v1_patch35_bigcue_curric_50k/final.zip` | 2552282 | `585fc5e92aeb9c620715a85c993148a05bee4982098fcdd475503f4855baaad6` |
| `brain/brain_v1_patch35_bigcue_curric_50k/train_config.json` | 1551 | `99a9a8715a28922c6ea1b45cd790fc315927e4bed88a8e02b87b3ffb3b806907` |
| `brain/brain_v1_patch36_recurrent_warm_100k/checkpoints/ckpt_100000_steps.zip` | 15192028 | `a9435720d46e1d797e32be9340fba0314d425c27023af298f5565b4f217bfa97` |
| `brain/brain_v1_patch36_recurrent_warm_100k/checkpoints/ckpt_25000_steps.zip` | 15189037 | `79c53c5dcdb901ae16461c08cc9930fed65c57ddfb8d484ec53abe6105c30a50` |
| `brain/brain_v1_patch36_recurrent_warm_100k/checkpoints/ckpt_50000_steps.zip` | 15190029 | `06231207427bc7ad1fc45d7bc991cc756831275c5c226904b19a5fab51a320c4` |
| `brain/brain_v1_patch36_recurrent_warm_100k/checkpoints/ckpt_75000_steps.zip` | 15191033 | `fe78ad62d9bffb2fdbf45247cbacac8429e5b6c4bbe110b80009e0461cbd13c9` |
| `brain/brain_v1_patch36_recurrent_warm_100k/final.zip` | 15192197 | `67ed54b3b81db0ac838044a42ff5f0e174d827e45e8df5ff9991b4066cf0e1fc` |
| `brain/brain_v1_patch36_recurrent_warm_100k/train_config.json` | 1580 | `caaee9fe94d76c52ba9db0e77a5a2dfeb9a549fd2c8c6b85bb3fab3c971d2577` |
| `brain/brain_v1_patch37a_priv_bc_smoke/final.pt` | 625183 | `de6c35db2bce134147da8124137743547bdf851d8cfa91eda82d0a022bb1110e` |
| `brain/brain_v1_patch37a_priv_bc_smoke/train_config.json` | 618 | `bff68564bf9e4f19f9e6193cedcbe0c800783589961d88d30421b72149f3b671` |
| `brain/brain_v1_patch37b_dagger_200k_seed1/final.pt` | 625183 | `dfec1274779035d3fc29329c4f901a704010eb348a49a2abca370a2f2ee636aa` |
| `brain/brain_v1_patch37b_dagger_200k_seed1/train_config.json` | 913 | `2912ba30b891ce3050b897d80e21725865b68f45cddf8624f773e8fa5ab4141a` |
| `brain/brain_v1_patch37b_dagger_smoke/final.pt` | 625183 | `287b9e6ac6cd30e4fa565b6d95614a15f585a4654c72e9dfdbe5aaa8005f3ef9` |
| `brain/brain_v1_patch37b_dagger_smoke/train_config.json` | 906 | `c4c0a3df6b0cf14e5b9d1c5e5b111c7ddd7cd4e49ea0bf8cf03b9082d8f53bce` |
| `brain/brain_v1_patch37b_oracle_100k_seed0/final.pt` | 625183 | `eefcdbbbbc125e54d33ae4882162cf5ec7f01d2a24964846d35f23b3535afd27` |
| `brain/brain_v1_patch37b_oracle_100k_seed0/train_config.json` | 819 | `1c93657c4455df5d0959d9dc325e7a48d9b820f80fce3c6125ac537f9ba3d799` |
| `brain/brain_v1_patch38a_preflight_20260618_143716/final.pt` | 619103 | `cb67d032c8ee50ef4539597e0da46c9c6402e4e90a410a92590c08761a52bdb3` |
| `brain/brain_v1_patch38a_preflight_20260618_143716/train_config.json` | 839 | `8559b3010f8b2c9a49dc231ccf9e09afa3696c8dccbccadaf64b24ed24874b94` |
| `brain/brain_v1_patch38b_visual_50k_seed0/final.pt` | 619103 | `f618c88aafdfa5fd2d23cefb1e861c4ff2573db600f8c7e82bda3d86997122b3` |
| `brain/brain_v1_patch38b_visual_50k_seed0/train_config.json` | 826 | `8223919154ad6dc206eeb0181119977685a9845b01ff1546d8c43ef485f47828` |
| `brain/brain_v1_patch38c_smoke_20260618_174813/final.pt` | 619103 | `c263af319544307b08c5eb3aae7143fdd527a823961ef2db8605b121f1d50eea` |
| `brain/brain_v1_patch38c_smoke_20260618_174813/train_config.json` | 1199 | `d0ec70fbd03adcabb07b1d76bebfb787a734c162844c416d7a9aa5aa4d32cb84` |
| `brain/brain_v1_patch38c_visual_dagger_80k_seed0/final.pt` | 619103 | `90b79bb0805ad243365fbbea72bf219ee6f01a122be17205a427934dc16ff7cb` |
| `brain/brain_v1_patch38c_visual_dagger_80k_seed0/train_config.json` | 1189 | `7074a6414acb176a1144477081f3a6ff390f5dadfe407d815832b78957c18c25` |
| `brain/brain_v1_resume_tiny/final.zip` | 2494465 | `b34151f912ac91512ce9c0bd972205b0825d2d02ae5c1475c337be05ae6a151e` |
| `brain/brain_v1_resume_tiny/train_config.json` | 1336 | `58b345a6279a252b7c46d8c67edad63292d910e1a267280b7183441a30ebc91c` |
| `brain/brain_v1_rewardfix_forward60_300k/final.zip` | 2552280 | `bfc8065dc9fb49f13714bbaa084ebf639dac7bb5bc71bde901815dd7522bf7b8` |
| `brain/brain_v1_rewardfix_forward60_300k/train_config.json` | 1377 | `c6fefefc6580615163ebcf160af5c79beb44dd479bcd4416e201740b5a840d39` |
| `brain/brain_v1_rewardfix_tiny/final.zip` | 2494465 | `333f580d15b4879ae7eee2dafca81b2f6786dee3034c47f6fc1b72cc5ffc9ab3` |
| `brain/brain_v1_rewardfix_tiny/train_config.json` | 1364 | `32120d4b6933661edbf9bdd3e98f4a8d9de81af25e73b8a48738a6a328e2f859` |
| `brain/brain_v1_smoke_300k/train_config.json` | 640 | `61378ddf894156a1bb4146f04ad3fb6ee75c171a258556dafc77d04934ccf3b1` |
| `brain/brain_v1_smoke_30k/final.zip` | 2470943 | `58884e27389243670aa4c47a7f08a9efae8ddd335c0a8a681e10244bc9262e39` |
| `brain/brain_v1_smoke_30k/train_config.json` | 638 | `c330fc52a4f3893bd687241eff6b081d2fab001ce72489380d36c137fefc1817` |
| `brain/brain_v1_taper_300k/train_config.json` | 950 | `1d423494a0df53ff384cd3e805160394ae1728f0ed4cc8bbb614b3c54e5437e2` |
| `v4_5b_speed_polish_1m/best_model.zip` | 2287322 | `083a9d8e660e0078cfe8177514862337e1af4c1ea5d78f1f975ad255e4ddd892` |
| `v4_5b_speed_polish_1m/ckpt_1000000_steps.zip` | 2287322 | `13ef6ef70e11c7497401819aa57455e3b88f85a978c666083a2ac13d1c4bebed` |
| `v4_5b_speed_polish_1m/ckpt_500000_steps.zip` | 2287308 | `359782284bc33fc96876d0f16c5df6af86cb508f9e4d8c9fad5efb149e7a6d93` |
| `v4_5b_speed_polish_1m/ckpt_vecnormalize_1000000_steps.pkl` | 10549 | `3bfb274819608e2328639ae0033056a066dcc449f9ce610e04bb6a149adeb22f` |
| `v4_5b_speed_polish_1m/ckpt_vecnormalize_500000_steps.pkl` | 10549 | `122ead2d77d11a3324231617ad3e154ab86a311f44741ad71a2bd9a99591bc10` |
| `v4_5b_speed_polish_1m/final.zip` | 2287323 | `77b7a99d56315f3e5edb1f352e465e13683b00ab986682c82abc55fac4832dfe` |
| `v4_5b_speed_polish_1m/vecnormalize.pkl` | 10549 | `32377b7763e901e3e078ec2323e7a2ae9e75e3e43cd9a3773b2cebdf98aea4dd` |

## Training configurations

### brain/brain_v1_camera_food_dropout_300k/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 256,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "episode_steps": 1000,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 4,
  "observation_mode": "privileged",
  "privileged_food_dropout_taper_enabled": true,
  "privileged_food_dropout_taper_steps": 300000,
  "privileged_food_end_dropout": 1.0,
  "privileged_food_end_scale": null,
  "privileged_food_scale": 1.0,
  "privileged_food_start_dropout": 0.0,
  "privileged_food_start_scale": null,
  "privileged_food_taper_enabled": false,
  "privileged_food_taper_steps": 0,
  "recurrent_ppo_available": true,
  "run_name": "brain_v1_camera_food_dropout_300k",
  "seed": 0,
  "total_steps": 300000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_curriculum_100k/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 256,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "episode_steps": 1000,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 4,
  "observation_mode": "privileged",
  "privileged_food_scale": 1.0,
  "recurrent_ppo_available": true,
  "run_name": "brain_v1_curriculum_100k",
  "seed": 0,
  "total_steps": 100000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch32_forward60_50k/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 256,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "continued_training": true,
  "eat_radius": 0.1,
  "episode_steps": 1000,
  "food_spawn_angle_deg": 60.0,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 4,
  "observation_mode": "privileged",
  "privileged_food_dropout_taper_enabled": true,
  "privileged_food_dropout_taper_steps": 50000,
  "privileged_food_end_dropout": 1.0,
  "privileged_food_end_scale": null,
  "privileged_food_scale": 1.0,
  "privileged_food_start_dropout": 1.0,
  "privileged_food_start_scale": null,
  "privileged_food_taper_enabled": false,
  "privileged_food_taper_steps": 0,
  "recurrent_ppo_available": true,
  "resumed_from_path": "/home/ubuntu/GeckoBrain/models/brain/brain_v1_camera_food_dropout_300k",
  "resumed_from_run": "brain_v1_camera_food_dropout_300k",
  "run_name": "brain_v1_patch32_forward60_50k",
  "seed": 0,
  "total_steps": 50000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch33_vislock_forward60_50k/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 256,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "checkpoint_freq": 25000,
  "continued_training": true,
  "eat_radius": 0.1,
  "episode_steps": 1000,
  "food_spawn_angle_deg": 60.0,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 4,
  "observation_mode": "privileged",
  "privileged_food_dropout_taper_enabled": true,
  "privileged_food_dropout_taper_steps": 50000,
  "privileged_food_end_dropout": 1.0,
  "privileged_food_end_scale": null,
  "privileged_food_scale": 1.0,
  "privileged_food_start_dropout": 1.0,
  "privileged_food_start_scale": null,
  "privileged_food_taper_enabled": false,
  "privileged_food_taper_steps": 0,
  "recurrent_ppo_available": true,
  "resumed_from_path": "/home/ubuntu/GeckoBrain/models/brain/brain_v1_camera_food_dropout_300k",
  "resumed_from_run": "brain_v1_camera_food_dropout_300k",
  "run_name": "brain_v1_patch33_vislock_forward60_50k",
  "seed": 0,
  "total_steps": 50000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch34_visualdelta_forward60_50k/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 256,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "checkpoint_freq": 25000,
  "checkpoint_freq_env_calls": 6250,
  "continued_training": true,
  "eat_radius": 0.1,
  "episode_steps": 1000,
  "food_spawn_angle_deg": 60.0,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 4,
  "observation_mode": "privileged",
  "privileged_food_dropout_taper_enabled": true,
  "privileged_food_dropout_taper_steps": 50000,
  "privileged_food_end_dropout": 1.0,
  "privileged_food_end_scale": null,
  "privileged_food_scale": 1.0,
  "privileged_food_start_dropout": 1.0,
  "privileged_food_start_scale": null,
  "privileged_food_taper_enabled": false,
  "privileged_food_taper_steps": 0,
  "recurrent_ppo_available": true,
  "resumed_from_path": "/home/ubuntu/GeckoBrain/models/brain/brain_v1_patch32_forward60_50k",
  "resumed_from_run": "brain_v1_patch32_forward60_50k",
  "run_name": "brain_v1_patch34_visualdelta_forward60_50k",
  "seed": 0,
  "total_steps": 50000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch35_bigcue_curric_50k/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 256,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "checkpoint_freq": 25000,
  "checkpoint_freq_env_calls": 6250,
  "continued_training": true,
  "eat_radius": 0.1,
  "episode_steps": 1000,
  "food_radius": 0.09,
  "food_radius_end": 0.035,
  "food_radius_start": 0.09,
  "food_radius_taper_steps": 40000,
  "food_spawn_angle_deg": 60.0,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 4,
  "observation_mode": "privileged",
  "privileged_food_dropout_taper_enabled": true,
  "privileged_food_dropout_taper_steps": 50000,
  "privileged_food_end_dropout": 1.0,
  "privileged_food_end_scale": null,
  "privileged_food_scale": 1.0,
  "privileged_food_start_dropout": 1.0,
  "privileged_food_start_scale": null,
  "privileged_food_taper_enabled": false,
  "privileged_food_taper_steps": 0,
  "recurrent_ppo_available": true,
  "resumed_from_path": "/home/ubuntu/GeckoBrain/models/brain/brain_v1_patch32_forward60_50k",
  "resumed_from_run": "brain_v1_patch32_forward60_50k",
  "run_name": "brain_v1_patch35_bigcue_curric_50k",
  "seed": 0,
  "total_steps": 50000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch36_recurrent_warm_100k/train_config.json

```json
{
  "algo": "recurrent_ppo",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 256,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "checkpoint_freq": 25000,
  "checkpoint_freq_env_calls": 6250,
  "continued_training": false,
  "eat_radius": 0.1,
  "episode_steps": 1000,
  "food_radius": 0.035,
  "food_radius_end": null,
  "food_radius_start": null,
  "food_radius_taper_steps": 0,
  "food_spawn_angle_deg": 60.0,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 4,
  "observation_mode": "privileged",
  "policy_class_used": "MultiInputLstmPolicy",
  "privileged_food_dropout_taper_enabled": true,
  "privileged_food_dropout_taper_steps": 100000,
  "privileged_food_end_dropout": 1.0,
  "privileged_food_end_scale": null,
  "privileged_food_scale": 1.0,
  "privileged_food_start_dropout": 1.0,
  "privileged_food_start_scale": null,
  "privileged_food_taper_enabled": false,
  "privileged_food_taper_steps": 0,
  "recurrent_ppo_available": true,
  "resumed_from_path": null,
  "resumed_from_run": null,
  "run_name": "brain_v1_patch36_recurrent_warm_100k",
  "seed": 0,
  "total_steps": 100000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m",
  "warm_start_extractor_run": "brain_v1_patch32_forward60_50k"
}
```

### brain/brain_v1_patch37a_priv_bc_smoke/train_config.json

```json
{
  "action_dim": 4,
  "algo": "behavior_cloning",
  "batch_size": 256,
  "best_val_loss": 0.1996961385011673,
  "body_features_dim": 96,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "dataset_name": "patch37a_smoke_5k",
  "eat_radius": 0.1,
  "epochs": 3,
  "food_radius": 0.035,
  "food_spawn_angle_deg": 60.0,
  "fused_features_dim": 256,
  "image_features_dim": 128,
  "lr": 0.0003,
  "proprio_dim": 92,
  "run_name": "brain_v1_patch37a_priv_bc_smoke",
  "seed": 0,
  "use_privileged_food": true,
  "val_frac": 0.1,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch37b_dagger_200k_seed1/train_config.json

```json
{
  "action_dim": 4,
  "algo": "behavior_cloning",
  "batch_size": 256,
  "best_val_loss": 0.00010346018880918103,
  "body_features_dim": 96,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "dataset_name": "patch37b_oracle_100k",
  "dataset_names": [
    "patch37b_oracle_100k",
    "patch37b_dagger_100k"
  ],
  "dataset_num_transitions": {
    "patch37b_dagger_100k": 100000,
    "patch37b_oracle_100k": 100000
  },
  "eat_radius": 0.1,
  "epochs": 20,
  "extra_dataset_names": [
    "patch37b_dagger_100k"
  ],
  "food_radius": 0.035,
  "food_spawn_angle_deg": 60.0,
  "fused_features_dim": 256,
  "image_features_dim": 128,
  "lr": 0.0003,
  "observation_mode": "privileged",
  "proprio_dim": 92,
  "run_name": "brain_v1_patch37b_dagger_200k_seed1",
  "seed": 1,
  "use_privileged_food": true,
  "val_frac": 0.1,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch37b_dagger_smoke/train_config.json

```json
{
  "action_dim": 4,
  "algo": "behavior_cloning",
  "batch_size": 256,
  "best_val_loss": 0.0852333630124728,
  "body_features_dim": 96,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "dataset_name": "patch37a_smoke_5k",
  "dataset_names": [
    "patch37a_smoke_5k",
    "patch37b_dagger_smoke_10k"
  ],
  "dataset_num_transitions": {
    "patch37a_smoke_5k": 5000,
    "patch37b_dagger_smoke_10k": 10000
  },
  "eat_radius": 0.1,
  "epochs": 5,
  "extra_dataset_names": [
    "patch37b_dagger_smoke_10k"
  ],
  "food_radius": 0.035,
  "food_spawn_angle_deg": 60.0,
  "fused_features_dim": 256,
  "image_features_dim": 128,
  "lr": 0.0003,
  "observation_mode": "privileged",
  "proprio_dim": 92,
  "run_name": "brain_v1_patch37b_dagger_smoke",
  "seed": 1,
  "use_privileged_food": true,
  "val_frac": 0.1,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch37b_oracle_100k_seed0/train_config.json

```json
{
  "action_dim": 4,
  "algo": "behavior_cloning",
  "batch_size": 256,
  "best_val_loss": 0.00037471359391929584,
  "body_features_dim": 96,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "dataset_name": "patch37b_oracle_100k",
  "dataset_names": [
    "patch37b_oracle_100k"
  ],
  "dataset_num_transitions": {
    "patch37b_oracle_100k": 100000
  },
  "eat_radius": 0.1,
  "epochs": 20,
  "extra_dataset_names": [],
  "food_radius": 0.035,
  "food_spawn_angle_deg": 60.0,
  "fused_features_dim": 256,
  "image_features_dim": 128,
  "lr": 0.0003,
  "observation_mode": "privileged",
  "proprio_dim": 92,
  "run_name": "brain_v1_patch37b_oracle_100k_seed0",
  "seed": 0,
  "use_privileged_food": true,
  "val_frac": 0.1,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch38a_preflight_20260618_143716/train_config.json

```json
{
  "action_dim": 4,
  "algo": "visual_distillation",
  "batch_size": 128,
  "best_epoch": 2,
  "best_val_loss": 0.5466712713241577,
  "body_features_dim": 96,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "dataset_name": "patch38_preflight_20260618_143716",
  "dataset_num_transitions": 1024,
  "eat_radius": 0.1,
  "epochs": 2,
  "food_radius": 0.035,
  "food_spawn_angle_deg": 60.0,
  "fused_features_dim": 256,
  "image_features_dim": 128,
  "lr": 0.0003,
  "observation_mode": "visual",
  "proprio_dim": 92,
  "run_name": "brain_v1_patch38a_preflight_20260618_143716",
  "seed": 38,
  "teacher_brain_run": "brain_v1_patch37b_dagger_200k_seed1",
  "use_privileged_food": false,
  "use_privileged_food_student": false,
  "val_frac": 0.1,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch38b_visual_50k_seed0/train_config.json

```json
{
  "action_dim": 4,
  "algo": "visual_distillation",
  "batch_size": 256,
  "best_epoch": 17,
  "best_val_loss": 0.002726367436116561,
  "body_features_dim": 96,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "dataset_name": "patch38b_visual_50k_seed0",
  "dataset_num_transitions": 50000,
  "eat_radius": 0.1,
  "epochs": 20,
  "food_radius": 0.035,
  "food_spawn_angle_deg": 60.0,
  "fused_features_dim": 256,
  "image_features_dim": 128,
  "lr": 0.0003,
  "observation_mode": "visual",
  "proprio_dim": 92,
  "run_name": "brain_v1_patch38b_visual_50k_seed0",
  "seed": 0,
  "teacher_brain_run": "brain_v1_patch37b_dagger_200k_seed1",
  "use_privileged_food": false,
  "use_privileged_food_student": false,
  "val_frac": 0.1,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch38c_smoke_20260618_174813/train_config.json

```json
{
  "action_dim": 4,
  "algo": "visual_distillation",
  "batch_size": 128,
  "best_epoch": 1,
  "best_val_loss": 0.8475667834281921,
  "body_features_dim": 96,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "dataset_name": "patch38b_visual_50k_seed0",
  "dataset_names": [
    "patch38b_visual_50k_seed0",
    "patch38c_dagger_smoke_20260618_174813"
  ],
  "dataset_num_transitions": 2048,
  "dataset_row_counts": {
    "patch38b_visual_50k_seed0": 50000,
    "patch38c_dagger_smoke_20260618_174813": 3000
  },
  "dataset_total_rows_before_max_rows": 53000,
  "eat_radius": 0.1,
  "epochs": 1,
  "extra_dataset_names": [
    "patch38c_dagger_smoke_20260618_174813"
  ],
  "food_radius": 0.035,
  "food_spawn_angle_deg": 60.0,
  "fused_features_dim": 256,
  "image_features_dim": 128,
  "lr": 0.0003,
  "observation_mode": "visual",
  "proprio_dim": 92,
  "run_name": "brain_v1_patch38c_smoke_20260618_174813",
  "seed": 38,
  "teacher_brain_run": "brain_v1_patch37b_dagger_200k_seed1",
  "train_obs": "visual",
  "use_privileged_food": false,
  "use_privileged_food_student": false,
  "val_frac": 0.1,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_patch38c_visual_dagger_80k_seed0/train_config.json

```json
{
  "action_dim": 4,
  "algo": "visual_distillation",
  "batch_size": 256,
  "best_epoch": 20,
  "best_val_loss": 0.0134959063725546,
  "body_features_dim": 96,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "dataset_name": "patch38b_visual_50k_seed0",
  "dataset_names": [
    "patch38b_visual_50k_seed0",
    "patch38c_visual_dagger_30k_seed0"
  ],
  "dataset_num_transitions": 80000,
  "dataset_row_counts": {
    "patch38b_visual_50k_seed0": 50000,
    "patch38c_visual_dagger_30k_seed0": 30000
  },
  "dataset_total_rows_before_max_rows": 80000,
  "eat_radius": 0.1,
  "epochs": 20,
  "extra_dataset_names": [
    "patch38c_visual_dagger_30k_seed0"
  ],
  "food_radius": 0.035,
  "food_spawn_angle_deg": 60.0,
  "fused_features_dim": 256,
  "image_features_dim": 128,
  "lr": 0.0003,
  "observation_mode": "visual",
  "proprio_dim": 92,
  "run_name": "brain_v1_patch38c_visual_dagger_80k_seed0",
  "seed": 0,
  "teacher_brain_run": "brain_v1_patch37b_dagger_200k_seed1",
  "train_obs": "visual",
  "use_privileged_food": false,
  "use_privileged_food_student": false,
  "val_frac": 0.1,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_resume_tiny/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 64,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "continued_training": true,
  "episode_steps": 1000,
  "food_spawn_angle_deg": 180.0,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 1,
  "observation_mode": "privileged",
  "privileged_food_dropout_taper_enabled": true,
  "privileged_food_dropout_taper_steps": 0,
  "privileged_food_end_dropout": 1.0,
  "privileged_food_end_scale": null,
  "privileged_food_scale": 1.0,
  "privileged_food_start_dropout": 1.0,
  "privileged_food_start_scale": null,
  "privileged_food_taper_enabled": false,
  "privileged_food_taper_steps": 0,
  "recurrent_ppo_available": true,
  "resumed_from_path": "/home/ubuntu/GeckoBrain/models/brain/brain_v1_camera_food_dropout_300k",
  "resumed_from_run": "brain_v1_camera_food_dropout_300k",
  "run_name": "brain_v1_resume_tiny",
  "seed": 0,
  "total_steps": 1000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_rewardfix_forward60_300k/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 256,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "continued_training": true,
  "eat_radius": 0.1,
  "episode_steps": 1000,
  "food_spawn_angle_deg": 60.0,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 4,
  "observation_mode": "privileged",
  "privileged_food_dropout_taper_enabled": true,
  "privileged_food_dropout_taper_steps": 300000,
  "privileged_food_end_dropout": 1.0,
  "privileged_food_end_scale": null,
  "privileged_food_scale": 1.0,
  "privileged_food_start_dropout": 1.0,
  "privileged_food_start_scale": null,
  "privileged_food_taper_enabled": false,
  "privileged_food_taper_steps": 0,
  "recurrent_ppo_available": true,
  "resumed_from_path": "/home/ubuntu/GeckoBrain/models/brain/brain_v1_camera_food_dropout_300k",
  "resumed_from_run": "brain_v1_camera_food_dropout_300k",
  "run_name": "brain_v1_rewardfix_forward60_300k",
  "seed": 0,
  "total_steps": 300000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_rewardfix_tiny/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 64,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "continued_training": true,
  "eat_radius": 0.1,
  "episode_steps": 1000,
  "food_spawn_angle_deg": 60.0,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 1,
  "observation_mode": "privileged",
  "privileged_food_dropout_taper_enabled": true,
  "privileged_food_dropout_taper_steps": 300000,
  "privileged_food_end_dropout": 1.0,
  "privileged_food_end_scale": null,
  "privileged_food_scale": 1.0,
  "privileged_food_start_dropout": 1.0,
  "privileged_food_start_scale": null,
  "privileged_food_taper_enabled": false,
  "privileged_food_taper_steps": 0,
  "recurrent_ppo_available": true,
  "resumed_from_path": "/home/ubuntu/GeckoBrain/models/brain/brain_v1_camera_food_dropout_300k",
  "resumed_from_run": "brain_v1_camera_food_dropout_300k",
  "run_name": "brain_v1_rewardfix_tiny",
  "seed": 0,
  "total_steps": 1000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_smoke_300k/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": false
  },
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "episode_steps": 1000,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "recurrent_ppo_available": true,
  "run_name": "brain_v1_smoke_300k",
  "seed": 0,
  "total_steps": 300000,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_smoke_30k/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": false
  },
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "episode_steps": 1000,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "recurrent_ppo_available": true,
  "run_name": "brain_v1_smoke_30k",
  "seed": 0,
  "total_steps": 30000,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

### brain/brain_v1_taper_300k/train_config.json

```json
{
  "algo": "PPO",
  "architecture": {
    "actor_layers": [
      128,
      64
    ],
    "body_features_dim": 96,
    "critic_layers": [
      128,
      64
    ],
    "fused_features_dim": 256,
    "image_features_dim": 128,
    "use_privileged": true
  },
  "batch_size": 256,
  "brain_action": [
    "target_dir_x",
    "target_dir_y",
    "target_distance",
    "engage"
  ],
  "brain_action_dim": 4,
  "episode_steps": 1000,
  "n_steps": 128,
  "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
  "num_envs": 4,
  "observation_mode": "privileged",
  "privileged_food_end_scale": 0.2,
  "privileged_food_scale": 1.0,
  "privileged_food_start_scale": 1.0,
  "privileged_food_taper_enabled": true,
  "privileged_food_taper_steps": 300000,
  "recurrent_ppo_available": true,
  "run_name": "brain_v1_taper_300k",
  "seed": 0,
  "total_steps": 300000,
  "use_privileged_food": true,
  "walker_run": "v4_5b_speed_polish_1m"
}
```

