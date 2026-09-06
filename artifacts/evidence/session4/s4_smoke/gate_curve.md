# Gate curve - s4_smoke

Every row is `realism_metrics.py` + `eval.session2_controller.gate2`,
the same code path as the base evidence. Reward is not a column here
on purpose: it is not the quantity being claimed.

Protocol: {"duration_s": 20.0, "realism_metrics_flags": ["--xml", "C:\\Users\\ziyad\\GeckoBrain\\morphology\\gecko_body_lab_v2.xml", "--gait-profile", "lab", "--episodes", "1", "--seed", "0", "--duration", "20.0", "--settle", "3.0", "--hind-stance-compensation", "all", "--residual-scale", "0.25"], "scored_by": "eval.session2_controller.gate2 on the rollout trace", "settle_s": 3.0, "train_config_sha256": "3d6075230a98cc59fa8c84482db32886b5a865db007da4dc7bdaa88df7b67d48"}

| step | fwd m/s | net/path | hind swing | front stance | hind duty | limb phase | CV | gates | note |
|---|---|---|---|---|---|---|---|---|---|
| base (policy off) | 0.0416 | 0.7427 | 0.0806 | 0.5841 | 0.7334 | 0.6347 | 0.0149 | 4/6 PPP.P. | stride CV over ceiling; not entrained (base is too) |
| 0 | 0.0415 | 0.7433 | 0.0677 | 0.5811 | 0.7342 | 0.6412 | 0.0087 | 4/6 PPP.P. | stride CV over ceiling; not entrained (base is too) |
| 10,000 | 0.0416 | 0.7370 | 0.1533 | 0.5535 | 0.7079 | 0.6451 | 0.0031 | 2/6 PP.... | stride CV over ceiling; not entrained (base is too) |
| 20,000 | 0.0410 | 0.7287 | 0.0333 | 0.5815 | 0.6926 | 0.6162 | 0.0022 | 3/6 PPP... | stride CV over ceiling; not entrained (base is too) |
| 20,480 | 0.0410 | 0.7363 | 0.0484 | 0.5855 | 0.6927 | 0.6286 | 0.0027 | 3/6 PPP... |  |
