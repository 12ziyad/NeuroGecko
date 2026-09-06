# Gate curve - s4_run2b_elbow008

Every row is `realism_metrics.py` + `eval.session2_controller.gate2`,
the same code path as the base evidence. Reward is not a column here
on purpose: it is not the quantity being claimed.

A row whose stride-period CV exceeds 0.1 is REJECTED and cannot be
selected as best, however many gates it passes: an irregular gait can raise
a contact-load check without walking any better. Same rule as
`tools/fit_gate2_official.py`.

Protocol: {"duration_s": 20.0, "realism_metrics_flags": ["--xml", "C:\\Users\\ziyad\\GeckoBrain\\morphology\\gecko_body_lab_v2.xml", "--gait-profile", "lab", "--episodes", "1", "--seed", "0", "--duration", "20.0", "--settle", "3.0", "--hind-stance-compensation", "all", "--residual-scale", "0.25"], "scored_by": "eval.session2_controller.gate2 on the rollout trace", "settle_s": 3.0, "train_config_sha256": "caa2bb855d7fad3634b86ba443eafa31eb45727e1a57a2225da5bbfbd18b9b5f"}

| step | fwd m/s | net/path | hind swing | front stance | hind duty | limb phase | CV | gates | note |
|---|---|---|---|---|---|---|---|---|---|
| base (policy off) | 0.0416 | 0.7427 | 0.0806 | 0.5841 | 0.7334 | 0.6347 | 0.0149 | 4/6 PPP.P. | registry CV check failed; not entrained (base is too) |
| 0 | 0.0415 | 0.7433 | 0.0677 | 0.5811 | 0.7342 | 0.6412 | 0.0087 | 4/6 PPP.P. | registry CV check failed; not entrained (base is too) |
| 100,000 | 0.0415 | 0.7471 | 0.1559 | 0.5732 | 0.7135 | 0.6333 | 0.0069 | 2/6 PP.... | registry CV check failed; not entrained (base is too) |
| 200,000 | 0.0412 | 0.7604 | 0.0570 | 0.5966 | 0.6921 | 0.6164 | 0.0054 | 3/6 PPP... | registry CV check failed; not entrained (base is too) |
| 300,000 | 0.0422 | 0.7679 | 0.1478 | 0.6128 | 0.7089 | 0.6092 | 0.0028 | 2/6 PP.... |  |
| 400,000 | 0.0425 | 0.7609 | 0.1641 | 0.6165 | 0.6712 | 0.6068 | 0.1780 | 2/6 PP.... | REJECTED: stride CV 0.17802522903312049 > 0.1; registry CV check failed |
| 500,000 | 0.0438 | 0.7671 | 0.1402 | 0.6281 | 0.6932 | 0.6066 | 0.1784 | 2/6 PP.... | REJECTED: stride CV 0.17843695521473266 > 0.1; registry CV check failed |
| 600,000 | 0.0437 | 0.7623 | 0.1293 | 0.6358 | 0.6937 | 0.5996 | 0.0041 | 2/6 PP.... | registry CV check failed; not entrained (base is too) |
| 700,000 | 0.0442 | 0.7893 | 0.1185 | 0.6653 | 0.6780 | 0.5726 | 0.1648 | 3/6 PP.P.. | REJECTED: stride CV 0.16479379475891123 > 0.1; registry CV check failed; not entrained (base is too) |
| 800,000 | 0.0456 | 0.7971 | 0.1087 | 0.6408 | 0.6308 | 0.5989 | 0.2928 | 2/6 PP.... | REJECTED: stride CV 0.2928165445007637 > 0.1; registry CV check failed; not entrained (base is too) |
| 900,000 | 0.0477 | 0.8161 | 0.1033 | 0.6422 | 0.6347 | 0.5924 | 0.2554 | 2/6 PP.... | REJECTED: stride CV 0.2553780930712241 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,000,000 | 0.0473 | 0.8208 | 0.1043 | 0.6593 | 0.6287 | 0.6348 | 0.2974 | 3/6 PP.P.. | REJECTED: stride CV 0.29743341837778753 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,100,000 | 0.0480 | 0.8268 | 0.0043 | 0.6740 | 0.5827 | 0.6802 | 0.3752 | 4/6 PPPP.. | REJECTED: stride CV 0.37521486590392145 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,200,000 | 0.0487 | 0.8258 | 0.0097 | 0.6613 | 0.5970 | 0.7522 | 0.3705 | 4/6 PPPP.. | REJECTED: stride CV 0.3704860798595661 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,300,000 | 0.0497 | 0.8277 | 0.0172 | 0.6364 | 0.6521 | 0.6095 | 0.3056 | 3/6 PPP... | REJECTED: stride CV 0.3055818406967357 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,400,000 | 0.0495 | 0.8263 | 0.0032 | 0.6398 | 0.5986 | 0.9175 | 0.3791 | 3/6 PPP... | REJECTED: stride CV 0.3791091198453002 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,500,000 | 0.0506 | 0.8225 | 0.0140 | 0.6324 | 0.6065 | 0.6313 | 0.3268 | 3/6 PPP... | REJECTED: stride CV 0.3267880277336622 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,600,000 | 0.0511 | 0.8203 | 0.0215 | 0.6108 | 0.6503 | 0.6702 | 0.2323 | 3/6 PPP... | REJECTED: stride CV 0.2322661126266419 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,700,000 | 0.0512 | 0.8206 | 0.0000 | 0.6152 | 0.6374 | 0.6645 | 0.2332 | 3/6 PPP... | REJECTED: stride CV 0.23324183878274815 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,800,000 | 0.0516 | 0.8105 | 0.0086 | 0.6152 | 0.6242 | 0.6044 | 0.2624 | 3/6 PPP... | REJECTED: stride CV 0.26237139331171544 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,900,000 | 0.0535 | 0.8110 | 0.0215 | 0.6125 | 0.6638 | 0.6173 | 0.2237 | 3/6 PPP... | REJECTED: stride CV 0.22372931526034018 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,000,000 | 0.0538 | 0.8135 | 0.0032 | 0.6115 | 0.6078 | 0.7043 | 0.3592 | 3/6 PPP... | REJECTED: stride CV 0.3592044163399243 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,100,000 | 0.0542 | 0.8197 | 0.0075 | 0.6185 | 0.6309 | 0.7375 | 0.3286 | 3/6 PPP... | REJECTED: stride CV 0.3286200468453205 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,200,000 | 0.0550 | 0.8157 | 0.0086 | 0.6231 | 0.6371 | 0.6724 | 0.3056 | 3/6 PPP... | REJECTED: stride CV 0.30560682651242305 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,300,000 | 0.0555 | 0.8148 | 0.0108 | 0.6283 | 0.6266 | 0.6605 | 0.3071 | 3/6 PPP... | REJECTED: stride CV 0.30705860606692065 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,400,000 | 0.0557 | 0.8116 | 0.0054 | 0.6181 | 0.6229 | 0.8095 | 0.3798 | 3/6 PPP... | REJECTED: stride CV 0.3798437253149427 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,500,000 | 0.0553 | 0.8064 | 0.0075 | 0.6182 | 0.6435 | 0.6623 | 0.3088 | 3/6 PPP... | REJECTED: stride CV 0.3088141920737817 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,600,000 | 0.0550 | 0.8108 | 0.0183 | 0.6297 | 0.6666 | 0.6232 | 0.0184 | 3/6 PPP... | registry CV check failed; not entrained (base is too) |
| 2,700,000 | 0.0550 | 0.8124 | 0.0161 | 0.6250 | 0.6750 | 0.6341 | 0.2364 | 3/6 PPP... | REJECTED: stride CV 0.23636836406082667 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,800,000 | 0.0560 | 0.8106 | 0.0376 | 0.6334 | 0.6533 | 0.6290 | 0.2738 | 3/6 PPP... | REJECTED: stride CV 0.27380666142496535 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,900,000 | 0.0558 | 0.8115 | 0.0161 | 0.6196 | 0.6543 | 0.6418 | 0.3110 | 3/6 PPP... | REJECTED: stride CV 0.31101598132208136 > 0.1; registry CV check failed; not entrained (base is too) |
| 3,000,000 | 0.0558 | 0.8050 | 0.0161 | 0.6172 | 0.6712 | 0.6520 | 0.2394 | 3/6 PPP... | REJECTED: stride CV 0.2394498405525062 > 0.1; registry CV check failed; not entrained (base is too) |
| 3,014,656 | 0.0557 | 0.8126 | 0.0161 | 0.6212 | 0.6285 | 0.6640 | 0.3443 | 3/6 PPP... | REJECTED: stride CV 0.3442987242437736 > 0.1; registry CV check failed; not entrained (base is too) |
