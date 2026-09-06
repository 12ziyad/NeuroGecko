# Gate curve - s4_run1_seed0

Every row is `realism_metrics.py` + `eval.session2_controller.gate2`,
the same code path as the base evidence. Reward is not a column here
on purpose: it is not the quantity being claimed.

A row whose stride-period CV exceeds 0.1 is REJECTED and cannot be
selected as best, however many gates it passes: an irregular gait can raise
a contact-load check without walking any better. Same rule as
`tools/fit_gate2_official.py`.

Protocol: {"duration_s": 20.0, "realism_metrics_flags": ["--xml", "C:\\Users\\ziyad\\GeckoBrain\\morphology\\gecko_body_lab_v2.xml", "--gait-profile", "lab", "--episodes", "1", "--seed", "0", "--duration", "20.0", "--settle", "3.0", "--hind-stance-compensation", "all", "--residual-scale", "0.25"], "scored_by": "eval.session2_controller.gate2 on the rollout trace", "settle_s": 3.0, "train_config_sha256": "2f1890511678f62fbdfc945761d77351acc16d53c595e6433b7ee0b70957ab84"}

| step | fwd m/s | net/path | hind swing | front stance | hind duty | limb phase | CV | gates | note |
|---|---|---|---|---|---|---|---|---|---|
| base (policy off) | 0.0416 | 0.7427 | 0.0806 | 0.5841 | 0.7334 | 0.6347 | 0.0149 | 4/6 PPP.P. | registry CV check failed; not entrained (base is too) |
| 0 | 0.0415 | 0.7433 | 0.0677 | 0.5811 | 0.7342 | 0.6412 | 0.0087 | 4/6 PPP.P. | registry CV check failed; not entrained (base is too) |
| 100,000 | 0.0406 | 0.7506 | 0.0613 | 0.5916 | 0.7126 | 0.6216 | 0.0058 | 3/6 PPP... | registry CV check failed; not entrained (base is too) |
| 200,000 | 0.0416 | 0.7514 | 0.0527 | 0.5832 | 0.7024 | 0.6320 | 0.0071 | 3/6 PPP... | registry CV check failed; not entrained (base is too) |
| 300,000 | 0.0425 | 0.7627 | 0.1108 | 0.5866 | 0.7227 | 0.6114 | 0.0032 | 2/6 PP.... | registry CV check failed; not entrained (base is too) |
| 400,000 | 0.0430 | 0.7579 | 0.0968 | 0.6017 | 0.7106 | 0.6112 | 0.0109 | 3/6 PPP... | registry CV check failed |
| 500,000 | 0.0442 | 0.7515 | 0.1108 | 0.5876 | 0.6892 | 0.6194 | 0.0087 | 2/6 PP.... |  |
| 600,000 | 0.0452 | 0.7584 | 0.1484 | 0.5940 | 0.6468 | 0.6268 | 0.1675 | 2/6 PP.... | REJECTED: stride CV 0.1674648678741888 > 0.1; registry CV check failed; not entrained (base is too) |
| 700,000 | 0.0465 | 0.7629 | 0.1667 | 0.5936 | 0.7042 | 0.6157 | 0.0151 | 2/6 PP.... | registry CV check failed; not entrained (base is too) |
| 800,000 | 0.0485 | 0.7534 | 0.1677 | 0.5836 | 0.7575 | 0.6022 | 0.0211 | 3/6 PP..P. | registry CV check failed |
| 900,000 | 0.0501 | 0.7598 | 0.1591 | 0.5799 | 0.7404 | 0.6197 | 0.2272 | 3/6 PP..P. | REJECTED: stride CV 0.227220020015802 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,000,000 | 0.0508 | 0.7626 | 0.1559 | 0.5745 | 0.7260 | 0.6489 | 0.3079 | 2/6 PP.... | REJECTED: stride CV 0.30789381048884634 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,100,000 | 0.0524 | 0.7723 | 0.1576 | 0.5792 | 0.7257 | 0.6481 | 0.3044 | 2/6 PP.... | REJECTED: stride CV 0.3043708976042615 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,200,000 | 0.0534 | 0.7682 | 0.1522 | 0.5776 | 0.7279 | 0.6865 | 0.3494 | 2/6 PP.... | REJECTED: stride CV 0.3493520790156129 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,300,000 | 0.0535 | 0.7659 | 0.1402 | 0.5521 | 0.7313 | 0.6494 | 0.2757 | 3/6 PP..P. | REJECTED: stride CV 0.27567784777081517 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,400,000 | 0.0530 | 0.7614 | 0.1312 | 0.5353 | 0.7023 | 0.8188 | 0.3757 | 2/6 PP.... | REJECTED: stride CV 0.37570522932864275 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,500,000 | 0.0532 | 0.7611 | 0.1097 | 0.5303 | 0.7032 | 0.7674 | 0.3399 | 2/6 PP.... | REJECTED: stride CV 0.33994021675682784 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,600,000 | 0.0543 | 0.7698 | 0.1129 | 0.5126 | 0.6985 | 0.6799 | 0.4115 | 2/6 PP.... | REJECTED: stride CV 0.41150149403352776 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,700,000 | 0.0545 | 0.7671 | 0.1043 | 0.5149 | 0.7089 | 0.7097 | 0.3015 | 2/6 PP.... | REJECTED: stride CV 0.30149443860546493 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,800,000 | 0.0550 | 0.7658 | 0.1129 | 0.5065 | 0.7027 | 0.6845 | 0.3380 | 2/6 PP.... | REJECTED: stride CV 0.33803967903234694 > 0.1; registry CV check failed; not entrained (base is too) |
| 1,900,000 | 0.0551 | 0.7762 | 0.1065 | 0.5102 | 0.6859 | 0.6577 | 0.3362 | 2/6 PP.... | REJECTED: stride CV 0.3361962795506125 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,000,000 | 0.0567 | 0.7697 | 0.1011 | 0.4985 | 0.6473 | 0.9577 | 0.4086 | 2/6 PP.... | REJECTED: stride CV 0.4085623461754522 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,100,000 | 0.0584 | 0.7669 | 0.1043 | 0.5012 | 0.6850 | 0.7810 | 0.3635 | 2/6 PP.... | REJECTED: stride CV 0.3634901036974103 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,200,000 | 0.0576 | 0.7630 | 0.0828 | 0.4941 | 0.6750 | 0.7382 | 0.3322 | 3/6 PPP... | REJECTED: stride CV 0.3322019633629199 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,300,000 | 0.0585 | 0.7676 | 0.0613 | 0.4891 | 0.6965 | 0.6863 | 0.3118 | 3/6 PPP... | REJECTED: stride CV 0.3118068770597868 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,400,000 | 0.0591 | 0.7697 | 0.0817 | 0.4951 | 0.6898 | 0.7520 | 0.3121 | 3/6 PPP... | REJECTED: stride CV 0.31208376470195587 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,500,000 | 0.0593 | 0.7772 | 0.1087 | 0.5062 | 0.6394 | 0.7165 | 0.3837 | 2/6 PP.... | REJECTED: stride CV 0.3836635484251544 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,600,000 | 0.0595 | 0.7758 | 0.0785 | 0.5079 | 0.6791 | 0.7083 | 0.2645 | 3/6 PPP... | REJECTED: stride CV 0.2644870221421507 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,700,000 | 0.0580 | 0.7862 | 0.1022 | 0.5146 | 0.6384 | 0.6855 | 0.3301 | 2/6 PP.... | REJECTED: stride CV 0.33010278247515135 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,800,000 | 0.0596 | 0.7858 | 0.0720 | 0.5132 | 0.6422 | 0.7015 | 0.3296 | 3/6 PPP... | REJECTED: stride CV 0.32960450051385204 > 0.1; registry CV check failed; not entrained (base is too) |
| 2,900,000 | 0.0598 | 0.7882 | 0.0591 | 0.5186 | 0.6410 | 0.6948 | 0.3473 | 3/6 PPP... | REJECTED: stride CV 0.34729562315469265 > 0.1; registry CV check failed; not entrained (base is too) |
| 3,000,000 | 0.0605 | 0.7979 | 0.1152 | 0.5283 | 0.5897 | 0.6633 | 0.4031 | 2/6 PP.... | REJECTED: stride CV 0.40310814999979927 > 0.1; registry CV check failed; not entrained (base is too) |
| 3,014,656 | 0.0607 | 0.7943 | 0.0548 | 0.5313 | 0.5986 | 0.8848 | 0.3973 | 3/6 PPP... | REJECTED: stride CV 0.3973079042496892 > 0.1; registry CV check failed; not entrained (base is too) |
