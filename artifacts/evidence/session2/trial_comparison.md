# Session2 retained trial comparison

Retained CPU-only manual trials; no biological validation. Gate2 remains failed; no further tuning, training, or plant changes are implied.

Each case is n=1 deterministic trajectory, not an independent-animal sample. Replays are not pooled and no across-case/replay SD is computed.

Root selected19_front_height as experimental lab defaults; final_candidate and verified_final are retained confirmations. Later failed alternatives are not hidden.

P/F denote saved engineering checks, not biological validation. Paired values are left/right. Loads are raw scalar touch-threshold fractions conditional on the executed held command; they are not world-vertical forces.

## Six decision metrics

Targets: forward ≥0.04m/s; net/path ≥0.5; each hind swing load <0.10; each front stance load ≥0.65; each hind duty0.78±0.05; each ipsilateral phase0.435±0.03. Timing validity and completing without a fall are also required.

| Case | Forward m/s | Net/path | Hind swing HL/HR | Front stance FL/FR | Hind duty HL/HR | Phase HL→FL/HR→FR | Gate2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00_reference | 0.0118 F | 0.2056 F | 0.5111/0.6909 F | 0.5037/0.5633 F | 0.4855/0.5672 F | 0.4699/0.5371 F | F |
| 00b_exact_reference | 0.0118 F | 0.2056 F | 0.5111/0.6909 F | 0.5037/0.5633 F | 0.4855/0.5672 F | 0.4699/0.5371 F | F |
| 01_hind_lift | 0.0007 F | 0.0927 F | 0.2404/0.3182 F | 0.5137/0.6084 F | 0.4869/0.3909 F | 0.4539/0.3606 F | F |
| 02_mirror | 0.0096 F | 0.2836 F | 0.4354/0.2495 F | 0.5493/0.5498 F | 0.5288/0.5011 F | 0.3537/0.4920 F | F |
| 03_tuck | 0.0056 F | 0.1206 F | 0.4495/0.4242 F | 0.5437/0.4808 F | 0.6302/0.6751 F | 0.5656/0.5750 F | F |
| 04_left_press | -0.0014 F | 0.0190 F | 0.5020/0.4596 F | 0.4883/0.5515 F | 0.5596/0.5376 F | 0.5587/0.4323 F | F |
| 05_right_press | -0.0048 F | 0.0769 F | 0.4899/0.5172 F | 0.5390/0.3788 F | 0.4916/0.6046 F | 0.5128/0.5703 F | F |
| 06_seek | -0.0067 F | 0.0818 F | 0.5303/0.5273 F | 0.5397/0.3455 F | 0.4336/0.6267 F | 0.4282/0.6353 F | F |
| 07_continuity | -0.0067 F | 0.0818 F | 0.5303/0.5273 F | 0.5397/0.3455 F | 0.4336/0.6267 F | 0.4282/0.6353 F | F |
| 08_front_lift | 0.0335 F | 0.7038 P | 0.0707/0.0051 P | 0.6507/0.5697 F | 0.6575/0.7134 F | 0.5647/0.6463 F | F |
| 09_fore_ratio | 0.0296 F | 0.6550 P | 0.0636/0.0101 P | 0.6423/0.5599 F | 0.6550/0.7068 F | 0.5731/0.6435 F | F |
| 10_hind_duty | 0.0295 F | 0.6319 P | 0.0591/0.0033 P | 0.6377/0.5562 F | 0.6467/0.7198 F | 0.5752/0.6519 F | F |
| 11_phase | 0.0299 F | 0.6324 P | 0.0667/0.0087 P | 0.6338/0.5518 F | 0.6422/0.7098 F | 0.5904/0.6550 F | F |
| 12_heading | 0.0319 F | 0.6376 P | 0.0613/0.0185 P | 0.6298/0.5639 F | 0.6134/0.6929 F | 0.5808/0.6539 F | F |
| 13_amplitude | 0.0397 F | 0.7373 P | 0.0409/0.0587 P | 0.6516/0.5747 F | 0.6206/0.6631 F | 0.5740/0.6167 F | F |
| 14_front_height | 0.0385 F | 0.7615 P | 0.0516/0.0609 P | 0.6570/0.5825 F | 0.6174/0.6808 F | 0.5788/0.6246 F | F |
| 15_right_support | 0.0393 F | 0.7531 P | 0.0462/0.0467 P | 0.6439/0.5815 F | 0.6205/0.6752 F | 0.5854/0.6092 F | F |
| 16_left_support | 0.0391 F | 0.7579 P | 0.0527/0.0500 P | 0.6533/0.5717 F | 0.6530/0.6853 F | 0.5890/0.6273 F | F |
| 17_amplitude | 0.0411 P | 0.7779 P | 0.0624/0.0533 P | 0.6610/0.5808 F | 0.6565/0.6595 F | 0.5761/0.5903 F | F |
| 18_stance_rotation | 0.0462 P | 0.7770 P | 0.0161/0.0120 P | 0.5816/0.5707 F | 0.6672/0.6677 F | 0.6128/0.6166 F | F |
| 19_front_height | 0.0424 P | 0.8253 P | 0.0129/0.0109 P | 0.6114/0.6064 F | 0.6408/0.6312 F | 0.5774/0.5690 F | F |
| 20_front_height | 0.0355 F | 0.7668 P | 0.1570/0.1391 F | 0.6037/0.6132 F | 0.6334/0.6854 F | 0.6346/0.5615 F | F |
| 21_hind_height | 0.0340 F | 0.7927 P | 0.1301/0.1424 F | 0.6251/0.6290 F | 0.6570/0.6894 F | 0.6360/0.5612 F | F |
| final_candidate | 0.0424 P | 0.8253 P | 0.0129/0.0109 P | 0.6114/0.6064 F | 0.6408/0.6312 F | 0.5774/0.5690 F | F |
| steering_left | 0.0426 P | 0.8235 P | 0.0108/0.0109 P | 0.6077/0.6057 F | 0.6357/0.6442 F | 0.5721/0.5853 F | F |
| steering_right | 0.0425 P | 0.8244 P | 0.0129/0.0120 P | 0.6057/0.6081 F | 0.6395/0.6374 F | 0.5921/0.5722 F | F |
| verified_final | 0.0424 P | 0.8253 P | 0.0129/0.0109 P | 0.6114/0.6064 F | 0.6408/0.6312 F | 0.5774/0.5690 F | F |

## Contact timing and acquisition

Each foot cell is observed contact cycles/s / periodCV%. These remain contact-cycle diagnostics. Failed entrainment is not repaired by merging events. Missing means unavailable, not zero.

| Case | HL rate/CV% | FL rate/CV% | HR rate/CV% | FR rate/CV% | All rates pass | All CV single digits | ≥200Hz | No fall/completed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_reference | 3.9836 / 28.5043 | 3.8173 / 25.4392 | 3.0879 / 31.9121 | 2.7227 / 37.0501 | F | F | P | P |
| 00b_exact_reference | 3.9836 / 28.5043 | 3.8173 / 25.4392 | 3.0879 / 31.9121 | 2.7227 / 37.0501 | F | F | P | P |
| 01_hind_lift | 3.6161 / 33.1305 | 4.1003 / 38.7287 | 3.6302 / 19.5174 | 3.6266 / 23.2752 | F | F | P | P |
| 02_mirror | 1.5633 / 46.1222 | 2.4958 / 57.3962 | 3.5655 / 25.2914 | 3.5638 / 22.3627 | F | F | P | P |
| 03_tuck | 2.4044 / 49.0905 | 3.5019 / 26.6157 | 2.5614 / 32.9269 | 3.0952 / 32.8354 | F | F | P | P |
| 04_left_press | 3.5638 / 14.1017 | 3.5808 / 9.9706 | 2.5559 / 42.1651 | 3.5638 / 31.8386 | F | F | P | P |
| 05_right_press | 3.5680 / 12.1725 | 3.6791 / 20.6913 | 3.5859 / 18.0087 | 3.5672 / 2.4262 | F | F | P | P |
| 06_seek | 3.5672 / 13.8023 | 4.5141 / 28.2960 | 3.5237 / 22.8639 | 3.6258 / 11.5553 | F | F | P | P |
| 07_continuity | 3.5672 / 13.8023 | 4.5141 / 28.2960 | 3.5237 / 22.8639 | 3.6258 / 11.5553 | F | F | P | P |
| 08_front_lift | 1.1887 / 1.7074 | 1.6659 / 49.7897 | 1.1890 / 0.2152 | 1.1899 / 2.0311 | F | F | P | P |
| 09_fore_ratio | 1.2519 / 16.4251 | 1.8479 / 60.1138 | 1.1890 / 3.2830 | 1.1896 / 3.5119 | F | F | P | P |
| 10_hind_duty | 1.1914 / 0.9023 | 1.9610 / 61.5551 | 1.1893 / 3.5655 | 1.1875 / 2.0175 | F | F | P | P |
| 11_phase | 1.1878 / 0.6788 | 1.4286 / 37.5819 | 1.1893 / 2.0710 | 1.1899 / 2.0186 | F | F | P | P |
| 12_heading | 1.1887 / 2.2021 | 1.3049 / 27.7708 | 1.1887 / 0.3191 | 1.1890 / 2.3237 | P | F | P | P |
| 13_amplitude | 1.1887 / 1.0030 | 1.1893 / 3.3943 | 1.1857 / 3.5782 | 1.1893 / 2.1368 | P | P | P | P |
| 14_front_height | 1.1890 / 1.6387 | 1.1899 / 3.5312 | 1.1887 / 3.9940 | 1.1890 / 2.2578 | P | P | P | P |
| 15_right_support | 1.1887 / 1.5206 | 1.1888 / 3.6381 | 1.1891 / 4.4986 | 1.1896 / 2.3271 | P | P | P | P |
| 16_left_support | 1.1872 / 2.2571 | 1.1876 / 2.4394 | 1.1890 / 2.9821 | 1.1899 / 2.3898 | P | P | P | P |
| 17_amplitude | 1.2513 / 17.2713 | 1.1888 / 2.6095 | 1.1888 / 2.1046 | 1.1869 / 3.0573 | P | F | P | P |
| 18_stance_rotation | 1.1890 / 1.6077 | 1.1868 / 2.0851 | 1.1893 / 1.9377 | 1.1878 / 2.3563 | P | P | P | P |
| 19_front_height | 1.1887 / 2.8750 | 1.1910 / 1.6712 | 1.1888 / 3.1280 | 1.1917 / 2.1528 | P | P | P | P |
| 20_front_height | 1.7972 / 47.9456 | 2.1998 / 30.2465 | 1.4265 / 36.5151 | 1.7730 / 37.2839 | F | F | P | P |
| 21_hind_height | 2.6035 / 63.6798 | 2.3770 / 14.6680 | 1.2497 / 19.0510 | 1.6049 / 35.0546 | F | F | P | P |
| final_candidate | 1.1887 / 2.8750 | 1.1910 / 1.6712 | 1.1888 / 3.1280 | 1.1917 / 2.1528 | P | P | P | P |
| steering_left | 1.1905 / 3.0284 | 1.1882 / 2.4303 | 1.1916 / 4.1195 | 1.1878 / 2.4808 | P | P | P | P |
| steering_right | 1.1872 / 4.1740 | 1.1905 / 1.7951 | 1.1848 / 3.4176 | 1.1869 / 2.7918 | P | P | P | P |
| verified_final | 1.1887 / 2.8750 | 1.1910 / 1.6712 | 1.1888 / 3.1280 | 1.1917 / 2.1528 | P | P | P | P |

## Provenance

Copy six gate values/pass flags from each saved report. Early reports may use earlier endpoint masks; source report hashes and missing validity fields remain explicit.

The companion JSON contains every original report SHA256, retained-trace SHA256 verification, body/source hashes, exact controller parameters, per-foot rate flags, and recorded requirements. Most exploratory trials retained reports only. No original evidence is changed.
