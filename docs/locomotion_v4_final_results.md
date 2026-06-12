# NeuroGecko locomotion V4 results

Champion model:
v4_5b_speed_polish_1m

V4.5B eval20:
speed=0.1100
single_render_speed=0.1132
FL=0.388
FR=0.320
hop=0.384
slip=0.0567
belly=0
falls=0

V4.6 relax de-stick:
front_seek_relax 0.35 -> 0.25
speed=0.1093
FL=0.388
FR=0.317
hop=0.383
slip=0.0569
Result: failed, no improvement.

V4.7 10M final polish:
speed=0.1035
single_render_speed=0.1055
FL=0.405
FR=0.343
hop=0.341
slip=0.0556
Result: cleaner but slower. More training pushes safe/sticky mode.

Final decision:
Use V4.5B as champion.
Do not continue V4.7.
Speed gate should likely be revised from 0.115 to 0.110 unless one structural speed fix is chosen later.
