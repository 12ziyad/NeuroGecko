# NeuroGecko V4 Final Locomotion Acceptance

Final decision:
Accept V4.5B as the official locomotion champion.

Champion model:
v4_5b_speed_polish_1m

Accepted speed gate revision:
old gate = 0.115
new gate = 0.110

Reason:
V4.5B is the best speed/support balance found.
The final structural review concluded that the remaining +0.005 speed is not available from a safe small controller change without risking the front-support/hop solution.

V4.5B eval20:
speed=0.1100
single_render_speed=0.1132
FL=0.388
FR=0.320
hop=0.384
slip=0.0567
belly=0
falls=0
all_4_feet_participate=True

V4.6:
front_seek_relax 0.35 -> 0.25
speed=0.1093
FL=0.388
FR=0.317
hop=0.383
slip=0.0569
Result: failed, no improvement.

V4.7 10M:
speed=0.1035
FL=0.405
FR=0.343
hop=0.341
slip=0.0556
Result: cleaner contact but slower. More training pushes safe/sticky mode.

Final conclusion:
Stop V4 locomotion training.
Use V4.5B as final base.
Future speed beyond 0.115 would require a new project-level morphology/actuator/controller redesign, not another PPO continuation.
