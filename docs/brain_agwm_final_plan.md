# Brain AGWM Final Plan

Brain Phase Patch 1 keeps locomotion frozen. The V4.5B walker remains the spinal
locomotion layer, loaded from `models/v4_5b_speed_polish_1m/final.zip` with its
matching `vecnormalize.pkl`. Do not reopen locomotion for this phase.

The AGWM brain sits above that frozen walker. It will eventually include:

- a vision encoder for the `head_cam` stream
- an anatomy graph over the gecko body
- a recurrent world model
- internal drives
- an actor-critic policy
- the frozen V4.5B walker underneath

The brain does not emit raw 25-joint actions. Its output is only the high-level
target/task command:

```text
[target_dir_x, target_dir_y, target_distance, engage]
```

That command is converted into the existing `GeckoWalkEnv.target` channel. The
frozen walker still owns all joint-level control.

The first brain task is vision-guided foraging: see food from the existing
`head_cam`, choose a target direction and engagement level, walk toward food, and
eat when close enough.

Safe training ladder:

1. Wiring test
2. Oracle rollout
3. 10k smoke
4. 300k-500k smoke
5. 1M-2M usable brain

Do not reopen locomotion while working through this ladder.
