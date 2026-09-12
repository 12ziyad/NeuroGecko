from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from brain.drives import DriveState
from common.provenance import parameter_value
from common.walker_pairing import check_pairing
from brain.strike import Strike
from envs.gecko_walk_env import GeckoWalkEnv

REPO = Path(__file__).resolve().parent.parent

#: Breaths per second. INVENTED: no breathing frequency has been published for
#: any gecko (#302). 0.4 Hz is one breath every 2.5 s, the order a keeper sees.
BREATH_HZ = 0.4
#: Throat excursion, degrees. INVENTED for the same reason.
GULAR_AMPLITUDE_DEG = 12.0   # was 5.0: with the skin bound to the throat the 5 was invisible (#343). INVENTED
#: Lid angle at a full blink, degrees. INVENTED; the ORGAN is published --
#: eublepharids are the only geckos with movable eyelids.
EYELID_CLOSED_DEG = 70.0


def _camera_scene_options(policy_camera_mode: str):
    """Keep camera contracts independent; sealing changes policy pixels.

    ``legacy`` preserves pre-seal render options for checkpoint comparisons,
    not the original MJCF itself. ``sealed`` masks group-1 body visuals, all
    sites and skins. World/light/camera/physical-body changes can still change
    observations; this is not a guarantee that arbitrary appearance edits are
    distribution-neutral. New vision experiments should select sealed mode
    explicitly, and checkpoint evaluations must record which mode was used.
    """
    if policy_camera_mode not in ("legacy", "sealed"):
        raise ValueError("policy_camera_mode must be 'legacy' or 'sealed'")
    policy = mujoco.MjvOption()
    human = mujoco.MjvOption()
    if policy_camera_mode == "sealed":
        policy.geomgroup[1] = 0
        policy.sitegroup[:] = 0
        policy.skingroup[:] = 0
    # Preserve ordinary human rendering, including body group 1 and shadows.
    # Do not enable collision group 3 or diagnostic measurement/site group 4.
    return policy, human


def _scene_obj(renderer):
    return getattr(renderer, "scene", getattr(renderer, "_scene", None))


def _add_scene_sphere(renderer, pos, radius=0.025, rgba=(1.0, 0.1, 0.05, 1.0)):
    scene = _scene_obj(renderer)
    if scene is None or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_scene_capsule(
    renderer,
    start,
    end,
    radius=0.006,
    rgba=(1.0, 0.7, 0.05, 0.9),
):
    scene = _scene_obj(renderer)
    if scene is None or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    try:
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(radius),
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
        scene.ngeom += 1
    except Exception:
        return


#: The colour the legacy marker was painted. Kept ONLY so the legacy path
#: reproduces bit for bit; nothing new should key on it.
LEGACY_MARKER_RGB = (0.10, 0.80, 0.15)


#: Chromatic tolerance for the placeholder food detector. DERIVED, not chosen:
#: it is half the chromatic distance from the prey to the nearest confusable
#: surface in the world. That nearest confuser is the GECKO'S OWN SPOTS at
#: 0.145 -- a brown cricket on sand really is camouflaged, which is why the
#: margin is thin and why a colour detector is the wrong instrument. Measured
#: distances: gecko spot 0.145, gecko skin 0.180, floor mat 0.219, sand texture
#: 0.243, belly 0.311.
FOOD_CHROMA_TOLERANCE = 0.07


def _rendered_rgb(model, geom_id):
    """The colour MuJoCo will actually DRAW this geom in.

    Not the same thing as `geom_rgba`, and the difference has now cost this
    project two identical bugs. A geom with a material and no rgba of its own
    carries MuJoCo's default 0.5 0.5 0.5 in `geom_rgba` while the renderer uses
    the material. Anything that asks the model what the prey looks like must
    ask this, so the detector and the renderer cannot answer differently.
    """
    matid = int(model.geom_matid[geom_id])
    rgba = model.geom_rgba[geom_id]
    if matid >= 0 and np.allclose(rgba, _MUJOCO_DEFAULT_RGBA):
        rgba = model.mat_rgba[matid]
    return tuple(float(x) for x in rgba[:3])


#: What MuJoCo puts in geom_rgba when the XML sets none. A geom sitting at
#: exactly this value has almost certainly not stated a colour at all.
_MUJOCO_DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)


def _food_visible_frac(image: np.ndarray, target_rgb=None,
                       tolerance: float = FOOD_CHROMA_TOLERANCE) -> float:
    """Fraction of pixels that look like the food, in a uint8 HxWx3 image.

    THIS IS A PLACEHOLDER FOR A RETINA and is labelled as one. A real eye does
    not find prey by matching a colour; brain module 4 replaces it.

    `target_rgb` is READ FROM THE MODEL by the caller rather than written here.
    That is the whole point of the argument. The previous version hard-coded a
    green test -- `g > 120 and g > r + 30 and g > b + 30` -- which was correct
    for the painted marker it was written against and returns EXACTLY 0.0 for
    the brown prey that replaced it, at every illumination from 0.4x to 3.0x.
    So the no-cheat world shipped with an animal that could not see its own
    food, and the detector and the world could disagree silently because each
    stated the colour separately. Deriving it from the model removes the class
    of bug rather than moving it to a new colour.

    Matching is on CHROMATICITY -- the pixel normalised by its own brightness
    -- so a shadowed cricket and a lit one are the same object. The old test
    was on absolute channel values and failed under illumination change even
    for its own marker: at gain 0.4 the painted sphere also scored 0.0.
    """
    if target_rgb is None:
        target_rgb = LEGACY_MARKER_RGB
    pixels = image.astype(np.float64)
    brightness = pixels.sum(axis=2, keepdims=True)
    target = np.asarray(target_rgb, dtype=np.float64)
    target_sum = float(target.sum())
    if target_sum <= 0:
        return 0.0
    # Unlit pixels carry no colour information; excluding them stops black from
    # matching everything once it is normalised.
    lit = brightness[:, :, 0] > 24.0
    with np.errstate(invalid="ignore", divide="ignore"):
        chroma = np.where(brightness > 0, pixels / brightness, 0.0)
    distance = np.abs(chroma - (target / target_sum)).sum(axis=2)
    mask = lit & (distance < tolerance)
    return float(mask.sum()) / float(mask.size)


class GeckoBrainEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        walker_run: str = "v4_5b_speed_polish_1m",
        frame_skip: int = 25,
        max_steps: int = 1000,
        brain_steps_per_action: int = 1,
        food_radius: float | None = None,
        food_spawn_radius: tuple[float, float] = (0.25, 0.70),
        food_spawn_angle_deg: float = 180.0,
        eat_radius: float | None = None,
        camera_width: int = 64,
        camera_height: int = 64,
        render_mode: str | None = None,
        seed: int | None = None,
        privileged_target: float = 1.0,
        privileged_food_dropout_prob: float = 0.0,
        walker_oracle: bool = True,
        strike: bool = False,
        gait_profile: str = "legacy",
        warn_pairing: bool = True,
        hunt_targeting: bool = False,
        use_policy: bool = True,
        control_mode: str = "cpg_residual",
        residual_scale: float = 0.25,
        front_stance_press: float = 0.40,
        front_swing_lift: float = 0.40,
        contact_thresh: float = 0.0564,
        show_debug_markers: bool = False,
        view_mode: str = "close",
        camera_smoothing: float = 0.0,
        policy_camera_mode: str = "legacy",
        walker_xml_path: str | Path | None = None,
        privileged_food_channel: bool = True,
        prey_parameters=None,
        homeostasis: bool = False,
        homeostasis_time_compression: float = 1.0,
        action_selection: bool = False,
        behaviour_control: bool = False,
        meals_owed_at_start: float = 0.0,
        mouth_goal: bool = False,
        smell: bool = False,
        circadian: bool = False,
        time_of_day_h: float = 0.0,
        eye: bool = False,
        eye_render_pixels: int | None = None,
        eye_receptor_pixels: int | None = None,
        gaze_pitch_gain: float = 0.0,
        gaze_pitch_rate: float = 0.25,
        gaze_pitch_max_deg: float = 45.0,
        gaze_pitch_decay: float = 0.0,
    ):
        super().__init__()
        #: Closed-loop gaze. 0.0 is OFF and is the default, so every existing
        #: run and every gate result is unchanged until this is switched on.
        self.gaze_pitch_gain = float(gaze_pitch_gain)
        self.gaze_pitch_rate = float(np.clip(gaze_pitch_rate, 0.0, 1.0))
        self.gaze_pitch_max_rad = math.radians(float(gaze_pitch_max_deg))
        #: How fast the aim relaxes when the eye reports nothing. INVENTED.
        #: 0.0 holds the aim, which is what the measurement above supports.
        self.gaze_pitch_decay = float(np.clip(gaze_pitch_decay, 0.0, 1.0))
        self._gaze_pitch_rad = 0.0
        self._gaze_yaw_rad = 0.0
        self._prev_gaze_yaw_rad = 0.0
        self._last_head_yaw_rate = 0.0
        self._breath_t = 0.0
        self.policy_camera_mode = str(policy_camera_mode)
        self._policy_scene_option, self._render_scene_option = _camera_scene_options(
            self.policy_camera_mode
        )
        self.walker_run = str(walker_run)
        self.show_debug_markers = bool(show_debug_markers)
        _valid_views = ("fixed", "chase", "close", "hunt", "static", "track")
        if str(view_mode).lower() not in _valid_views:
            raise ValueError(f"view_mode must be one of {_valid_views}, got '{view_mode}'")
        self._view_mode = str(view_mode).lower()
        self._camera_smoothing = float(max(0.0, min(1.0, camera_smoothing)))
        self._smooth_lookat: np.ndarray | None = None
        self._static_anchor: np.ndarray | None = None
        self._static_azimuth: float | None = None
        self._smooth_azimuth: float | None = None
        self.max_steps = int(max_steps)
        self.brain_steps_per_action = int(brain_steps_per_action)
        if self.brain_steps_per_action < 1:
            raise ValueError("brain_steps_per_action must be >= 1")
        # These were 0.035 m and 0.10 m: a 7 cm-wide food item, eaten from 10 cm.
        # The published strike trigger distance for evasive prey is 2.03 cm at
        # 0.384 SVL, which is 4.07 cm on this body -- so the old eat radius was
        # about 2.5x the distance a real gecko launches from, and the old food
        # was four times the width of the prey it stands for. None now resolves
        # both from the registry, where each carries its own provenance. Passing
        # the old numbers explicitly still reproduces the legacy behaviour.
        from common.provenance import parameter_value
        self.food_radius = float(parameter_value("prey_radius_m")
                                 if food_radius is None else food_radius)
        self.food_spawn_radius = (
            float(food_spawn_radius[0]),
            float(food_spawn_radius[1]),
        )
        self.food_spawn_angle_deg = float(food_spawn_angle_deg)
        self.eat_radius = float(parameter_value("prey_capture_distance_m")
                                if eat_radius is None else eat_radius)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.render_mode = render_mode
        self.privileged_target = float(privileged_target)
        self.privileged_food_dropout_prob = float(np.clip(privileged_food_dropout_prob, 0.0, 1.0))
        # privileged_target scales the channel and can be curriculum'd to zero,
        # but the five slots remain in the observation. This removes the block
        # entirely, which is the honest configuration and which no recovered
        # checkpoint fits -- that is the point, not an oversight.
        self.privileged_food_channel = bool(privileged_food_channel)
        self._rng = np.random.default_rng(seed)

        walker_max_steps = max(1, self.max_steps * self.brain_steps_per_action)
        self.walk_env = GeckoWalkEnv(
            xml_path=walker_xml_path,
            frame_skip=frame_skip,
            max_steps=walker_max_steps,
            control_mode=control_mode,
            residual_scale=residual_scale,
            contact_thresh=contact_thresh,
            front_stance_press=front_stance_press,
            front_swing_lift=front_swing_lift,
            render_mode=render_mode,
            seed=seed,
            privileged_target=bool(walker_oracle),
            gait_profile=gait_profile,
            # WHICH PART OF THE ANIMAL HAS TO ARRIVE, and this is the fix for
            # what #174 misdiagnosed as a broken walker.
            #
            # The walker's goal test measured from the TRUNK and called 4 cm
            # arrival. Nothing in it mentioned the mouth. Told instead that the
            # goal is the mouth at the published 20.3 mm strike distance, the
            # SAME frozen checkpoint arrives 75 times in 160 seconds -- one
            # every 2.1 s, mean closest 20.5 mm. It could always do this.
            # `mouth_goal` gives the autonomous animal the same arrival test the
            # oracle-driven one always had -- the MOUTH within strike range --
            # without the oracle. Measuring your own arrival from your own nose
            # is proprioception, not privileged information (#285).
            **({"approach_site": "mouth",
                "reach_dist": float(parameter_value("strike_trigger_distance_m"))}
               if (hunt_targeting or mouth_goal) else {}),
        )
        # THE PROFILE THE BRAIN ACTUALLY WALKS ON, NOW NAMED RATHER THAN
        # INHERITED. This constructor never passed `gait_profile`, so
        # GeckoWalkEnv's "legacy" default won -- and the 4/6 gate evidence was
        # measured under "lab" (artifacts/evidence/lab_frozen/report.json,
        # gait_profile: lab). Same body, and the same frozen checkpoint: model
        # sha256 77b7a99d is byte-identical to the one the evidence used. What
        # differs is the CPG scaffolding the residual rides on. So everything
        # built on this environment has been running the UN-GATED profile.
        # That is the third time a default has won because nobody named the
        # parameter -- see the walker oracle above, and #26/#33 before it.
        #
        # The default stays "legacy" for the same reason the oracle stays on:
        # every brain checkpoint in models/brain/ was trained against it, and
        # flipping a default silently is how this class of defect gets made in
        # the first place. Measured cost of the difference is in ledger #180.
        # What changes is that it can no longer be silent.
        self.use_policy = bool(use_policy)
        self.gait_profile = self.walk_env.gait_profile
        # AIM AT THE PREY, NOT ROUGHLY THAT WAY.
        #
        # `_brain_action_to_target` places the walker's goal 0.05-0.80 m ahead
        # in the BODY FRAME, derived from the brain's action. That is a fine
        # abstraction for wandering and it cannot express "go to the cricket":
        # the brain can only ask for a heading and a range, and the last few
        # centimetres -- the ones that decide whether a strike is possible --
        # are exactly where a heading-and-range request loses.
        #
        # With hunt_targeting on, the target is placed AT the prey whenever
        # prey exists, and the walker's own goal machinery does the rest. Off
        # by default: it changes what the action means, so every existing
        # checkpoint and every existing evidence run keeps the old behaviour.
        self.hunt_targeting = bool(hunt_targeting)
        # This env always loads a frozen residual, so `using_policy` is True.
        # The guard exists because Session 9 assembled the one broken pairing
        # out of two independent defaults and nothing objected -- see
        # common/walker_pairing.py for the measurements.
        if warn_pairing:
            #  rather than a hard True: lab WITHOUT the residual is
            # the accepted walker, and warning about it would train the reader
            # to ignore the warning.
            check_pairing(self.gait_profile, self.use_policy, context="GeckoBrainEnv")
        # THE WALKER'S OWN ORACLE, NOW PASSED RATHER THAN DEFAULTED.
        #
        # GeckoWalkEnv puts five numbers into its 92-long proprioception vector:
        # exact egocentric direction, distance and bearing to the goal, read out
        # of the physics engine, noiseless, at any range. It defaults to True and
        # this constructor never passed the flag, so it was unconditionally on
        # and nothing here could turn it off. All twelve checkpoints in the
        # repository record proprio_dim 92, including the ones described as
        # pure-vision. This is the THIRD time a privileged channel has survived a
        # "removed everywhere" declaration -- see FAILURE_MAP #26 and #33.
        #
        # It is exposed rather than removed, and it still defaults to True,
        # because tools/oracle_ablation.py measured what removing it costs:
        # zeroing the five slots leaves the walker moving just as far (0.086 ->
        # 0.110 m) while progress TOWARD THE GOAL collapses 0.0664 -> 0.0060 m,
        # a 91 % loss. The legs are fine; the navigation was the oracle. So the
        # channel is load-bearing, turning it off needs a retrained walker, and
        # flipping the default silently would break every checkpoint here.
        #
        # What changes is that it can no longer be claimed absent: the flag is
        # real, and step() reports the truth in info["walker_oracle"].
        self.walker_oracle = bool(walker_oracle)
        # Prey needs a world with somewhere to put it. When the model carries a
        # prey mocap body the food IS that body, so the camera sees a real geom
        # rather than a sphere drawn onto the scene after rendering.
        self.prey = None
        # THE STRIKE. Off by default, like the eye, and for the same reason:
        # a new module does not become the live path until it is shown to
        # work. With it off, capture is proximity -- walk within the eat
        # radius and the prey is eaten, which is what every existing
        # checkpoint was trained against. With it on, capture requires a
        # strike that LANDS, and the published miss rate applies: 82.9 % on
        # evasive crickets, so roughly one in five attempts fails.
        self.strike = Strike(rng=self._rng) if strike else None
        self._mouth_sid = mujoco.mj_name2id(self.walk_env.model, mujoco.mjtObj.mjOBJ_SITE, "mouth")
        self._tongue_sid = mujoco.mj_name2id(self.walk_env.model, mujoco.mjtObj.mjOBJ_SITE, "tongue_tip")
        self._jaw_ok = getattr(self.walk_env, "_jaw_act", -1) >= 0 and self._tongue_sid >= 0
        # Lever from the neck pitch axis to the snout, for turning a published
        # head DROP (a translation) into the pitch that delivers it. MEASURED
        # from the body at construction (neck body origin to nose_tip site in
        # the stand pose), not assumed: the first version hard-coded 45 mm and
        # the body says 37 mm (#300).
        _m, _d = self.walk_env.model, self.walk_env.data
        _neck = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, "neck")
        _nose = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_SITE, "nose_tip")
        self._snout_lever_m = (float(np.linalg.norm(_d.site_xpos[_nose] - _d.xpos[_neck]))
                               if _neck >= 0 and _nose >= 0 else 0.037)
        self._strike_touched = False
        self._strike_base_pitch = None
        self._breath_t = 0.0
        self._prey_mocap = -1
        #: What the food actually looks like, read from the model so the
        #: detector cannot disagree with the world. None until a prey geom is
        #: found, in which case the legacy marker colour is used.
        self._food_rgb = None
        if prey_parameters is not None:
            from envs.prey import FleeingPrey
            body = mujoco.mj_name2id(self.walk_env.model, mujoco.mjtObj.mjOBJ_BODY, "prey")
            if body < 0:
                raise ValueError("This walker model has no prey body. Pass walker_xml_path to a "
                                 "world generated by utils/build_world.py --prey-radius.")
            self._prey_mocap = int(self.walk_env.model.body_mocapid[body])
            if self._prey_mocap < 0:
                raise ValueError("The prey body must be mocap so it stays out of qpos.")
            self.prey = FleeingPrey(prey_parameters, height_m=prey_parameters.radius_m,
                                    rng=self._rng,
                                    # If a strike exists it owns capture. Leaving
                                    # proximity capture on would respawn the prey
                                    # the moment it entered striking range, so no
                                    # strike could ever fire -- measured, 0 strikes
                                    # across 1050 frames with the prey placed in
                                    # range on every step.
                                    proximity_capture=not strike)
            self.food_radius = float(prey_parameters.radius_m)
            self.eat_radius = float(prey_parameters.capture_distance_m)
            # Read the prey's actual colour out of the model. One source of
            # truth: if the world is regenerated in a different colour the
            # detector follows it without anyone remembering to.
            #
            # THE MATERIAL WINS, AND GETTING THAT WRONG WAS THE SAME BUG TWICE.
            # Session 8 fixed a detector that tested green against a brown prey
            # by reading the colour from the model instead of hard-coding it --
            # and read `geom_rgba`. But the prey geom in
            # morphology/gecko_world_v1.xml carries `material="prey"` and sets
            # no rgba, so `geom_rgba` is MuJoCo's *default* 0.5 0.5 0.5 grey
            # while the renderer draws 0.55 0.35 0.18. Measured: the detector
            # scored 0.0 on a patch of the colour the world actually shows.
            # A geom's own rgba overrides its material's, so that is checked
            # first and the material is the fallback -- the same precedence
            # the renderer uses.
            geom = mujoco.mj_name2id(self.walk_env.model,
                                     mujoco.mjtObj.mjOBJ_GEOM, "prey_geom")
            if geom < 0:
                raise ValueError(
                    "The prey body has no geom named 'prey_geom', so its "
                    "appearance cannot be read and the food detector would "
                    "silently key on the wrong colour.")
            self._food_rgb = _rendered_rgb(self.walk_env.model, geom)
        self._nose_sid = mujoco.mj_name2id(
            self.walk_env.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "nose_tip",
        )
        if self._nose_sid < 0:
            raise RuntimeError("XML site 'nose_tip' not found; required for mouth-based eating.")

        # THE ACCEPTED WALKER WAS UNREACHABLE FROM HERE, and that is why every
        # clip this session was filmed with a rejected one.
        #
        # This env always loaded the frozen residual and defaulted to
        # gait_profile="legacy". The walker the gates accepted is `lab` with
        # ZERO residual, and no combination of arguments could ask for it.
        # Measured with one consistent method, fraction of the step each foot
        # carries load, against published targets of 0.70 fore and 0.765 hind:
        #
        #     lab + no policy     0.770 0.764 0.787 0.790   even to 0.006
        #     legacy + policy     0.704 0.601 0.435 0.367   hind feet at half
        #
        # The second line is what was on screen. The front pair looked fine,
        # which is why it survived: the only duty metric being checked was the
        # front one, and the defect was in the hind feet.
        #
        # With use_policy=False the CPG base drives the legs directly. Under
        # `lab` the base steers off the target bearing itself, so the animal
        # still goes where the brain points it.
        self.walk_model, self.walk_norm = (
            self._load_frozen_walker(self.walker_run) if use_policy else (None, None))
        self.walk_obs_dim = int(np.prod(self.walk_env.observation_space.shape))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    0,
                    255,
                    shape=(self.camera_height, self.camera_width, 3),
                    dtype=np.uint8,
                ),
                "proprio": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.walk_obs_dim,),
                    dtype=np.float32,
                ),
                "drives": spaces.Box(0.0, 1.0,
                                     shape=(4,) if homeostasis else (6,), dtype=np.float32),
                "prev_action": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                **({"privileged": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)}
                   if privileged_food_channel else {}),
            }
        )

        # brain/drives.py has hunger saturating in 67 s from invented constants.
        # The hypothalamus replaces it with the published energy budget, at which
        # point hunger moves on a timescale of days and the drives vector is four
        # channels rather than six. Opt-in, because no recovered checkpoint fits
        # the smaller vector.
        # Brain module 2, reading brain module 1. REPORTING ONLY: the selector
        # names the behaviour the animal would choose and that name goes into
        # info, but it does not steer anything. It must not, yet -- only one
        # behaviour has a controller. Hooking a six-way chooser to a
        # one-behaviour body would look like integration and mean nothing.
        # What this DOES buy is the two modules running together on live data
        # every step, which is the only way the seam gets exercised.
        # Brain module 4. When present it REPLACES the colour matcher: prey
        # visibility becomes a small-moving-target response over a retinotopic
        # map, and a bearing comes with it, which the colour matcher never
        # produced. Off by default, because turning it on changes what the
        # policy is given and no trained checkpoint has seen it.
        self.eye = None
        self._eye_render_px = None
        if eye:
            from brain.tectum import Eye
            # SUPERSAMPLE, then limit to the animal. Rendering above the
            # animal's own resolving power is not a superpower: the image is
            # low-passed to what its receptors can carry before anything reads
            # it. What it buys is the removal of ALIASING -- fine floor texture
            # that the eye could never resolve otherwise survives as false
            # structure and moves when the animal moves, which is
            # indistinguishable from prey to a motion detector.
            render_px = int(eye_render_pixels or 64)
            receptor_px = int(eye_receptor_pixels or 64)
            self._eye_render_px = render_px
            self.eye = Eye(fovy_deg=70.0, pixels=receptor_px,
                           cells=min(receptor_px, 64),
                           render_pixels=render_px)

        self.selector = None
        if action_selection:
            if not homeostasis:
                raise ValueError(
                    "action_selection needs homeostasis=True: the selector's "
                    "salience is computed from the hypothalamus drive vector.")
            from brain.gecko_selector import GeckoSelector
            self.selector = GeckoSelector()

        # THE WIRE (#265). Until session 12 the selector chose a behaviour and
        # the env put it in the info dict beside `behaviour_controls_nothing`,
        # because the body could only do one thing. It can now do three -- search,
        # chase, hold still -- so selection is allowed to mean something.
        #
        # OFF BY DEFAULT, and `step(action)` is untouched whether it is on or
        # off: the closed loop runs through `autonomous_step()`, which is a
        # separate entry point. Every checkpoint, gate and policy that drives
        # this env with its own actions behaves exactly as before.
        self.programs = None
        self.brainstem = None
        self.search = None
        self.evidence = None
        if behaviour_control:
            missing = [n for n, v in (("homeostasis", homeostasis),
                                      ("action_selection", action_selection),
                                      ("eye", eye)) if not v]
            if missing:
                raise ValueError(
                    "behaviour_control needs " + ", ".join(f"{m}=True" for m in missing)
                    + ": the loop runs hunger -> basal ganglia -> motor program "
                      "-> body, and a missing link would be filled with a "
                      "placeholder, which is what this flag exists to stop.")
            from brain.programs import MotorPrograms
            from brain.search import FixationEvidence, SearchPattern
            self.search = SearchPattern()
            self.evidence = FixationEvidence(
                adaptive=True, fixate_steps=SearchPattern.FIXATE_STEPS)
            try:
                from brain.brainstem import Brainstem
                from brain.spinal_cpg import SpinalCPG, FEET
                cpg = self.walk_env.cpg
                delays = {f: cpg.profile.touchdown_delays_cycle[i]
                          for i, f in enumerate(FEET)}
                # The cord is the brainstem's OUTPUT TARGET, not the walker's
                # rhythm source: `attach_spinal_cord` is deliberately not called,
                # so the accepted 4/6 walker keeps running on the closed form it
                # was gated on. What the brainstem contributes here is its
                # behaviour-to-effort map, which is the published-in-direction
                # part; the rhythm stays where the evidence is.
                self.brainstem = Brainstem(SpinalCPG(cpg.freq, delays))
            except Exception:
                self.brainstem = None       # falls back to LOCOMOTOR directly
            self.programs = MotorPrograms(self.search, self.evidence,
                                          brainstem=self.brainstem)

        # Brain module 5: smell. Built, tested, and until now read by nothing.
        # For THIS species chemoreception is what gates costly defence --
        # reaction to snake scent 0.21 against 0.07 for the sight of a snake,
        # n = 42 -- so a threat signal computed from vision alone was modelling
        # the wrong sense. See brain/vomeronasal.py for the statistics.
        self.nose = None
        if smell:
            from brain.vomeronasal import Vomeronasal
            self.nose = Vomeronasal()

        # Brain module 6: the day/night clock. CONNECTED (#352). It was built
        # and tested in session 9w and consumed by nothing (#262); session 12
        # added this flag, which constructed the object, reset it, and reported
        # an `arousal` the object never produced, because `clock.step` was
        # never called and `self._arousal` was the constant 1.0 (#347).
        #
        # It is stepped below in `step()` and its output is the `rest` channel's
        # salience, through `brain/gecko_selector.py`. That is the only
        # modulation the sources speak to: the two published numbers this
        # species has about arousal -- activity onset after lights-out, and an
        # evening peak window -- are measurements of WHEN IT IS ACTIVE AND WHEN
        # IT IS NOT, which is what rest is. Anything else arousal could be made
        # to modulate would be invention, and #347 was right to refuse it.
        #
        # `time_of_day_h` sets the phase the episode starts in. It is a SCENARIO
        # setting, not biology -- the same shape as `meals_owed_at_start` -- and
        # it exists because a day is 86400 s and an episode is tens of seconds,
        # so the phase has to be chosen rather than waited for. The clock
        # advances on the SAME declared `homeostasis_time_compression` as the
        # rest of the slow physiology; at its default of 1.0 the clock barely
        # moves inside an episode, which is correct and not a defect.
        self._time_of_day_s = float(time_of_day_h) * 3600.0
        self._time_compression = float(homeostasis_time_compression)
        self._arousal = 1.0
        #: Whether the rest channel shut the lids last step, so that opening
        #: them again is this module's business and not a blanket write that
        #: would fight the strike's own blink.
        self._eyes_were_shut = False
        self._last_eye_bearing = None
        self.clock = None
        if circadian:
            from brain.arousal import Arousal
            self.clock = Arousal()
            self._arousal = float(self.clock.arousal_at(self._time_of_day_s))

        self.homeostasis = None
        if homeostasis:
            from brain.hypothalamus import Homeostasis, Physiology
            mass = float(np.sum(self.walk_env.model.body_mass[1:]))
            physiology = Physiology.from_registry(body_mass_kg=mass)
            self.homeostasis = Homeostasis(
                physiology,
                time_compression=float(homeostasis_time_compression),
            )
            # A NOCTURNAL FORAGER EMERGES HUNGRY (#268), and without this the
            # animal is never hungry inside an episode at all. Measured: the
            # hypothalamus reports ~3292 HOURS of reserve remaining, so over a
            # 600-step rollout the energy deficit reaches 0.0000 and the basal
            # ganglia releases NOTHING on every step -- which is the real reason
            # the selector was never wired to anything.
            #
            # The alternative is `homeostasis_time_compression`, and it takes a
            # value near 20000 before any behaviour is released, i.e. starving
            # the animal over three days inside twelve seconds. Starting it with
            # an energy debt is the same statement made honestly: this species
            # eats every few days, so an animal emerging at dusk is some number
            # of meals behind. That number is the parameter, it is INVENTED at
            # its default of 0.0 meaning "starts full", and it is reported.
            self.meals_owed_at_start = float(meals_owed_at_start)
            if self.meals_owed_at_start:
                self.homeostasis.energy_J = float(
                    self.homeostasis.energy_setpoint_J
                    - self.meals_owed_at_start * self.homeostasis.meal_scale_J)
                self._initial_energy_J = self.homeostasis.energy_J
        self.drives = DriveState()
        self.food_xy = np.zeros(2, dtype=np.float64)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._brain_target_xy = np.zeros(2, dtype=np.float64)
        self._step = 0
        self._head_renderer = None
        self._render_renderer = None
        self._last_info = {}

    def set_privileged_food_scale(self, scale: float) -> None:
        self.privileged_target = float(scale)

    def set_privileged_food_dropout_prob(self, prob: float) -> None:
        self.privileged_food_dropout_prob = float(np.clip(prob, 0.0, 1.0))

    def set_food_radius(self, radius: float) -> None:
        self.food_radius = float(max(radius, 1e-3))

    def _load_frozen_walker(self, walker_run: str):
        run_dir = REPO / "models" / walker_run
        model_path = run_dir / "final.zip"
        vec_path = run_dir / "vecnormalize.pkl"
        missing = [p for p in (model_path, vec_path) if not p.exists()]
        if missing:
            paths = "\n".join(f"  - {p}" for p in missing)
            raise FileNotFoundError(
                "Missing frozen walker artifact(s):\n"
                f"{paths}\n"
                "Brain Phase Patch 1 expects a frozen walker with final.zip and "
                "vecnormalize.pkl. No locomotion training is started here."
            )

        vec_env = DummyVecEnv([lambda: self.walk_env])
        norm = VecNormalize.load(str(vec_path), vec_env)
        norm.training = False
        norm.norm_reward = False
        model = PPO.load(str(model_path), device="cpu")
        return model, norm

    def _trunk_xy(self) -> np.ndarray:
        return self.walk_env.data.xpos[self.walk_env._trunk][:2].copy()

    def _trunk_rot(self) -> np.ndarray:
        return self.walk_env.data.xmat[self.walk_env._trunk].reshape(3, 3).copy()

    def _write_prey(self) -> None:
        self.walk_env.data.mocap_pos[self._prey_mocap] = self.prey.mocap_position
        mujoco.mj_forward(self.walk_env.model, self.walk_env.data)

    def _spawn_food(self) -> None:
        if self.prey is not None:
            # Prey places itself, never inside its own flee radius, so the
            # animal always has to travel to it.
            self.food_xy = self.prey.reset(self._nose_xy(), rng=self._rng)
            self._write_prey()
            return
        lo, hi = self.food_spawn_radius
        if hi < lo:
            lo, hi = hi, lo
        if self.food_spawn_angle_deg >= 180.0:
            angle = self._rng.uniform(-np.pi, np.pi)
        else:
            a = math.radians(max(self.food_spawn_angle_deg, 0.0))
            forward_world = self._trunk_rot() @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            heading = math.atan2(forward_world[1], forward_world[0])
            angle = heading + self._rng.uniform(-a, a)
        radius = self._rng.uniform(max(lo, 0.0), max(hi, 0.0))
        offset = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        self.food_xy = self._trunk_xy() + offset

    def _food_delta_body(self):
        trunk_xy = self._trunk_xy()
        d_world = np.array(
            [self.food_xy[0] - trunk_xy[0], self.food_xy[1] - trunk_xy[1], 0.0],
            dtype=np.float64,
        )
        dist = float(np.linalg.norm(d_world[:2])) + 1e-9
        d_body = self._trunk_rot().T @ d_world
        ego = d_body[:2] / dist
        heading = float(np.arctan2(ego[1], ego[0]))
        return ego.astype(np.float32), dist, heading

    def food_egocentric(self):
        ego, dist, heading = self._food_delta_body()
        return ego.copy(), float(dist), float(heading)

    def oracle_action(self, engage: float = 1.0) -> np.ndarray:
        ego, dist, _ = self._food_delta_body()
        dist_cmd = 2.0 * (np.clip(dist, 0.05, 0.80) - 0.05) / 0.75 - 1.0
        engage_cmd = 2.0 * np.clip(float(engage), 0.0, 1.0) - 1.0
        return np.array([ego[0], ego[1], dist_cmd, engage_cmd], dtype=np.float32)

    def _food_distance(self) -> float:
        return float(np.linalg.norm(self.food_xy - self._trunk_xy()))

    def _nose_xy(self) -> np.ndarray:
        return np.asarray(self.walk_env.data.site_xpos[self._nose_sid][:2], dtype=np.float64)

    def _mouth_food_distance(self) -> float:
        return float(np.linalg.norm(self.food_xy - self._nose_xy()))

    def _set_walk_target(self, target: np.ndarray) -> None:
        self.walk_env.target = np.asarray(target, dtype=np.float64).copy()
        _, self.walk_env._prev_dist, _ = self.walk_env._target_egocentric()

    def _run_strike(self, dt, proximity_capture):
        """Advance any strike in flight, and launch one if the prey is close.

        Returns whether the prey was actually caught this step. Proximity alone
        no longer counts: `proximity_capture` is discarded except as the signal
        that the prey has already respawned itself, which the strike must not
        then also claim.
        """
        # HORIZONTAL distance, and the reason is published. The strike is
        # mostly a DOWNWARD lunge: Delheusy, Brillet & Bels 1995 measured the
        # head translating about 8 mm horizontally and about 28 mm VERTICALLY
        # over the 80 ms capture cycle in this species. Our nose sits 13.7 mm
        # above a prey that is 9 mm off the floor, so a 3-D trigger of 20.3 mm
        # can never be met by walking -- the vertical gap alone eats most of
        # it, and closing that gap is exactly what the strike is for. Measured:
        # with a 3-D trigger the animal stalls at 40.8 mm and never strikes.
        offset = np.asarray(self.food_xy)[:2] - self._nose_xy()
        mouth_distance = float(np.linalg.norm(offset))
        fraction, finished, dice_hit = self.strike.step(dt)

        # THE STRIKE IS A MOVEMENT, AND THE CATCH IS GEOMETRY (#292). Until
        # session 12 nothing moved: `fire()` rolled a die against the published
        # 82.9 % and the cricket vanished. Now, while a strike is in flight, the
        # mouth opens along the published gape profile, the head drops along
        # the published ~27 mm, and the prey is caught if -- and only if -- its
        # centre passes inside the mouth while the jaws are open. The 82.9 %
        # stops being a number typed in and becomes a number the animal has to
        # earn. The statistical path is kept for bodies without a jaw, so every
        # older evidence run still reproduces.
        # BREATHING, every step, strike or not. The visible pulsing under a
        # gecko's chin is the gular pump, and it is what makes a still animal
        # look alive rather than paused (#302). Geckos breathe in single breaths
        # or short bursts separated by breath-holds (Milsom 1984, Gekkonidae,
        # abstract only; Dial & Schwenk 1996 record "buccal pulsing" in
        # Coleonyx brevis, SAME FAMILY, as an olfactory sniff before a defensive
        # display). NO breathing frequency, gular excursion or throat
        # displacement has ever been published for any gecko -- the session-12
        # sweep closed that search -- so BREATH_HZ and the amplitude are
        # INVENTED and say so here.
        self._breath_t += dt
        breath = math.sin(2.0 * math.pi * BREATH_HZ * self._breath_t)
        self.walk_env.gular_rad = math.radians(GULAR_AMPLITUDE_DEG) * breath

        # THE TONGUE (#343). Two published behaviours, both this species, and
        # nothing else moves it:
        #   * post-feeding labial licking at 0.165 licks/s for the minutes
        #     after a meal (Cooper, DePerno & Steele 1996, n = 16; the registry
        #     entry post_feeding_labial_lick_rate_per_s). The window length is
        #     INVENTED inside their "several minutes": LICK_WINDOW_S.
        #   * tongue-flicking at the rate the nose already computes from prey
        #     odour (tongue_flicks_per_min), which is published for this species.
        # Each flick is a short protrusion; its shape (a sharpened sine) and
        # its reach are INVENTED and declared here.
        from common.provenance import parameter_value
        LICK_WINDOW_S = 120.0
        LICK_REACH_M, FLICK_REACH_M = 0.010, 0.006
        t = self._breath_t
        ext = 0.0
        if t < getattr(self, "_lick_until", -1.0):
            rate = float(parameter_value("post_feeding_labial_lick_rate_per_s"))
            ext = max(ext, LICK_REACH_M * max(0.0, math.sin(2.0 * math.pi * rate * t)) ** 3)
        if self.nose is not None:
            per_min = float(self.nose.last.get("tongue_flicks_per_min", 0.0))
            if per_min > 0.0:
                ext = max(ext, FLICK_REACH_M
                          * max(0.0, math.sin(2.0 * math.pi * (per_min / 60.0) * t)) ** 4)
        self.walk_env.tongue_m = float(ext)

        physical = self._mouth_sid >= 0 and self._jaw_ok
        caught = False
        if physical and (self.strike.active or finished):
            gape_deg = self.strike.gape_profile(fraction)
            self.walk_env.jaw_rad = math.radians(gape_deg)
            # The published drop is a head translation; the neck delivers it
            # here as a pitch, capped at the joints' travel.
            drop = self.strike.head_drop_profile(fraction)
            # SET, not accumulate (#296): the profile is cumulative, so adding it
            # every step over-integrated the drop and pinned the head at the
            # pitch stop within a few steps of every launch.
            # THE DROP IS THE GAZE, NOT ON TOP OF IT (#297). At launch the gaze
            # loop has already pitched the head ~35 deg to look at a cricket
            # 17 mm away and 13 mm below the nose, and the joints stop at 45.
            # Adding a 27 mm drop (34 deg on this lever) to that pinned the head
            # at the stop on every strike. The strike now COMMANDS the pitch
            # that delivers the published drop from LEVEL, so looking at the
            # prey and dropping onto it are the same movement -- which is what
            # a jaws-only capture with no slow-open phase describes.
            if self._strike_base_pitch is None:
                self._strike_base_pitch = float(self._gaze_pitch_rad)
            want = drop / max(self._snout_lever_m, 1e-3)
            self._gaze_pitch_rad = float(np.clip(
                max(want, self._strike_base_pitch * (1.0 - fraction)),
                0.0, self.gaze_pitch_max_rad))
            self.walk_env.gaze_pitch_rad = self._gaze_pitch_rad
            # LUNGE. `snap` is head and neck only; `bag_jump` adds the
            # hindlimb push. Both names and the split by distance are from the
            # published ethogram for this species (see Strike.mode_for); which
            # one runs is decided there, not here.
            self.walk_env.lunge = (self.strike.velocity_profile(fraction)
                                   if self.strike.mode == "bag_jump" else 0.0)
            # BLINK. The eyes shut through the strike and open after it. That
            # eublepharids CAN shut their eyes is published -- they are the only
            # geckos with movable eyelids -- but whether a gecko closes them ON
            # a strike is UNRECORDED in every feeding study the session-12 sweep
            # opened, including two Coleonyx papers filmed at 500-1288 fps. This
            # is therefore an INVENTED behaviour on a published organ, and it is
            # the honest place to say so.
            self.walk_env.eyelid_rad = math.radians(
                EYELID_CLOSED_DEG * min(1.0, 2.5 * fraction))
            if gape_deg > 10.0 and self._prey_in_mouth():
                self._strike_touched = True
        if finished:
            if physical:
                hit = bool(self._strike_touched)
                # replace the die's tally with what actually happened
                self.strike.hits += int(hit) - int(dice_hit)
                self.strike.misses += int(not hit) - int(not dice_hit)
                self.walk_env.jaw_rad = 0.0
                self.walk_env.eyelid_rad = 0.0
                self.walk_env.lunge = 0.0
                self._strike_touched = False
                self._strike_base_pitch = None
            else:
                hit = dice_hit
            if hit and not proximity_capture:
                caught = True
                if self.prey is not None:
                    self.prey.captures += 1
                    self.food_xy = self.prey.reset(self._nose_xy(), rng=self._rng)
                    self._write_prey()
        if not self.strike.active:
            self.strike.fire(mouth_distance)
        return bool(caught)

    def _prey_in_mouth(self):
        """Is the prey's centre inside the open mouth? Pure geometry.

        The mouth volume is a capsule from the `mouth` site (the commissure
        region on the mandible) to the `tongue_tip` site, with a radius of the
        prey's own radius plus a small margin. No probability enters.
        """
        d = self.walk_env.data
        a = d.site_xpos[self._mouth_sid]
        b = d.site_xpos[self._tongue_sid]
        p = np.array([self.food_xy[0], self.food_xy[1], float(self.prey.height_m)])
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
        gap = float(np.linalg.norm(p - (a + t * ab)))
        return gap <= float(self.prey.height_m) + 0.004   # height_m IS the prey radius (env sets height_m=radius_m)

    def _brain_action_to_target(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        dir_body = np.asarray(action[:2], dtype=np.float64)
        dir_norm = float(np.linalg.norm(dir_body))
        if dir_norm < 1e-6:
            dir_body = np.array([1.0, 0.0], dtype=np.float64)
        else:
            dir_body = dir_body / dir_norm

        distance = 0.05 + (float(action[2]) + 1.0) * 0.5 * (0.80 - 0.05)
        engage = (float(action[3]) + 1.0) * 0.5
        engage = float(np.clip(engage, 0.0, 1.0))
        # THE 5 cm FLOOR (#285). With the goal measured from the trunk, a stop
        # distance under 5 cm put the goal inside the animal. Measured from the
        # MOUTH that floor is wrong by the whole length of the head: the nose
        # sits ~5 cm ahead of the trunk, so a goal that may never come closer
        # than 5 cm to the trunk can never come closer than ~0 to the nose --
        # and the strike needs 20.3 mm. The floor is kept for trunk goals and
        # dropped for mouth goals.
        _floor = 0.05 if self.walk_env.approach_site == "trunk" else 0.0
        stop_dist = max(float(self.walk_env.reach_dist) + 0.01, _floor)
        target_dist = stop_dist + engage * (distance - stop_dist)

        body_vec = np.array([dir_body[0], dir_body[1], 0.0], dtype=np.float64)
        world_vec = self._trunk_rot() @ body_vec
        world_dir = world_vec[:2]
        world_norm = float(np.linalg.norm(world_dir))
        if world_norm < 1e-6:
            world_dir = np.array([1.0, 0.0], dtype=np.float64)
        else:
            world_dir = world_dir / world_norm

        if self.hunt_targeting and self.prey is not None:
            # The prey's own position, in world coordinates. The direction the
            # brain asked for is ignored while prey exists, because the brain
            # asking for a direction is what could not reach it.
            target = np.asarray(self.food_xy, dtype=np.float64)[:2]
        else:
            target = self._trunk_xy() + target_dist * world_dir
        self._set_walk_target(target)
        self._brain_target_xy = self.walk_env.target.copy()
        return self._brain_target_xy.copy(), engage

    # ------------------------------------------------------------------
    # THE REST OF THE WORLD (#333)
    #
    # Measured before this was written: over 900 autonomous steps, four of the
    # selector's six channels had a gate of EXACTLY 0.0000 on every step. Not
    # losing a competition -- never entered into one. `flee` and `bask` because
    # their inputs did not exist; `rest` because a strolling gecko does not tire
    # (#284, and that one is correct physiology); `groom` because explore's
    # 0.35 x hunger beats its 0.05 tonic on every step the animal is hungry.
    #
    # Everything below is INERT WHEN THE BODY IS ABSENT. `gecko_world_v1.xml`
    # has no shelter, warm patch or threat, so on that world every method here
    # returns None and the animal behaves exactly as it did before -- which is
    # what keeps the gates, the checkpoints and the accepted walker untouched.
    # ------------------------------------------------------------------

    #: Substrate temperature away from the warm patch. DERIVED: Hastings et al.
    #: 2023 cycled the substrate 25 -> 15 -> 25 C and body temperature tracked
    #: it at r2 = 0.97, so 25 C is that protocol's baseline ground rather than a
    #: number chosen here. It sits 4.5 C below the preferred band's floor, which
    #: is the whole point: a gecko emerging onto cooled ground is cold.
    AMBIENT_SUBSTRATE_C = 25.0
    #: How fast body temperature follows the ground it is lying on. INVENTED.
    #: The correlation is published, the time constant is not. Compressed so a
    #: warm-up that takes a real animal tens of minutes takes tens of seconds
    #: here, for the same reason #268 gave the animal meals owed at the start
    #: rather than starving it for three days inside twelve seconds.
    THERMAL_TAU_S = 220.0
    THERMAL_TIME_COMPRESSION = 300.0

    def _world_body_xy(self, name):
        """Ground position of a named world body, or None if it is not there."""
        import mujoco
        model = self.walk_env.model
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return None
        return np.array(self.walk_env.data.xpos[bid][:2], dtype=np.float64)

    def _bearing_to_deg(self, target_xy):
        """Body-relative bearing to a point, degrees, or None."""
        if target_xy is None:
            return None
        delta = np.asarray(target_xy, float) - self._trunk_xy()
        if float(np.linalg.norm(delta)) < 1e-9:
            return 0.0
        world_deg = math.degrees(math.atan2(delta[1], delta[0]))
        return (world_deg - self._trunk_yaw_deg() + 180.0) % 360.0 - 180.0

    def _predator_distance_m(self):
        """Distance to the threat body, or None when the world has none.

        Returning None rather than a large number is deliberate: the nose's
        predator channel then reads zero because there is nothing to smell,
        which is a different statement from the animal being safe.
        """
        threat = self._world_body_xy("threat")
        if threat is None:
            return None
        return float(np.linalg.norm(threat - self._trunk_xy()))

    def _substrate_temperature_C(self):
        """The temperature of the ground under the animal, or None.

        This species is THIGMOTHERMIC: it takes heat from the ground by lying on
        it, not from light. Body temperature tracks substrate at r2 = 0.97
        against 0.92 for air (Hastings et al. 2023, n = 12). So the warm patch
        is a patch of ground, and standing on it is what warms the animal.
        """
        import mujoco
        model = self.walk_env.model
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "warm_patch")
        if bid < 0:
            return None
        centre = np.array(self.walk_env.data.xpos[bid][:2], dtype=np.float64)
        half = 0.05
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == bid:
                half = float(model.geom_size[g][0])
                break
        from common.provenance import parameter_value
        warm = float(parameter_value("warm_surface_temperature_C"))
        on_patch = bool(np.all(np.abs(self._trunk_xy() - centre) <= half))
        return warm if on_patch else self.AMBIENT_SUBSTRATE_C

    def _on_warm_ground(self):
        """Is the animal standing on ground at or above its preferred floor?"""
        substrate = self._substrate_temperature_C()
        if substrate is None or self.homeostasis is None:
            return False
        low, _ = self.homeostasis.physiology.preferred_temperature_C
        return bool(substrate >= low)

    def _step_body_temperature(self, dt_s):
        """Relax body temperature toward the ground it is standing on."""
        if self.homeostasis is None:
            return
        substrate = self._substrate_temperature_C()
        if substrate is None:
            return                      # no thermal field: leave it pinned
        tau = self.THERMAL_TAU_S / max(1e-9, self.THERMAL_TIME_COMPRESSION)
        alpha = 1.0 - math.exp(-float(dt_s) / tau)
        body = float(self.homeostasis.body_temperature_C)
        self.homeostasis.body_temperature_C = body + (substrate - body) * alpha

    def brain_command(self):
        """What the animal's own brain says to do this step. Reads no oracle.

        hunger and fatigue (brain 1) -> basal ganglia (brain 2) -> motor program
        -> heading, head yaw and locomotor drive. The eye's contribution enters
        as a BEARING only, through the fixation evidence, and never as a
        position.
        """
        if self.programs is None:
            raise RuntimeError("brain_command() needs behaviour_control=True")
        behaviour = self.selector.selected()
        urgency = 0.0
        if behaviour is not None:
            gates = self.selector.gates()
            urgency = float(gates[list(self.selector.channels).index(behaviour)])
        # THE EYE'S LAST REPORT, AND NOTHING IF IT SAID NOTHING (#271).
        #
        # The first version fell back to the PREVIOUS bearing whenever the eye
        # was silent, meaning to cover the one-step lag. What it actually did
        # was manufacture evidence: the accumulator received the same number
        # every step, that number agreed with itself perfectly, and the animal
        # committed on it. Measured: the eye fired on 27.1 % of believed steps
        # while the accumulator's own report-rate estimate read 1.0, the
        # adaptive quorum ran to 36, and it still committed 663 times -- every
        # one of them with the prey off screen.
        #
        # Silence is data. A step on which the eye reports nothing must reach
        # the accumulator AS nothing.
        seen = self.eye.last if isinstance(getattr(self.eye, "last", None), dict) else {}
        bearing = (seen.get("prey_bearing_deg")
                   if float(seen.get("prey_salience", 0.0)) > 0.0 else None)
        # WHERE TO FLEE TO AND WHERE TO WARM UP. Both are None on a world that
        # has neither, which is the committed one, and the programs fall back to
        # searching exactly as before. On a furnished world they are real
        # bearings and the flee and bask channels finally have somewhere to go
        # (#267, #333).
        return self.programs.step(
            behaviour, urgency,
            bearing_deg=bearing,
            trunk_yaw_deg=self._trunk_yaw_deg(),
            shelter_bearing_deg=self._bearing_to_deg(
                self._world_body_xy("shelter")),
            warm_bearing_deg=self._bearing_to_deg(
                self._world_body_xy("warm_patch")),
            on_warm_ground=self._on_warm_ground())

    def _trunk_yaw_deg(self):
        q = self.walk_env.data.qpos[3:7]
        return math.degrees(math.atan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))

    def autonomous_step(self):
        """One step with the animal driving itself. No action is passed in.

        Deliberately a SEPARATE entry point from `step(action)`. Every gate,
        checkpoint and trained policy keeps handing this env an action and
        keeps getting the behaviour it was validated with; nothing about that
        path changes whether behaviour_control is on or off.
        """
        cmd = self.brain_command()
        heading = math.radians(float(cmd["heading_deg"]))
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        action[0] = math.cos(heading)
        action[1] = math.sin(heading)
        # The eye reports no range, so the goal is a fixed span ahead and the
        # animal walks ALONG a bearing rather than toward a point.
        action[2] = (0.35 - 0.05) / 0.375 - 1.0
        action[3] = 1.0 if cmd["locomotor_drive"] > 0.0 else -1.0
        self._gaze_yaw_rad = math.radians(float(cmd["head_yaw_deg"]))
        self.walk_env.locomotor_drive = float(cmd["locomotor_drive"] > 0.0)
        out = self.step(action)
        self._last_eye_bearing = out[4].get("prey_bearing_deg")
        out[4]["brain_command"] = cmd
        return out

    def _walker_obs_raw(self) -> np.ndarray:
        return self.walk_env._obs().astype(np.float32)

    def _walker_obs_normalized(self) -> np.ndarray:
        raw = self._walker_obs_raw().reshape(1, -1)
        return self.walk_norm.normalize_obs(raw.copy())

    def _head_cam_image(self) -> np.ndarray:
        if self._head_renderer is None:
            self._head_renderer = mujoco.Renderer(
                self.walk_env.model,
                self.camera_height,
                self.camera_width,
            )
        try:
            self._head_renderer.update_scene(
                self.walk_env.data, camera="head_cam",
                scene_option=self._policy_scene_option,
            )
            # MuJoCo 3.9: shadows are MjvScene render flags, not MjvOption
            # attributes. Never mutate the model's shared light_castshadow.
            scene = _scene_obj(self._head_renderer)
            if scene is None:
                raise RuntimeError("Policy renderer does not expose its scene")
            scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = (
                self.policy_camera_mode == "legacy"
            )
            # With a real prey geom in the world there is nothing to draw: the
            # camera already sees it, in the prey material's own dull brown
            # rather than the bright green marker this used to paint on. Drawing
            # one too would put two food items in front of the animal.
            if self.prey is None:
                _add_scene_sphere(
                    self._head_renderer,
                    np.array([self.food_xy[0], self.food_xy[1], self.food_radius],
                             dtype=np.float64),
                    radius=self.food_radius, rgba=(0.1, 0.95, 0.25, 1.0),
                )
            image = self._head_renderer.render()
        except Exception as exc:
            raise RuntimeError(
                "Failed to render 64x64 RGB image from camera 'head_cam'. "
                "Check that MuJoCo can create a renderer in this process."
            ) from exc

        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(
                f"head_cam render returned invalid image shape {image.shape}; "
                "expected HxWx3 RGB."
            )
        image = image[:, :, :3]
        if image.shape[:2] != (self.camera_height, self.camera_width):
            raise RuntimeError(
                f"head_cam render returned {image.shape[:2]}, expected "
                f"{(self.camera_height, self.camera_width)}."
            )
        return image.astype(np.uint8, copy=False)

    def _privileged_vector(self) -> np.ndarray:
        ego, dist, heading = self._food_delta_body()
        privileged = np.array(
            [
                ego[0],
                ego[1],
                np.clip(dist, 0.0, 2.0),
                np.cos(heading),
                np.sin(heading),
            ],
            dtype=np.float32,
        )
        privileged = privileged * np.float32(self.privileged_target)
        if self.privileged_food_dropout_prob > 0.0 and self._rng.random() < self.privileged_food_dropout_prob:
            return np.zeros(5, dtype=np.float32)
        return privileged

    def _obs(self):
        return {
            "image": self._head_cam_image(),
            "proprio": self._walker_obs_raw(),
            "drives": (self.homeostasis.vector() if self.homeostasis is not None
                       else self.drives.vector()),
            "prev_action": self._prev_action.copy(),
            **({"privileged": self._privileged_vector()}
               if self.privileged_food_channel else {}),
        }

    def reset(self, *, seed: int | None = None, options=None):
        self._gaze_pitch_rad = 0.0
        self._gaze_yaw_rad = 0.0
        self._prev_gaze_yaw_rad = 0.0
        self._last_head_yaw_rate = 0.0
        if getattr(self, "walk_env", None) is not None:
            self.walk_env.gaze_pitch_rad = 0.0
            self.walk_env.gaze_yaw_rad = 0.0
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.walk_env.reset(seed=seed)
        self.drives.reset()
        if self.homeostasis is not None:
            # Energy deliberately survives the episode: a gecko does not become
            # full because a rollout ended.
            self.homeostasis.reset()
            if getattr(self, "meals_owed_at_start", 0.0):
                self.homeostasis.energy_J = float(self._initial_energy_J)
        if self.selector is not None:
            self.selector.reset()
        for _m in (self.programs, self.nose):
            if _m is not None:
                _m.reset()
        if self.clock is not None:
            # The clock resets to the phase the episode was CONFIGURED for, not
            # to zero. `Arousal.reset()` defaults to time_of_day_s=0.0, and the
            # bare `_m.reset()` in the loop above used to send it there --
            # harmless only because nothing read the result (#347).
            self.clock.reset(self._time_of_day_s)
            self._arousal = float(self.clock.arousal_at(self._time_of_day_s))
        self._eyes_were_shut = False
        if self.eye is not None:
            self.eye.reset()
        self._step = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._smooth_lookat = None
        self._static_anchor = None
        self._static_azimuth = None
        self._smooth_azimuth = None
        self._spawn_food()
        self._brain_action_to_target(np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32))
        self._last_info = {
            "food_dist": self._food_distance(),
            "mouth_food_dist": self._mouth_food_distance(),
            "ate": False,
            "hunger": self.drives.hunger,
            "energy": self.drives.energy,
            "fear": self.drives.fear,
            "danger": self.drives.danger,
            "engage": 0.0,
            "walker_forward_speed": 0.0,
            "belly_contact": 0.0,
            "fallen": False,
            "brain_target_xy": self._brain_target_xy.copy(),
            "food_xy": self.food_xy.copy(),
        }
        return self._obs(), dict(self._last_info)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        food_dist_before = self._food_distance()
        mouth_dist_before = self._mouth_food_distance()
        trunk_before = self._trunk_xy()
        _, engage = self._brain_action_to_target(action)

        walker_terminated = False
        walker_truncated = False
        last_walker_info = {}
        steps_run = 0
        for _ in range(self.brain_steps_per_action):
            self._set_walk_target(self._brain_target_xy)
            if self.walk_model is None:
                # ZERO RESIDUAL. The hand-written CPG base is the whole
                # controller -- which is the configuration the 4/6 gates
                # accepted, and the one this env could not previously express.
                walker_action = np.zeros(
                    self.walk_env.action_space.shape, dtype=np.float32)
            else:
                norm_obs = self._walker_obs_normalized()
                walker_action, _ = self.walk_model.predict(norm_obs, deterministic=True)
                walker_action = np.asarray(walker_action, dtype=np.float32).reshape(-1)
            _, _, terminated, truncated, info = self.walk_env.step(walker_action)
            last_walker_info = info
            steps_run += 1
            walker_terminated = walker_terminated or bool(terminated)
            walker_truncated = walker_truncated or bool(truncated)
            if walker_terminated or walker_truncated:
                break

        self._set_walk_target(self._brain_target_xy)
        total_dt = max(steps_run, 1) * float(self.walk_env.dt)
        prey_captured = False
        if self.prey is not None:
            # The predator the prey reacts to is the mouth, not the body centre,
            # and the prey's own capture distance is the eat radius, so there is
            # one definition of "close enough to eat" rather than two.
            self.food_xy, prey_captured = self.prey.step(total_dt, self._nose_xy())
            self._write_prey()
        if self.strike is not None:
            # Proximity no longer feeds the animal. It has to strike, and the
            # strike has to land.
            prey_captured = self._run_strike(total_dt, prey_captured)
        food_dist_after = self._food_distance()
        mouth_dist_after = self._mouth_food_distance()
        ate = bool(prey_captured) if self.prey is not None else bool(mouth_dist_after <= self.eat_radius)
        fallen = bool(walker_terminated or last_walker_info.get("fallen", 0.0) > 0.5)
        belly_contact = float(last_walker_info.get("belly_contact", 0.0))
        danger = float(np.clip(0.65 * belly_contact + (1.0 if fallen else 0.0), 0.0, 1.0))
        moving_speed = float(np.linalg.norm(self._trunk_xy() - trunk_before) / total_dt)
        moving_drive = float(np.clip(moving_speed / 0.25, 0.0, 1.0))

        if ate:
            self._lick_until = self._breath_t + 120.0   # LICK_WINDOW_S, see the tongue drive
        self.drives.update(total_dt, ate=ate, danger=danger, moving=moving_drive)
        # BRAIN 6 ADVANCES HERE, before the drive vector is read, for the same
        # reason body temperature does: a selector reading last step's arousal
        # is a step behind the time of day it is meant to be deciding in.
        if self.clock is not None:
            self._arousal = float(
                self.clock.step(total_dt * self._time_compression)["arousal"])
        if self.homeostasis is not None:
            from common.provenance import parameter_value
            # Activity cost is the published cost of transport at the speed the
            # body actually moved, not a guess about effort.
            # The ground first, then the rest. Body temperature has to move
            # BEFORE the drive vector is read, or the thermal error the selector
            # sees is a step behind the ground the animal is standing on.
            self._step_body_temperature(total_dt)
            self.homeostasis_reward = self.homeostasis.step(
                total_dt,
                meal_wet_mass_kg=(float(parameter_value("prey_item_wet_mass_kg")) if ate else 0.0),
                activity_power_W=self.homeostasis.physiology.locomotion_power_W(moving_speed),
                speed_m_s=moving_speed,
            )

        progress = mouth_dist_before - mouth_dist_after
        r_progress = 12.0 * progress
        r_eat = 10.0 if ate else 0.0
        r_close = 0.5 * max(0.0, 1.0 - mouth_dist_after / 0.20) if ate else 0.0
        r_time = -0.01 * max(steps_run, 1)
        r_danger = -0.35 * danger + (-2.0 if fallen else 0.0)

        if ate and self.prey is None:
            self._spawn_food()
            food_dist_after = self._food_distance()
            mouth_dist_after = self._mouth_food_distance()
        elif ate:
            # FleeingPrey respawns itself on capture; food_xy already followed it.
            food_dist_after = self._food_distance()
            mouth_dist_after = self._mouth_food_distance()

        self._step += 1
        self._prev_action = action.copy()
        terminated = bool(fallen)
        truncated = bool((self._step >= self.max_steps or walker_truncated) and not terminated)

        obs = self._obs()
        food_visible_frac = _food_visible_frac(obs["image"], self._food_rgb)
        prey_bearing_deg = None
        if self.eye is not None:
            # The eye supersedes the colour matcher. Its salience is already
            # in [0, 1] and is what the selector consumes; the colour figure
            # stays in info so the two can be compared on the same frames.
            # EFFERENCE COPY. The eye is handed what the animal's own motor
            # system is doing, so it can subtract the flow its own walking
            # explains. Without it the tectum reported the same salience in a
            # world with prey and a world with none -- d = 0.036.
            # THE CAMERA IS ON THE HEAD, AND SO NOW IS THE ORGAN THAT
            # PREDICTS WHAT THE CAMERA WILL SEE (#247).
            #
            # Until session 12 this read `gyro_trunk` and `vel_trunk` -- the
            # animal predicted its own visual flow from sensors mounted above
            # its hips, while the eye rode on a head that yaws +-70 deg
            # relative to that trunk and bobs with every step. Measured: the
            # head's own yaw rate is 3 to 4 times the trunk's even walking
            # straight (median 0.37 deg/s against 0.13, 95th percentile 2.75
            # against 0.73). The prediction was systematically too small, the
            # residual was read as the world moving, and 30.4 % of frames
            # produced a false alarm with the animal standing still.
            #
            # The fix is anatomy, not a gain. The semicircular canals of every
            # vertebrate sit in the SKULL, beside the eye. `gyro_head` and
            # `vel_head` are mounted on the site the camera already occupies.
            # This is not privileged information -- a gecko has a vestibular
            # system, and knowing that your own head is turning is what it is
            # for.
            #
            # Both yaw and pitch of the head translate the whole image, so the
            # rate fed to the uniform term is the magnitude of the two
            # together rather than yaw alone.
            g, v = self.walk_env.head_velocity()
            self._last_head_yaw_rate = float(np.degrees(g[2]))
            self_motion = {
                "yaw_rate_deg_s": float(np.degrees(
                    math.hypot(float(g[1]), float(g[2])))),
                "forward_m_s": float(v[0]),
                # The animal's own gaze command. Not a world reading: a gecko
                # knows where it is pointing its head. Lets the elevation rule
                # test the WORLD horizon rather than the frame's.
                "gaze_pitch_deg": float(math.degrees(self._gaze_pitch_rad)),
            }
            seen = self.eye.step(obs["image"], total_dt, self_motion=self_motion)
            food_visible_frac = float(seen["prey_salience"])
            # None when the eye reported nothing (#271). Kept numeric here so
            # the info field and every existing reader keep the shape they had,
            # and guarded everywhere the VALUE is acted on.
            _b = seen.get("prey_bearing_deg")
            # None when the eye reported nothing (#271), which is what the
            # non-eye colour-match path already reports and what every existing
            # reader already guards for with `food_visible_frac > 0`.
            prey_bearing_deg = float(_b) if _b is not None else None
            # GAZE. The eye aims its own head. When the tectum reports a target
            # low in the frame, the neck and head pitch down to bring it back
            # toward the axis -- a foveation loop that consumes ONLY the eye's
            # own output and never the prey's true position.
            #
            # WHY. Measured over 2400 steps of a real hunt, the prey's
            # elevation falls from -4 deg at 0.3-0.5 m to -35 deg inside 4 cm,
            # and at fovy 70 the frame stops at -35: the fraction of frames
            # with the prey on the image collapses from 100 % at 4-8 cm to
            # 45 % inside 4 cm. The animal goes blind exactly where the strike
            # has to fire, and it closed to 21.0 mm against a 20.3 mm trigger.
            #
            # The gain is INVENTED. The behaviour is not: Delheusy et al. 1995
            # measured this species' head translating ~28 mm vertically through
            # a capture.
            if self.gaze_pitch_gain:
                el = seen.get("prey_elevation_deg")
                if el is not None and el < 0.0:
                    want = math.radians(-float(el)) * float(self.gaze_pitch_gain)
                    self._gaze_pitch_rad += self.gaze_pitch_rate * (
                        want - self._gaze_pitch_rad)
                else:
                    # NOTHING REPORTED. The first version relaxed toward level
                    # here, reasoning that a lost target should not leave the
                    # animal staring at the ground. Measured: that is wrong and
                    # it defeats the loop. Inside 4 cm the eye reports on ~1 %
                    # of frames, so 99 % of steps were un-aiming the head, and
                    # the gaze never exceeded 9.1 deg against a target sitting
                    # at -35. The loop was starved by the blindness it exists
                    # to remove.
                    #
                    # A target lost LOW in the frame is evidence the head is
                    # aimed too high, not evidence there is no target. So the
                    # aim is HELD by default (decay 0.0) and the decay is a
                    # declared, INVENTED parameter rather than an assumption
                    # baked into the branch.
                    self._gaze_pitch_rad *= (1.0 - self.gaze_pitch_decay)
                self.walk_env.gaze_pitch_rad = float(
                    np.clip(self._gaze_pitch_rad, 0.0, self.gaze_pitch_max_rad))
        # Head yaw is commanded straight through by the caller -- there is no
        # closed loop on it, because the thing that would close it (a search
        # pattern) lives in brain/search.py and not in the environment.
        self.walk_env.gaze_yaw_rad = float(self._gaze_yaw_rad)
        # The colour matcher reports an AREA FRACTION, so it is rescaled: a
        # 3 px prey covers under 1% of the frame. The eye reports a salience
        # that is already in [0, 1], and putting it through the same divisor
        # would saturate it to 1.0 for anything visible at all.
        food_visible_signal = (float(food_visible_frac) if self.eye is not None
                               else min(food_visible_frac / 0.012, 1.0))

        # Brain 1 -> brain 2, on live data. Threat is the same danger signal the
        # reward uses; prey_visible is what the retina-substitute actually
        # reports, so the selector sees the world the animal sees.
        # SMELL FEEDS THREAT (#266). For this species defence is gated by
        # chemoreception, not by sight, so `danger` -- which is a geometric
        # proximity signal -- is combined with what the nose reports rather
        # than used alone. With smell off this is exactly the old value.
        if self.nose is not None:
            self.nose.step(prey_distance_m=float(food_dist_after),
                           predator_distance_m=self._predator_distance_m())
            # THE VISUAL TERM IS OFF, AND IT WAS WIRED TO THE WRONG SENSE
            # (#355). This read `predator_visible=bool(danger > 0.0)`, and
            # `danger` is `0.65 * belly_contact + fallen` -- the animal's belly
            # touching the ground, or it having fallen over. That is a
            # MECHANOSENSORY and postural state, not seeing anything. So the
            # model was adding Frydlova et al. 2026's published VISUAL reaction
            # probability of 0.07 on a mechanosensory trigger, and the same
            # factorial measured the mechanosensory term at chi2 < 0.01,
            # p > 0.9, verbatim: "neither independently elicited overt
            # defensive behaviour". The sum it produced -- 0.28 -- appears
            # nowhere in `config/proxies.yaml` and was reported in the ledger as
            # though the paper printed it.
            #
            # The honest reading of that paper is the one this map's own
            # "currently blocked" table already had: smell 0.21, sight 0.
            # An animal that could genuinely SEE a predator would be a
            # different argument, but this one cannot -- prey-finding is NOT
            # ACCEPTED and the threat body is not detected by the eye at all.
            # So flee's ceiling is the one published number, 0.21.
            danger = max(danger, float(self.nose.defensive_probability(
                self.nose.last["predator_odour"],
                predator_visible=False)))

        # PREY_VISIBLE IS A DECISION, NOT A PIXEL COUNT (#265). The hunt channel
        # is hunger x prey_visible, so feeding it a raw salience makes the animal
        # lunge at renderer noise -- which is measurably what the eye reports on
        # two frames in three at the shipped threshold (#254). When the full loop
        # is running, the signal is the eye's own COMMITTED evidence: several
        # agreeing looks taken while the head was still. Otherwise it is the old
        # value, unchanged.
        prey_visible_signal = float(food_visible_signal)
        if self.programs is not None:
            prey_visible_signal = (
                1.0 if self.evidence.last.get("committed_bearing_deg") is not None
                else 0.0)
        self._prey_visible_signal = prey_visible_signal

        if self.selector is not None:
            # BRAIN 6 -> BRAIN 2. With no clock this is 1.0, the rest channel's
            # salience is 0.0, and every other channel is untouched.
            self.selector.step_from_homeostasis(
                self.homeostasis.vector(),
                threat=danger,
                prey_visible=prey_visible_signal,
                arousal=self._arousal)

            # A SLEEPING GECKO SHUTS ITS EYES (#356), and this is the one part
            # of resting that is published rather than inferred. Bergel et al.
            # 2026, Nat Neurosci 29:543-550, recorded sleep in seven lizard
            # species with EOG electrodes placed UNDER EACH EYELID of this
            # animal -- n = 2 E. macularius -- which is possible because
            # eublepharids are the only geckos with movable lids and this one
            # closes them to sleep, where a tokay cannot. The same paper is why
            # `sleep_cycle_period_s` is null: it establishes that this animal
            # sleeps without giving a cycle period anyone can defend (#198).
            #
            # IT FOLLOWS THE CLOCK, NOT THE REST GATE, and the first version of
            # this did the second (#358). Measured over 300 light-phase steps:
            # rest releases cleanly at gate 1.0 for about eighteen steps, then
            # the animal lying still on cold ground drives BASK's salience up to
            # 1.0 as well, and with two channels tied at maximum salience the
            # published model part-releases both at 0.2210 and 0.2209. That is
            # the authors' "dual" outcome and the network settles on 100 % of
            # steps -- it is not an oscillation -- but `argmax` turns the tie
            # into a coin flip, so a lid driven from the rest gate FLUTTERS at a
            # 16.7 deg mean instead of closing. Sleep is a state of the clock,
            # not the outcome of a competition, and no source says an animal
            # shuts its eyes because one channel out-competed another.
            #
            # The ANGLE is the already-declared INVENTED `EYELID_CLOSED_DEG`,
            # reused rather than added to -- no lid excursion has ever been
            # published for this species or any lizard. The CONJUNCTION with
            # locomotion is also INVENTED and is the minimum needed to stop the
            # animal sleepwalking: a gecko whose legs are running has its eyes
            # open, so anything that moves the body opens the lids whatever the
            # hour. Skipped while a strike is in flight, because the strike owns
            # the lids during its own blink and runs earlier in this same step.
            if self.strike is None or not self.strike.active:
                asleep = (1.0 - float(np.clip(self._arousal, 0.0, 1.0))
                          if float(self.walk_env.locomotor_drive) <= 0.0
                          else 0.0)
                if asleep > 0.0:
                    self.walk_env.eyelid_rad = math.radians(
                        EYELID_CLOSED_DEG * asleep)
                elif self._eyes_were_shut:
                    self.walk_env.eyelid_rad = 0.0
                self._eyes_were_shut = asleep > 0.0
        reward = r_progress + r_eat + r_close + r_time + r_danger

        info = {
            "food_dist": float(food_dist_after),
            "mouth_food_dist": float(mouth_dist_after),
            "ate": ate,
            "hunger": float(self.drives.hunger),
            "energy": float(self.drives.energy),
            "fear": float(self.drives.fear),
            "danger": float(danger),
            "engage": float(engage),
            "walker_forward_speed": float(last_walker_info.get("forward_speed", 0.0)),
            "belly_contact": float(belly_contact),
            "fallen": bool(fallen),
            "brain_target_xy": self._brain_target_xy.copy(),
            "food_xy": self.food_xy.copy(),
            "progress": float(progress),
            "moving_speed": float(moving_speed),
            "food_visible_frac": float(food_visible_frac),
            "prey_bearing_deg": prey_bearing_deg,
            "vision_source": "eye" if self.eye is not None else "colour_match",
            "gaze_pitch_deg": float(math.degrees(self._gaze_pitch_rad)),
            "gaze_pitch_gain": float(self.gaze_pitch_gain),
            "gaze_yaw_deg": float(math.degrees(self._gaze_yaw_rad)),
            "head_yaw_rate_deg_s": float(self._last_head_yaw_rate),
            # Reported every step so no run can later be described as
            # oracle-free without the record contradicting it.
            "walker_oracle": self.walker_oracle,
            "gait_profile": self.gait_profile,
            "hunt_targeting": self.hunt_targeting,
            "use_policy": self.use_policy,
            "accepted_walker": (not self.use_policy) and self.gait_profile == "lab",
            "gate_validated_profile": self.gait_profile == "lab",
            "strike_active": bool(self.strike.active) if self.strike else False,
            "strike_mode": self.strike.mode if self.strike else None,
            "strikes": self.strike.strikes if self.strike else 0,
            "strike_hits": self.strike.hits if self.strike else 0,
            "capture_requires_a_strike": self.strike is not None,
            "food_oracle_scale": float(self.privileged_target),
            "food_visible_signal": float(food_visible_signal),
            "food_radius": float(self.food_radius),
            "reward_progress": float(r_progress),
            "reward_eat_bonus": float(r_eat),
            "reward_close_bonus": float(r_close),
            "reward_time_penalty": float(r_time),
            "reward_danger_penalty": float(r_danger),
        }
        if self.selector is not None:
            # Reported, not obeyed. `behaviour` is what the animal WOULD do.
            info["behaviour"] = self.selector.selected()
            info["behaviour_gates"] = self.selector.gates().copy()
            info["behaviour_committed"] = bool(self.selector.committed())
            # True until session 12, and the field is kept rather than deleted
            # so that any reader of an older record can see when it changed.
            info["behaviour_controls_nothing"] = self.programs is None
            info["prey_visible_signal"] = float(
                getattr(self, "_prey_visible_signal", 0.0))
        if self.programs is not None:
            info["program"] = self.programs.last.get("program")
            info["locomotor_drive"] = float(self.walk_env.locomotor_drive)
            info["committed_bearing_deg"] = self.programs.last.get(
                "committed_bearing_deg")
        if self.nose is not None:
            info["smell"] = self.nose.state()
        if self.clock is not None:
            # This now reports what the clock produced. Until #352 it reported
            # the constant 1.0 assigned at construction, for a module whose
            # `step` was never called.
            info["arousal"] = float(self._arousal)
            info["clock"] = dict(self.clock.last)
        self._last_info = dict(info)
        return obs, float(reward), terminated, truncated, info

    def _wide_camera_lookat(self, lookahead: float = 0.8) -> np.ndarray:
        trunk = self.walk_env.data.xpos[self.walk_env._trunk].copy()
        lookat = trunk.copy()
        delta = self._brain_target_xy - trunk[:2]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-9:
            lookat[:2] += delta / norm * float(lookahead)
            return lookat
        forward = self._trunk_rot()[:, 0]
        lookat[:2] += forward[:2] * float(lookahead)
        return lookat

    def _add_render_markers(self, renderer) -> None:
        food_xyz = np.array(
            [self.food_xy[0], self.food_xy[1], self.food_radius], dtype=np.float64
        )
        if self.prey is None:
            _add_scene_sphere(renderer, food_xyz, radius=self.food_radius,
                              rgba=(0.1, 0.95, 0.25, 1.0))
        if not self.show_debug_markers:
            return

        trunk = self.walk_env.data.xpos[self.walk_env._trunk].copy()
        trunk_ground = np.array([trunk[0], trunk[1], 0.045], dtype=np.float64)
        target_xyz = np.array(
            [self._brain_target_xy[0], self._brain_target_xy[1], 0.045], dtype=np.float64
        )
        _add_scene_sphere(renderer, target_xyz, radius=0.022,
                          rgba=(1.0, 0.15, 0.05, 0.9))
        _add_scene_capsule(renderer, trunk_ground, target_xyz,
                           rgba=(1.0, 0.65, 0.05, 0.9))
        _add_scene_capsule(renderer, target_xyz, food_xyz, radius=0.003,
                           rgba=(0.1, 0.8, 1.0, 0.55))

    def _render_camera_params(self):
        """Return (lookat, azimuth, distance, elevation) for the current view mode."""
        trunk = self.walk_env.data.xpos[self.walk_env._trunk].copy()
        rot = self._trunk_rot()
        forward = rot[:, 0]
        heading_deg = math.degrees(math.atan2(forward[1], forward[0]))

        if self._view_mode == "fixed":
            distance = 2.0
            height = 0.70
            target_azimuth = 130.0
            target_lookat = np.array([trunk[0], trunk[1], 0.05], dtype=np.float64)
        elif self._view_mode == "chase":
            distance = 2.2
            height = 0.70
            # camera sits directly behind: heading_deg + 180 in MuJoCo azimuth space
            target_azimuth = heading_deg + 180.0
            target_lookat = np.array([trunk[0], trunk[1], 0.05], dtype=np.float64)
        elif self._view_mode == "track":
            # FOLLOWS POSITION, NEVER ROTATES. This is the one that looks right,
            # and working out why corrected a mistake of mine: I blamed the
            # sliding ground for the shake in the earlier clips. It was not the
            # ground. `close` and `chase` set azimuth = heading + 180, so the
            # camera SWINGS every time the gecko turns, and a walking gecko
            # yaws constantly. Holding the azimuth fixed while tracking the
            # animal removes the shake completely and keeps it in frame -- the
            # ground slides past exactly as it should when something walks.
            if self._static_azimuth is None:
                self._static_azimuth = heading_deg + 128.0
            distance = 0.62
            height = 0.30
            target_azimuth = self._static_azimuth
            target_lookat = np.array([trunk[0], trunk[1], 0.03], dtype=np.float64)
        elif self._view_mode == "static":
            # A camera bolted to the world. Every other mode tracks the trunk,
            # so the animal stays centred and the GROUND slides -- which reads
            # as camera shake and makes it impossible to tell whether the gecko
            # moved or the view did. This one does not move at all. It is
            # anchored where the episode began rather than at the origin, so it
            # frames wherever the action actually is.
            if self._static_anchor is None:
                # Anchored ONCE, from the animal's heading at that moment, then
                # never touched again. A hard-coded azimuth frames the gecko
                # differently depending on which way it happened to start, and
                # put the prey behind it as often as in front. Computing it
                # once keeps the camera genuinely fixed while still guaranteeing
                # a side view with the head -- and therefore the strike -- facing
                # into frame. Slightly ahead of the trunk, because the mouth is
                # what this is filming.
                self._static_anchor = np.array(
                    [trunk[0] + forward[0] * 0.05, trunk[1] + forward[1] * 0.05, 0.02])
                self._static_azimuth = heading_deg + 90.0
            distance = 0.34
            height = 0.115
            target_azimuth = self._static_azimuth
            target_lookat = self._static_anchor.copy()
        elif self._view_mode == "hunt":
            # Tight enough to see a 9 mm cricket and an 80 ms strike. The
            # existing "close" view sits 1.5 m back, which renders the whole
            # animal about twenty pixels tall and the prey as a single dot --
            # fine for watching gait, useless for watching a strike.
            distance = 0.34
            height = 0.13
            target_azimuth = heading_deg + 205.0
            lookahead = forward[:2] * 0.07
            target_lookat = np.array(
                [trunk[0] + lookahead[0], trunk[1] + lookahead[1], 0.02],
                dtype=np.float64)
        else:  # "close" — closer 3/4-rear view
            distance = 1.5
            height = 0.50
            target_azimuth = heading_deg + 180.0
            # slight forward bias so the gecko stays in lower-half of frame
            lookahead = forward[:2] * 0.15
            target_lookat = np.array(
                [trunk[0] + lookahead[0], trunk[1] + lookahead[1], 0.05],
                dtype=np.float64,
            )

        alpha = 0.0 if self._view_mode in ('static', 'track') else self._camera_smoothing
        if alpha > 0.0 and self._smooth_lookat is not None and self._smooth_azimuth is not None:
            lookat = alpha * self._smooth_lookat + (1.0 - alpha) * target_lookat
            # wrap azimuth delta to [-180, 180] to avoid spinning through 360
            delta_az = (target_azimuth - self._smooth_azimuth + 180.0) % 360.0 - 180.0
            azimuth = self._smooth_azimuth + (1.0 - alpha) * delta_az
        else:
            lookat = target_lookat.copy()
            azimuth = target_azimuth

        self._smooth_lookat = lookat.copy()
        self._smooth_azimuth = float(azimuth)

        elevation = -math.degrees(math.asin(min(height / distance, 0.9999)))
        return lookat, float(azimuth), float(distance), float(elevation)

    def render(self):
        if self._render_renderer is None:
            self._render_renderer = mujoco.Renderer(self.walk_env.model, 480, 640)
        lookat, azimuth, distance, elevation = self._render_camera_params()
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        self._render_renderer.update_scene(
            self.walk_env.data, camera=cam, scene_option=self._render_scene_option
        )
        _scene_obj(self._render_renderer).flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        self._add_render_markers(self._render_renderer)
        return self._render_renderer.render()

    def close(self):
        if self._head_renderer is not None:
            self._head_renderer.close()
            self._head_renderer = None
        if self._render_renderer is not None:
            self._render_renderer.close()
            self._render_renderer = None
        if getattr(self, "walk_norm", None) is not None:
            self.walk_norm.close()
        elif getattr(self, "walk_env", None) is not None:
            self.walk_env.close()
