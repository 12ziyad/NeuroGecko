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
        eye: bool = False,
        eye_render_pixels: int | None = None,
        eye_receptor_pixels: int | None = None,
    ):
        super().__init__()
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
            **({"approach_site": "mouth",
                "reach_dist": float(parameter_value("strike_trigger_distance_m"))}
               if hunt_targeting else {}),
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

        self.homeostasis = None
        if homeostasis:
            from brain.hypothalamus import Homeostasis, Physiology
            mass = float(np.sum(self.walk_env.model.body_mass[1:]))
            self.homeostasis = Homeostasis(
                Physiology.from_registry(body_mass_kg=mass),
                time_compression=float(homeostasis_time_compression),
            )
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
        _, finished, hit = self.strike.step(dt)
        caught = False
        if finished and hit and not proximity_capture:
            caught = True
            if self.prey is not None:
                self.prey.captures += 1
                self.food_xy = self.prey.reset(self._nose_xy(), rng=self._rng)
                self._write_prey()
        if not self.strike.active:
            self.strike.fire(mouth_distance)
        return bool(caught)

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
        stop_dist = max(float(self.walk_env.reach_dist) + 0.01, 0.05)
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
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.walk_env.reset(seed=seed)
        self.drives.reset()
        if self.homeostasis is not None:
            # Energy deliberately survives the episode: a gecko does not become
            # full because a rollout ended.
            self.homeostasis.reset()
        if self.selector is not None:
            self.selector.reset()
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

        self.drives.update(total_dt, ate=ate, danger=danger, moving=moving_drive)
        if self.homeostasis is not None:
            from common.provenance import parameter_value
            # Activity cost is the published cost of transport at the speed the
            # body actually moved, not a guess about effort.
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
            gyro = self.walk_env.data.sensor("gyro_trunk").data                 if hasattr(self.walk_env.data, "sensor") else None
            vel = self.walk_env._s("vel_trunk")
            self_motion = {
                "yaw_rate_deg_s": float(np.degrees(self.walk_env._s("gyro_trunk")[2])),
                "forward_m_s": float(vel[0]),
            }
            seen = self.eye.step(obs["image"], total_dt, self_motion=self_motion)
            food_visible_frac = float(seen["prey_salience"])
            prey_bearing_deg = float(seen["prey_bearing_deg"])
        # The colour matcher reports an AREA FRACTION, so it is rescaled: a
        # 3 px prey covers under 1% of the frame. The eye reports a salience
        # that is already in [0, 1], and putting it through the same divisor
        # would saturate it to 1.0 for anything visible at all.
        food_visible_signal = (float(food_visible_frac) if self.eye is not None
                               else min(food_visible_frac / 0.012, 1.0))

        # Brain 1 -> brain 2, on live data. Threat is the same danger signal the
        # reward uses; prey_visible is what the retina-substitute actually
        # reports, so the selector sees the world the animal sees.
        if self.selector is not None:
            self.selector.step_from_homeostasis(
                self.homeostasis.vector(),
                threat=danger,
                prey_visible=float(food_visible_signal))
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
            info["behaviour_controls_nothing"] = True
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
