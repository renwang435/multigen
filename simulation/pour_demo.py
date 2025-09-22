"""This script is used to test the static scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import math
import os
from typing import Literal

import imageio.v2 as iio
import numpy as np
import rootutils
import torch
import tyro
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolverConfig
from loguru import logger as log
from rich.logging import RichHandler
from torchvision.utils import make_grid

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.cfg.objects import FluidObjCfg, PrimitiveCubeCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import TensorState, state_tensor_to_nested


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaaclab"
    )
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


class ObsSaver:
    """Save the observations to a video."""

    def __init__(self, video_path: str | None = None):
        self.video_path = video_path
        self.images: list[np.array] = []

        self.image_idx = 0

    def add(self, state: TensorState):
        """Add the observation to the video."""
        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        """Save the video."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=20)


import datetime
import random

# random_color = lambda: (random.uniform(0.6, 0.99), random.uniform(0.2, 0.7), random.uniform(0.1, 0.4))
random_color_ = (random.uniform(0.6, 0.99), random.uniform(0.2, 0.7), random.uniform(0.1, 0.4))
# breakpoint()
random_water_height = random.randint(5, 25)
path_name = f"tmp_{random_color_}_{random_water_height}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
obs_saver = ObsSaver(video_path=f"outputs/water/{path_name}.mp4")
meta_data = {
    "random_color": random_color_,
    "random_water_height": random_water_height,
}
STEP_NOW = 0

args = tyro.cli(Args)
scenario = ScenarioCfg(
    robot=args.robot,
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

scenario.objects = [
    PrimitiveCubeCfg(
        name="table",
        size=(0.7366, 1.4732, 0.0254),
        color=random_color_,
        physics=PhysicStateType.GEOM,
    ),
    RigidObjCfg(
        name="cup1",
        usd_path="metasim/data/pouring/Tall_Glass_5.usd",
        physics=PhysicStateType.RIGIDBODY,
        scale=0.008,
        default_position=(0.4, 0.3, 0.6943 + 0.0127),
    ),
    FluidObjCfg(
        name="water",
        numParticlesX=10,
        numParticlesY=10,
        numParticlesZ=random_water_height,
        density=0.0,
        particle_mass=0.0001,
        particleSpacing=0.004,
        viscosity=0.1,
        default_position=(0.4, 0.3, 0.6943 + 0.0127 + 0.03),
    ),
    RigidObjCfg(
        name="cup2",
        usd_path="metasim/data/pouring/Tall_Glass_5.usd",
        physics=PhysicStateType.RIGIDBODY,
        scale=0.008,
        default_position=(0.42, 0.15, 0.6943 + 0.0127),
    ),
    # PrimitiveFrameCfg(name="frame", scale=0.1, base_link=("kinova_gen3_robotiq_2f85", "end_effector_link")),
]
scenario.cameras = [
    PinholeCameraCfg(
        name="rgb",
        width=1920,
        height=1080,
        pos=(0.0, 0.05, 1.69),
        look_at=(0.35, 0.05, 0.707),
    )
]

env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "table": {
                "pos": torch.tensor([0.3683, 0.1234, 0.6943]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "wall": {
                "pos": torch.tensor([0.7616, 0.1234, 1.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "cup1": {},
            "cup2": {},
        },
        "robots": {
            "kinova_gen3_robotiq_2f85": {
                "pos": torch.tensor([-0.05, 0.05, 1.6891]),
                "rot": torch.tensor([0.2706, -0.65328, -0.65328, -0.2706]),
                "dof_pos": {
                    "joint_1": -26 / 180 * math.pi,
                    "joint_2": -41 / 180 * math.pi,
                    "joint_3": -77 / 180 * math.pi,
                    "joint_4": 52 / 180 * math.pi,
                    "joint_5": 9.3 / 180 * math.pi,
                    "joint_6": 55 / 180 * math.pi,
                    "joint_7": 0.0,
                    "finger_joint": 0.0,
                },
            },
        },
    }
    for _ in range(args.num_envs)
]


robot = scenario.robot

tensor_args = TensorDeviceType()
config_file = load_yaml(join_path(get_robot_path(), robot.curobo_ref_cfg_name))["robot_cfg"]
curobo_robot_cfg = RobotConfig.from_dict(config_file, tensor_args)
world_cfg = WorldConfig(
    cuboid=[
        Cuboid(
            name="ground",
            pose=[0.0, 0.0, -0.4, 1, 0.0, 0.0, 0.0],
            dims=[10.0, 10.0, 0.8],
        )
    ]
)
ik_config = IKSolverConfig.load_from_robot_config(
    curobo_robot_cfg,
    world_cfg,
    rotation_threshold=0.05,
    position_threshold=0.005,
    num_seeds=20,
    self_collision_check=True,
    self_collision_opt=True,
    tensor_args=tensor_args,
    use_cuda_graph=True,
)

*_, robot_ik = get_curobo_models(robot)
curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
ee_n_dof = len(robot.gripper_release_q)

env.reset(states=init_states)


def reach_target_try(ee_pos: torch.Tensor, ee_quat: torch.Tensor, decimation: int = 3):
    """Reach the target position and orientation."""
    states = env.handler.get_states()
    ee_joint_idx = env.handler.get_joint_names(robot.name).index("finger_joint")
    ee_q = states.robots[robot.name].joint_pos[:, ee_joint_idx]
    curr_robot_q = states.robots[robot.name].joint_pos.cuda()
    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])
    ee_pos_target_global = torch.tensor(ee_pos, device="cuda").repeat(args.num_envs, 1)
    ee_quat_target_global = torch.tensor(ee_quat, device="cuda").repeat(args.num_envs, 1)
    robot_base_pos = states.robots[robot.name].root_state[:, :3].cuda()
    robot_base_quat = states.robots[robot.name].root_state[:, 3:7].cuda()
    ee_pos_target_local = quat_apply(
        quat_inv(robot_base_quat),
        ee_pos_target_global - robot_base_pos,
    )
    ee_quat_target_local = quat_mul(quat_inv(robot_base_quat), ee_quat_target_global)
    result = robot_ik.solve_batch(Pose(ee_pos_target_local, ee_quat_target_local), seed_config=seed_config)

    q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda")
    ik_succ = result.success.squeeze(1)
    q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    actions = [
        {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))} for i_env in range(scenario.num_envs)
    ]
    actions[0]["dof_pos_target"]["finger_joint"] = ee_q[0]

    for i_step in range(decimation):
        obs, _, _, _, _ = env.step(actions)
        global STEP_NOW
        STEP_NOW += 1
        obs_saver.add(obs)


def reach_target_dedicated(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    quat_atol: float = 0.03,
    pos_atol: float = 0.008,
):
    """Reach the target position and orientation."""
    states = env.handler.get_states()
    ee_idx = states.robots[robot.name].body_names.index(env.handler.robot.ee_body_name)
    cur_ee_pos = states.robots[robot.name].body_state[:, ee_idx, :3]
    cur_ee_quat = states.robots[robot.name].body_state[:, ee_idx, 3:7]
    flag = 0
    while not torch.allclose(cur_ee_pos, ee_pos, atol=pos_atol) or not torch.allclose(
        matrix_from_quat(cur_ee_quat), matrix_from_quat(ee_quat), atol=quat_atol
    ):
        flag += 1
        log.debug(f"Cur pos: {cur_ee_pos}")
        log.debug(f"Cur quat: {cur_ee_quat}")
        log.debug(f"Target pos: {ee_pos}")
        log.debug(f"Target quat: {ee_quat}")
        log.debug(f"pos close: {torch.allclose(cur_ee_pos, ee_pos, atol=pos_atol)}")
        log.debug(
            f"quat close: {torch.allclose(matrix_from_quat(cur_ee_quat), matrix_from_quat(ee_quat), atol=quat_atol)}"
        )

        reach_target_try(ee_pos, ee_quat)

        states = env.handler.get_states()
        ee_idx = states.robots[robot.name].body_names.index(env.handler.robot.ee_body_name)
        cur_ee_pos = states.robots[robot.name].body_state[:, ee_idx, :3]
        cur_ee_quat = states.robots[robot.name].body_state[:, ee_idx, 3:7]
        if flag > 100:
            break

    log.info(f"Reach target in {flag} steps")
    if flag > 100:
        return False
    return True


def close_gripper():
    """Close the gripper."""
    states = env.handler.get_states()
    state_nested = state_tensor_to_nested(env.handler, states)
    cur_robot_dof = state_nested[0]["robots"][robot.name]["dof_pos"]
    random_finger_joint = random.uniform(0.32, 0.35)
    random_finger_joint = 0.31
    cur_robot_dof["finger_joint"] = random_finger_joint
    actions = [{"dof_pos_target": cur_robot_dof}] * scenario.num_envs
    for _ in range(20):
        obs, _, _, _, _ = env.step(actions)
        global STEP_NOW
        STEP_NOW += 1
        obs_saver.add(obs)


def rotate_arm(steps: int = 100):
    """Rotate the arm."""
    states = env.handler.get_states()
    state_nested = state_tensor_to_nested(env.handler, states)
    cur_robot_dof = state_nested[0]["robots"][robot.name]["dof_pos"]
    cur_robot_dof["joint_7"] = -math.pi / 2
    actions = [{"dof_pos_target": cur_robot_dof}] * scenario.num_envs
    for _ in range(steps):
        obs, _, _, _, _ = env.step(actions)
        global STEP_NOW
        STEP_NOW += 1
        obs_saver.add(obs)


def rotate_arm_back(steps: int = 100):
    """Rotate the arm."""
    states = env.handler.get_states()
    state_nested = state_tensor_to_nested(env.handler, states)
    cur_robot_dof = state_nested[0]["robots"][robot.name]["dof_pos"]
    cur_robot_dof["joint_7"] = 0
    actions = [{"dof_pos_target": cur_robot_dof}] * scenario.num_envs
    for _ in range(steps):
        obs, _, _, _, _ = env.step(actions)
        global STEP_NOW
        STEP_NOW += 1
        obs_saver.add(obs)


def rotate_joint46(deg4: float, deg6: float):
    """Rotate the joint 4."""
    states = env.handler.get_states()
    state_nested = state_tensor_to_nested(env.handler, states)

    cur_robot_dof = state_nested[0]["robots"][robot.name]["dof_pos"]
    origin_joint_4 = cur_robot_dof["joint_4"]
    target_joint_4 = deg4 / 180 * math.pi
    origin_joint_6 = cur_robot_dof["joint_6"]
    target_joint_6 = deg6 / 180 * math.pi
    origin_finger_joint = cur_robot_dof["finger_joint"]
    target_finger_joint = 0.3095
    for i in range(20):
        cur_robot_dof["joint_4"] = origin_joint_4 + (target_joint_4 - origin_joint_4) * i / 20
        cur_robot_dof["joint_6"] = origin_joint_6 + (target_joint_6 - origin_joint_6) * i / 20
        cur_robot_dof["finger_joint"] = origin_finger_joint + (target_finger_joint - origin_finger_joint) * i / 20
        actions = [{"dof_pos_target": cur_robot_dof}] * scenario.num_envs
        obs, _, _, _, _ = env.step(actions)
        obs_saver.add(obs)
        global STEP_NOW
        STEP_NOW += 1


def rotate_joint3(deg3: float):
    """Rotate the joint 3."""
    states = env.handler.get_states()
    state_nested = state_tensor_to_nested(env.handler, states)
    cur_robot_dof = state_nested[0]["robots"][robot.name]["dof_pos"]
    cur_robot_dof["joint_3"] = deg3 / 180 * math.pi
    cur_robot_dof["finger_joint"] = 0.3095
    actions = [{"dof_pos_target": cur_robot_dof}] * scenario.num_envs
    for _ in range(20):
        obs, _, _, _, _ = env.step(actions)
        global STEP_NOW
        STEP_NOW += 1
        obs_saver.add(obs)


log.info("reaching")
flag = reach_target_dedicated(torch.tensor([[0.17, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.18, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.19, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.20, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.21, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.22, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.23, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.24, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.25, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.26, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.27, 0.30, 0.8]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()

log.info("closing gripper")
close_gripper()

log.info("lifting")
meta_data["start_lifting_steps"] = STEP_NOW
flag = reach_target_dedicated(torch.tensor([[0.25, 0.30, 0.81]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.25, 0.30, 0.82]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.25, 0.30, 0.83]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.25, 0.30, 0.84]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
flag = reach_target_dedicated(torch.tensor([[0.25, 0.30, 0.85]]), torch.tensor([[0.0, 0.707, 0.0, 0.707]]))
if not flag:
    exit()
meta_data["start_rotate_joint46"] = STEP_NOW
rotate_joint46(70, 37)
log.info("moving")
meta_data["start_rotate_joint3"] = STEP_NOW
rotate_joint3(-85)
obs_saver.save()

import random

steps = random.randint(30, 100)
meta_data["start_rotate_arm"] = STEP_NOW

rotate_arm(steps)
meta_data["start_rotate_arm_back"] = STEP_NOW
rotate_arm_back(steps)
meta_data["start_end_steps"] = STEP_NOW
for _ in range(50):
    env.handler.env.sim.step()
    # global STEP_NOW
    STEP_NOW += 1
    env.handler.refresh_render()
    obs = env.handler.get_states()
    obs_saver.add(obs)

obs_saver.save()
import json

json.dump(meta_data, open(f"outputs/water/{path_name}.json", "w"))
