"""Sub-module containing utilities for loading and saving trajectories in v2 format."""

from __future__ import annotations

import os
from glob import glob

import torch
from loguru import logger as log

from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg

from .loader import load_traj_file


def get_traj_v2(task: BaseTaskCfg, robot: BaseRobotCfg):
    """Get the trajectory data.

    Args:
        task: The task cfg instance.
        robot: The robot cfg instance.

    Returns:
        The trajectory data.
    """
    ## Load trajectory data
    assert os.path.exists(task.traj_filepath)
    if os.path.isfile(task.traj_filepath):
        assert (
            task.traj_filepath.endswith("_v2.pkl")
            or task.traj_filepath.endswith("_v2.pkl.gz")
            or task.traj_filepath.endswith("_v2.json")
            or task.traj_filepath.endswith("_v2.yaml")
        )
        data = load_traj_file(task.traj_filepath)[robot.name]
    else:
        assert task.traj_filepath.find("v2") != -1
        paths = glob(os.path.join(task.traj_filepath, f"{robot.name}_v2.*"))
        assert len(paths) >= 1
        path = paths[0]
        log.info(f"Loading trajectory from {path}")
        data = load_traj_file(path)[robot.name]

    ## Parse initial states
    if "init_state" in data[0]:
        init_states = [traj["init_state"] for traj in data]
    else:
        raise ValueError("No init_state found in the trajectory data")
    for demo_idx, init_state in enumerate(init_states):
        for obj_name in init_state:
            init_states[demo_idx][obj_name]["pos"] = torch.tensor(init_states[demo_idx][obj_name]["pos"])
            init_states[demo_idx][obj_name]["rot"] = torch.tensor(init_states[demo_idx][obj_name]["rot"])

    ## Parse actions
    if "actions" in data[0]:
        all_actions = [traj["actions"] for traj in data]
    else:
        log.error("No actions found in the trajectory data")
        all_actions = None

    ## Parse states
    if "states" in data[0] and data[0]["states"] is not None:
        all_states = [traj["states"] for traj in data]
        for demo_idx, states in enumerate(all_states):
            for step_idx, state in enumerate(states):
                for obj_name in state:
                    all_states[demo_idx][step_idx][obj_name]["pos"] = torch.tensor(state[obj_name]["pos"])
                    all_states[demo_idx][step_idx][obj_name]["rot"] = torch.tensor(state[obj_name]["rot"])
    else:
        log.error("No states found in the trajectory data")
        all_states = None

    return init_states, all_actions, all_states
