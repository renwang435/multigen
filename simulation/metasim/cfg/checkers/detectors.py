from __future__ import annotations

from dataclasses import MISSING

import torch
from loguru import logger as log

from metasim.cfg.objects import BaseObjCfg, PrimitiveCubeCfg
from metasim.constants import PhysicStateType
from metasim.utils.configclass import configclass
from metasim.utils.math import matrix_from_quat, quat_from_matrix
from metasim.utils.tensor_util import tensor_to_str

try:
    from metasim.sim import BaseSimHandler
except:
    pass


@configclass
class BaseDetector:
    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        raise NotImplementedError

    def is_detected(self, handler: BaseSimHandler, obj_name: str) -> torch.BoolTensor:
        raise NotImplementedError

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        raise NotImplementedError


@configclass
class RelativeBboxDetector(BaseDetector):
    """
    Check if the object is in the bounding box detector
    - The bbox detector is defined by `relative_pos` and `relative_quat` to the base object specified by `base_obj_name`
    - The bbox size is defined by `checker_lower` and `checker_upper`
    - If `ignore_base_ori` is True, the base object orientation is ignored
    """

    base_obj_name: str = MISSING
    relative_pos: tuple[float, float, float] = MISSING
    relative_quat: tuple[float, float, float, float] = MISSING
    checker_lower: tuple[float, float, float] = MISSING
    checker_upper: tuple[float, float, float] = MISSING
    ignore_base_ori: bool = False
    debug_vis: bool = False
    """Visualize the bounding box. Only supported with IsaacLab."""
    name: str = "bbox_detector"  # TODO: This is used for obj meta cfg name, need to handle multiple detectors
    fixed: bool = True
    """The pose of the bounding box is fixed once reset. Otherwise, it will be updated every step in correspond to the base object. Default to True."""

    def _update_checker(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(handler.num_envs))

        relative_rot_mat = matrix_from_quat(
            torch.tensor(self.relative_quat, dtype=torch.float32, device=handler.device)
        )  # [3, 3]
        relative_pos = torch.tensor(self.relative_pos, dtype=torch.float32, device=handler.device)  # [3]
        base_pos = handler.get_pos(self.base_obj_name, env_ids=env_ids)
        if self.ignore_base_ori:
            base_quat = torch.zeros((len(env_ids), 4), dtype=torch.float32, device=handler.device)
            base_quat[:, 0] = 1.0
        else:
            base_quat = handler.get_rot(self.base_obj_name, env_ids=env_ids)
        base_rot_mat = matrix_from_quat(base_quat)  # [n_env, 3, 3]
        checker_pos = base_pos + torch.matmul(base_rot_mat, relative_pos.unsqueeze(-1)).squeeze(-1)  # [n_env, 3]
        checker_rot_mat = torch.matmul(base_rot_mat, relative_rot_mat)  # [n_env, 3, 3]

        if not hasattr(self, "checker_pos"):
            self.checker_pos = torch.zeros((handler.num_envs, 3), dtype=torch.float32, device=handler.device)
        if not hasattr(self, "checker_rot_mat"):
            self.checker_rot_mat = torch.zeros((handler.num_envs, 3, 3), dtype=torch.float32, device=handler.device)
        if not hasattr(self, "checker_quat"):
            self.checker_quat = torch.zeros((handler.num_envs, 4), dtype=torch.float32, device=handler.device)

        self.checker_pos[env_ids] = checker_pos
        self.checker_rot_mat[env_ids] = checker_rot_mat
        self.checker_quat[env_ids] = quat_from_matrix(checker_rot_mat)

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(handler.num_envs))
        self._update_checker(handler, env_ids)

        self.checker_lower = torch.tensor(self.checker_lower, dtype=torch.float32, device=handler.device)
        self.checker_upper = torch.tensor(self.checker_upper, dtype=torch.float32, device=handler.device)

        ## Reset debug viewer
        self.reset_debug_viewer(handler, env_ids)

    def is_detected(self, handler: BaseSimHandler, obj_name: str) -> torch.BoolTensor:
        if not self.fixed:
            self._update_checker(handler)

        obj_pos = handler.get_pos(obj_name)

        # [n_env, 3]
        obj_pos_checker_local = torch.matmul(
            self.checker_rot_mat.transpose(-1, -2), (obj_pos - self.checker_pos).unsqueeze(-1)
        ).squeeze(-1)
        # [n_env, 1]
        object_in_checker = (
            (obj_pos_checker_local < self.checker_upper.unsqueeze(0))
            & (obj_pos_checker_local > self.checker_lower.unsqueeze(0))
        ).all(dim=-1)
        log.debug(
            f"Object {obj_name} local position in checker based on {self.base_obj_name}:"
            f" {tensor_to_str(obj_pos_checker_local)}"
        )
        return object_in_checker

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        if self.debug_vis:
            scale = torch.tensor(self.checker_upper) - torch.tensor(self.checker_lower)
            viewer = PrimitiveCubeCfg(
                name=self.name,
                size=scale,
                physics=PhysicStateType.XFORM,
                color=(1.0, 0.0, 0.0),
            )
            return [viewer]
        else:
            return []

    def reset_debug_viewer(self, handler: BaseSimHandler, env_ids: list[int]):
        if self.debug_vis:
            pos = self.checker_pos + torch.matmul(
                self.checker_rot_mat,
                torch.tensor((self.checker_lower + self.checker_upper) / 2, dtype=torch.float32).unsqueeze(-1),
            ).squeeze(-1)
            rot = self.checker_quat
            handler.set_pose(self.name, pos, rot)


@configclass
class Relative2DSphereDetector(BaseDetector):
    base_obj_name: str = MISSING
    relative_pos: tuple[float, float, float] = MISSING
    aixs: tuple[int, int] = MISSING
    radius: float = MISSING
    debug_vis: bool = False

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(handler.num_envs))

        relative_pos = torch.tensor(self.relative_pos, dtype=torch.float32)  # [3]

        base_pos = handler.get_pos(self.base_obj_name, env_ids=env_ids)

        self.checker_pos = base_pos + relative_pos

    def is_detected(self, handler: BaseSimHandler, obj_name: str) -> torch.BoolTensor:
        obj_pos = handler.get_pos(obj_name)

        object_in_checker = (
            torch.norm(obj_pos[:, self.aixs] - self.checker_pos[:, self.aixs], p=2, dim=-1) < self.radius
        )
        if object_in_checker.shape[0] != handler.num_envs:
            raise ValueError(
                f"Object {obj_name} in checker {self.name} is not in the correct shape: {object_in_checker.shape}"
            )

        return object_in_checker

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        return []


@configclass
class Relative3DSphereDetector(BaseDetector):
    base_obj_name: str = MISSING
    relative_pos: tuple[float, float, float] = MISSING
    radius: float = MISSING
    debug_vis: bool = False

    def reset(self, handler: BaseSimHandler, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = list(range(handler.num_envs))

        relative_pos = torch.tensor(self.relative_pos, dtype=torch.float32)  # [3]
        base_pos = handler.get_pos(self.base_obj_name, env_ids=env_ids)

        self.checker_pos = base_pos + relative_pos

    def is_detected(self, handler: BaseSimHandler, obj_name: str) -> torch.BoolTensor:
        obj_pos = handler.get_pos(obj_name)

        object_in_checker = torch.norm(obj_pos - self.checker_pos, p=2, dim=-1) < self.radius

        return object_in_checker

    def get_debug_viewers(self) -> list[BaseObjCfg]:
        return []
