# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


def position_tracking(env, target_pos,  start_time: float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of position in the world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    episode_time=env.episode_length_buf * env.step_dt
    #print('episode step:',episode_time)
    # pos_error = torch.norm(env.command_manager.get_command(command_name)[:, :2]
    #                        +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)
    #return 1-0.5*pos_error
    target_pos = torch.tensor(target_pos, device=env.device)
    pos_error_square = torch.sum(torch.square(target_pos[:2]
                            +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2]), dim=1)
    #pos_error=torch.sqrt(pos_error_square)
    # print('pos error:',pos_error)
    # print('pos tracking reward:',1/(1+pos_error_square))
    # print('pos command:',env.command_manager.get_command(command_name)[:, :2])
    # print('pos robot:',asset.data.root_pos_w[:, :2]-env.scene.env_origins[:,:2])
    # print('pos error square',pos_error_square)
    #print('pos error rew',torch.where(episode_time>2,1/(1+pos_error_square),torch.zeros_like(pos_error_square)))
    return torch.where(episode_time>start_time,1/(1+pos_error_square),torch.zeros_like(pos_error_square))

def wait_penalty(env, target_pos,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for waiting."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    norm_vel=asset.data.root_lin_vel_w.norm(dim=1)
    target_pos = torch.tensor(target_pos, device=env.device)
    pos_error = torch.norm(target_pos[ :2]
                           +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)
    #print('wait penalty:',torch.where(torch.logical_and(norm_vel<0.15, pos_error>0.2),torch.ones_like(norm_vel),torch.zeros_like(norm_vel)))
    return torch.where(torch.logical_and(norm_vel<0.15, pos_error>0.2),
                       torch.ones_like(norm_vel),torch.zeros_like(norm_vel))
class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_link_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_link_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        if "asset_cfg" not in cfg.params:
            cfg.params["asset_cfg"] = SceneEntityCfg("robot")
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, threshold: float, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        if "asset_cfg" not in cfg.params:
            cfg.params["asset_cfg"] = SceneEntityCfg("robot")
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)
