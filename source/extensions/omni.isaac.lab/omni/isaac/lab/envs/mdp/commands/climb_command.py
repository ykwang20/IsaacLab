# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.managers import SceneEntityCfg



if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import NormalVelocityCommandCfg,  ClimbCommandCfg


class ClimbCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: ClimbCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: ClimbCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)
       

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        num_bodies = self.robot.num_bodies

   
        self.climb_command = torch.zeros(self.num_envs, device=self.device)

        self.min_dist_to_box = None
        self.max_avg_height = torch.zeros(self.num_envs, device=self.device)
        self.max_com_x = torch.zeros(self.num_envs, device=self.device)
        self.mass = self.robot.root_physx_view.get_masses().clone().to(self.device)


    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "ClimbCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        # msg += f"\tHeading command: {self.cfg.heading_command}\n"
        # if self.cfg.heading_command:
        #     msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        # msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return self.climb_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

        

   
    def _resample_command(self, env_ids: Sequence[int]):
        if hasattr(self._env, "reset_buf"):

            # If the reset buffer is present, use it to determine which environments to reset
            reset_buf = self._env.reset_buf
            reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        else:
            # If the reset buffer is not present, reset all environments
            reset_env_ids = torch.arange(self.num_envs, device=self.device)
        self.climb_command[env_ids] = 1
        if self.cfg.activated:
            self.climb_command[reset_env_ids] =  0

        self.mass.copy_(self.robot.root_physx_view.get_masses())

        # reset avg height of all rigid bodies
        self.max_avg_height[reset_env_ids] = torch.mean(self.robot.data.body_pos_w[reset_env_ids, :, 2].clip(max=0.02), dim=1)
        if self.min_dist_to_box is None:
            self.min_dist_to_box = torch.abs(self.robot.data.body_pos_w[:, :, 2]).clone()
        else:
            self.min_dist_to_box[reset_env_ids] = torch.abs(self.robot.data.body_pos_w[reset_env_ids, :, 2])
        body_coms = self.robot.data.body_pos_w[reset_env_ids, :, 0] - self._env.scene.env_origins[reset_env_ids, 0].unsqueeze(1)
        self.max_com_x[reset_env_ids] =  torch.sum(body_coms * self.mass[reset_env_ids], dim=1) / torch.sum(self.mass[reset_env_ids], dim=1)

    def update_max_com_x(self, current_com: torch.Tensor):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.max_com_x = torch.max(self.max_com_x, current_com,)
    
    def update_min_dist_to_box(self, current_dist: torch.Tensor):
        """Update the minimum distance to the box in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the min distance
        if self.min_dist_to_box is None:
            self.min_dist_to_box = current_dist.clone()
        else:      
            self.min_dist_to_box = torch.min(self.min_dist_to_box, current_dist,)

    def update_max_avg_height(self, current_avg: torch.Tensor):
        """Update the average height of the robot in the environment.

        Args:
            env_ids: The indices of the environments to update.
        """
        # update the max height
        self.max_avg_height = torch.max(self.max_avg_height, current_avg,)
    

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        pass
       
   