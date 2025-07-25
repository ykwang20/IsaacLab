# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter
import carb
import omni.isaac.lab.sim as sim_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def terrain_levels_box_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > 2.5
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def terrain_levels_target(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], reached_distance: float, minimal_covered: float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the of the robot to the target pos.

    This term is used to increase the difficulty of the terrain when the robot walks close enough to the target and decrease the
    difficulty ...

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    target_pos_e = env.command_manager.get_command("target_pos_e")
    target_pos_w=target_pos_e+env.scene.env_origins
    target_pos_w[:,2]=0
    # compute the distance the robot walked
    covered_distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    remaining_distance = torch.norm(target_pos_w[env_ids, :2] - asset.data.root_pos_w[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = remaining_distance < reached_distance
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = covered_distance < minimal_covered
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def terrain_levels_height(
    env: ManagerBasedRLEnv, env_ids: Sequence[int],update_prob:float=0.8,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the of the robot to the target pos.

    This term is used to increase the difficulty of the terrain when the robot walks close enough to the target and decrease the
    difficulty ...

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    
    lowest_height =  asset.data.body_pos_w[env_ids, :, 2].min(dim=1)[0] 
    # robots that walked far enough progress to harder terrains
    move_up = lowest_height > 0.02
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = (asset.data.root_link_pos_w[env_ids, 2]-env.scene.env_origins[env_ids,2]) < 0.45
    move_down *= ~move_up
    # update terrain levels
    #terrain.update_env_origins(env_ids, move_up, move_down)
    terrain.update_env_origins_prob(env_ids, move_up, move_down, update_prob)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def modify_push_vel(env: ManagerBasedRLEnv, env_ids: Sequence[int], vel_min: float, vel_max: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg('push_robot')
        # update term settings
        term_cfg.params = {"velocity_range": {"x": (vel_min, vel_max), "y": (vel_min, vel_max)}}
        env.event_manager.set_term_cfg('push_robot', term_cfg)

def gravity_annealing(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], min_gravity_ratio: float=0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the of the robot to the target pos.

    This term is used to increase the difficulty of the terrain when the robot walks close enough to the target and decrease the
    difficulty ...

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target_pos_e = env.command_manager.get_command("target_pos_e")
    target_pos_w=target_pos_e+env.scene.env_origins
    target_pos_w[:,2]=0
    # compute the distance the robot walked
    remaining_distance = torch.norm(target_pos_w[env_ids, :2] - asset.data.root_pos_w[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    gravity_up = remaining_distance.mean(dim=0) < 0.1
    physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
    current_g: carb.Float3 = physics_sim_view.get_gravity()
    g_z =current_g.z
    if gravity_up:
        

        if current_g.z > -9.8:  
            g_z = current_g.z - 0.05*9.81
            new_g = carb.Float3(
                current_g.x ,
                current_g.y ,
                g_z
            )
            physics_sim_view.set_gravity(new_g)
    
    return g_z

