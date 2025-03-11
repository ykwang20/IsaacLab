# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_com_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_com_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)

def position_tracking(env, command_name: str,  start_time: float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of position in the world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    episode_time=env.episode_length_buf * env.step_dt
    #print('episode step:',episode_time)
    # pos_error = torch.norm(env.command_manager.get_command(command_name)[:, :2]
    #                        +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)
    #return 1-0.5*pos_error
    pos_error_square = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2]
                            +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2]), dim=1)
    pos_error=torch.sqrt(pos_error_square)
    # print('pos error:',pos_error)
    # print('pos tracking reward:',1/(1+pos_error_square))
    # print('pos command:',env.command_manager.get_command(command_name)[:, :2])
    # print('pos robot:',asset.data.root_pos_w[:, :2]-env.scene.env_origins[:,:2])
    # print('pos error square',pos_error_square)
    #print('pos error rew',torch.where(episode_time>2,1/(1+pos_error_square),torch.zeros_like(pos_error_square)))
    return torch.where(episode_time>start_time,1/(1+pos_error_square),torch.zeros_like(pos_error_square))
    

def wait_penalty(env, command_name: str,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for waiting."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    norm_vel=asset.data.root_lin_vel_w.norm(dim=1)
    pos_error = torch.norm(env.command_manager.get_command(command_name)[:, :2]
                           +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)
    #print('wait penalty:',torch.where(torch.logical_and(norm_vel<0.15, pos_error>0.2),torch.ones_like(norm_vel),torch.zeros_like(norm_vel)))
    return torch.where(torch.logical_and(norm_vel<0.15, pos_error>0.2),
                       torch.ones_like(norm_vel),torch.zeros_like(norm_vel))

def move_in_direction(env, command_name: str,  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward moving in a specified direction."""
    # extract the used quantities (to enable type-hinting)
    # TODO: only xy?
    asset = env.scene[asset_cfg.name]
    target_pos_e=env.command_manager.get_command(command_name)
    target_pos_w=target_pos_e+env.scene.env_origins
    target_pos_w[:,2]=0
    direction=target_pos_w-asset.data.root_pos_w
    vel=asset.data.root_lin_vel_w
    raw_reward=torch.cosine_similarity(direction[:,:3],vel[:,:3],dim=1)
    #print('raw reward:',raw_reward)
    condition=torch.logical_and(torch.norm(vel,dim=-1)<0.1,raw_reward>0)
    # print('velocity:',torch.norm(vel))
    # print('move in direction:',torch.where(condition,torch.zeros_like(raw_reward),raw_reward))
    return torch.where(condition,torch.zeros_like(raw_reward),raw_reward)

def joint_velocity_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset=env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0)
    return torch.sum(out_of_limits, dim=1)

def base_lin_ang_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset=env.scene[asset_cfg.name]
    root_id=asset.find_bodies("torso_link")[0]
    
    return (torch.sum(torch.square(asset.data.body_lin_acc_w[:, root_id,:]), dim=-1).squeeze(-1)
            +0.02*torch.sum(torch.square(asset.data.body_ang_acc_w[:,root_id, :]), dim=-1).squeeze(-1))

def base_lin_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset=env.scene[asset_cfg.name]
    # print('base lin vel:',torch.norm(asset.data.root_com_lin_vel_b[:, :],dim=-1))
    # print('base ang vel:',torch.norm(asset.data.root_com_ang_vel_b[:, :],dim=-1))
    return torch.sum(torch.square(asset.data.root_com_lin_vel_b[:, :]), dim=1)+0.02*torch.sum(torch.square(asset.data.root_com_lin_vel_b[:, :]), dim=1)

def feet_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the acceleration of the feet using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    feet_ids = asset.find_bodies(".*_ankle_roll_link")[0]
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, feet_ids, :],dim=-1), dim=-1)

def feet_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the acceleration of the feet using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    feet_ids = asset.find_bodies(".*_ankle_roll_link")[0]
    # print('feet height',asset.data.body_pos_w[:, feet_ids, 2])
    # print('body height',asset.data.root_link_pos_w[:, 2])
    return torch.sum(asset.data.body_pos_w[:, feet_ids, 2].clip(max=0.02), dim=-1)

def body_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the acceleration of the feet using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    print('body height',asset.data.root_link_pos_w[:, 2])
    return asset.data.root_link_pos_w[:, 2]

def stand_at_target(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty=torch.sum(torch.abs(angle), dim=1)
    reach_target= torch.norm(env.command_manager.get_command(command_name)[:, :2]
                    +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)<0.25
    return torch.where(reach_target,penalty,torch.zeros_like(penalty))

def stable_at_target(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    feet_ids = asset.find_bodies(".*_ankle_roll_link")[0]
    feet_on_air=torch.sum(asset.data.body_pos_w[:, feet_ids, 2], dim=-1)
    move_quickly=torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    reach_target= torch.norm(env.command_manager.get_command(command_name)[:, :2]
                    +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)<0.1
    return torch.where(reach_target,feet_on_air+move_quickly,torch.zeros_like(feet_on_air))





def contact_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    value=env.termination_manager.get_term('base_contact').float()

    return value

def stepped_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    value=env.termination_manager.get_term('success').float()
    return value

def air_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    value=env.termination_manager.get_term('on_air').float()
    return value

def body_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    bodies_air_time=contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    air_time=bodies_air_time.min(dim=1)[0]
    # print('body ids:',sensor_cfg.body_ids)
    # print('bodies air time:',bodies_air_time)
    # print('air time:',air_time)
    # print('reward:',torch.exp(20*air_time)-1)
    return torch.exp(20*air_time)-1
    # net_contact_forces = contact_sensor.data.net_forces_w_history
    # # check if any contact force exceeds the threshold
    # on_air=~torch.any(
    #     torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    # )
    # on_air = torch.logical_and(on_air, env.episode_length_buf>20)
    # #print('on air:',on_air)
    # return on_air

def success_bonus(
    env: ManagerBasedRLEnv,sensor_cfg: SceneEntityCfg, success_distance:float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
   
    # extract the used quantities (to enable type-hinting) 
    asset = env.scene[asset_cfg.name]
   
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact=torch.all(torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1, dim=1)
    target_pos_e = env.command_manager.get_command("target_pos_e")
    target_pos_w=target_pos_e+env.scene.env_origins
    target_pos_w[:,2]=0
    
    remaining_distance = torch.norm(target_pos_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    #print('remaining distance:',remaining_distance)
    near = remaining_distance < success_distance
    
    stepped=torch.logical_and(near, contact)
    # print('near', near)
    # print('contact', contact)
    return stepped


def curiosity(env: ManagerBasedRLEnv):
    curio_obs = env.obs_buf['curiosity']
    assert curio_obs.shape[1] == env.cfg.curiosity.obs_dim
    return env.curiosity_handler.update_curiosity(curio_obs)

def curiosity_cnt(env: ManagerBasedRLEnv):
    curio_obs = env.obs_buf['curiosity']
    assert curio_obs.shape[1] == env.cfg.curiosity.obs_dim
    return env.curiosity_handler.update_curiosity(curio_obs)