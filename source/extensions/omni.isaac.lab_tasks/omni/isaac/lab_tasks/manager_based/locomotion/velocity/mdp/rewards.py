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
from omni.isaac.lab.sensors import ContactSensor, ContactSensorZ
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
    #print('root pos:',asset.data.root_pos_w[:, :2]-env.scene.env_origins[:,:2])
    pos_error=torch.sqrt(pos_error_square)
    #print('pos error:',pos_error)
    #print('pos tracking reward:',1/(1+pos_error_square))
    # print('pos command:',env.command_manager.get_command(command_name)[:, :2])
    # print('pos robot:',asset.data.root_pos_w[:, :2]-env.scene.env_origins[:,:2])
    # print('pos error square',pos_error_square)
    #print('pos error rew',torch.where(episode_time>2,1/(1+pos_error_square),torch.zeros_like(pos_error_square)))
    return torch.where(episode_time>start_time,1/(1+pos_error_square),torch.zeros_like(pos_error_square))

def position_tracking_cos(env, command_name: str,  start_time: float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of position in the world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    episode_time=env.episode_length_buf * env.step_dt
    #print('episode step:',episode_time)
    # pos_error = torch.norm(env.command_manager.get_command(command_name)[:, :2]
    #                        +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)
    #return 1-0.5*pos_error
    pos_error = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2]
                            +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2],dim=-1)
    
    rew=0.3+0.7*torch.cos(pos_error*torch.pi/2)
    # print('pos command:',env.command_manager.get_command(command_name)[:, :2])
    # print('pos robot:',asset.data.root_pos_w[:, :2]-env.scene.env_origins[:,:2])
    # print('pos error square',pos_error_square)
    #print('pos error rew',torch.where(episode_time>2,1/(1+pos_error_square),torch.zeros_like(pos_error_square)))
    return torch.where(episode_time>start_time,rew,torch.zeros_like(rew))
    

# def wait_penalty(env, command_name: str,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalty for waiting."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     norm_vel=asset.data.root_lin_vel_w.norm(dim=1)
#     vel_x=asset.data.root_lin_vel_w[:,0]
#     pos_error = torch.norm(env.command_manager.get_command(command_name)[:, :2]
#                            +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)
#     # print('norm vel:',norm_vel)
#     # print('vel x:',vel_x)
#     # print('pos error:',pos_error)
#     # print('wait penalty:',torch.where(torch.logical_and(vel_x<0.3, pos_error>0.2),torch.ones_like(norm_vel),torch.zeros_like(norm_vel)))
#     return torch.where(torch.logical_and(vel_x<0.15, pos_error>0.2),
#                        torch.ones_like(norm_vel),torch.zeros_like(norm_vel))
def downward_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    max_avg_height = env.command_manager.get_term('target_pos_e').max_avg_height
    # input("Input Enter")
    # print('max_avg_height:',max_avg_height)
    current_height = torch.mean(asset.data.body_pos_w[:, :, 2].clip(max=0.02), dim=1)
    # print('current_height:',current_height)
    downward=max_avg_height-current_height
    success = torch.logical_and(asset.data.body_pos_w[:, :, 2].min(dim=1)[0] > 0.02, 
                                asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0] > 1.6)
    #print('downward:',downward)
    
    #print('lowest height:',asset.data.body_pos_w[:, :, 2].min(dim=1)[0])
    # print('root pos:',asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0])
    penalty=torch.logical_or(downward>0 , downward.abs()<0.00001)
    penalty=penalty.logical_and(current_height<0.02)
    # print('penalty:',penalty)   
    penalty=penalty.logical_and(~success)
    # print('penalty after:',penalty)
    #print('penalty:',penalty)
    env.command_manager.get_term('target_pos_e').update_max_avg_height(current_height)
    return torch.where(penalty,torch.ones_like(downward),torch.zeros_like(downward))

def com_backward_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    max_com_x = env.command_manager.get_term('target_pos_e').max_com_x
    #print('max com x:',max_com_x)
    #compute body com X position relative to the env origin
    body_coms = asset.data.body_pos_w[:, :, 0] - env.scene.env_origins[:, 0].unsqueeze(1) #shape: (num_envs, num_bodies)
    masses= env.command_manager.get_term('target_pos_e').mass
    #calculate center of mass of all bodies, which means the mean of body coms multiplied by the mass of each body
    current_com_x = torch.sum(body_coms * masses, dim=1) / torch.sum(masses, dim=1)
    #print('com:',current_com_x)
    backward = max_com_x - current_com_x
    success = torch.logical_and(asset.data.body_pos_w[:, :, 2].min(dim=1)[0] > 0.02, 
                                asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0] > 2)
    #print('current root x:',asset.data.root_pos_w[:, 0] - env.scene.env_origins[:,0])
    #penalty = torch.logical_or(backward > 0, backward.abs() < 0.00001)
    penalty = torch.logical_and(backward > 0,~success)
    # input("Input Enter")
    # print('backward:',backward)

    env.command_manager.get_term('target_pos_e').update_max_com_x(current_com_x)

    
    
   
    return torch.where(penalty, torch.ones_like(backward), torch.zeros_like(backward))

def wait_penalty(env, command_name: str,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for waiting."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    norm_vel=asset.data.root_lin_vel_w.norm(dim=1)
    #print('norm vel:',norm_vel) 
    current_height = torch.mean(asset.data.body_pos_w[:, :, 2].clip(max=0.), dim=1)
    #print('current_height:',current_height)
    # pos_error = torch.norm(env.command_manager.get_command(command_name)[:, :2]
    #                        +env.scene.env_origins[:,:2]-asset.data.root_pos_w[:, :2], dim=1)
    #print('wait penalty:',torch.where(torch.logical_and(norm_vel<0.15, pos_error>0.2),torch.ones_like(norm_vel),torch.zeros_like(norm_vel)))
    # return torch.where(torch.logical_and(norm_vel<0.15, pos_error>0.2),
    #                    torch.ones_like(norm_vel),torch.zeros_like(norm_vel))
    #print('wait penalty:',torch.where(torch.logical_and(norm_vel<0.15, current_height<-0.01),torch.ones_like(norm_vel),torch.zeros_like(norm_vel)))
    return torch.where(torch.logical_and(norm_vel<0.15, current_height<-0.2),
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
    pos_error_sqaure= torch.sum(torch.square(direction[:, :2]), dim=1)
    vel=asset.data.root_lin_vel_w
    raw_reward=torch.cosine_similarity(direction[:,:3],vel[:,:3],dim=1)
    #print('raw reward:',raw_reward)
    #condition=torch.logical_and(torch.norm(vel,dim=-1)<0.1,raw_reward>0)
    condition=pos_error_sqaure < 0.05  # if the robot is very close to the target position
    reward = torch.where(condition, torch.ones_like(raw_reward), raw_reward)
    # print('velocity:',torch.norm(vel))
    # if reward[0]< 0:
    #     print('pos error square:', pos_error_sqaure[0])
    #     user_input = input("Input Enter")
    #print('move in direction:',torch.where(condition,torch.zeros_like(raw_reward),raw_reward))
    #print('move in direction reward:', torch.where(reward > 0, reward, 2*reward))
    # print('move in direction reward:', torch.where(reward > 0, reward, 4*reward*vel.norm(dim=-1)))
    # print('vel norm:',vel.norm(dim=-1))
    # user_input = input("Input Enter")
    #print('move in direction reward:', torch.where(reward > 0, reward, reward))
    return torch.where(reward > 0, reward, reward)  # Penalize negative reward sharply
    #return torch.where(condition,torch.zeros_like(raw_reward),raw_reward)

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

def base_lin_vel_clip(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset=env.scene[asset_cfg.name]
    norm_lin_vel=(torch.sum(torch.square(asset.data.root_com_lin_vel_b[:, :]), dim=1)-1).clip(min=0)
    
    #input("Input Enter")
    #print('base lin vel:',norm_lin_vel)
    return norm_lin_vel

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

def knee_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    knee_ids = asset.find_bodies(".*knee_link")[0]
    #print('knee heights:',asset.data.body_pos_w[:, knee_ids, 2])
    max_knee = asset.data.body_pos_w[:, knee_ids, 2].max(dim=-1)
    max_height = max_knee[0]
    high_knee_ids = max_knee[1]
    knee_x = asset.data.body_pos_w[:, knee_ids, 0]        # (B, N_knee)
    batch_idx = torch.arange(knee_x.size(0), device=knee_x.device)
    high_knee_x = knee_x[batch_idx, high_knee_ids] - env.scene.env_origins[:, 0]
    #relative_height=max_height-env.scene.env_origins[:,2]
    #print('max height:',max_height)
    lift_bonus=max_height.clip(max=0.01)-env.scene.env_origins[:,2]
    higher_penalty=5*(0.15-max_height).clip(max=0)
    backward_penalty=(high_knee_x-1.25).clip(max=0.25)
    # print('root pos:',asset.data.root_link_pos_w[:, 0]-env.scene.env_origins[:,0])
    # print('backward penalty:',backward_penalty)
    return lift_bonus+higher_penalty+backward_penalty

def body_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the acceleration of the feet using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    #print('body height',asset.data.root_link_pos_w[:, 2])
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
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # contact_forces = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    air_time=bodies_air_time.min(dim=1)[0]
    if air_time > 0.0001:
        user_input = input("Input Enter")
    else:
        #print('bodies air time:', bodies_air_time)
        idx_list = torch.nonzero(bodies_air_time[0] < 0.001).squeeze(-1).tolist()
        #print('idx list:', idx_list)

        print('contact body:', [contact_sensor.body_names[i] for i in idx_list])
        #print('contact forces:', contact_forces[0, idx_list])
        print('contact forces:',net_contact_forces[0, 0,idx_list, :])

   
    return torch.exp(20*air_time)-1
    # net_contact_forces = contact_sensor.data.net_forces_w_history
    # # check if any contact force exceeds the threshold
    # on_air=~torch.any(
    #     torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    # )
    # on_air = torch.logical_and(on_air, env.episode_length_buf>20)
    # #print('on air:',on_air)
    # return on_air

def group_air_time(env: ManagerBasedRLEnv, upper_sensor_cfg: SceneEntityCfg, lower_sensor_cfg: SceneEntityCfg,threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # penelize the time spent when there are at least one group of bodies in the air
    asset = env.scene[asset_cfg.name]

    contact_sensor: ContactSensorZ = env.scene.sensors[upper_sensor_cfg.name]

    upper_bodies_air_time=contact_sensor.data.current_air_time[:, upper_sensor_cfg.body_ids]
    lower_bodies_air_time=contact_sensor.data.current_air_time[:, lower_sensor_cfg.body_ids]

    

    upper_air_time=upper_bodies_air_time.min(dim=1)[0]
    lower_air_time=lower_bodies_air_time.min(dim=1)[0]

    air_time=torch.maximum(upper_air_time, lower_air_time)
    root_pos=asset.data.root_pos_w[:, :]-env.scene.env_origins[:,:]
    #print('root height:',root_height)
    #activated = torch.logical_and(root_pos[:,2]> 0.55,contact_sensor.activated)
    #activated = torch.logical_and(contact_sensor.activated, root_pos[:,0]> 1.35)
    activated = contact_sensor.activated
    air_time=torch.where(activated, air_time, torch.zeros_like(air_time))
    bodies_id_list=upper_sensor_cfg.body_ids + lower_sensor_cfg.body_ids
    # print('contact activated:',contact_sensor.activated)
    # print('root pos:',root_pos[:,0])
    # print('activated:',activated)
    # print('air time:', air_time)
    # print('root pos:',root_pos[:,0])
    # print('upper body names:', [contact_sensor.body_names[i] for i in upper_sensor_cfg.body_ids])
    # print('lower body names:', [contact_sensor.body_names[i] for i in lower_sensor_cfg.body_ids])
    #if air_time > 0.0001:
    #user_input = input("Input Enter")
    #print('air time:', air_time)
    #else:
    #print('bodies air time:', bodies_air_time)
    # bodies_air_time=contact_sensor.data.current_air_time[:, :]
    # idx_list= torch.nonzero(bodies_air_time[0] < 0.001).squeeze(-1).tolist()
    # lower_idx_list = torch.nonzero(lower_bodies_air_time[0] < 0.001).squeeze(-1).tolist()
    # upper_idx_list = torch.nonzero(upper_bodies_air_time[0] < 0.001).squeeze(-1).tolist()
    # #print('idx list:', idx_list)

    #print('contact body:', [contact_sensor.body_names[i] for i in idx_list if i in bodies_id_list])
    #     #print('contact forces:', contact_forces[0, idx_list])
    # net_contact_forces = contact_sensor.data.net_forces_w_history

    # print('contact forces:',net_contact_forces[0, 0,lower_sensor_cfg.body_ids, :])
    # print('air time:', air_time)
   
    return (torch.exp(20*air_time)-1).clip(max=200.0)

def respective_air_time(env: ManagerBasedRLEnv, upper_sensor_cfg: SceneEntityCfg, lower_sensor_cfg: SceneEntityCfg,threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # penelize the time spent when there are at least one group of bodies in the air
    asset = env.scene[asset_cfg.name]

    contact_sensor: ContactSensorZ = env.scene.sensors[upper_sensor_cfg.name]

    upper_bodies_air_time=contact_sensor.data.current_air_time[:, upper_sensor_cfg.body_ids]
    lower_bodies_air_time=contact_sensor.data.current_air_time[:, lower_sensor_cfg.body_ids]

    upper_air_time=upper_bodies_air_time.min(dim=1)[0]
    lower_air_time=lower_bodies_air_time.min(dim=1)[0]

    #air_time=torch.maximum(upper_air_time, lower_air_time)
    root_pos=asset.data.root_pos_w[:, :]-env.scene.env_origins[:,:]
    activated = torch.logical_and(root_pos[:,2]> 0.55,contact_sensor.activated)
    activated = torch.logical_and(activated, root_pos[:,0]> 1.35)

    upper_air_time=torch.where(activated, upper_air_time, torch.zeros_like(upper_air_time))
    lower_air_time=torch.where(activated, lower_air_time, torch.zeros_like(lower_air_time))
    #air_time=torch.where(activated, air_time, torch.zeros_like(air_time))
    
   
   
    return (torch.exp(20*upper_air_time)-1).clip(max=200.0)+(torch.exp(20*lower_air_time)-1).clip(max=200.0)

def feet_in_air(env: ManagerBasedRLEnv, feet_sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_air=torch.max(torch.norm(net_contact_forces[:, :, feet_sensor_cfg.body_ids], dim=-1), dim=1)[0] < threshold
    feet_air=torch.sum(feet_air.int(), dim=1)
    #print('feet air:', feet_air)
    return feet_air

def knee_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, feet_sensor_cfg:SceneEntityCfg, torso_bodies_sensor_cfg:SceneEntityCfg,threshold: float,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_contact_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
    torso_bodies_sensor: ContactSensor = env.scene.sensors[torso_bodies_sensor_cfg.name]

    bodies_air_time=contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    net_contact_forces = feet_contact_sensor.data.net_forces_w_history
    feet_air=torch.any(torch.max(torch.norm(net_contact_forces[:, :, feet_sensor_cfg.body_ids], dim=-1), dim=1)[0] < 1, dim=1)
    #print('feet air:', feet_air)
    root_pos=asset.data.root_pos_w[:, 0]-env.scene.env_origins[:,0]
    #user_input = input("Input Enter")
    torso_bodies_force=torso_bodies_sensor.data.force_matrix_w.norm(dim=-1).max(dim=-1)[0].squeeze(-1)
    #print('torso_bodies_force:', torso_bodies_force)
    #print('bodies air time:', bodies_air_time)
    torso_self_contact=torso_bodies_force>threshold
    max_air_time=bodies_air_time.max(dim=1)[0]
    tmp_air_time=bodies_air_time.clone()
    tmp_air_time[:,-1]=max_air_time
    bodies_air_time=torch.where(torso_self_contact.unsqueeze(-1), tmp_air_time, bodies_air_time)
    air_time=bodies_air_time.min(dim=1)[0]
    #print('body names:',[contact_sensor.body_names[i] for i in sensor_cfg.body_ids])
    #print('body ids:',sensor_cfg.body_ids)
    #print('bodies air time:', bodies_air_time)
    # print('air time:', air_time)
    #if air_time > 0.0001:
    # #     print('pos error square:', pos_error_sqaure[0])
          #user_input = input("Input Enter")
          #print('root pos:',root_pos)
    #print('knee air time:', (torch.exp(20*air_time)-1))
    reward=(torch.exp(20*air_time)-1).clip(max=2000.0)
    #print('root pos:',root_pos)
    #print('reward:',torch.where(root_pos>1.5, reward, torch.zeros_like(reward)))
    return torch.where(root_pos>1.68, reward, torch.zeros_like(reward))

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
    curio_obs = env.obs_buf['rnd_state']
    assert curio_obs.shape[1] == env.cfg.curiosity.obs_dim
    return env.curiosity_handler.update_curiosity(curio_obs)

def curiosity_cnt(env: ManagerBasedRLEnv):
    curio_obs = env.obs_buf['rnd_state']
    assert curio_obs.shape[1] == env.cfg.curiosity.obs_dim
    return env.curiosity_handler.update_curiosity(curio_obs)