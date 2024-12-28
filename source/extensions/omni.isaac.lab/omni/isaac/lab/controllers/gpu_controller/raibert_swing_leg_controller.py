"""Implements the Raibert Swing Leg controller in Isaac."""
from typing import Any
import torch
from .base_robot import BaseRobot
from .gait_config import GaitConfig
from .phase_gait_generator import PhaseGaitGenerator

@torch.jit.script
def cubic_bezier(x0: torch.Tensor, x1: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
  progress = t**3 + 3 * t**2 * (1 - t)
  return x0 + progress * (x1 - x0)


@torch.jit.script
def _gen_swing_foot_trajectory(input_phase: torch.Tensor,
                               start_pos: torch.Tensor, mid_pos: torch.Tensor,
                               end_pos: torch.Tensor, highest_phase: torch.Tensor) -> torch.Tensor:
  # how about giving desired velocity?
  cutoff = highest_phase
  input_phase = torch.stack([input_phase] * 3, dim=-1)
  desired_foot_pos= torch.zeros_like(start_pos)
  desired_foot_pos[...,:2]=cubic_bezier(start_pos[...,:2], end_pos[...,:2], input_phase[...,:2])
  desired_foot_pos[...,2]=torch.where(input_phase[...,2]<cutoff, cubic_bezier(start_pos[...,2], mid_pos[...,2], input_phase[...,2]/cutoff), 
                                      cubic_bezier(mid_pos[...,2], end_pos[...,2], (input_phase[...,2]-cutoff)/(1-cutoff)))

  # return torch.where(
  #     input_phase < cutoff,
  #     cubic_bezier(start_pos, mid_pos, input_phase / cutoff),
  #     cubic_bezier(mid_pos, end_pos, (input_phase - cutoff) / (1 - cutoff)))
  return desired_foot_pos

@torch.jit.script
def cross_quad(v1, v2):
  """Assumes v1 is nx3, v2 is nx4x3"""
  v1 = torch.stack([v1, v1, v1, v1], dim=1)
  shape = v1.shape
  v1 = v1.reshape((-1, 3))
  v2 = v2.reshape((-1, 3))
  return torch.cross(v1, v2).reshape((shape[0], shape[1], 3))


# @torch.jit.script
def compute_desired_foot_positions(
    base_rot_mat,
    base_height,
    hip_positions_in_body_frame,
    base_velocity_world_frame,
    base_angular_velocity_body_frame,
    projected_gravity,
    desired_base_height: float,
    foot_height: float,
    foot_landing_clearance: float,
    stance_duration,
    normalized_phase,
    phase_switch_foot_positions,
    mid_residual,
    land_residual,
    highest_phase,
):
  indices = torch.tensor([0], device=base_rot_mat.device)

  hip_position = torch.matmul(base_rot_mat,
                              hip_positions_in_body_frame.transpose(
                                  1, 2)).transpose(1, 2)
  # test_base_rot_mat = torch.eye(3, device=base_rot_mat.device).repeat((hip_position.shape[0], 1, 1))
  # hip_position = torch.matmul(test_base_rot_mat,
  #                             hip_positions_in_body_frame.transpose(
  #                                 1, 2)).transpose(1, 2)

  # print("Global Hip Position:\n", hip_position[indices])

  # Mid-air position
  mid_position = torch.clone(hip_position)
  mid_position[..., 2] = (-base_height[:, None] + foot_height)
  #/ projected_gravity[:, 2])[:, None]
  # print("Mid Position World Frame:\n", mid_position[indices])

  # Land position
  base_velocity = base_velocity_world_frame
  # print("Base Velocity World Frame:\n", base_velocity[indices])
  hip_velocity_body_frame = cross_quad(base_angular_velocity_body_frame,
                                       hip_positions_in_body_frame)
  #print("Hip Velocity Body Frame:\n", hip_velocity_body_frame[indices])
  hip_velocity = base_velocity[:, None, :] + torch.matmul(
      base_rot_mat, hip_velocity_body_frame.transpose(1, 2)).transpose(1, 2)
  # print("Hip Velocity World Frame:\n", hip_velocity[indices])

  land_position = hip_velocity * stance_duration[:, :, None] / 2
  land_position[..., 0] = torch.clip(land_position[..., 0], -0.15, 0.15)
  land_position[..., 1] = torch.clip(land_position[..., 1], -0.08, 0.08)
  # print("Land Position Hip Frame:\n", land_position[indices])
  # print("Global Hip Position:\n", hip_position[indices])
  land_position += hip_position
  # print("Land Position Body Frame:\n", land_position[indices])
  land_position[..., 2] = (-base_height[:, None] + foot_landing_clearance)
  # -land_position[..., 0] * projected_gravity[:, 0, None]
  # -land_position[..., 1] * projected_gravity[:, 1, None]
  # ) / projected_gravity[:, 2, None]

  

  mid_position+=mid_residual
  land_position+=land_residual
  # print("Mid residual World Frame:\n", mid_residual[0])
  # print("Land residual World Frame:\n", land_residual[0])
  

  foot_position = _gen_swing_foot_trajectory(normalized_phase,
                                             phase_switch_foot_positions,
                                             mid_position, land_position, highest_phase)
  
  # print("Base Rot Mat:\n", base_rot_mat[indices])
  # print("Normalized Phase:\n", normalized_phase[indices])
  # print("Global Sta Position:\n", phase_switch_foot_positions[indices])
  # print("Global Mid Position:\n", mid_position[indices])
  # print("Global Land Position:\n", land_position[indices])
  # print("Global Des Position:\n", foot_position[indices])
  # input("Any Key...")

  return foot_position


class RaibertSwingLegController:
  """Controls the swing leg position using Raibert's formula.
  For details, please refer to chapter 2 in "Legged robbots that balance" by
  Marc Raibert. The key idea is to stablize the swing foot's location based on
  the CoM moving speed.
  """
  def __init__(self,
               robot: BaseRobot,
               gait_generator: PhaseGaitGenerator,
               gait_config: GaitConfig,):
    self._robot = robot
    self._device = self._robot._device
    self._num_envs = self._robot.num_envs
    self._gait_generator = gait_generator
    self._last_leg_state = gait_generator.desired_contact_state
    self._foot_landing_clearance = gait_config.foot_landing_clearance
    self._desired_base_height = gait_config.desired_base_height
    self._foot_height = gait_config.foot_height
    self._phase_switch_foot_positions = None

    self.mid_residual=torch.zeros_like(self._robot.foot_positions_in_base_frame)
    self.land_residual=torch.zeros_like(self._robot.foot_positions_in_base_frame) 
    self.highest_phase=torch.ones_like(self._gait_generator.normalized_phase)*0.5
    
    # self.mid_x_range=[-0.035,0.035]
    # self.mid_y_range=[-0.1,0.1]
    # self.mid_z_range=[-0.05,0.05]
    # self.land_x_range=[-0.03, 0.07]
    # self.land_y_range=[-0.1,0.1]

    self.mid_x_range=[-0.0,0.0]
    self.mid_y_range=[-0.,0.]
    self.mid_z_range=[-0.,0.]
    self.land_x_range=[-0.03, 0.07]
    self.land_y_range=[-0.1,0.1]
    self.highest_phase_range=[0.2,0.8]


    self.reset()

  def reset(self) -> None:
    self._last_leg_state = torch.clone(
        self._gait_generator.desired_contact_state)
    base_quat = self._robot.base_orientation_quat
    self._phase_switch_foot_positions = torch.matmul(
        self._robot.base_rot_mat,
        self._robot.foot_positions_in_base_frame.transpose(1, 2)).transpose(1, 2)
    # print("Phase Switch Foot Positions in Reset:\n", self._phase_switch_foot_positions)

  def reset_idx(self, env_ids) -> None:
    self._last_leg_state[env_ids] = torch.clone(
        self._gait_generator.desired_contact_state[env_ids])
    self._phase_switch_foot_positions[env_ids] = torch.matmul(
        self._robot.base_rot_mat[env_ids],
        self._robot.foot_positions_in_base_frame[env_ids].transpose(1, 2)).transpose(1, 2)
    # print("Phase Switch Foot Positions in Reset_idx:\n", self._phase_switch_foot_positions)

  def update(self, swing_command) -> None:

    cmd_mid_z_residual=swing_command[:,:4]
    cmd_land_x_residual=swing_command[:,4:8]
    cmd_land_y_residual=swing_command[:,8:12]
    cmd_highest_phase=swing_command[:,12].unsqueeze(1)

    new_leg_state = torch.clone(self._gait_generator.desired_contact_state)
    new_foot_positions = torch.matmul(
        self._robot.base_rot_mat,
        self._robot.foot_positions_in_base_frame.transpose(1, 2)).transpose(1, 2)
    # print(f"foot_positions_in_base_frame\n: {self._robot.foot_positions_in_base_frame}")
    # print(f"new_foot_positions: {new_foot_positions}")
    # input("Any Key...")
    self._phase_switch_foot_positions = torch.where(
        torch.tile((self._last_leg_state == new_leg_state)[:, :, None],
                   [1, 1, 3]), self._phase_switch_foot_positions,
        new_foot_positions)
    

    new_mid_residual=torch.clone(self.mid_residual)
    new_land_residual=torch.clone(self.land_residual)


    new_highest_phase=torch.rand_like(self._gait_generator.normalized_phase) * (self.highest_phase_range[1] - self.highest_phase_range[0]) + self.highest_phase_range[0]
    new_mid_residual[...,0]= torch.rand_like(self.mid_residual[...,0]) * (self.mid_x_range[1] - self.mid_x_range[0]) + self.mid_x_range[0]
    new_mid_residual[...,1]= torch.rand_like(self.mid_residual[...,1]) * (self.mid_y_range[1] - self.mid_y_range[0]) + self.mid_y_range[0]
    new_mid_residual[...,2]= torch.rand_like(self.mid_residual[...,2]) * (self.mid_z_range[1] - self.mid_z_range[0]) + self.mid_z_range[0]
    new_land_residual[...,0]= torch.rand_like(self.land_residual[...,0]) * (self.land_x_range[1] - self.land_x_range[0]) + self.land_x_range[0]
    new_land_residual[...,1]= torch.rand_like(self.land_residual[...,1]) * (self.land_y_range[1] - self.land_y_range[0]) + self.land_y_range[0]

    #print('new_mid_residual:', new_mid_residual[0])
    new_highest_phase=cmd_highest_phase
    new_mid_residual[...,2]=cmd_mid_z_residual
    new_land_residual[...,0]=cmd_land_x_residual
    new_land_residual[...,1]=cmd_land_y_residual

    self.highest_phase=torch.where(self._last_leg_state == new_leg_state,
                    self.highest_phase, new_highest_phase)
    self.mid_residual=torch.where(torch.tile((self._last_leg_state == new_leg_state)[:, :, None],
                   [1, 1, 3]), self.mid_residual, new_mid_residual)
    self.land_residual=torch.where(torch.tile((self._last_leg_state == new_leg_state)[:, :, None],
                   [1, 1, 3]), self.land_residual, new_land_residual)
    #print('mid_residual:', self.mid_residual[0])

    self._last_leg_state = new_leg_state

  @property
  def desired_foot_positions(self):
    """Computes desired foot positions in WORLD frame centered at robot base.

    Note: it returns an invalid position for stance legs.
    """
    # print('mid_residual:', self.mid_residual)
    # print('land_residual:', self.land_residual)
    # print('highest_phase:', self.highest_phase)
    return compute_desired_foot_positions(
        self._robot.base_rot_mat,
        self._robot.base_position[:, 2],
        self._robot.hip_positions_in_body_frame,
        self._robot.base_velocity_world_frame,
        self._robot.base_angular_velocity_body_frame,
        self._robot.projected_gravity,
        self._desired_base_height,
        self._foot_height,
        self._foot_landing_clearance,
        self._gait_generator.stance_duration,
        self._gait_generator.normalized_phase,
        self._phase_switch_foot_positions,
        self.mid_residual,
        self.land_residual,
        self.highest_phase,
    )
