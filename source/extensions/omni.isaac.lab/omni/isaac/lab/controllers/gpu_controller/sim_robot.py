
from omni.isaac.lab.utils.array import convert_to_torch
import torch
from .controller_observation import ControllerObservation
from .base_robot import BaseRobot


@torch.jit.script
def motor_angles_from_foot_positions(foot_local_positions,
                                     hip_offset,
                                     device: str = "cuda"):
  foot_positions_in_hip_frame = foot_local_positions - hip_offset
  l_up = 0.213
  l_low = 0.233
  l_hip = 0.08 * torch.tensor([-1, 1, -1, 1], device=device)

  x = foot_positions_in_hip_frame[:, :, 0]
  y = foot_positions_in_hip_frame[:, :, 1]
  z = foot_positions_in_hip_frame[:, :, 2]
  theta_knee = -torch.arccos(
      torch.clip((x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
                 (2 * l_low * l_up), -1, 1))
  l = torch.sqrt(
      torch.clip(l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee),
                 1e-7, 1))
  theta_hip = torch.arcsin(torch.clip(-x / l, -1, 1)) - theta_knee / 2
  c1 = l_hip * y - l * torch.cos(theta_hip + theta_knee / 2) * z
  s1 = l * torch.cos(theta_hip + theta_knee / 2) * y + l_hip * z
  theta_ab = torch.arctan2(s1, c1)

  # thetas: num_envs x 4
  joint_angles = torch.stack([theta_ab, theta_hip, theta_knee], dim=2)
  return joint_angles.reshape((-1, 12))


class SimRobot(BaseRobot):
    def __init__(self, obs: ControllerObservation):
        super().__init__(obs)
        com_offset = -convert_to_torch([0.011611, 0.004437, 0.000108],
                            device=self._device)
        self._hip_offset = convert_to_torch(
            [[0.1881, -0.04675, 0.], [0.1881, 0.04675, 0.], [-0.1881, -0.04675, 0.],
            [-0.1881, 0.04675, 0.]],
            device=self._device) + com_offset
        delta_x, delta_y = 0.0, 0.0
        hip_position_single = convert_to_torch((
            (0.1835 + delta_x, -0.131 - delta_y, 0),
            (0.1835 + delta_x, 0.122 + delta_y, 0),
            (-0.1926 - delta_x, -0.131 - delta_y, 0),
            (-0.1926 - delta_x, 0.122 + delta_y, 0),), device=self._device)
        self._hip_positions_in_body_frame = torch.stack([hip_position_single] *
                                                        self._num_envs,
                                                        dim=0)
        
        self._hip_offset = convert_to_torch(
            [[0.1881, -0.04675, 0.], [0.1881, 0.04675, 0.], [-0.1881, -0.04675, 0.],
            [-0.1881, 0.04675, 0.]],
            device=self._device)
        hip_position_single = convert_to_torch([[0.1881, -0.12675, 0],
                                        [0.1881, 0.12675, 0],
                                        [-0.1881, -0.12675, 0],
                                        [-0.1881, 0.12675, 0]], device=self._device)
        self._hip_positions_in_body_frame = torch.stack([hip_position_single] * self._num_envs, dim=0)

    def get_motor_angles_from_foot_positions(self, foot_local_positions):
        return motor_angles_from_foot_positions(foot_local_positions,
                                                self.hip_offset,
                                                device=self._device)








