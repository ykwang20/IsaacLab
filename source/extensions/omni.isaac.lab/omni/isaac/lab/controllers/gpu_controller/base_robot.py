from omni.isaac.lab.utils.array import convert_to_torch
import torch
from .rotation_utils import quat_to_rot_mat, get_euler_xyz_from_quaternion, angle_normalize
from .controller_observation import ControllerObservation
from .motors import MotorControlMode, MotorGroup, MotorModel, MotorCommand


class BaseRobot:
    def __init__(self, obs: ControllerObservation):
        self._obs = obs
        self._num_envs = obs.num_robot
        self._device = obs.device
        self._motors = MotorGroup(device=self._device,
                            num_envs=self._num_envs,
                            motors=(
                                MotorModel(
                                    name="FR_hip_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=0.0,
                                    min_position=-0.802851455917,
                                    max_position=0.802851455917,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="FR_thigh_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=0.9,
                                    min_position=-1.0471975512,
                                    max_position=4.18879020479,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="FR_calf_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=-1.8,
                                    min_position=-2.6965336943,
                                    max_position=-0.916297857297,
                                    min_velocity=-20,
                                    max_velocity=20,
                                    min_torque=-35.55,
                                    max_torque=35.55,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="FL_hip_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=0.0,
                                    min_position=-0.802851455917,
                                    max_position=0.802851455917,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="FL_thigh_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=0.9,
                                    min_position=-1.0471975512,
                                    max_position=4.18879020479,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="FL_calf_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=-1.8,
                                    min_position=-1.0471975512,
                                    max_position=4.18879020479,
                                    min_velocity=-20,
                                    max_velocity=20,
                                    min_torque=-35.55,
                                    max_torque=35.55,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RR_hip_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=0.0,
                                    min_position=-0.802851455917,
                                    max_position=0.802851455917,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RR_thigh_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=0.9,
                                    min_position=-1.0471975512,
                                    max_position=4.18879020479,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RR_calf_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=-1.8,
                                    min_position=-2.6965336943,
                                    max_position=-0.916297857297,
                                    min_velocity=-20,
                                    max_velocity=20,
                                    min_torque=-35.55,
                                    max_torque=35.55,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RL_hip_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=0.0,
                                    min_position=-0.802851455917,
                                    max_position=0.802851455917,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RL_thigh_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=0.9,
                                    min_position=-1.0471975512,
                                    max_position=4.18879020479,
                                    min_velocity=-30,
                                    max_velocity=30,
                                    min_torque=-23.7,
                                    max_torque=23.7,
                                    kp=100,
                                    kd=1,
                                ),
                                MotorModel(
                                    name="RL_calf_joint",
                                    motor_control_mode=MotorControlMode.HYBRID,
                                    init_position=-1.8,
                                    min_position=-2.6965336943,
                                    max_position=-0.916297857297,
                                    min_velocity=-20,
                                    max_velocity=20,
                                    min_torque=-35.55,
                                    max_torque=35.55,
                                    kp=100,
                                    kd=1,
                                ),
                            ),
                            torque_delay_steps=False)
        
        self._base_rot_mat = quat_to_rot_mat(self._obs.base_quat)
        self._base_rot_mat_t = torch.transpose(self._base_rot_mat, 1, 2)
        self._gravity_vec = torch.stack([convert_to_torch([0., 0., 1.], device=self._device)] * self._num_envs)
        self._projected_gravity = torch.bmm(self._base_rot_mat_t, self._gravity_vec[:, :, None])[:, :, 0]

    def update(self):
        self._base_rot_mat = quat_to_rot_mat(self._obs.base_quat)
        # print("Base Rotation Matrix:\n", self._base_rot_mat)
        self._base_rot_mat_t = torch.transpose(self._base_rot_mat, 1, 2)
        self._projected_gravity = torch.bmm(self._base_rot_mat_t, self._gravity_vec[:, :, None])[:, :, 0]

    def get_motor_angles_from_foot_positions(self, foot_local_positions):
        raise NotImplementedError()

    def compute_troques_from_motor_actions(self, motor_command: MotorCommand):
        return self._motors.convert_to_torque(motor_command, self.motor_positions, self.motor_velocities)
    
    @property
    def time_since_reset(self):
        return torch.clone(self._obs.time_since_reset)

    @property
    def base_position(self):
        return self._obs.base_position

    @property
    def base_orientation_quat(self):
        return self._obs.base_quat

    @property
    def base_orientation_rpy(self):
        return angle_normalize(
            get_euler_xyz_from_quaternion(self._obs.base_quat))
    
    @property
    def base_rot_mat(self):
        return self._base_rot_mat

    @property
    def base_rot_mat_t(self):
        return self._base_rot_mat_t

    @property
    def foot_positions_in_base_frame(self):
        foot_positions_world_frame = self._obs.foot_positions
        base_position_world_frame = self._obs.base_position
        # num_env x 4 x 3
        foot_position = (foot_positions_world_frame -
                        base_position_world_frame[:, None, :])
        # print("Base Position in world frame:\n", base_position_world_frame)
        # print("Foot Position in world frame:\n", foot_positions_world_frame)
        # print("Foot Position relative in world frame:\n", foot_position)
        return torch.matmul(self._base_rot_mat_t,
                            foot_position.transpose(1, 2)).transpose(1, 2)

    @property
    def hip_offset(self):
        """Position of hip offset in base frame, used for IK only."""
        return self._hip_offset
    
    @property
    def hip_positions_in_body_frame(self):
        return self._hip_positions_in_body_frame


    @property
    def base_velocity_world_frame(self):
        return self._obs.base_lin_vel_world

    @property
    def base_velocity_body_frame(self):
        return torch.bmm(self._base_rot_mat_t, self._obs.base_lin_vel_world[:, :, None])[:, :, 0]

    @property
    def base_angular_velocity_body_frame(self):
        return torch.bmm(self._base_rot_mat_t, self._obs.base_ang_vel_world[:, :, None])[:, :, 0]

    @property
    def motor_positions(self):
        return torch.clone(self._obs.motor_positions)

    @property
    def motor_velocities(self):
        return torch.clone(self._obs.motor_velocities)

    @property
    def motor_group(self):
        return self._motors

    @property
    def projected_gravity(self):
        return self._projected_gravity

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def all_foot_jacobian(self):
        rot_mat_t = self.base_rot_mat_t
        jacobian = torch.zeros((self._num_envs, 12, 12), device=self._device)
        jacobian[:, :3, :3] = torch.bmm(rot_mat_t, self._obs.jacobian[:, 4, :3, 6:9])
        jacobian[:, 3:6, 3:6] = torch.bmm(rot_mat_t, self._obs.jacobian[:, 8, :3, 9:12])
        jacobian[:, 6:9, 6:9] = torch.bmm(rot_mat_t, self._obs.jacobian[:, 12, :3, 12:15])
        jacobian[:, 9:12, 9:12] = torch.bmm(rot_mat_t, self._obs.jacobian[:, 16, :3, 15:18])
        # jacobian[:, :3, :3] = torch.bmm(rot_mat_t, self._obs.jacobian[:, 8, :3, 9:12])
        # jacobian[:, 3:6, 3:6] = torch.bmm(rot_mat_t, self._obs.jacobian[:, 4, :3, 6:9])
        # jacobian[:, 6:9, 6:9] = torch.bmm(rot_mat_t, self._obs.jacobian[:, 16, :3, 15:18])
        # jacobian[:, 9:12, 9:12] = torch.bmm(rot_mat_t, self._obs.jacobian[:, 12, :3, 12:15])
        #print("Jacobian:\n", jacobian[:, :,:])
        #input("Any Key...")
        return jacobian









