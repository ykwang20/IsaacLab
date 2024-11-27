import torch
from .controller_observation import ControllerObservation
from .sim_robot import SimRobot
from .phase_gait_generator import PhaseGaitGenerator
from .qp_torque_optimizer import QPTorqueOptimizer
from .raibert_swing_leg_controller import RaibertSwingLegController
from .gait_config import GaitConfig
from .motors import MotorCommand
import matplotlib.pyplot as plt
import numpy as np


class LocomotionController:
    def __init__(self, env, asset):
        self._obs = ControllerObservation(env, asset)
        self._robot = SimRobot(self._obs)
        self._gait_config = GaitConfig()
        self._gait_generator = PhaseGaitGenerator(self._robot, self._gait_config)
        self._swing_leg_controller = RaibertSwingLegController(self._robot, self._gait_generator, self._gait_config)
        self._torque_optimizer = QPTorqueOptimizer(self._robot,
                                                   foot_friction_coef=0.4,
                                                   desired_body_height=self._gait_config.desired_base_height,
                                                   use_full_qp=False,
                                                   clip_grf=True)
        
        self._velocity_command = torch.zeros((self._robot._num_envs, 3), device=self._robot._device)
        self._max_velocity = torch.tensor(self._gait_config.max_velocity, device=self._robot._device)
        self._zero = torch.zeros(self._robot._num_envs, device=self._robot._device)
        self._motor_command: MotorCommand = None
        self._torques = torch.zeros((self._robot._num_envs, 12), device=self._robot._device)
        self.torques_logger=[]
        self.angles_logger=[]
        self.time_steps=500

    def set_command(self, velocity_command: torch.Tensor):
        #TODO: the frequency between these two needs to be tuned
        self._obs.update()
        self._robot.update()
        self._gait_generator.update()
        self._swing_leg_controller.update()
        self._velocity_command += velocity_command
        self._velocity_command[:] = torch.clip(self._velocity_command, -self._max_velocity, self._max_velocity)
        # print("Velocity Command: ", self._velocity_command)
        self._velocity_command[:, 0] = 0.4
        self._velocity_command[:, 1] = 0.
        self._velocity_command[:, 2] = 0
        self._torque_optimizer.desired_linear_velocity = torch.stack((self._velocity_command[:, 0], self._velocity_command[:, 1], self._zero), dim=1)
        self._torque_optimizer.desired_angular_velocity = torch.stack((self._zero, self._zero, self._velocity_command[:, 2]), dim=1)

    def compute_torques(self):
        #TODO: the frequency between these two needs to be tuned
        
        #print('desired contact state:', self._gait_generator.desired_contact_state)
        self._motor_command, _, _, _, _ = self._torque_optimizer.get_action(
            foot_contact_state=self._gait_generator.desired_contact_state,
            swing_foot_position=self._swing_leg_controller.desired_foot_positions)
        # print("Motor Command: ", self._motor_command)
        torques=self.motor_command_to_torques()
        #self.add_torque_data(torques[0], self._obs.motor_positions[0])
        lab_torques=torques[:,[3,0,9,6,4,1,10,7,5,2,11,8]]
        # lab_torques[:,1]=-lab_torques[:,1]
        # lab_torques[:,3]=-lab_torques[:,3]
       
        gym_torques=lab_torques[:,self._obs.gym_joint_id]
        # print('lab torques:', lab_torques)
        # print('gym torques:', gym_torques)
        # print('difference:', torch.sum(gym_torques-torques))
        #torques=torch.zeros_like(torques)
        return lab_torques

    def reset(self, env_ids):
        robot_ids = torch.empty(2 * len(env_ids), dtype=env_ids.dtype, device=env_ids.device)
        robot_ids[0::2] = env_ids * 2
        robot_ids[1::2] = env_ids * 2 + 1
        self._obs.update()
        self._robot.update()
        self._velocity_command[robot_ids, :] = 0
        self._swing_leg_controller.reset_idx(robot_ids)
        self._gait_generator.reset_idx(robot_ids)
        # print("Robot IDs: ", robot_ids)
        # print("Velocity Command: ", self._velocity_command)

    def motor_command_to_torques(self, update_obs=False):
        if update_obs:
            self._obs.update()
        self._torques, _ = self._robot.compute_troques_from_motor_actions(self._motor_command)
        return self._torques



    def add_torque_data(self,new_torque,new_angle):
        self.torques_logger.append(new_torque.cpu().numpy())
        self.angles_logger.append(new_angle.cpu().numpy().copy())
        if len(self.torques_logger) >= self.time_steps:
            # 当数据达到 1000 个数据点时，绘制并存储图像
            self.plot_and_save_torques()

    def plot_and_save_torques(self):
        # 转换为 numpy 数组以便于处理
        torques_array = np.array(self.torques_logger)
        angles_array = np.array(self.angles_logger)

        # print('toeque shape',torques_array.shape)
        # print('amgle shape',angles_array.shape)
        #print('torque log',self.angles_logger)
        # 获取前三个维度的力矩值
        # torque_1 = torques_array[:, 3]
        # torque_2 = torques_array[:, 4]
        # torque_3 = torques_array[:, 5]
        # angle_1 = angles_array[:, 3]
        # angle_2 = angles_array[:, 4]
        # angle_3 = angles_array[:, 5]
        torque_1 = torques_array[:, 0]
        torque_2 = torques_array[:, 1]
        torque_3 = torques_array[:, 2]
        angle_1 = angles_array[:, 0]
        angle_2 = angles_array[:, 1]
        angle_3 = angles_array[:, 2]
        print('angle_array',angles_array)
        print('angle_1',angle_1)
        # 生成时间步
        time = np.arange(torque_1.shape[0])

        # 绘制三个关节的曲线，每个关节在单独的图中
        plt.figure(figsize=(15, 12))

       # 绘制第一个关节的力矩和角度曲线
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(torque_1, label='Torque 1', color='b')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Torque Value', color='b')
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(angle_1, label='Angle 1', color='c', linestyle='--')
        ax2.set_ylabel('Angle Value', color='c')

        # 绘制第二个关节的力矩和角度曲线
        ax3 = plt.subplot(3, 1, 2)
        ax3.plot(torque_2, label='Torque 2', color='g')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Torque Value', color='g')
        ax3.grid(True)
        ax4 = ax3.twinx()
        ax4.plot(angle_2, label='Angle 2', color='y', linestyle='--')
        ax4.set_ylabel('Angle Value', color='y')

        # 绘制第三个关节的力矩和角度曲线
        ax5 = plt.subplot(3, 1, 3)
        ax5.plot(torque_3, label='Torque 3', color='r')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Torque Value', color='r')
        ax5.grid(True)
        ax6 = ax5.twinx()
        ax6.plot(angle_3, label='Angle 3', color='m', linestyle='--')
        ax6.set_ylabel('Angle Value', color='m')

        # 调整布局并保存图表
        plt.tight_layout()
        plt.savefig('torques_and_angles_plot.png')
        plt.close()




