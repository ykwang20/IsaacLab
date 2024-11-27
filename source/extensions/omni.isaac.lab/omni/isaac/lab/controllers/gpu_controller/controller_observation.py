import torch
# from mqe.envs.go1.go1 import Go1
import omni.isaac.lab.utils.math as math_utils
import numpy as np

class ControllerObservation:
    def __init__(self, env,asset):

        self.env = env
        self.num_robot = self.env.num_envs
        self.device = self.env.device
        self.asset=asset
        self.feet_names = [".*_foot"]
        self.feet_ids=self.asset.find_bodies(self.feet_names)[0]
        print("Feets: ", self.asset.find_bodies(self.feet_names))
        self.feet_ids=[14,13,16,15]
        print("Feet IDs: ", self.asset.find_bodies(self.feet_names))
        self.FR_joint_ids=self.asset.find_joints('FR_.*')[0]
        self.FR_body_ids=self.asset.find_bodies('FR_.*')[0]
        print("FR IDS: ", self.asset.find_joints('FR_.*'))
        print("FR BODY IDS: ", self.asset.find_bodies('FR_.*'))
        self.FL_joint_ids=self.asset.find_joints('FL_.*')[0]
        self.FL_body_ids=self.asset.find_bodies('FL_.*')[0]
        print("FL IDS: ", self.asset.find_joints('FL_.*'))
        print("FL BODY IDS: ", self.asset.find_bodies('FL_.*'))
        self.RR_joint_ids=self.asset.find_joints('RR_.*')[0]
        self.RR_body_ids=self.asset.find_bodies('RR_.*')[0]
        print("RR IDS: ", self.asset.find_joints('RR_.*'))
        print("RR BODY IDS: ", self.asset.find_bodies('RR_.*'))
        self.RL_joint_ids=self.asset.find_joints('RL_.*')[0]
        self.RL_body_ids=self.asset.find_bodies('RL_.*')[0]
        print("RL IDS: ", self.asset.find_joints('RL_.*'))
        print("RL BODY IDS: ", self.asset.find_bodies('RL_.*'))
        self.gym_joint_id=self.FR_joint_ids+self.FL_joint_ids+self.RR_joint_ids+self.RL_joint_ids
        self.gym_body_id=[0]+self.FR_body_ids+self.FL_body_ids+self.RR_body_ids+self.RL_body_ids
        print("Gym Joint ID: ", self.gym_joint_id)
        print("Gym Body ID: ", self.gym_body_id)
        
        self.update(True)

    def update(self, initializing=False):
        if not initializing:
            self.time_since_reset = (self.env.episode_length_buf*self.env.step_dt).view(-1)
        else:
            self.time_since_reset=torch.zeros((self.num_robot), device=self.device)
        self.base_position = self.asset.data.root_pos_w
        self.base_quat = self.asset.data.root_quat_w[:,[1,2,3,0]]
        self.base_lin_vel_world = self.asset.data.root_lin_vel_w
        self.base_ang_vel_world = self.asset.data.root_ang_vel_w
        self.motor_positions = self.asset.data.joint_pos[:,self.gym_joint_id].view(self.num_robot, -1)
        self.motor_velocities = self.asset.data.joint_vel[:,self.gym_joint_id].view(self.num_robot, -1)
        self.foot_positions=self.asset.data.body_pos_w[:, self.feet_ids].view(self.num_robot, -1, 3)
        #print('body pos shape:',self.asset.data.body_pos_w.shape)
        jacobian_raw=self.asset.root_physx_view.get_jacobians()
        gym_joint_id_extended=[0,1,2,3,4,5]+[x+6 for x in self.gym_joint_id]
        # print("first shape:",jacobian_raw[:,self.gym_body_id,:,:6].shape)
        # print("jacobian shape:",jacobian_raw.shape)
        # print('length of gym_joint_id_extended:',len(gym_joint_id_extended), gym_joint_id_extended)
        # print("gym_body_id shape:", torch.tensor(self.gym_body_id).shape)
        # print("jacobian_raw shape:", jacobian_raw.shape)
        # print("gym_joint_id_extended length:", len(gym_joint_id_extended))

        #print("second shape:",jacobian_raw[:,self.gym_body_id,:,gym_joint_id_extended].shape)
        #self.jacobian = torch.cat((jacobian_raw[:,self.gym_body_id,:,:6],jacobian_raw[:,self.gym_body_id,:,gym_joint_id_extended]),dim=-1)
        self.jacobian = jacobian_raw[:,:,:,gym_joint_id_extended]
        self.jacobian = self.jacobian[:,self.gym_body_id, :,:]
        indices = torch.tensor([0,], device=self.device)
    
        # print("Time Since Reset: ", self.time_since_reset[indices])
        # print("Base Position:\n", self.base_position[indices])
        # print("Base Quat:\n", self.base_quat[indices])
        # # print("Base Linear Velocity World:\n", self.base_lin_vel_world[indices])
        # # print("Base Angular Velocity World:\n", self.base_ang_vel_world[indices])
        # print("Foot Positions:\n", self.foot_positions[indices])
        # print("Motor Positions:\n", self.motor_positions[indices])
        # print("Motor Velocities:\n", self.motor_velocities[indices])
        # print("Jacobian size:\n", self.jacobian[indices].shape)
        # print("Jacobian root body:\n", self.jacobian[indices,0,:,:])
        # print("Jacobian FR:\n", self.jacobian[indices,1:5,:,:])
        # print("Jacobian FL:\n", self.jacobian[indices,5:9,:,:])
        # print("Jacobian RR:\n", self.jacobian[indices,9:13,:,:])
        # print("Jacobian RL:\n", self.jacobian[indices,13:17,:,:])


        torch.set_printoptions(profile="full")
        torch.set_printoptions(sci_mode=False)
        indices = torch.tensor([1], device=self.device)
        # print("Foot Positions:\n", self.foot_positions[indices])
        # print("Time Since Reset: ", self.time_since_reset[indices])
        # print("Base Position:\n", self.base_position[indices])
        # print("Base Quat:\n", self.base_quat[indices])
        # print("Base Linear Velocity World:\n", self.base_lin_vel_world[indices])
        # print("Base Angular Velocity World:\n", self.base_ang_vel_world[indices])
        # print("Foot Positions:\n", self.foot_positions[indices])
        # print("Motor Positions:\n", self.motor_positions[indices])
        # print("Motor Velocities:\n", self.motor_velocities[indices])
        # print("Jacobian:\n", self.jacobian[indices])
        # input("Any Key...")
        # print("The Shape of all the variables are as follows:")
        # print("time_since_reset: ", self.time_since_reset.shape)
        # print("base_position: ", self.base_position.shape)
        # print("base_quat: ", self.base_quat.shape)
        # print("base_lin_vel_world: ", self.base_lin_vel_world.shape)
        # print("base_ang_vel_world: ", self.base_ang_vel_world.shape)
        # print("motor_positions: ", self.motor_positions.shape)
        # print("motor_velocities: ", self.motor_velocities.shape)
        # print("foot_positions: ", self.foot_positions.shape)

        # self.time_since_reset = torch.zeros(self._num_robot, device=self._device)
        # self.base_position = torch.zeros((self.num_robot, 3), device=self.device)
        # self.base_quat = torch.tensor([[0., 0., 0., 1.]] * self.num_robot, device=self.device)
        # self.base_lin_vel_world = torch.zeros((self.num_robot, 3), device=self.device)
        # self.base_ang_vel_world = torch.zeros((self.num_robot, 3), device=self.device)
        # self.motor_positions = torch.zeros((self.num_robot, 12), device=self.device)
        # self.motor_velocities = torch.zeros((self.num_robot, 12), device=self.device)
        # self.foot_positions = torch.zeros((self.num_robot, 4, 3), device=self.device)
        # self.jacobian = torch.zeros((self.num_robot, 17, 6, 18), device=self.device)

