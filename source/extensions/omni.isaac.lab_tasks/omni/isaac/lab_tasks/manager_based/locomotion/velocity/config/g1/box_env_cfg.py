# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils.noise import UniformEulerNoiseOnQuatCfg as EulerNoise
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.sensors import  patterns

from omni.isaac.lab.envs.common import ViewerCfg

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets import G1_MINIMAL_CFG, G1_CFG, G1_29_CFG,G1_29_MINIMAL_CFG,G1_29_MODIFIED_CFG,G1_29_MODIFIED_MIN_CFG,G1_29_ANNEAL_23_CFG,G1_29_ANNEAL_23_MODIFIED_CFG , G1_29_MODIFIED_714_CFG# isort: skip
import omni.isaac.lab.terrains as terrain_gen
import math
import random


@configclass
class G1Rewards:
    """Reward terms for the MDP."""
    #TODO: add termination penalty and replace alive reward
    # -- task
    # position_tracking = RewTerm(func=mdp.position_tracking, weight=4.,
    #                              params={"command_name": "target_pos_e","start_time": 0})
    # position_tracking_cos = RewTerm(func=mdp.position_tracking_cos, weight=20.,
    #                               params={"command_name": "target_pos_e","start_time": 1})
    # standing_joint = RewTerm(func=mdp.standing_joint_deviation, weight=0.5,#-0.005,
    #                          params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})#[".*shoulder.*",".*elbow.*",".*wrist.*"])})
    # standing_orientation = RewTerm(func=mdp.standing_flat_orientation, weight=0.5,params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),})
    # standing_lin_vel = RewTerm(func=mdp.standing_lin_vel, weight=0.5,params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),})
    # standing_ang_vel = RewTerm(func=mdp.standing_ang_vel, weight=0.5,params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),})
    # standing_height = RewTerm(func=mdp.standing_height_l2, weight=0.5,params={"desired_height":0.78})
    standing_joint = RewTerm(func=mdp.standing_joint_deviation, weight=2,#-0.005,
                             params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})#[".*shoulder.*",".*elbow.*",".*wrist.*"])})
    standing_orientation = RewTerm(func=mdp.standing_flat_orientation, weight=2,params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),})
    standing_lin_vel = RewTerm(func=mdp.standing_lin_vel, weight=2,params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),})
    standing_ang_vel = RewTerm(func=mdp.standing_ang_vel, weight=2,params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),})
    standing_height = RewTerm(func=mdp.standing_height_l2, weight=1,params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),"desired_height":0.78})

    downward_penalty = RewTerm(func=mdp.downward_penalty, weight=-4, params={"asset_cfg" :SceneEntityCfg("robot", body_names=[".*wrist.*", ".*elbow_link", "head_link"]),})
    backward_penalty = RewTerm(func=mdp.com_backward_penalty, weight=-2, params={"wall_x" :1.5})#-2
    # downward_penalty = RewTerm(func=mdp.downward_to_torso_penalty, weight=-4, params={"asset_cfg" :SceneEntityCfg("robot", body_names=[".*hip.*", ".*knee.*", ".*ankle.*"]),
    #                                                                          "torso_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),})
    #down_and_back_penalty = RewTerm(func=mdp.down_and_back_penalty, weight = -6, params={"wall_x":1.5})
    alive_reward=RewTerm(func=mdp.is_alive, weight=9)
    wait_penalty = RewTerm(func=mdp.wait_penalty, weight=-1,params={"command_name": "target_pos_e"}) #weight=-1
    #move_in_direction = RewTerm(func=mdp.move_in_direction, weight=5.0,params={"command_name": "target_pos_e"})
    #move_in_direction = RewTerm(func=mdp.move_in_direction, weight=1.0,params={"command_name": "target_pos_e"})
    #termination_penalty = RewTerm(func=mdp.contact_terminated, weight=-200.0)


    # Contact penalties
    pressure_penalty=RewTerm(func=mdp.body_pressure, weight = -1, params = {"asset_cfg" :SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
                                                                            "sensor_cfg": SceneEntityCfg("bodies_ground_contact",
                                                                            body_names=[ ".*_ankle_roll_link"]),})
    contact_on_wall_penalty = RewTerm(func=mdp.contact_on_wall, weight=-1,params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*"]), 
                                      "wall_x": 1.5})
    contact_exp_penalty=RewTerm(func=mdp.contact_forces_exp, weight=-1,
                            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*"]), 
                                     "threshold": 1000.0,"grad_scale":0.01})
                                    #"threshold": 500.0,"grad_scale":0.01})
    # air_penalty = RewTerm(func=mdp.body_air_time, weight=-1,params={"sensor_cfg": SceneEntityCfg("contact_forces",
    #                                          body_names=[ ".*"]), "threshold": 1.0,})
    head_contact_penalty = RewTerm(func=mdp.contact_forces_exp, weight=-1,params={"sensor_cfg": SceneEntityCfg("contact_forces",  body_names=["head_link"]), 
                                      "threshold": 0,"grad_scale":0.1})
    group_air_penalty = RewTerm(func=mdp.group_air_time, weight=-1,params={"upper_sensor_cfg": SceneEntityCfg("bodies_ground_contact",
                                                                            body_names=[ ".*wrist.*",".*shoulder.*",".*elbow_link",]), 
                                                                            "lower_sensor_cfg": SceneEntityCfg("bodies_ground_contact",
                                                                            body_names=[ ".*_hip_yaw_link",".*_knee_link",".*_ankle_roll_link"]),
                                                                            "feet_sensor_cfg": SceneEntityCfg("bodies_ground_contact",
                                                                            body_names=[ ".*_ankle_roll_link",]),
                                                                            "threshold": 1.0,})
    # knee_air_time_penalty = RewTerm(func=mdp.knee_air_time, weight=-1,params={
    #                                         "sensor_cfg": SceneEntityCfg("contact_forces",
    #                                         #  body_names=[ ".*_hip_roll_link",".*_hip_yaw_link",".*_knee_link", "torso_link","pelvis_contour_link"]),
    #                                          body_names=[ ".*_hip_yaw_link",".*_knee_link",]),
    #                                          "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*_ankle_roll_link"]),
    #                                           "torso_bodies_sensor_cfg": SceneEntityCfg("torso_bodies_contact",
    #                                          body_names=[ "torso_link"]),
    #                                          "threshold": 1.0,})
    #this leads to quick motion
    #knee_height_reward = RewTerm(func=mdp.knee_height, weight=1) 

    # body_drag_penalty = RewTerm(func=mdp.body_dragging, weight=-1,params={"vel_threshold": 0.1 ,"asset_cfg" :SceneEntityCfg("robot", body_names=[".*"]),
    #                                                                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*"])})
    body_slipping_penalty = RewTerm(func=mdp.body_slipping, weight=-0.1,params={"asset_cfg" :SceneEntityCfg("robot", body_names=[".*"]),
                                                                                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*"])})
    # body_slipping_penalty = RewTerm(func=mdp.body_slipping, weight=-1,params={"asset_cfg" :SceneEntityCfg("robot", body_names=[".*"]),
    #                                                                             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*"])})
    
    # Joint related penalties
    #action_rate_penalty=RewTerm(func=mdp.processed_action_rate_l2, weight=-0.0002,params={"action_name":"joint_pos"})
    #action_rate_penalty=RewTerm(func=mdp.processed_action_rate_l2, weight=-0.002,params={"action_name":"joint_pos"})
    action_rate_penalty=RewTerm(func=mdp.processed_action_rate_l2, weight=-0.2,params={"action_name":"joint_pos"})

    #joint_deviation=RewTerm(func=mdp.joint_deviation_l1, weight=-0.005,#-0.005,
    #                         params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    # wrist_deviation_exp=RewTerm(func=mdp.joint_deviation_exp, weight=-1.,
    #                             params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*wrist.*"]), "scale": 10, "threshold": 0.5})
    joint_pos_limits =RewTerm(func=mdp.joint_pos_limits, weight=-10,)#-1,)
    #joint_vel_penalty=RewTerm(func=mdp.joint_vel_l2, weight=-0.0001,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])} )
    joint_vel_penalty=RewTerm(func=mdp.joint_vel_l2, weight=-0.001,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])} )

    #joint_vel_clip_penalty=RewTerm(func=mdp.joint_vel_clip, weight=-1,params={"threshold": 3, "asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])} )
    #joint_vel_exp_penalty=RewTerm(func=mdp.joint_vel_exp, weight=-1,params={"grad_scale": 1,"threshold": 10, "asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])} )
    joint_vel_lim_penalty=RewTerm(func=mdp.joint_velocity_limits, weight=-1, params={"soft_ratio": 0.9, "asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    joint_acc_penalty=RewTerm(func=mdp.joint_acc_l2, weight=-2e-8,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    #joint_vel_lim_penalty=RewTerm(func=mdp.joint_velocity_limits, weight=-0.1, params={"soft_ratio": 1., "asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})

    torque_penalty=RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-5,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    #torque_lim_penalty=RewTerm(func=mdp.applied_torque_limits, weight=-0.002,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])} )

    power_penalty=RewTerm(func=mdp.power_consumption, weight=-0.00001)

    # Rigid body penalties
    #base_vel_penalty=RewTerm(func=mdp.base_lin_vel_clip, weight=-5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.005)    
    base_acc_penalty=RewTerm(func=mdp.base_lin_ang_acc, weight=-0.0001)
    #feet_acc_penalty=RewTerm(func=mdp.feet_acc, weight=-0.00002)
    rigid_body_acc_penalty=RewTerm(func=mdp.body_lin_acc_l2, weight=-0.0002,params={"asset_cfg" :SceneEntityCfg("robot", 
                                            body_names=[".*"])})

   
    #curiosity_rnd = RewTerm(func=mdp.curiosity, weight=200)
    #curiosity_cnt = RewTerm(func=mdp.curiosity_cnt, weight=2000)



    

@configclass
class TargetCommandsCfg:
    """Command specifications for the MDP."""

    target_pos_e = mdp.TargetCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        #radius_range=((2., 3.)),
        radius_range=((1.75, 3.)),
        debug_vis=True,
        success_threshold=0.06,
    )

@configclass
class ClimbCommandsCfg:
    """Command specifications for the MDP."""

    climb_command = mdp.ClimbCommandCfg(
        activated=True,
        asset_name="robot",
        resampling_time_range=(1., 1.),
        debug_vis=False,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0), lin_vel_y=(-0., 0.), ang_vel_z=(0.0, 0.0), heading=(-0.1*math.pi, 0.1*math.pi)
        ),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    height_low = DoneTerm(func=mdp.root_height_below_minimum, params={'minimum_height':0.45})
    #TODO: only for standing
    # base_contact = DoneTerm(
    #     func=mdp.standing_illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link",".*hip.*"]), "threshold": 1.0},     
    # )
    # success = DoneTerm(func=mdp.stepped_on,params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "threshold": 1.0,
    #                   "platform_width": 3,"reached_distance": 0.06,} )
    # max_consecutive_success = DoneTerm(
    #     func=mdp.max_consecutive_success, params={"num_success": 1, }#params={"num_success": 50, }
    # )
    # max_contact= DoneTerm(func=mdp.max_contact_force, 
    #                         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*"]), "threshold": 2000.0})
    # on_air= DoneTerm(func=mdp.on_air,params={"sensor_cfg":
    #                                           SceneEntityCfg("contact_forces", body_names=[ ".*_elbow_link",".*_wrist_yaw_link",".*_hip_yaw_link",".*_ankle_roll_link",".*_hip_pitch_link","torso_link","pelvis"]), "threshold": 1.0})

BOX_AND_PIT_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=20,
    # num_rows=1,
    # num_cols=1,
    # border_width=2.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pit": terrain_gen.MeshPitTerrainCfg(proportion=1., pit_depth_range=(0.55, 0.55), platform_width=3),
        #"pit": terrain_gen.MeshPitTerrainCfg(proportion=1., pit_depth_range=(0., 0.), platform_width=3),
    },
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
 
    @configclass
    class PolicyCfg(ObsGroup):
        #TODO: Always check the symmetry of the observation terms
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.01, n_max=0.01))

        #root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.2, n_max=0.2))
        # root_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.2, n_max=0.2))

        torso_lin_vel = ObsTerm(func=mdp.body_lin_vel_w, params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),},noise=Unoise(n_min=-0.2, n_max=0.2))
        torso_ang_vel = ObsTerm(func=mdp.body_ang_vel_w, params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),},noise=Unoise(n_min=-0.2, n_max=0.2))
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        #base_yaw = ObsTerm(func=mdp.base_yaw_w, noise=Unoise(n_min=-0.05, n_max=0.05))
        # base_quat = ObsTerm(func=mdp.root_quat_w, params={"make_quat_unique":True},
        #                     noise=EulerNoise(n_min=[-0.05,-0.05,-0.05], n_max=[0.05,0.05,0.05]))
        # base_pos = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.01, n_max=0.01))

        torso_quat = ObsTerm(func=mdp.body_quat_w, params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]), "make_quat_unique":True},
                            noise=EulerNoise(n_min=[-0.05,-0.05,-0.05], n_max=[0.05,0.05,0.05]))
        torso_pos = ObsTerm(func=mdp.body_pos_w, params={"asset_cfg" :SceneEntityCfg("robot", body_names=["torso_link"]),}, noise=Unoise(n_min=-0.01, n_max=0.01))  
        #target_commands = ObsTerm(func=mdp.target_pos_root_frame, params={"command_name": "target_pos_e"})
        #target_commands = ObsTerm(func=mdp.target_pos_w, params={"command_name": "target_pos_e"})
        #velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_processed_action,params={"action_name":"joint_pos"})
        box_height = ObsTerm(func=mdp.box_height, noise=Unoise(n_min=-0.01, n_max=0.01))
        climb_command = ObsTerm(func=mdp.climb_command)
        #trunk_mass = ObsTerm(func=mdp.body_mass, params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])})

        time = ObsTerm(func=mdp.time)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.01, n_max=0.01), 
        #     clip=(-1.0, 1.0),
        # )
        #contact_forces = ObsTerm(func=mdp.body_contact_forces, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*"])} )


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 6

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    # @configclass
    # class CuriosityCfg(ObsGroup):
    #     """Observations for policy group."""

    #     # observation terms (order preserved)
    #     base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    #     base_pos = ObsTerm(func=mdp.root_pos_target,params={"command_name": "target_pos_e"})
    #     base_quat = ObsTerm(func=mdp.root_quat_w)
    #     joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized)
    #     contact_forces = ObsTerm(func=mdp.body_contact_forces, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*"])} )
        
    #     # # Temporary for testing
    #     # max_contact_force = ObsTerm(func=mdp.max_contact_forces,params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*",])} )
    #     # base_acc = ObsTerm(func=mdp.base_acc)

    #     def __post_init__(self):
    #         self.enable_corruption = False#True
    #         self.concatenate_terms =True

    # # observation groups
    # rnd_state: CuriosityCfg = CuriosityCfg()

@configclass
class TargetCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_target, 
                              params={"reached_distance": 0.1, "minimal_covered": 1.5})
    #gravity_mag = CurrTerm(func=mdp.gravity_annealing)

@configclass
class HeightCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_height, 
                              params={"update_prob": 0.8})
    #push_vel = CurrTerm(func=mdp.modify_push_vel, params={"vel_min": -0.5, "vel_max": 0.5, "num_steps": 24000})
    
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True,
                                           clip = {".*_joint":(-100., 100.)})

@configclass
class CuriosityCfg:
    type ="rnd" #"nhash"
    use_curiosity = False#True
    obs_dim =54
    hidden_sizes_pred = [256,128]
    hidden_sizes_target = [256,128]
    hidden_sizes_hash=[32]
    pred_dim = 16
    lr= 1e-3
    adaptive_lr = True
    obs_lb = [-1,-2,-2,-4,-4,-4]+[-0.1,-0.1,-0.4]+[-1,-1,-1,-1]+ [-1.11111 for _ in range(29)]+[0 for _ in range(12)]
    obs_ub =[2,2,2,4,4,4]+[1.5,0.1,0.]+[1,1,1,1]+[1.11111 for _ in range(29)]+[1200 for _ in range(12)]
   
@configclass
class G1BoxEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1Rewards = G1Rewards()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: ClimbCommandsCfg = ClimbCommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    curriculum: HeightCurriculumCfg = HeightCurriculumCfg()
    actions: ActionsCfg = ActionsCfg()
    curiosity: CuriosityCfg = CuriosityCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(1.5, 2.5, 1), lookat=(1.5, 0.0, 0.0))
    def __post_init__(self):
        # post init of parent
        # Scene
        #self.curiosity=True
        #self.scene.robot = G1_29_ANNEAL_23_MODIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = G1_29_MODIFIED_714_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #self.scene.robot = G1_29_MODIFIED_MIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions =True#False
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        #self.scene.contact_forces.history_length = 16
        # TODO: always check this
       
        self.scene.terrain.terrain_generator = BOX_AND_PIT_CFG
        #self.scene.height_scanner.pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.6, 1.0])
       
        super().__post_init__()
        self.episode_length_s =7#7.5#10#20
        # Randomization

        # self.events.push_robot.interval_range_s=(4.,6.) #= None
        # self.events.push_robot.params={"velocity_range": {"x": (-0., 0.), "y": (-0., 0.)}}

        self.events.add_base_mass.params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        }

        self.events.physics_material.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (1.25, 1.25),
            "make_consistent": True,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }

        self.events.push_robot= None
        #self.events.add_base_mass= None


        self.events.reset_robot_joints.params["position_range"] = (0.85, 1.15)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (1.25, 1.3), "y": (-0.6, 0.6),"z":(0.0,0.0),"yaw":(-math.pi/6, math.pi/6)},#"yaw": (math.pi, math.pi)},
            #"pose_range": {"x": (0.35, 0.35), "y": (-1.2, 1.2),"z":(0.03,0.03), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 40
        self.sim.physx.gpu_temp_buffer_capacity = 167772160
        self.sim.physx.gpu_max_rigid_patch_count = 327680
        #self.sim.gravity = (0.0, 0.0, -9.81*0.5)




@configclass
class G1BoxEnvCfg_Play(G1BoxEnvCfg):
    def __post_init__(self):
        # post init of parent
        
        super().__post_init__()

        # make a smaller scene for play
        self.scene.contact_forces.debug_vis = True
        self.scene.bodies_ground_contact.debug_vis = True

        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s =5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        # self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque #= None


        self.events.add_base_mass= None
        #self.events.push_robot = None
        self.events.physics_material.params ={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        }
    
        #self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
