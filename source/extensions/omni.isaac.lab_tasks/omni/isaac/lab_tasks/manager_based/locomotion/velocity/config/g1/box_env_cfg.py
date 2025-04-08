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
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.sensors import  patterns



import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets import G1_MINIMAL_CFG, G1_CFG, G1_29_CFG,G1_29_MINIMAL_CFG,G1_29_MODIFIED_CFG,G1_29_ANNEAL_23_CFG,G1_29_ANNEAL_23_MODIFIED_CFG # isort: skip
import omni.isaac.lab.terrains as terrain_gen
import math
import random


@configclass
class G1Rewards:
    """Reward terms for the MDP."""
    #TODO: add termination penalty and replace alive reward
    # -- task
    #track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    #track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    position_tracking = RewTerm(func=mdp.position_tracking, weight=4.,
                                  params={"command_name": "target_pos_e","start_time": 0})
    # position_tracking_cos = RewTerm(func=mdp.position_tracking_cos, weight=20.,
    #                               params={"command_name": "target_pos_e","start_time": 1})
    wait_penalty = RewTerm(func=mdp.wait_penalty, weight=-1,params={"command_name": "target_pos_e"}) #weight=-2
    #move_in_direction = RewTerm(func=mdp.move_in_direction, weight=5.0,params={"command_name": "target_pos_e"})
    move_in_direction = RewTerm(func=mdp.move_in_direction, weight=1.0,params={"command_name": "target_pos_e"})
    #termination_penalty = RewTerm(func=mdp.contact_terminated, weight=-200.0)
    #success_rew = RewTerm(func=mdp.stepped_terminated, weight=20000)
    #air_term_penalty = RewTerm(func=mdp.air_terminated, weight=-1000)
    #termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.)
    # success_bonus = RewTerm(
    #     func=mdp.success_bonus,
    #     weight=2000,#400.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #     "success_distance": 0.06,}
    # )
    joint_vel_penalty=RewTerm(func=mdp.joint_vel_l2, weight=-0.0001,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])} )
    torque_penalty=RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-5,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    #torque_penalty=RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-4,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    joint_vel_lim_penalty=RewTerm(func=mdp.joint_velocity_limits, weight=-0.1, params={"soft_ratio": 1., "asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    #torque_lim_penalty=RewTerm(func=mdp.applied_torque_limits, weight=-0.002,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])} )
    joint_acc_penalty=RewTerm(func=mdp.joint_acc_l2, weight=-2e-8,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    base_acc_penalty=RewTerm(func=mdp.base_lin_ang_acc, weight=-0.0001)
    feet_acc_penalty=RewTerm(func=mdp.feet_acc, weight=-0.00002)
    rigid_body_acc_penalty=RewTerm(func=mdp.body_lin_acc_l2, weight=-0.0002,params={"asset_cfg" :SceneEntityCfg("robot", 
                                            body_names=[".*"])})
    # contact_penalty=RewTerm(func=mdp.contact_forces, weight=-0.005,
    #                         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*_elbow_link",".*_wrist_yaw_link",".*_hip_yaw_link",".*_ankle_roll_link",".*_hip_pitch_link","torso_link","pelvis"]), 
    #                                 "threshold": 650.0})
    contact_exp_penalty=RewTerm(func=mdp.contact_forces_exp, weight=-0.1,
                            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*"]), 
                                    "threshold": 500.0,"grad_scale":0.0025})
    #alive_reward=RewTerm(func=mdp.is_alive, weight=5)#100
    air_penalty = RewTerm(func=mdp.body_air_time, weight=-1,params={"sensor_cfg": SceneEntityCfg("contact_forces",
                                             body_names=[ ".*"]), "threshold": 1.0,})
    knee_air_time_penalty = RewTerm(func=mdp.knee_air_time, weight=-1,params={
                                            "sensor_cfg": SceneEntityCfg("contact_forces",
                                             body_names=[ ".*_hip_roll_link",".*_hip_yaw_link",".*_knee_link", "torso_link","pelvis_contour_link"]),
                                             "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*_ankle_roll_link"]),
                                              "torso_bodies_sensor_cfg": SceneEntityCfg("torso_bodies_contact",
                                             body_names=[ "torso_link"]),
                                             "threshold": 1.0,})
    #feet_air =RewTerm(func=mdp.feet_in_air, weight=0.1,params={"feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*_ankle_roll_link"]),"threshold": 1.0,})
    
    #feet_height = RewTerm(func=mdp.feet_height, weight=0.5)
    #TODO: base vel
    base_vel_penalty=RewTerm(func=mdp.base_lin_ang_vel, weight=-0.001)
    power_penalty=RewTerm(func=mdp.power_consumption, weight=-0.00001)
    #body_height = RewTerm(func=mdp.body_height, weight=1.2)
    action_rate_penalty=RewTerm(func=mdp.processed_action_rate_l2, weight=-0.0002,params={"action_name":"joint_pos"})

    #stand_at_target=RewTerm(func=mdp.stand_at_target, weight=-0.5,
    #                params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"]), "command_name": "target_pos_e"})
    # stable_at_target=RewTerm(func=mdp.stable_at_target, weight=-0.5,
    #                          params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"]), "command_name": "target_pos_e"})
    # joint_deviation=RewTerm(func=mdp.joint_deviation_l1, weight=-0.05,#weight=-0.05,
    #                         params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    joint_deviation=RewTerm(func=mdp.joint_deviation_l1, weight=-0.005,#-0.005,
                            params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.25,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.4,
    #     },
    # )
    #flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1)
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    # )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.005)
    #curiosity_rnd = RewTerm(func=mdp.curiosity, weight=200)
    #curiosity_cnt = RewTerm(func=mdp.curiosity_cnt, weight=2000)
    joint_pos_limits =RewTerm(func=mdp.joint_pos_limits, weight=-0.1,)


# @configclass
# class RoughRewards:

    

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
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},     
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
        "pit": terrain_gen.MeshPitTerrainCfg(proportion=1., pit_depth_range=(0.55, 0.8), platform_width=3),
        #"pit": terrain_gen.MeshPitTerrainCfg(proportion=1., pit_depth_range=(0.4, 0.8), platform_width=3),
    },
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
 
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.01, n_max=0.01))

        root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.01, n_max=0.01))
        root_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.01, n_max=0.01))
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        base_quat = ObsTerm(func=mdp.root_quat_w)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        #target_commands = ObsTerm(func=mdp.target_pos_root_frame, params={"command_name": "target_pos_e"})
        target_commands = ObsTerm(func=mdp.target_pos_w, params={"command_name": "target_pos_e"})
        #velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        actions = ObsTerm(func=mdp.last_processed_action,params={"action_name":"joint_pos"})
        box_height = ObsTerm(func=mdp.box_height, noise=Unoise(n_min=-0.01, n_max=0.01))

        time = ObsTerm(func=mdp.time)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.01, n_max=0.01), 
        #     clip=(-1.0, 1.0),
        # )
        #contact_forces = ObsTerm(func=mdp.body_contact_forces, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*"])} )


        def __post_init__(self):
            self.enable_corruption = False#True
            self.concatenate_terms = True
            self.history_length = 6

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CuriosityCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_target,params={"command_name": "target_pos_e"})
        base_quat = ObsTerm(func=mdp.root_quat_w)
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized)
        contact_forces = ObsTerm(func=mdp.body_contact_forces, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*_elbow_link",".*_wrist_yaw_link",".*_hip_yaw_link",".*_ankle_roll_link",".*_hip_pitch_link","torso_link","pelvis"])} )


        def __post_init__(self):
            self.enable_corruption = False#True
            self.concatenate_terms = True

    # observation groups
    rnd_state: CuriosityCfg = CuriosityCfg()

@configclass
class TargetCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_target, 
                              params={"reached_distance": 0.1, "minimal_covered": 1.5})
    
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
    use_curiosity = True
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
    commands: TargetCommandsCfg = TargetCommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    curriculum: TargetCurriculumCfg = TargetCurriculumCfg()
    actions: ActionsCfg = ActionsCfg()
    curiosity: CuriosityCfg = CuriosityCfg()


    def __post_init__(self):
        # post init of parent
        # Scene
        #self.curiosity=True
        self.scene.robot = G1_29_ANNEAL_23_MODIFIED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        #self.scene.contact_forces.history_length = 16
        # TODO: always check this
        self.scene.torso_bodies_contact.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Robot/left_elbow_link",
                                                                  "{ENV_REGEX_NS}/Robot/right_elbow_link",
                                                                    "{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
                                                                    "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
                                                                    "{ENV_REGEX_NS}/Robot/left_wrist_pitch_link",
                                                                    "{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
                                                                    "{ENV_REGEX_NS}/Robot/left_wrist_roll_link",
                                                                    "{ENV_REGEX_NS}/Robot/right_wrist_roll_link",]
        self.scene.terrain.terrain_generator = BOX_AND_PIT_CFG
        #self.scene.height_scanner.pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.6, 1.0])
       
        super().__post_init__()
        self.episode_length_s =5#10#20
        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (0.7, 1.3)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (1.2, 1.35), "y": (-1., 1.),"z":(0.03,0.03), "yaw": (0, 0)},
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




@configclass
class G1BoxEnvCfg_Play(G1BoxEnvCfg):
    def __post_init__(self):
        # post init of parent
        
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s =5# 2.5#10.0
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
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        #self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
