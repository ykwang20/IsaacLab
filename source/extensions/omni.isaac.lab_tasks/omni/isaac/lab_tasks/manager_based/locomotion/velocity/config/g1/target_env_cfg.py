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



import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets import G1_MINIMAL_CFG, G1_CFG, G1_29_CFG  # isort: skip
import omni.isaac.lab.terrains as terrain_gen
import math


@configclass
class G1Rewards:
    """Reward terms for the MDP."""
    # -- task
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    # position_tracking = RewTerm(func=mdp.position_tracking, weight=10.,
    #                              params={"command_name": "target_pos_e"})
    # wait_penalty = RewTerm(func=mdp.wait_penalty, weight=-1,params={"command_name": "target_pos_e"})
    # move_in_direction = RewTerm(func=mdp.move_in_direction, weight=10.0,params={"command_name": "target_pos_e"})

    #termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.)
    joint_vel_penalty=RewTerm(func=mdp.joint_vel_l2, weight=-0.000001,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])} )
    torque_penalty=RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    joint_vel_lim_penalty=RewTerm(func=mdp.joint_velocity_limits, weight=-1, params={"soft_ratio": 1., "asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])})
    torque_lim_penalty=RewTerm(func=mdp.applied_torque_limits, weight=-0.002,params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"])} )
    base_acc_penalty=RewTerm(func=mdp.base_lin_ang_acc, weight=-0.001)
    feet_acc_penalty=RewTerm(func=mdp.feet_acc, weight=-0.000002)
    action_rate_penalty=RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # stand_at_target=RewTerm(func=mdp.stand_at_target, weight=-0.5,
    #                 params={"asset_cfg" :SceneEntityCfg("robot", joint_names=[".*"]), "command_name": "target_pos_e"})
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

# @configclass
# class RoughRewards:

    

@configclass
class TargetCommandsCfg:
    """Command specifications for the MDP."""

    target_pos_e = mdp.TargetCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        radius_range=(2.0, 5.0),
        debug_vis=True,
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
            lin_vel_x=(0., 1.0), lin_vel_y=(-0., 0.), ang_vel_z=(-1.0, 1.0), heading=(-0.1*math.pi, 0.1*math.pi)
        ),
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},     
    )

BOX_AND_PIT_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=20,
    # num_rows=2,
    # num_cols=2,
    # border_width=2.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "box": terrain_gen.MeshBoxTerrainCfg(proportion=0.5, box_height_range=(0.05, 0.65), platform_width=3),
        "pit": terrain_gen.MeshPitTerrainCfg(proportion=0.5, pit_depth_range=(0.05, 0.65), platform_width=3),
    },
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        #target_commands = ObsTerm(func=mdp.target_pos_root_frame, params={"command_name": "target_pos_e"})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class TargetCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_target, 
                              params={"reached_distance": 0.4, "minimal_covered": 1.25})
    
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

@configclass
class G1TargetEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1Rewards = G1Rewards()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        
        # Scene
        #self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot = G1_29_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.terrain.terrain_generator = BOX_AND_PIT_CFG
        super().__post_init__()
        self.episode_length_s = 20
        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
        # self.sim.physx.gpu_temp_buffer_capacity = 167772160
        self.sim.physx.gpu_max_rigid_patch_count = 327680




@configclass
class G1TargetEnvCfg_PLAY(G1TargetEnvCfg):
    def __post_init__(self):
        # post init of parent
        
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2
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
