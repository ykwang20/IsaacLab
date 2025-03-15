# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm

import omni.isaac.lab.terrains as terrain_gen


import omni.isaac.lab_tasks.manager_based.classic.humanoid.mdp as mdp

##
# Scene definition
##

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
        "pit": terrain_gen.MeshPitTerrainCfg(proportion=1., pit_depth_range=(0.6, 1.1), platform_width=3),
        #"pit": terrain_gen.MeshPitTerrainCfg(proportion=1., pit_depth_range=(0.4, 0.8), platform_width=3),
    },
    )

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
        terrain_generator=BOX_AND_PIT_CFG
    )

    # robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Humanoid/humanoid_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=None,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.34),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={
                    ".*_waist.*": 20.0,
                    ".*_upper_arm.*": 10.0,
                    "pelvis": 10.0,
                    ".*_lower_arm": 2.0,
                    ".*_thigh:0": 10.0,
                    ".*_thigh:1": 20.0,
                    ".*_thigh:2": 10.0,
                    ".*_shin": 5.0,
                    ".*_foot.*": 2.0,
                },
                damping={
                    ".*_waist.*": 5.0,
                    ".*_upper_arm.*": 5.0,
                    "pelvis": 5.0,
                    ".*_lower_arm": 1.0,
                    ".*_thigh:0": 5.0,
                    ".*_thigh:1": 5.0,
                    ".*_thigh:2": 5.0,
                    ".*_shin": 0.1,
                    ".*_foot.*": 1.0,
                },
            ),
        },
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)



##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_waist.*": 67.5,
            ".*_upper_arm.*": 67.5,
            "pelvis": 67.5,
            ".*_lower_arm": 45.0,
            ".*_thigh:0": 45.0,
            ".*_thigh:1": 135.0,
            ".*_thigh:2": 45.0,
            ".*_shin": 90.0,
            ".*_foot.*": 22.5,
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (2.8, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (2.8, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0., 0.),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    #progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (2.8, 0.0, 0.0)})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # (3) Reward for non-upright posture
    #upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    # (4) Reward for moving in the right direction
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (2.8, 0.0, 0.0)}
    )
    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # (6) Penalty for energy consumption
    energy = RewTerm(
        func=mdp.power_consumption,
        weight=-0.005,
        params={
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            }
        },
    )
    # (7) Penalty for reaching close to joint limits
    joint_limits = RewTerm(
        func=mdp.joint_limits_penalty_ratio,
        weight=-0.25,
        params={
            "threshold": 0.98,
            "gear_ratio": {
                ".*_waist.*": 67.5,
                ".*_upper_arm.*": 67.5,
                "pelvis": 67.5,
                ".*_lower_arm": 45.0,
                ".*_thigh:0": 45.0,
                ".*_thigh:1": 135.0,
                ".*_thigh:2": 45.0,
                ".*_shin": 90.0,
                ".*_foot.*": 22.5,
            },
        },
    )
    position_tracking = RewTerm(func=mdp.position_tracking, weight=10.,
                                   params={"target_pos": (2.8, 0.0, 0.0),"start_time": 0})
    wait_penalty = RewTerm(func=mdp.wait_penalty, weight=-2,params={"target_pos": (2.8, 0.0, 0.0),})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    #torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.8})
    max_contact= DoneTerm(func=mdp.max_contact_force, 
                           params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ ".*"]), "threshold": 3000.0})


@configclass
class TargetCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_target, 
                              params={"target_pos": (2.8, 0.0, 0.0),"reached_distance": 0.1, "minimal_covered": 1.5})
    
@configclass
class HumanoidBoxEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: TargetCurriculumCfg = TargetCurriculumCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        print('contact_forces', self.scene.contact_forces.prim_path)
        self.events.reset_base.params = {
            "pose_range": {"x": (1.2, 1.2), "y": (-0.1, 0.1),"z":(0.01,0.01), "yaw": (0, 0)},
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
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
