# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, QPActionsCfg, ActionsCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
import math


##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_GO1_CFG  # isort: skip
from omni.isaac.lab.actuators import IdealPDActuatorCfg
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.sensors import  RayCasterCfg, patterns

from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=4.0,
    #border_width=20.0,
    # num_rows=10,
    # num_cols=20,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
       # "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.),
        # "objects": terrain_gen.MeshRepeatedBoxesTerrainCfg(
        #     object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=80, height=0.05, size=(0.1, 0.1), max_yx_angle=60.0, degrees=True
        #     ),
        #     object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        #         num_objects=380, height=0.1, size=(0.1, 0.1), max_yx_angle=60.0, degrees=True
        #     ), platform_width=0.

        # )
        "objects": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                #num_objects=0, height=0.05, size=(0.1, 0.1), max_yx_angle=60.0, degrees=True
                num_objects=480, height=0.1, size=(0.1, 0.1), max_yx_angle=60.0, degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=480, height=0.1, size=(0.1, 0.1), max_yx_angle=60.0, degrees=True
            ), platform_width=0.

        )
    },
)
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class QPCommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,#True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.6), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.4, 0.4), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )

@configclass
class QPRewardsCfg:
    """Reward terms for the MDP."""
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    foot_on_obj_penalty=RewTerm(func=mdp.foot_on_obj, weight=-1,  
                                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"), "threshold": 1.0},)
    

import torch
@configclass
class UnitreeGo1QPEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: QPActionsCfg = QPActionsCfg()
    commands: QPCommandsCfg = QPCommandsCfg()
    rewards: QPRewardsCfg = QPRewardsCfg()
    

    def __post_init__(self):
        self.episode_length_s = 10.0
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print('self.scene.robot:', self.scene.robot)
        print("visual file:", f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl")

        self.scene.robot=self.scene.robot.replace(init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.28),
            joint_pos={
                ".*L_hip_joint": 0.,
                ".*R_hip_joint": 0.,
                "F[L,R]_thigh_joint": 0.9,
                "R[L,R]_thigh_joint": 0.9,
                ".*_calf_joint": -1.8,
            },
            joint_vel={".*": 0.0},
        ))
        self.scene.robot=self.scene.robot.replace(actuators={
        "base_legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_thigh_joint", ".*_calf_joint"],
            effort_limit=33.5,
            velocity_limit=21.0,
            stiffness=0.,
            damping=0.,
            friction=0.,
        ),
        "hip_legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=33.5,
            velocity_limit=21.0,
            stiffness=0.,
            damping=0.,
            friction=0.0,
        ),
    },)
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            #visual_material=None,
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                #mdl_path="/home/legrobot/IsaacLab/asset/ground/Figure_1.png",
                project_uvw=True,
                #texture_scale=(0.25, 0.25),
                texture_scale=(5., 5.),
            ),
            debug_vis=True,
        )

        print('self.scene.robot:', self.scene.robot)
    #     self.scene.camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/trunk/lower_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.FisheyeCameraCfg(
    #         projection_type="fisheyeEquidistant",
    #         clipping_range=(0.01, 10000.0),
    #         #TODO： modify the focal length and FOV
    #         #focal_length=1.93, 
    #         focal_length=0.1,
    #         focus_distance=0.6, 
    #         f_stop=2.0,
    #         horizontal_aperture=3.896,
    #         vertical_aperture=2.453,
    #         horizontal_aperture_offset=0.0,
    #         vertical_aperture_offset=0.0,
    #         fisheye_nominal_width=1936,
    #         fisheye_nominal_height=1216,
    #         fisheye_optical_centre_x=970.94244,
    #         fisheye_optical_centre_y=600.374,
    #         #fisheye_max_fov=100.6,
    #         fisheye_max_fov=160,
    #     ),
    #     #TODO: the convention is not reasonable
    #     offset=CameraCfg.OffsetCfg(pos=(0., 0., -0.08), rot=(1., 0., 0., 0.), convention="opengl"),
    # )
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=[1., 0.6]),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
    )
        # reduce action scale
        self.decimation = 5#100
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (0., 0.), "y": (-0., 0.), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["trunk", ".*thigh"]

        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        #self.curriculum.terrain_levels = None


@configclass
class UnitreeGo1QPEnvCfg_PLAY(UnitreeGo1QPEnvCfg):
    actions: QPActionsCfg = QPActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        #make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # # disable randomization for play
        # self.observations.policy.enable_corruption = False
        # # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
