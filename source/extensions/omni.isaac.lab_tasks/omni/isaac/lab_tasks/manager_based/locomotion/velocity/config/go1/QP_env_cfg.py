# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, QPActionsCfg, ActionsCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.lab.sim as sim_utils


##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_GO1_CFG  # isort: skip
from omni.isaac.lab.actuators import IdealPDActuatorCfg


@configclass
class UnitreeGo1QPEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: QPActionsCfg = QPActionsCfg()

    

    def __post_init__(self):
        self.episode_length_s = 5.0
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print('self.scene.robot:', self.scene.robot)
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
        print('self.scene.robot:', self.scene.robot)
        self.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk/lower_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        # spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd",
        # activate_contact_sensors=True,
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #     disable_gravity=False,
        #     retain_accelerations=False,
        #     linear_damping=0.0,
        #     angular_damping=0.0,
        #     max_linear_velocity=1000.0,
        #     max_angular_velocity=1000.0,
        #     max_depenetration_velocity=1.0,
        # ),
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        # ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0., rest_offset=-0.02)),
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheye_equidistant",
            clipping_range=(0.01, 10000.0),
            focal_length=1.93, 
            focus_distance=0.6, 
            f_stop=2.0,
            horizontal_aperture=3.896,
            vertical_aperture=2.453,
            horizontal_aperture_offset=0.0,
            vertical_aperture_offset=0.0,
            fisheye_nominal_width=1936,
            fisheye_nominal_height=1216,
            fisheye_optical_centre_x=970.94244,
            fisheye_optical_centre_y=600.374,
            fisheye_max_fov=100.6,
        ),
        offset=CameraCfg.OffsetCfg(pos=(0., 0.0, -0.08), rot=(1., 0., 0., 0.), convention="world"),
    )
        # reduce action scale
        self.decimation = 1
        self.sim.dt = 0.002
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
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo1QPEnvCfg_PLAY(UnitreeGo1QPEnvCfg):
    actions: QPActionsCfg = QPActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
