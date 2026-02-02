# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the bimanual lift environment.

Objects (cube or fruit) are dropped randomly on the table:
- Cube → Left hand picks up → Drop in left_square_pail
- Fruit → Right hand picks up → Drop in right_square_pail
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

import math

# Local mug assets (with collision meshes)
from source.openarm.openarm.tasks.manager_based.openarm_manipulation import OPENARM_ROOT_DIR
MUG_ASSETS = [
    f"{OPENARM_ROOT_DIR}/usds/mugs/1.usd",
    f"{OPENARM_ROOT_DIR}/usds/mugs/2.usd",
    f"{OPENARM_ROOT_DIR}/usds/mugs/3.usd",
    f"{OPENARM_ROOT_DIR}/usds/mugs/4.usd",
]

# SimReady fruit assets from Omniverse
SIMREADY_ASSETS_URL = "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props"
FRUIT_ASSETS = [
    f"{SIMREADY_ASSETS_URL}/pomegranate01/pomegranate01.usd",
    f"{SIMREADY_ASSETS_URL}/orange_02/orange_02.usd",
    f"{SIMREADY_ASSETS_URL}/lemon_02/lemon_02.usd",
    f"{SIMREADY_ASSETS_URL}/lime01/lime01.usd",
    f"{SIMREADY_ASSETS_URL}/avocado01/avocado01.usd",
]

# Number of mug assets (for determining object type in rewards)
NUM_MUG_ASSETS = len(MUG_ASSETS)

##
# Scene definition
##


@configclass
class BimanualLiftSceneCfg(InteractiveSceneCfg):
    """Configuration for the bimanual lift scene with robot, pails, and objects.
    
    Objects spawn randomly (cube or fruit):
    - Cube → Left hand picks up → Drop in left_square_pail
    - Fruit → Right hand picks up → Drop in right_square_pail
    
    Note: The factory USD already contains the "Danny" table (surface at z≈0).
    """

    # Robot: will be populated by agent env cfg (use factory USD with built-in table)
    robot: ArticulationCfg = MISSING
    
    # End-effector frames: will be populated by agent env cfg
    left_ee_frame: FrameTransformerCfg = MISSING
    right_ee_frame: FrameTransformerCfg = MISSING
    
    # Target object: will be populated by agent env cfg (cube or fruit)
    object: RigidObjectCfg = MISSING

    # Left pail for mugs (positioned on left side of table, clear of table edge)
    left_pail = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LeftPail",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.4, 0.45, 0.0], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path="https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/squarepail_a01/squarepail_a01.usd",
        ),
    )
    
    # Right pail for fruit (positioned on right side of table, clear of table edge)
    right_pail = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RightPail",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.4, -0.45, 0.0], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path="https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/squarepail_a01/squarepail_a01.usd",
        ),
    )

    # Note: Table "Danny" and lighting are part of the factory USD
    
    # Warehouse environment (centered on the environment grid)
    # Visual only - no collisions to avoid physics overflow
    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    
    # Ground plane for physics (invisible, just for collisions)
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP.
    
    The goal position for the lifted object.
    """

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    
    Both arms and grippers are controlled.
    """

    left_arm_action: mdp.JointPositionActionCfg = MISSING
    right_arm_action: mdp.JointPositionActionCfg = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Left arm joint state
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "openarm_left_joint1",
                    "openarm_left_joint2",
                    "openarm_left_joint3",
                    "openarm_left_joint4",
                    "openarm_left_joint5",
                    "openarm_left_joint6",
                    "openarm_left_joint7",
                ])
            },
        )
        
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "openarm_left_joint1",
                    "openarm_left_joint2",
                    "openarm_left_joint3",
                    "openarm_left_joint4",
                    "openarm_left_joint5",
                    "openarm_left_joint6",
                    "openarm_left_joint7",
                ])
            },
        )
        
        left_gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"])
            },
        )

        # Right arm joint state
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "openarm_right_joint1",
                    "openarm_right_joint2",
                    "openarm_right_joint3",
                    "openarm_right_joint4",
                    "openarm_right_joint5",
                    "openarm_right_joint6",
                    "openarm_right_joint7",
                ])
            },
        )
        
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "openarm_right_joint1",
                    "openarm_right_joint2",
                    "openarm_right_joint3",
                    "openarm_right_joint4",
                    "openarm_right_joint5",
                    "openarm_right_joint6",
                    "openarm_right_joint7",
                ])
            },
        )
        
        right_gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])
            },
        )

        # Object observations
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        
        left_ee_to_object = ObsTerm(func=mdp.left_ee_to_object_distance)
        right_ee_to_object = ObsTerm(func=mdp.right_ee_to_object_distance)
        
        # Object type: 1=cube (left hand), 0=fruit (right hand)
        object_type = ObsTerm(func=mdp.object_type_indicator)
        
        # Target pail position based on object type
        target_pail = ObsTerm(func=mdp.target_pail_position)
        
        # Previous actions
        left_arm_actions = ObsTerm(
            func=mdp.last_action, params={"action_name": "left_arm_action"}
        )
        right_arm_actions = ObsTerm(
            func=mdp.last_action, params={"action_name": "right_arm_action"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomize object position and orientation on table (Danny table surface at z≈0)
    # Object spawns in front of robot, biased left or right so one arm is closer
    # Random yaw rotation so mugs/fruit face different directions
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.15, 0.15),
                "y": (-0.25, 0.25),
                "z": (0.1, 0.1),  # Above table surface (table is at z≈0)
                "yaw": (-3.14159, 3.14159),  # Full 360° rotation around Z
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),  # No body_names - use root
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.
    
    Object type determines correct arm:
    - Cube → Left arm → left_pail
    - Fruit → Right arm → right_pail
    """

    # Correct arm reaching toward object (based on object type)
    reaching_object = RewTerm(
        func=mdp.correct_arm_reaching_object,
        params={"std": 0.1},
        weight=1.5,
    )
    
    # Fine-grained reaching reward
    reaching_object_fine = RewTerm(
        func=mdp.closest_arm_object_ee_distance,
        params={"std": 0.05},
        weight=0.5,
    )
    
    # Penalize wrong arm for approaching
    wrong_arm = RewTerm(
        func=mdp.wrong_arm_penalty,
        params={"std": 0.1},
        weight=-0.5,
    )

    # Lifting the object
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.06},
        weight=15.0,
    )

    # Moving object toward correct pail (once lifted)
    object_to_pail = RewTerm(
        func=mdp.object_to_correct_pail,
        params={"std": 0.3, "minimal_height": 0.06},
        weight=16.0,
    )

    object_to_pail_fine = RewTerm(
        func=mdp.object_to_correct_pail,
        params={"std": 0.05, "minimal_height": 0.06},
        weight=5.0,
    )
    
    # Big reward for successfully placing in correct pail
    success = RewTerm(
        func=mdp.object_in_correct_pail,
        params={"threshold": 0.1},
        weight=50.0,
    )

    # Action penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "openarm_left_joint.*", "openarm_left_finger_joint.*"
            ])
        },
    )
    
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "openarm_right_joint.*", "openarm_right_finger_joint.*"
            ])
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if object falls off table
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )

    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -1e-1, "num_steps": 10000},
    )
    
    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


##
# Environment configuration
##


@configclass
class BimanualLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual lifting environment."""

    # Scene settings
    scene: BimanualLiftSceneCfg = BimanualLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 8.0
        
        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
