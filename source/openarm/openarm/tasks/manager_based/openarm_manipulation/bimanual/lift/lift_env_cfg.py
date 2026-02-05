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

Two-cube bimanual task: each arm picks up its assigned cube and moves it to a target.
- Left arm: picks up left cube (spawned on left side), moves to left target
- Right arm: picks up right cube (spawned on right side), moves to right target
Scene assets (factory USD, table, pails) remain unchanged.
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
    
    Two-cube bimanual task: left cube for left arm, right cube for right arm.
    Note: The factory USD already contains the "Danny" table (surface at z=0.255).
    """

    # Robot: will be populated by agent env cfg (use factory USD with built-in table)
    robot: ArticulationCfg = MISSING
    
    # End-effector frames: will be populated by agent env cfg
    left_ee_frame: FrameTransformerCfg = MISSING
    right_ee_frame: FrameTransformerCfg = MISSING
    
    # Target objects: will be populated by agent env cfg
    # Left cube spawns on left side of table (positive Y)
    object_left: RigidObjectCfg = MISSING
    # Right cube spawns on right side of table (negative Y)
    object_right: RigidObjectCfg = MISSING

    # Left pail for mugs (on ground beside table)
    left_pail = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LeftPail",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.3, 0.45, 0.0], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path="https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/squarepail_a01/squarepail_a01.usd",
        ),
    )
    
    # Right pail for fruit (on ground beside table)
    right_pail = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RightPail",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.3, -0.45, 0.0], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path="https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/squarepail_a01/squarepail_a01.usd",
        ),
    )

    # Note: Table "Danny" and lighting are part of the factory USD
    
    # Warehouse environment for visualization (visual-only, no collision)
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
    
    Two goal positions: one for each arm's cube.
    Left target on left side (positive Y, extends over left pail).
    Right target on right side (negative Y, extends over right pail).
    """

    # Left arm target (left side of table, extends over left pail at y=0.45)
    left_object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg (left hand)
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(0.05, 0.45),   # Left side, extends over left pail
            pos_z=(0.35, 0.6),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    # Right arm target (right side of table, extends over right pail at y=-0.45)
    right_object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg (right hand)
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.4),
            pos_y=(-0.45, -0.05),  # Right side, extends over right pail
            pos_z=(0.35, 0.6),
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
        """Observations for policy group.
        
        Both arms observe their joint states, their assigned cube, and their target.
        """

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

        # Left cube observations (returns zeros when left arm inactive)
        left_object_position = ObsTerm(
            func=mdp.left_object_position_conditional,
            params={"object_cfg": SceneEntityCfg("object_left")},
        )
        # Target position also zeros when left arm inactive
        target_left_object_position = ObsTerm(
            func=mdp.left_target_position_conditional,
            params={"command_name": "left_object_pose"},
        )

        # Right cube observations (returns zeros when right arm inactive)
        right_object_position = ObsTerm(
            func=mdp.right_object_position_conditional,
            params={"object_cfg": SceneEntityCfg("object_right")},
        )
        # Target position also zeros when right arm inactive
        target_right_object_position = ObsTerm(
            func=mdp.right_target_position_conditional,
            params={"command_name": "right_object_pose"},
        )

        # Object assignment flags (for future use - can make arm stand still if no object)
        # Currently always 1.0 since both arms always have cubes
        left_arm_has_object = ObsTerm(func=mdp.left_arm_has_object)
        right_arm_has_object = ObsTerm(func=mdp.right_arm_has_object)
        
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
    """Configuration for events.
    
    Event order matters! Events run in the order defined here.
    For curriculum learning:
    1. randomize_active_arms - sets which arms are active (FIRST)
    2. reset_all, reset_robot_joints - standard resets
    3. store_initial_joint_positions - capture initial poses for stay-at-pose reward
    4. reset_left/right_object_position - randomize cube positions
    5. hide_inactive_cubes - move inactive cubes out of scene (LAST)
    """

    # STEP 1: Curriculum Learning - decide which arms are active FIRST
    # 45% left only, 45% right only, 10% both
    randomize_active_arms = EventTerm(
        func=mdp.randomize_active_arms,
        mode="reset",
        params={
            "left_only_prob": 0.45,
            "right_only_prob": 0.45,
            "both_prob": 0.10,
        },
    )

    # STEP 2: Standard scene reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    # STEP 3: Randomize robot joint positions on reset (both arms identically)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.25, 0.25),  # Random offset in radians
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "openarm_left_joint.*",
                    "openarm_left_finger_joint.*",
                    "openarm_right_joint.*",
                    "openarm_right_finger_joint.*",
                ],
            ),
        },
    )

    # STEP 4: Store initial joint positions for stay-at-pose reward (after joint randomization)
    store_initial_positions = EventTerm(
        func=mdp.store_initial_joint_positions,
        mode="reset",
    )

    # Reset the one-way gate for lift target tracking
    reset_lift_gate = EventTerm(
        func=mdp.reset_lift_target_gate,
        mode="reset",
    )
    
    # STEP 5: Randomize left cube position on left side of table
    reset_left_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("object_left"),
        },
    )

    # STEP 6: Randomize right cube position on right side of table
    reset_right_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("object_right"),
        },
    )

    # STEP 7: Hide inactive cubes (move below table) - MUST BE LAST
    hide_inactive_cubes = EventTerm(
        func=mdp.hide_inactive_cubes,
        mode="reset",
        params={
            "hidden_position": (0.0, 0.0, -5.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.
    
    Bimanual lift with curriculum learning:
    - 45% episodes: only left arm active (left cube on table, right cube hidden)
    - 45% episodes: only right arm active (right cube on table, left cube hidden)
    - 10% episodes: both arms active (standard bimanual task)
    
    Gated rewards only apply when the arm is active.
    Stay-at-pose rewards encourage inactive arms to remain still.
    """

    # === Left arm rewards (gated by active flag) ===
    
    # Left arm reaching toward left object (only when active)
    left_reaching_object = RewTerm(
        func=mdp.left_arm_object_distance_tanh_gated,
        params={"std": 0.2, "object_cfg": SceneEntityCfg("object_left")},
        weight=3.0,
    )

    # Lifting the left object (only when active)
    left_lifting_object = RewTerm(
        func=mdp.left_object_is_lifted_relative_gated,
        params={"min_delta": 0.015, "object_cfg": SceneEntityCfg("object_left")},
        weight=30.0,
    )

    # Moving left object toward left goal (only when active)
    left_goal_tracking = RewTerm(
        func=mdp.left_object_goal_distance_relative_gated,
        params={"std": 0.3, "min_delta": 0.015, "command_name": "left_object_pose", "object_cfg": SceneEntityCfg("object_left")},
        weight=25.0,
    )

    left_goal_tracking_fine_grained = RewTerm(
        func=mdp.left_object_goal_distance_relative_gated,
        params={"std": 0.05, "min_delta": 0.015, "command_name": "left_object_pose", "object_cfg": SceneEntityCfg("object_left")},
        weight=10.0,
    )

    # Left arm penalty for drifting from default pose (only when INACTIVE)
    # Uses default_joint_pos (home pose) - negative weight = penalty for distance
    left_stay_at_pose = RewTerm(
        func=mdp.left_arm_distance_from_default,
        params={},
        weight=-0.5,  # Gentle penalty for distance from default
    )

    # === Right arm rewards (gated by active flag) ===

    # Right arm reaching toward right object (only when active)
    right_reaching_object = RewTerm(
        func=mdp.right_arm_object_distance_tanh_gated,
        params={"std": 0.2, "object_cfg": SceneEntityCfg("object_right")},
        weight=3.0,
    )

    # Lifting the right object (only when active)
    right_lifting_object = RewTerm(
        func=mdp.right_object_is_lifted_relative_gated,
        params={"min_delta": 0.015, "object_cfg": SceneEntityCfg("object_right")},
        weight=30.0,
    )

    # Moving right object toward right goal (only when active)
    right_goal_tracking = RewTerm(
        func=mdp.right_object_goal_distance_relative_gated,
        params={"std": 0.3, "min_delta": 0.015, "command_name": "right_object_pose", "object_cfg": SceneEntityCfg("object_right")},
        weight=25.0,
    )

    right_goal_tracking_fine_grained = RewTerm(
        func=mdp.right_object_goal_distance_relative_gated,
        params={"std": 0.05, "min_delta": 0.015, "command_name": "right_object_pose", "object_cfg": SceneEntityCfg("object_right")},
        weight=10.0,
    )

    # Right arm penalty for drifting from default pose (only when INACTIVE)
    # Uses default_joint_pos (home pose) - negative weight = penalty for distance
    right_stay_at_pose = RewTerm(
        func=mdp.right_arm_distance_from_default,
        params={},
        weight=-0.5,  # Gentle penalty for distance from default
    )

    # === Action penalties (both arms) ===
    
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
    """Termination terms for the MDP.
    
    Uses gated terminations for curriculum learning - only terminate if
    an ACTIVE arm's cube drops. Inactive cubes are intentionally hidden.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if left object falls below minimum height (only when left arm active)
    left_object_dropping = DoneTerm(
        func=mdp.left_object_dropped_gated,
        params={"minimum_height": -0.05, "object_cfg": SceneEntityCfg("object_left")},
    )

    # Terminate if right object falls below minimum height (only when right arm active)
    right_object_dropping = DoneTerm(
        func=mdp.right_object_dropped_gated,
        params={"minimum_height": -0.05, "object_cfg": SceneEntityCfg("object_right")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.
    
    Gradually ramp up penalties for both arms over training.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 50000},
    )

    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -1e-2, "num_steps": 50000},
    )

    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -1e-2, "num_steps": 50000},
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
        self.episode_length_s = 5.0
        
        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 128 * 1024  # Increased for bimanual robot + objects
        self.sim.physx.friction_correlation_distance = 0.00625
