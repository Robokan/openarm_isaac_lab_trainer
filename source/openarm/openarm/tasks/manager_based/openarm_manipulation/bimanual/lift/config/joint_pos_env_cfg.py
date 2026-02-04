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

"""OpenArm-specific configuration for bimanual lift environment.

Simplified: right arm reaches a cube (train_lift style).
"""

import math

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import (
    RigidBodyPropertiesCfg,
    CollisionPropertiesCfg,
    MassPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from source.openarm.openarm.tasks.manager_based.openarm_manipulation.bimanual.lift.lift_env_cfg import (
    BimanualLiftEnvCfg,
)
from source.openarm.openarm.tasks.manager_based.openarm_manipulation.bimanual.lift import mdp
from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_FACTORY_HIGH_PD_CFG,
)


@configclass
class OpenArmBimanualCubeLiftEnvCfg(BimanualLiftEnvCfg):
    """OpenArm bimanual cube/fruit lift environment configuration.
    
    Objects are randomly spawned:
    - Cube (index 0) → Left hand → left_pail
    - Fruit (indices 1-5) → Right hand → right_pail
    """

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Set OpenArm bimanual as robot (factory USD includes Danny table)
        # Enable gravity for lift task (unlike reach where it's disabled)
        # Start with arms at default position (hanging down) - reward shaping guides arms up
        # Set OpenArm bimanual as robot (factory USD includes Danny table)
        # Start with right arm above the desk: shoulder back + elbow bent
        self.scene.robot = OPEN_ARM_FACTORY_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    # Right arm raised pose
                    "openarm_right_joint1": 0.0,   # shoulder pitch (backwards, within limits)
                    "openarm_right_joint2": 1.0,  # shoulder outward to the sides
                    "openarm_right_joint3": 0.0,  # shoulder twist
                    "openarm_right_joint4": 1.6,  # elbow bent more
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 0.0,
                    # Left arm stays at default (all zeros)
                    "openarm_left_joint1": 0.0,
                    "openarm_left_joint2": 0.0,
                    "openarm_left_joint3": 0.0,
                    "openarm_left_joint4": 0.0,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": 0.0,
                    # Grippers open
                    "openarm_left_finger_joint.*": 0.044,
                    "openarm_right_finger_joint.*": 0.044,
                },
            ),
        )
        self.scene.robot.spawn.rigid_props.disable_gravity = False

        # Left arm action
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_left_joint1",
                "openarm_left_joint2",
                "openarm_left_joint3",
                "openarm_left_joint4",
                "openarm_left_joint5",
                "openarm_left_joint6",
                "openarm_left_joint7",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # Right arm action
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_right_joint1",
                "openarm_right_joint2",
                "openarm_right_joint3",
                "openarm_right_joint4",
                "openarm_right_joint5",
                "openarm_right_joint6",
                "openarm_right_joint7",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # Left gripper action
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            open_command_expr={"openarm_left_finger_joint.*": 0.044},
            close_command_expr={"openarm_left_finger_joint.*": 0.0},
        )

        # Right gripper action
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )

        # Set the body name for the command (right hand, train_lift style)
        self.commands.object_pose.body_name = "openarm_right_hand"
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Left end-effector frame
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
                    name="left_ee",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1],
                    ),
                ),
            ],
        )

        # Right end-effector frame
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
                    name="right_ee",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1],
                    ),
                ),
            ],
        )

        # Spawn a cube on the table (train_lift style)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.3, 0.0, 0.36], rot=[1, 0, 0, 0]  # Centered above table
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                collision_props=CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                mass_props=MassPropertiesCfg(
                    mass=0.1,  # 100g - light enough to grasp
                ),
            ),
        )

        # Visualization markers
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.left_ee_frame.visualizer_cfg = marker_cfg
        self.scene.right_ee_frame.visualizer_cfg = marker_cfg.copy()


@configclass
class OpenArmBimanualCubeLiftEnvCfg_PLAY(OpenArmBimanualCubeLiftEnvCfg):
    """Play configuration with single environment for visualization."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Single environment for visualization
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        
        # Disable randomization for cleaner visualization
        self.observations.policy.enable_corruption = False
        
        # Warehouse is now in base scene config (BimanualLiftSceneCfg)
