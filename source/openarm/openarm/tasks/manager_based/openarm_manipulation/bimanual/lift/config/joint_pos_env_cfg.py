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

Object types:
- Index 0: Cube → Left hand picks up → left_pail
- Index 1-5: Fruit → Right hand picks up → right_pail
"""

import math

from isaaclab.assets import RigidObjectCfg
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
    MUG_ASSETS,
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
        self.scene.robot = OPEN_ARM_FACTORY_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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

        # Set the body name for the command (using left hand as reference)
        self.commands.object_pose.body_name = "openarm_left_hand"
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

        # Spawn a mug on the table
        # Using single mug type for now - all mugs go to left hand/left pail
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.35, 0, 0.3], rot=[1, 0, 0, 0]  # Above table surface (table at z=0.255)
            ),
            spawn=UsdFileCfg(
                usd_path=MUG_ASSETS[0],  # Use first mug for all environments
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_angular_velocity=100.0,  # Reduced to prevent spinning out
                    max_linear_velocity=10.0,    # Reduced to prevent shooting away
                    max_depenetration_velocity=2.0,  # Balanced: prevents ejection but allows collision response
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
    """Play configuration with fewer environments and no domain randomization."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Reduce number of environments for visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Disable randomization for cleaner visualization
        self.observations.policy.enable_corruption = False
