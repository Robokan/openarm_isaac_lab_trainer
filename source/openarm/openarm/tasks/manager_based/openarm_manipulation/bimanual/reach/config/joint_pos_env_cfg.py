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

import math

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .. import mdp
from ..reach_env_cfg import (
    ReachEnvCfg,
)

from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)
from isaaclab.assets.articulation import ArticulationCfg

##
# Environment configuration
##


@configclass
class OpenArmReachEnvCfg(ReachEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to OpenArm
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_left_joint1": 0.0,
                    "openarm_left_joint2": 0.0,
                    "openarm_left_joint3": 0.0,
                    "openarm_left_joint4": 0.0,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": 0.0,
                    "openarm_right_joint1": 0.0,
                    "openarm_right_joint2": 0.0,
                    "openarm_right_joint3": 0.0,
                    "openarm_right_joint4": 0.0,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 0.0,
                    "openarm_left_finger_joint.*": 0.0,
                    "openarm_right_finger_joint.*": 0.0,
                },  # Close the gripper
            ),
        )

        # override rewards
        self.rewards.left_end_effector_position_tracking.params["asset_cfg"].body_names = ["openarm_left_hand"]
        self.rewards.left_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["openarm_left_hand"]
        self.rewards.left_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["openarm_left_hand"]

        self.rewards.right_end_effector_position_tracking.params["asset_cfg"].body_names = ["openarm_right_hand"]
        self.rewards.right_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["openarm_right_hand"]
        self.rewards.right_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["openarm_right_hand"]

        # override actions
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

        # override command generator body
        # end-effector is along z-direction
        self.commands.left_ee_pose.body_name = "openarm_left_hand"
        self.commands.right_ee_pose.body_name = "openarm_right_hand"


@configclass
class OpenArmReachEnvCfg_PLAY(OpenArmReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # add warehouse environment for visualization
        self.scene.warehouse = AssetBaseCfg(
            prim_path="/World/Warehouse",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse.usd",
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
        )


@configclass
class OpenArmReachEnvCfg_TELEOP(OpenArmReachEnvCfg):
    """Reach configuration with expanded workspace for teleoperation training."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Expand reach workspace to cover full arm reach
        self.commands.left_ee_pose.ranges.pos_x = (0.05, 0.6)
        self.commands.left_ee_pose.ranges.pos_y = (0.05, 0.45)
        self.commands.left_ee_pose.ranges.pos_z = (0.15, 0.65)
        self.commands.left_ee_pose.ranges.roll = (-math.pi, math.pi)
        self.commands.left_ee_pose.ranges.pitch = (-math.pi, math.pi)
        self.commands.left_ee_pose.ranges.yaw = (-math.pi, math.pi)

        self.commands.right_ee_pose.ranges.pos_x = (0.05, 0.6)
        self.commands.right_ee_pose.ranges.pos_y = (-0.45, -0.05)
        self.commands.right_ee_pose.ranges.pos_z = (0.15, 0.65)
        self.commands.right_ee_pose.ranges.roll = (-math.pi, math.pi)
        self.commands.right_ee_pose.ranges.pitch = (-math.pi, math.pi)
        self.commands.right_ee_pose.ranges.yaw = (-math.pi, math.pi)

        # Randomize starting joint pose (arms only, smaller offsets)
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params["position_range"] = (-0.5, 0.5)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "openarm_left_joint[1-7]",
                "openarm_right_joint[1-7]",
            ],
        )
