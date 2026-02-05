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

"""Observation functions for bimanual lift task.

Includes object type indicator for cube vs fruit sorting.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Position of the object in the robot's root frame.
    
    Since robot is fixed at origin, this is just the world position.
    Returns: (num_envs, 3)
    """
    from isaaclab.utils.math import subtract_frame_transforms
    
    robot = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    # root_pos_w is (num_envs, 7) - slice to get just xyz
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return object_pos_b


def left_ee_to_object_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
) -> torch.Tensor:
    """Vector from left end-effector to object.
    
    Returns: (num_envs, 3)
    """
    object: RigidObject = env.scene[object_cfg.name]
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
    
    # root_pos_w is (num_envs, 7) - slice to get just xyz
    object_pos_w = object.data.root_pos_w[:, :3]
    # target_pos_w is (num_envs, num_targets, 3) - take first target
    left_ee_pos_w = left_ee_frame.data.target_pos_w[:, 0, :]
    
    return object_pos_w - left_ee_pos_w


def right_ee_to_object_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Vector from right end-effector to object.
    
    Returns: (num_envs, 3)
    """
    object: RigidObject = env.scene[object_cfg.name]
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]
    
    # root_pos_w is (num_envs, 7) - slice to get just xyz
    object_pos_w = object.data.root_pos_w[:, :3]
    # target_pos_w is (num_envs, num_targets, 3) - take first target
    right_ee_pos_w = right_ee_frame.data.target_pos_w[:, 0, :]
    
    return object_pos_w - right_ee_pos_w


def object_type_indicator(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Returns 1 if object is mug (use left hand), 0 if fruit (use right hand).
    
    This tells the policy which arm should be used for the current object.
    Object type is randomized at reset via the randomize_object_type event.
    """
    from .events import is_mug_object
    is_mug = is_mug_object(env, num_mug_assets=4)
    
    return is_mug.float().unsqueeze(-1)


def left_pail_position(
    env: ManagerBasedRLEnv,
    pail_pos: tuple = (0.3, 0.35, 0.1),  # On ground beside table
) -> torch.Tensor:
    """Position of the left pail (for cubes)."""
    pos = torch.tensor(pail_pos, device=env.device)
    return pos.unsqueeze(0).expand(env.num_envs, -1)


def right_pail_position(
    env: ManagerBasedRLEnv,
    pail_pos: tuple = (0.3, -0.35, 0.1),  # On ground beside table
) -> torch.Tensor:
    """Position of the right pail (for fruit)."""
    pos = torch.tensor(pail_pos, device=env.device)
    return pos.unsqueeze(0).expand(env.num_envs, -1)


def target_pail_position(
    env: ManagerBasedRLEnv,
    left_pail_pos: tuple = (0.3, 0.35, 0.1),   # On ground beside table
    right_pail_pos: tuple = (0.3, -0.35, 0.1),  # On ground beside table
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Position of the target pail based on object type.
    
    - Mug → left pail
    - Fruit → right pail
    
    Object type is randomized at reset via the randomize_object_type event.
    """
    from .events import is_mug_object
    is_mug = is_mug_object(env, num_mug_assets=4)
    
    left_goal = torch.tensor(left_pail_pos, device=env.device).unsqueeze(0)
    right_goal = torch.tensor(right_pail_pos, device=env.device).unsqueeze(0)
    
    # Select pail based on object type: mugs → left, fruit → right
    target = torch.where(is_mug.unsqueeze(-1), left_goal, right_goal)
    
    return target


def closest_arm_indicator(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Legacy: Returns 1 if left arm is closer, 0 if right arm is closer.
    
    Deprecated: Use object_type_indicator instead for this task.
    """
    # For this task, we use object type, not closest arm
    return object_type_indicator(env, object_cfg)


# ===== Lift Target Observations (Phase 1) =====

def left_ee_to_lift_target(
    env: ManagerBasedRLEnv,
    lift_target: tuple = (0.0, 0.2, 0.355),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
) -> torch.Tensor:
    """Vector from left end-effector to its lift target position.
    
    The lift target is the position where the arm should go before reaching for the object.
    This is the Phase 1 target for the left arm.
    
    Returns: (num_envs, 3)
    """
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
    left_ee_pos_w = left_ee_frame.data.target_pos_w[:, 0, :]
    
    target = torch.tensor(lift_target, device=env.device)
    
    return target - left_ee_pos_w


def right_ee_to_lift_target(
    env: ManagerBasedRLEnv,
    lift_target: tuple = (0.0, -0.2, 0.355),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Vector from right end-effector to its lift target position.
    
    The lift target is the position where the arm should go before reaching for the object.
    This is the Phase 1 target for the right arm.
    
    Returns: (num_envs, 3)
    """
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]
    right_ee_pos_w = right_ee_frame.data.target_pos_w[:, 0, :]
    
    target = torch.tensor(lift_target, device=env.device)
    
    return target - right_ee_pos_w


# ===== Object Assignment Observations =====

def left_arm_has_object(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Returns 1.0 if the left arm has an assigned object to pick up.
    
    Uses the _arm_active flag set by randomize_active_arms event.
    When left arm is inactive (flag=0), the policy should keep it still.
    
    Returns: (num_envs, 1)
    """
    from .events import get_arm_active_flags
    left_active, _ = get_arm_active_flags(env)
    return left_active.float().unsqueeze(-1)


def right_arm_has_object(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Returns 1.0 if the right arm has an assigned object to pick up.
    
    Uses the _arm_active flag set by randomize_active_arms event.
    When right arm is inactive (flag=0), the policy should keep it still.
    
    Returns: (num_envs, 1)
    """
    from .events import get_arm_active_flags
    _, right_active = get_arm_active_flags(env)
    return right_active.float().unsqueeze(-1)


# ===== Conditional Object Position Observations =====

def left_object_position_conditional(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_left"),
) -> torch.Tensor:
    """Position of left object, returns zeros when left arm is inactive.
    
    When left arm is inactive, the cube is moved far away and we return
    zeros to avoid confusing the policy with invalid positions.
    
    Returns: (num_envs, 3)
    """
    from .events import get_arm_active_flags
    left_active, _ = get_arm_active_flags(env)
    
    # Get actual object position
    real_pos = object_position_in_robot_root_frame(env, robot_cfg, object_cfg)
    
    # Return zeros for inactive arms
    return torch.where(left_active.unsqueeze(-1), real_pos, torch.zeros_like(real_pos))


def right_object_position_conditional(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_right"),
) -> torch.Tensor:
    """Position of right object, returns zeros when right arm is inactive.
    
    When right arm is inactive, the cube is moved far away and we return
    zeros to avoid confusing the policy with invalid positions.
    
    Returns: (num_envs, 3)
    """
    from .events import get_arm_active_flags
    _, right_active = get_arm_active_flags(env)
    
    # Get actual object position
    real_pos = object_position_in_robot_root_frame(env, robot_cfg, object_cfg)
    
    # Return zeros for inactive arms
    return torch.where(right_active.unsqueeze(-1), real_pos, torch.zeros_like(real_pos))


# ===== Conditional Target Position Observations =====

def left_target_position_conditional(
    env: ManagerBasedRLEnv,
    command_name: str = "left_object_pose",
) -> torch.Tensor:
    """Target position for left arm, returns zeros when left arm is inactive.
    
    Prevents the policy from seeing a valid target when the arm should stay still.
    
    Returns: (num_envs, 6) - position (3) + orientation (3)
    """
    from .events import get_arm_active_flags
    left_active, _ = get_arm_active_flags(env)
    
    # Get the command (target pose)
    command = env.command_manager.get_command(command_name)
    
    # Return zeros for inactive arms
    return torch.where(left_active.unsqueeze(-1), command, torch.zeros_like(command))


def right_target_position_conditional(
    env: ManagerBasedRLEnv,
    command_name: str = "right_object_pose",
) -> torch.Tensor:
    """Target position for right arm, returns zeros when right arm is inactive.
    
    Prevents the policy from seeing a valid target when the arm should stay still.
    
    Returns: (num_envs, 6) - position (3) + orientation (3)
    """
    from .events import get_arm_active_flags
    _, right_active = get_arm_active_flags(env)
    
    # Get the command (target pose)
    command = env.command_manager.get_command(command_name)
    
    # Return zeros for inactive arms
    return torch.where(right_active.unsqueeze(-1), command, torch.zeros_like(command))
