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

"""Custom event functions for bimanual lift task.

Currently using only mugs - all objects go to left hand.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def is_mug_object(env: ManagerBasedEnv, num_mug_assets: int = 4) -> torch.Tensor:
    """Check if each environment's object is a mug.
    
    Currently all objects are mugs (only 4 mug assets, no fruit).
    All objects should be picked up with the left hand and go to left pail.
    
    Args:
        env: The environment instance.
        num_mug_assets: Number of mug assets (unused, always returns True).
        
    Returns:
        Boolean tensor, always True (all mugs → use left hand).
    """
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def reset_lift_target_gate(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset the one-way gate tracking for velocity toward lift target.
    
    Called on episode reset to clear the "arm has reached lift target" flags
    for the environments being reset.
    """
    if hasattr(env, '_arm_reached_lift_target'):
        env._arm_reached_lift_target[env_ids] = False


def randomize_active_arms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    left_only_prob: float = 0.45,
    right_only_prob: float = 0.45,
    both_prob: float = 0.10,
) -> None:
    """Randomize which arms are active for curriculum learning (Phase 1: set flags).
    
    At episode reset, randomly chooses:
    - 45% chance: left arm active only (right arm stays still, right cube hidden)
    - 45% chance: right arm active only (left arm stays still, left cube hidden)
    - 10% chance: both arms active (standard bimanual task)
    
    Sets:
    - env._arm_active: (num_envs, 2) bool tensor [left_active, right_active]
    
    NOTE: This event should run BEFORE object position events.
    The hide_inactive_cubes event should run AFTER to actually move cubes.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices being reset.
        left_only_prob: Probability of left arm only active.
        right_only_prob: Probability of right arm only active. 
        both_prob: Probability of both arms active.
    """
    num_reset = len(env_ids)
    device = env.device
    
    # Initialize tracking tensors if not exist
    if not hasattr(env, '_arm_active'):
        env._arm_active = torch.ones(env.num_envs, 2, dtype=torch.bool, device=device)
    
    # Generate random values to determine active configuration
    rand_vals = torch.rand(num_reset, device=device)
    
    # Determine active arms for each resetting environment
    # < left_only_prob: left only
    # < left_only_prob + right_only_prob: right only  
    # else: both
    left_only = rand_vals < left_only_prob
    right_only = (rand_vals >= left_only_prob) & (rand_vals < left_only_prob + right_only_prob)
    both_active = rand_vals >= (left_only_prob + right_only_prob)
    
    # Set active flags
    # Left is active when: left_only OR both
    env._arm_active[env_ids, 0] = left_only | both_active
    # Right is active when: right_only OR both
    env._arm_active[env_ids, 1] = right_only | both_active


def store_initial_joint_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Store initial joint positions for stay-at-pose reward.
    
    Should run AFTER robot joint randomization.
    
    Sets:
    - env._arm_initial_joint_pos: (num_envs, 2, 7) initial positions per arm
    """
    device = env.device
    
    if not hasattr(env, '_arm_initial_joint_pos'):
        # Store initial joint positions for each arm (7 joints each)
        env._arm_initial_joint_pos = torch.zeros(env.num_envs, 2, 7, device=device)
    
    robot = env.scene["robot"]
    
    # Get joint indices for each arm
    left_joint_names = [
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6", 
        "openarm_left_joint7"
    ]
    right_joint_names = [
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
        "openarm_right_joint7"
    ]
    
    left_joint_ids = [robot.find_joints(name)[0][0] for name in left_joint_names]
    right_joint_ids = [robot.find_joints(name)[0][0] for name in right_joint_names]
    
    # Store initial positions
    env._arm_initial_joint_pos[env_ids, 0, :] = robot.data.joint_pos[env_ids][:, left_joint_ids]
    env._arm_initial_joint_pos[env_ids, 1, :] = robot.data.joint_pos[env_ids][:, right_joint_ids]


def hide_inactive_cubes(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    hidden_position: tuple = (0.0, 0.0, -5.0),
) -> None:
    """Move inactive cubes below the table (Phase 2: after object reset).
    
    Should run LAST after all other object position events.
    Uses the _arm_active flags set by randomize_active_arms.
    
    Uses a fixed hidden position rather than offsetting current position,
    since data buffers may not be updated yet after other reset events.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices being reset.
        hidden_position: Fixed position to place hidden cubes (below table).
    """
    if not hasattr(env, '_arm_active'):
        return  # No active flags set, nothing to do
    
    device = env.device
    object_left = env.scene["object_left"]
    object_right = env.scene["object_right"]
    
    # Fixed hidden pose (position + identity quaternion)
    hidden_pos = torch.tensor(hidden_position, device=device)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # w, x, y, z
    
    # For envs where left arm is NOT active, move left cube to hidden position
    left_inactive_mask = ~env._arm_active[env_ids, 0]
    num_left_inactive = left_inactive_mask.sum().item()
    if num_left_inactive > 0:
        inactive_env_ids = env_ids[left_inactive_mask]
        # Create pose tensor: (num_inactive, 7) = [x, y, z, qw, qx, qy, qz]
        hidden_pose = torch.cat([hidden_pos, identity_quat]).unsqueeze(0).expand(num_left_inactive, -1)
        object_left.write_root_pose_to_sim(hidden_pose, inactive_env_ids)
    
    # For envs where right arm is NOT active, move right cube to hidden position
    right_inactive_mask = ~env._arm_active[env_ids, 1]
    num_right_inactive = right_inactive_mask.sum().item()
    if num_right_inactive > 0:
        inactive_env_ids = env_ids[right_inactive_mask]
        # Create pose tensor: (num_inactive, 7) = [x, y, z, qw, qx, qy, qz]
        hidden_pose = torch.cat([hidden_pos, identity_quat]).unsqueeze(0).expand(num_right_inactive, -1)
        object_right.write_root_pose_to_sim(hidden_pose, inactive_env_ids)


def get_arm_active_flags(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the active flags for both arms.
    
    Returns:
        left_active: (num_envs,) bool tensor
        right_active: (num_envs,) bool tensor
    """
    if not hasattr(env, '_arm_active'):
        # Default: both arms active
        return (
            torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
            torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        )
    return env._arm_active[:, 0], env._arm_active[:, 1]
