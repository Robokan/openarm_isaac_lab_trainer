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

"""Reward functions for bimanual lift task.

Object type determines which arm picks up:
- Mug → Left hand → Drop in left_pail
- Fruit → Right hand → Drop in right_pail

The object type is stored in env.scene["object"].cfg and determined at spawn time.
We use the object's asset index to select the correct arm:
- Indices 0 to (NUM_MUG_ASSETS-1) = mugs (left hand)
- Indices NUM_MUG_ASSETS and above = fruits (right hand)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_object_type_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Determine object type per environment.
    
    Returns:
        is_mug: Boolean tensor, True if object is mug (use left hand)
                False if object is fruit (use right hand)
    
    Object type is randomized at reset via the randomize_object_type event.
    """
    from .events import is_mug_object
    return is_mug_object(env, num_mug_assets=4)


def _get_arm_distances(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    left_ee_frame_cfg: SceneEntityCfg,
    right_ee_frame_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute distance from each arm to the object.
    
    Returns:
        left_dist: Distance from left EE to object
        right_dist: Distance from right EE to object
    """
    object: RigidObject = env.scene[object_cfg.name]
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]
    
    object_pos_w = object.data.root_pos_w[:, :3]
    left_ee_pos_w = left_ee_frame.data.target_pos_w[:, 0, :]
    right_ee_pos_w = right_ee_frame.data.target_pos_w[:, 0, :]
    
    left_dist = torch.norm(object_pos_w - left_ee_pos_w, dim=-1)
    right_dist = torch.norm(object_pos_w - right_ee_pos_w, dim=-1)
    
    return left_dist, right_dist


def left_arm_object_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_left"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
) -> torch.Tensor:
    """Tanh-kernel reward for left arm reaching its object."""
    object: RigidObject = env.scene[object_cfg.name]
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w[:, :3]
    left_ee_pos_w = left_ee_frame.data.target_pos_w[:, 0, :]
    dist = torch.norm(object_pos_w - left_ee_pos_w, dim=-1)

    return 1.0 - torch.tanh(dist / std)


def right_arm_object_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_right"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Tanh-kernel reward for right arm reaching its object."""
    object: RigidObject = env.scene[object_cfg.name]
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w[:, :3]
    right_ee_pos_w = right_ee_frame.data.target_pos_w[:, 0, :]
    dist = torch.norm(object_pos_w - right_ee_pos_w, dim=-1)

    return 1.0 - torch.tanh(dist / std)


def correct_arm_reaching_object(
    env: ManagerBasedRLEnv,
    std: float,
    left_lift_target: tuple = (0.0, 0.2, 0.355),
    right_lift_target: tuple = (0.0, -0.2, 0.355),
    lift_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Reward the correct arm (based on object type) for reaching toward the object.
    
    ONLY active when the correct arm is at its lift target position (within threshold).
    This ensures arm lifts to ready position before reaching.
    
    - Cube → Left arm should reach (only if left arm at lift target)
    - Fruit → Right arm should reach (only if right arm at lift target)
    
    Uses a Gaussian kernel to reward proximity, gated by being at lift target.
    """
    is_cube = _get_object_type_mask(env)
    
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]
    
    left_ee_pos = left_ee_frame.data.target_pos_w[:, 0, :]
    right_ee_pos = right_ee_frame.data.target_pos_w[:, 0, :]
    
    left_dist, right_dist = _get_arm_distances(
        env, object_cfg, left_ee_frame_cfg, right_ee_frame_cfg
    )
    
    # Check if arms are at their lift targets
    left_target_t = torch.tensor(left_lift_target, device=left_ee_pos.device)
    right_target_t = torch.tensor(right_lift_target, device=right_ee_pos.device)
    
    left_at_target = torch.norm(left_ee_pos - left_target_t, dim=-1) < lift_threshold
    right_at_target = torch.norm(right_ee_pos - right_target_t, dim=-1) < lift_threshold
    
    # Gate: correct arm must be at its lift target
    correct_at_target = torch.where(is_cube, left_at_target, right_at_target)
    
    # Correct arm: left for cube, right for fruit
    correct_dist = torch.where(is_cube, left_dist, right_dist)
    reward = torch.exp(-correct_dist / std)
    
    # Only give reward if correct arm is at lift target
    reward = reward * correct_at_target.float()
    
    return reward


def wrong_arm_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Penalize the wrong arm for getting too close to the object.
    
    ONLY active during Phase 2 (when BOTH arms have reached lift targets).
    During Phase 1, we just want arms to reach lift targets without penalties.
    
    - Cube → Penalize right arm for approaching
    - Fruit → Penalize left arm for approaching
    """
    # Check if we're in Phase 2 (both arms reached lift target)
    if not hasattr(env, '_arm_reached_lift_target'):
        # Phase 1 not initialized yet, no penalty
        num_envs = env.num_envs
        device = env.device
        return torch.zeros(num_envs, device=device)
    
    both_at_target = env._arm_reached_lift_target[:, 0] & env._arm_reached_lift_target[:, 1]
    
    is_cube = _get_object_type_mask(env)
    left_dist, right_dist = _get_arm_distances(
        env, object_cfg, left_ee_frame_cfg, right_ee_frame_cfg
    )
    
    # Wrong arm: right for cube, left for fruit
    wrong_dist = torch.where(is_cube, right_dist, left_dist)
    
    # Penalty increases as wrong arm gets closer (inverted Gaussian)
    penalty = torch.exp(-wrong_dist / std)
    
    # Only apply penalty during Phase 2
    penalty = penalty * both_at_target.float()
    
    return penalty


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def _get_object_spawn_height(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get the object's configured spawn height from init_state.
    
    Uses object.data.default_root_state which contains the init_state.pos
    from the configuration. This is a fixed value, not the actual object position.
    """
    object: RigidObject = env.scene[object_cfg.name]
    # default_root_state is [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, ...]
    # Index 2 is the z-coordinate from init_state.pos
    return object.data.default_root_state[:, 2]


def object_is_lifted_relative(
    env: ManagerBasedRLEnv,
    min_delta: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward when object is lifted above its spawn height by min_delta."""
    object: RigidObject = env.scene[object_cfg.name]
    spawn_z = _get_object_spawn_height(env, object_cfg)
    return torch.where(object.data.root_pos_w[:, 2] > spawn_z + min_delta, 1.0, 0.0)


def object_to_correct_pail(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    left_pail_pos: tuple = (0.3, 0.35, 0.1),   # On ground beside table
    right_pail_pos: tuple = (0.3, -0.35, 0.1),  # On ground beside table
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for moving object toward the correct pail (only when lifted).
    
    - Cube → Move toward left_pail
    - Fruit → Move toward right_pail
    """
    is_cube = _get_object_type_mask(env)
    object: RigidObject = env.scene[object_cfg.name]
    object_pos = object.data.root_pos_w[:, :3]
    
    # Convert pail positions to tensors
    left_goal = torch.tensor(left_pail_pos, device=env.device).unsqueeze(0)
    right_goal = torch.tensor(right_pail_pos, device=env.device).unsqueeze(0)
    
    # Select correct pail based on object type
    goal_pos = torch.where(is_cube.unsqueeze(-1), left_goal, right_goal)
    
    distance = torch.norm(goal_pos - object_pos, dim=-1)
    
    # Only reward if object is lifted
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    reward = torch.exp(-distance / std) * is_lifted.float()
    
    return reward


def object_in_correct_pail(
    env: ManagerBasedRLEnv,
    left_pail_pos: tuple = (0.3, 0.35, 0.0),
    right_pail_pos: tuple = (0.3, -0.35, 0.0),
    threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Large reward when object is dropped in the correct pail.
    
    - Cube in left_pail = success
    - Fruit in right_pail = success
    """
    is_cube = _get_object_type_mask(env)
    object: RigidObject = env.scene[object_cfg.name]
    object_pos = object.data.root_pos_w[:, :3]
    
    # Pail positions (XY only for horizontal distance)
    left_goal_xy = torch.tensor(left_pail_pos[:2], device=env.device)
    right_goal_xy = torch.tensor(right_pail_pos[:2], device=env.device)
    
    # Distance to each pail (XY plane)
    left_dist = torch.norm(object_pos[:, :2] - left_goal_xy, dim=-1)
    right_dist = torch.norm(object_pos[:, :2] - right_goal_xy, dim=-1)
    
    # Check if in correct pail
    cube_in_left = is_cube & (left_dist < threshold)
    fruit_in_right = (~is_cube) & (right_dist < threshold)
    
    success = cube_in_left | fruit_in_right
    
    return success.float()


# Keep old functions for backwards compatibility
def closest_arm_reaching_object(
    env: ManagerBasedRLEnv,
    std: float,
    left_lift_target: tuple = (0.0, 0.2, 0.355),
    right_lift_target: tuple = (0.0, -0.2, 0.355),
    lift_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Legacy: Reward the closest arm for reaching (deprecated, use correct_arm_reaching_object)."""
    return correct_arm_reaching_object(env, std, left_lift_target, right_lift_target, lift_threshold, object_cfg, left_ee_frame_cfg, right_ee_frame_cfg)


def closest_arm_object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    left_lift_target: tuple = (0.0, 0.2, 0.355),
    right_lift_target: tuple = (0.0, -0.2, 0.355),
    lift_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Tanh-based reward for correct arm EE distance, gated by being at lift target."""
    is_cube = _get_object_type_mask(env)
    
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]
    
    left_ee_pos = left_ee_frame.data.target_pos_w[:, 0, :]
    right_ee_pos = right_ee_frame.data.target_pos_w[:, 0, :]
    
    left_dist, right_dist = _get_arm_distances(
        env, object_cfg, left_ee_frame_cfg, right_ee_frame_cfg
    )
    
    # Check if arms are at their lift targets
    left_target_t = torch.tensor(left_lift_target, device=left_ee_pos.device)
    right_target_t = torch.tensor(right_lift_target, device=right_ee_pos.device)
    
    left_at_target = torch.norm(left_ee_pos - left_target_t, dim=-1) < lift_threshold
    right_at_target = torch.norm(right_ee_pos - right_target_t, dim=-1) < lift_threshold
    
    # Gate: correct arm must be at its lift target
    correct_at_target = torch.where(is_cube, left_at_target, right_at_target)
    
    correct_dist = torch.where(is_cube, left_dist, right_dist)
    reward = 1.0 - torch.tanh(correct_dist / std)
    
    # Only give reward if correct arm is at lift target
    return reward * correct_at_target.float()


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for moving object toward command goal position (train_lift style)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )

    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (
        1.0 - torch.tanh(distance / std)
    )


def object_goal_distance_relative(
    env: ManagerBasedRLEnv,
    std: float,
    min_delta: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Goal tracking reward gated by relative lift height."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )

    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    spawn_z = _get_object_spawn_height(env, object_cfg)
    is_lifted = object.data.root_pos_w[:, 2] > spawn_z + min_delta
    return is_lifted.float() * (1.0 - torch.tanh(distance / std))


def ee_above_table_shaping(
    env: ManagerBasedRLEnv,
    table_height: float = 0.255,
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Reward for EE being at or above table height.
    
    Simple reward: 1.0 when above table, proportional when below.
    This encourages arms to lift up.
    """
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]
    
    left_ee_pos = left_ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)
    right_ee_pos = right_ee_frame.data.target_pos_w[:, 0, :]
    
    # Reward for being at table height or above
    # Normalize: 0 at ground, 1 at table height, capped at 1
    left_height_reward = torch.clamp(left_ee_pos[:, 2] / table_height, 0.0, 1.0)
    right_height_reward = torch.clamp(right_ee_pos[:, 2] / table_height, 0.0, 1.0)
    
    # Average both arms
    return (left_height_reward + right_height_reward) / 2.0


def lift_target_distance(
    env: ManagerBasedRLEnv,
    left_target: tuple = (0.0, 0.2, 0.355),
    right_target: tuple = (0.0, -0.2, 0.355),
    target_threshold: float = 0.05,
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Raw distance from EE to lift target (for penalty, like reach task).
    
    Uses raw distance penalty like the WORKING reach task.
    Raw distance gives CONSTANT gradient regardless of how far away,
    unlike exp() which has weak gradient when far.
    
    ONE-WAY GATE: Once an arm reaches its target, distance returns 0
    permanently for that arm until episode reset.
    
    Returns sum of distances (0 when both arms have reached targets).
    Use with negative weight to penalize distance.
    """
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]
    
    left_ee_pos = left_ee_frame.data.target_pos_w[:, 0, :]
    right_ee_pos = right_ee_frame.data.target_pos_w[:, 0, :]
    
    num_envs = left_ee_pos.shape[0]
    device = left_ee_pos.device
    
    # Initialize one-way gate tracking if not exists
    if not hasattr(env, '_arm_reached_lift_target'):
        env._arm_reached_lift_target = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
    
    # Target positions as tensors
    left_target_t = torch.tensor(left_target, device=device)
    right_target_t = torch.tensor(right_target, device=device)
    
    # Distance to target
    left_dist = torch.norm(left_ee_pos - left_target_t, dim=-1)
    right_dist = torch.norm(right_ee_pos - right_target_t, dim=-1)
    
    # Check if currently at target (within threshold)
    left_at_target_now = left_dist < target_threshold
    right_at_target_now = right_dist < target_threshold
    
    # Update one-way gate: once reached, stays reached
    env._arm_reached_lift_target[:, 0] = env._arm_reached_lift_target[:, 0] | left_at_target_now
    env._arm_reached_lift_target[:, 1] = env._arm_reached_lift_target[:, 1] | right_at_target_now
    
    # Use the one-way gate
    left_reached = env._arm_reached_lift_target[:, 0]
    right_reached = env._arm_reached_lift_target[:, 1]
    
    # Zero out distance if arm has reached target
    left_dist = torch.where(left_reached, torch.zeros_like(left_dist), left_dist)
    right_dist = torch.where(right_reached, torch.zeros_like(right_dist), right_dist)
    
    return left_dist + right_dist


def lift_target_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    left_target: tuple = (0.0, 0.2, 0.355),
    right_target: tuple = (0.0, -0.2, 0.355),
    target_threshold: float = 0.05,
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Reward for being close to lift target (tanh kernel).
    
    ONE-WAY GATE: Once an arm reaches its target, reward returns 1.0
    permanently for that arm until episode reset.
    
    Returns sum of (1 - tanh(dist/std)) for each arm.
    Use with positive weight.
    """
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]
    
    left_ee_pos = left_ee_frame.data.target_pos_w[:, 0, :]
    right_ee_pos = right_ee_frame.data.target_pos_w[:, 0, :]
    
    num_envs = left_ee_pos.shape[0]
    device = left_ee_pos.device
    
    # Initialize one-way gate tracking if not exists
    if not hasattr(env, '_arm_reached_lift_target'):
        env._arm_reached_lift_target = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
    
    # Target positions as tensors
    left_target_t = torch.tensor(left_target, device=device)
    right_target_t = torch.tensor(right_target, device=device)
    
    # Distance to target
    left_dist = torch.norm(left_ee_pos - left_target_t, dim=-1)
    right_dist = torch.norm(right_ee_pos - right_target_t, dim=-1)
    
    # Check if currently at target (within threshold)
    left_at_target_now = left_dist < target_threshold
    right_at_target_now = right_dist < target_threshold
    
    # Update one-way gate: once reached, stays reached
    env._arm_reached_lift_target[:, 0] = env._arm_reached_lift_target[:, 0] | left_at_target_now
    env._arm_reached_lift_target[:, 1] = env._arm_reached_lift_target[:, 1] | right_at_target_now
    
    # Use the one-way gate
    left_reached = env._arm_reached_lift_target[:, 0]
    right_reached = env._arm_reached_lift_target[:, 1]
    
    # Tanh reward: 1 when close, 0 when far
    left_reward = 1.0 - torch.tanh(left_dist / std)
    right_reward = 1.0 - torch.tanh(right_dist / std)
    
    # Once reached, give max reward (1.0)
    left_reward = torch.where(left_reached, torch.ones_like(left_reward), left_reward)
    right_reward = torch.where(right_reached, torch.ones_like(right_reward), right_reward)
    
    return left_reward + right_reward


def at_lift_target(
    env: ManagerBasedRLEnv,
    left_target: tuple = (0.0, 0.2, 0.355),
    right_target: tuple = (0.0, -0.2, 0.355),
    threshold: float = 0.05,
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Check if both arms are at their lift target positions.
    
    Returns 1.0 if BOTH arms are within threshold of their targets, 0.0 otherwise.
    This can be used to gate the reaching reward.
    """
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]
    
    left_ee_pos = left_ee_frame.data.target_pos_w[:, 0, :]
    right_ee_pos = right_ee_frame.data.target_pos_w[:, 0, :]
    
    left_target_t = torch.tensor(left_target, device=left_ee_pos.device)
    right_target_t = torch.tensor(right_target, device=right_ee_pos.device)
    
    left_dist = torch.norm(left_ee_pos - left_target_t, dim=-1)
    right_dist = torch.norm(right_ee_pos - right_target_t, dim=-1)
    
    both_at_target = (left_dist < threshold) & (right_dist < threshold)
    
    return both_at_target.float()


# ===== Curriculum Learning: Gated Rewards =====

def _get_arm_active_flags(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Get active flags for both arms from the curriculum system."""
    from .events import get_arm_active_flags
    return get_arm_active_flags(env)


def left_arm_object_distance_tanh_gated(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_left"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
) -> torch.Tensor:
    """Tanh-kernel reward for left arm reaching its object - only when active.
    
    Returns 0 when left arm is inactive (curriculum learning).
    """
    left_active, _ = _get_arm_active_flags(env)
    
    # Get base reward
    reward = left_arm_object_distance_tanh(env, std, object_cfg, left_ee_frame_cfg)
    
    # Gate by active flag
    return reward * left_active.float()


def right_arm_object_distance_tanh_gated(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_right"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Tanh-kernel reward for right arm reaching its object - only when active.
    
    Returns 0 when right arm is inactive (curriculum learning).
    """
    _, right_active = _get_arm_active_flags(env)
    
    # Get base reward
    reward = right_arm_object_distance_tanh(env, std, object_cfg, right_ee_frame_cfg)
    
    # Gate by active flag
    return reward * right_active.float()


def left_object_is_lifted_relative_gated(
    env: ManagerBasedRLEnv,
    min_delta: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_left"),
) -> torch.Tensor:
    """Reward when left object is lifted - only when left arm active."""
    left_active, _ = _get_arm_active_flags(env)
    
    reward = object_is_lifted_relative(env, min_delta, object_cfg)
    return reward * left_active.float()


def right_object_is_lifted_relative_gated(
    env: ManagerBasedRLEnv,
    min_delta: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_right"),
) -> torch.Tensor:
    """Reward when right object is lifted - only when right arm active."""
    _, right_active = _get_arm_active_flags(env)
    
    reward = object_is_lifted_relative(env, min_delta, object_cfg)
    return reward * right_active.float()


def left_object_goal_distance_relative_gated(
    env: ManagerBasedRLEnv,
    std: float,
    min_delta: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_left"),
) -> torch.Tensor:
    """Goal tracking for left object - only when left arm active."""
    left_active, _ = _get_arm_active_flags(env)
    
    reward = object_goal_distance_relative(env, std, min_delta, command_name, robot_cfg, object_cfg)
    return reward * left_active.float()


def right_object_goal_distance_relative_gated(
    env: ManagerBasedRLEnv,
    std: float,
    min_delta: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_right"),
) -> torch.Tensor:
    """Goal tracking for right object - only when right arm active."""
    _, right_active = _get_arm_active_flags(env)
    
    reward = object_goal_distance_relative(env, std, min_delta, command_name, robot_cfg, object_cfg)
    return reward * right_active.float()


# ===== Stay-at-Pose Rewards for Inactive Arms =====

def left_arm_distance_from_default(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Distance penalty for left arm when inactive.
    
    Returns the L2 distance from default pose when arm is INACTIVE.
    Returns 0 when arm is ACTIVE (no penalty for active arm).
    
    Use with NEGATIVE weight to penalize drifting from default pose.
    """
    left_active, _ = _get_arm_active_flags(env)
    
    robot = env.scene["robot"]
    
    # Get left arm joint indices
    left_joint_names = [
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6", 
        "openarm_left_joint7"
    ]
    left_joint_ids = [robot.find_joints(name)[0][0] for name in left_joint_names]
    
    # Current positions
    current_pos = robot.data.joint_pos[:, left_joint_ids]
    
    # Default positions from robot config (typically zeros)
    default_pos = robot.data.default_joint_pos[:, left_joint_ids]
    
    # Distance from default pose
    dist = torch.norm(current_pos - default_pos, dim=-1)
    
    # Only apply penalty when arm is INACTIVE
    return dist * (~left_active).float()


# Keep old name for backwards compatibility
def left_arm_stay_at_default_pose(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
) -> torch.Tensor:
    """Deprecated: Use left_arm_distance_from_default with negative weight."""
    return left_arm_distance_from_default(env)


def left_arm_stay_at_initial_pose(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
) -> torch.Tensor:
    """Deprecated: Use left_arm_distance_from_default with negative weight."""
    return left_arm_distance_from_default(env)


def right_arm_distance_from_default(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Distance penalty for right arm when inactive.
    
    Returns the L2 distance from default pose when arm is INACTIVE.
    Returns 0 when arm is ACTIVE (no penalty for active arm).
    
    Use with NEGATIVE weight to penalize drifting from default pose.
    """
    _, right_active = _get_arm_active_flags(env)
    
    robot = env.scene["robot"]
    
    # Get right arm joint indices
    right_joint_names = [
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
        "openarm_right_joint7"
    ]
    right_joint_ids = [robot.find_joints(name)[0][0] for name in right_joint_names]
    
    # Current positions
    current_pos = robot.data.joint_pos[:, right_joint_ids]
    
    # Default positions from robot config (typically zeros)
    default_pos = robot.data.default_joint_pos[:, right_joint_ids]
    
    # Distance from default pose
    dist = torch.norm(current_pos - default_pos, dim=-1)
    
    # Only apply penalty when arm is INACTIVE
    return dist * (~right_active).float()


# Keep old name for backwards compatibility
def right_arm_stay_at_default_pose(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
) -> torch.Tensor:
    """Deprecated: Use right_arm_distance_from_default with negative weight."""
    return right_arm_distance_from_default(env)


def right_arm_stay_at_initial_pose(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
) -> torch.Tensor:
    """Deprecated: Use right_arm_distance_from_default with negative weight."""
    return right_arm_distance_from_default(env)


# ===== Termination Functions =====

def left_object_dropped_gated(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_left"),
) -> torch.Tensor:
    """Terminate if left object drops below minimum height - only when left arm is active.
    
    If left arm is inactive, the cube is intentionally moved below the table,
    so we don't want to terminate for that.
    """
    left_active, _ = _get_arm_active_flags(env)
    
    object: RigidObject = env.scene[object_cfg.name]
    is_below = object.data.root_pos_w[:, 2] < minimum_height
    
    # Only terminate if arm is active AND cube dropped
    return is_below & left_active


def right_object_dropped_gated(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_right"),
) -> torch.Tensor:
    """Terminate if right object drops below minimum height - only when right arm is active.
    
    If right arm is inactive, the cube is intentionally moved below the table,
    so we don't want to terminate for that.
    """
    _, right_active = _get_arm_active_flags(env)
    
    object: RigidObject = env.scene[object_cfg.name]
    is_below = object.data.root_pos_w[:, 2] < minimum_height
    
    # Only terminate if arm is active AND cube dropped
    return is_below & right_active


def phase2_not_reached_timeout(
    env: ManagerBasedRLEnv,
    timeout_s: float = 2.0,
) -> torch.Tensor:
    """Terminate if arms haven't reached lift targets (phase 2) within timeout.
    
    Uses the one-way gate tracked by the lift target rewards.
    If both arms haven't triggered the gate by timeout, terminate.
    """
    num_envs = env.num_envs
    device = env.device
    
    # Check if one-way gate exists and both arms have reached
    if hasattr(env, '_arm_reached_lift_target'):
        left_reached = env._arm_reached_lift_target[:, 0]
        right_reached = env._arm_reached_lift_target[:, 1]
        both_reached = left_reached & right_reached
    else:
        # Gate not initialized yet, assume not reached
        both_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    # Check if past timeout
    time_exceeded = env.episode_length_buf * env.step_dt > timeout_s
    
    # Terminate if time exceeded and not in phase 2 (both arms at lift targets)
    return time_exceeded & (~both_reached)
