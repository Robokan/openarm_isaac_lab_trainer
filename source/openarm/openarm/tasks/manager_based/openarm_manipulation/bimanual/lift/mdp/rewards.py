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


def correct_arm_reaching_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Reward the correct arm (based on object type) for reaching toward the object.
    
    - Cube → Left arm should reach
    - Fruit → Right arm should reach
    
    Uses a Gaussian kernel to reward proximity.
    """
    is_cube = _get_object_type_mask(env)
    left_dist, right_dist = _get_arm_distances(
        env, object_cfg, left_ee_frame_cfg, right_ee_frame_cfg
    )
    
    # Correct arm: left for cube, right for fruit
    correct_dist = torch.where(is_cube, left_dist, right_dist)
    reward = torch.exp(-correct_dist / std)
    
    return reward


def wrong_arm_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Penalize the wrong arm for getting too close to the object.
    
    - Cube → Penalize right arm for approaching
    - Fruit → Penalize left arm for approaching
    """
    is_cube = _get_object_type_mask(env)
    left_dist, right_dist = _get_arm_distances(
        env, object_cfg, left_ee_frame_cfg, right_ee_frame_cfg
    )
    
    # Wrong arm: right for cube, left for fruit
    wrong_dist = torch.where(is_cube, right_dist, left_dist)
    
    # Penalty increases as wrong arm gets closer (inverted Gaussian)
    penalty = torch.exp(-wrong_dist / std)
    
    return penalty


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_to_correct_pail(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    left_pail_pos: tuple = (0.4, 0.45, 0.1),
    right_pail_pos: tuple = (0.4, -0.45, 0.1),
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
    left_pail_pos: tuple = (0.4, 0.45, 0.0),
    right_pail_pos: tuple = (0.4, -0.45, 0.0),
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
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Legacy: Reward the closest arm for reaching (deprecated, use correct_arm_reaching_object)."""
    return correct_arm_reaching_object(env, std, object_cfg, left_ee_frame_cfg, right_ee_frame_cfg)


def closest_arm_object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Legacy: Tanh-based reward for closest EE distance."""
    is_cube = _get_object_type_mask(env)
    left_dist, right_dist = _get_arm_distances(
        env, object_cfg, left_ee_frame_cfg, right_ee_frame_cfg
    )
    correct_dist = torch.where(is_cube, left_dist, right_dist)
    return 1.0 - torch.tanh(correct_dist / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Legacy: Reward for moving object toward command goal position."""
    return object_to_correct_pail(env, std, minimal_height, object_cfg=object_cfg)
