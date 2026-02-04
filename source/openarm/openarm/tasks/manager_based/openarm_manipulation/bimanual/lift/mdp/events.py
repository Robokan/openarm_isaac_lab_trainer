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
