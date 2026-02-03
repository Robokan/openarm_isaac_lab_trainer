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

"""Spherical pose command generator for sampling target poses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SphericalPoseCommand(UniformPoseCommand):
    """Command generator for pose commands within a sphere.

    Positions are sampled uniformly within a sphere defined by
    a center point and radius. An optional constraint limits
    sampling to a half-space (e.g., x >= min_x).

    For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts to quaternion (w, x, y, z).
    """

    cfg: "SphericalPoseCommandCfg"
    """Configuration for the command generator."""

    def __init__(self, cfg: "SphericalPoseCommandCfg", env: "ManagerBasedEnv"):
        """Initialize the command generator."""
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the given environment indices."""
        n = len(env_ids)
        device = self.device

        center = torch.tensor(
            self.cfg.sphere_center, device=device, dtype=torch.float32
        )
        radius = self.cfg.sphere_radius
        min_x = self.cfg.min_x

        # Sample until we get n valid poses (geometric constraint only)
        positions = torch.zeros(n, 3, device=device)
        remaining = torch.ones(n, dtype=torch.bool, device=device)

        for _ in range(100):  # Max attempts
            num_remaining = remaining.sum().item()
            if num_remaining == 0:
                break

            # Sample uniformly in sphere using cube root for radius
            u = torch.rand(num_remaining, device=device)
            r = radius * (u ** (1.0 / 3.0))  # Uniform in volume

            # Sample uniformly on sphere surface
            cos_theta = 2.0 * torch.rand(num_remaining, device=device) - 1.0
            theta = torch.acos(cos_theta)
            phi = 2.0 * math.pi * torch.rand(num_remaining, device=device)

            # Convert to Cartesian coordinates
            sin_theta = torch.sin(theta)
            x = r * sin_theta * torch.cos(phi)
            y = r * sin_theta * torch.sin(phi)
            z = r * cos_theta

            # Apply center offset
            new_pos = torch.stack([x, y, z], dim=-1) + center

            # Check geometric constraint: x >= min_x
            valid = new_pos[:, 0] >= min_x
            remaining_idx = torch.where(remaining)[0]
            valid_idx = torch.where(valid)[0]

            for vi in valid_idx:
                positions[remaining_idx[vi]] = new_pos[vi]
                remaining[remaining_idx[vi]] = False

        # Sample orientations
        euler_angles = torch.zeros(n, 3, device=device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        orientations = quat_from_euler_xyz(
            euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        )
        if self.cfg.make_quat_unique:
            orientations = quat_unique(orientations)

        # Set position and orientation commands
        self.pose_command_b[env_ids, 0] = positions[:, 0]
        self.pose_command_b[env_ids, 1] = positions[:, 1]
        self.pose_command_b[env_ids, 2] = positions[:, 2]
        self.pose_command_b[env_ids, 3:] = orientations


@configclass
class SphericalPoseCommandCfg(CommandTermCfg):
    """Configuration for spherical pose command generator."""

    class_type: type = SphericalPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment."""

    body_name: str = MISSING
    """Name of the body for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique. Defaults to False."""

    # Sphere parameters
    sphere_center: tuple[float, float, float] = MISSING
    """Center of the sphere (x, y, z) in the robot base frame."""

    sphere_radius: float = MISSING
    """Radius of the sphere in meters."""

    min_x: float = 0.0
    """Minimum x coordinate. Only positions with x >= min_x are sampled."""

    # Kept for compatibility but not used
    validate_ik: bool = False
    """Deprecated. IK validation is no longer supported."""

    @configclass
    class Ranges:
        """Distribution ranges for the orientation commands."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the orientation commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    """The configuration for the goal pose visualization marker."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker."""

    # Set the scale of the visualization markers
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
