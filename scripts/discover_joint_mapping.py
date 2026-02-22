#!/usr/bin/env python3
"""Discover the joint index mapping for the OpenArm bimanual robot.

Run inside the Isaac Lab container to print which indices in robot.data.joint_pos
correspond to which named joints. Outputs a JSON mapping file.

Usage (inside container):
    python scripts/discover_joint_mapping.py
    python scripts/discover_joint_mapping.py --output joint_mapping.json
"""

import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default=None)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import openarm.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

task_name = "Isaac-Reach-OpenArm-Bi-Teleop-v0"
env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=1)
env_cfg.scene.num_envs = 1

env = gym.make(task_name, cfg=env_cfg)
unwrapped = env.unwrapped
robot = unwrapped.scene["robot"]

left_arm_ids, left_arm_names = robot.find_joints([
    "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
    "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
    "openarm_left_joint7",
])
right_arm_ids, right_arm_names = robot.find_joints([
    "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
    "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
    "openarm_right_joint7",
])
left_grip_ids, left_grip_names = robot.find_joints("openarm_left_finger_joint.*")
right_grip_ids, right_grip_names = robot.find_joints("openarm_right_finger_joint.*")

all_names = robot.joint_names
total_joints = robot.data.joint_pos.shape[1]

print(f"\n{'='*60}")
print(f"OpenArm Joint Mapping Discovery")
print(f"{'='*60}")
print(f"Total joints in robot.data.joint_pos: {total_joints}")
print(f"\nAll joint names (in order):")
for i, name in enumerate(all_names):
    print(f"  [{i:2d}] {name}")

print(f"\nLeft arm IDs:     {list(left_arm_ids)}")
print(f"Left arm names:   {list(left_arm_names)}")
print(f"Right arm IDs:    {list(right_arm_ids)}")
print(f"Right arm names:  {list(right_arm_names)}")
print(f"Left gripper IDs: {list(left_grip_ids)}")
print(f"Left gripper names: {list(left_grip_names)}")
print(f"Right gripper IDs:  {list(right_grip_ids)}")
print(f"Right gripper names: {list(right_grip_names)}")

mapping = {
    "total_joints": total_joints,
    "all_joint_names": list(all_names),
    "left_arm_ids": list(left_arm_ids),
    "right_arm_ids": list(right_arm_ids),
    "left_gripper_ids": list(left_grip_ids),
    "right_gripper_ids": list(right_grip_ids),
    "left_arm_names": list(left_arm_names),
    "right_arm_names": list(right_arm_names),
    "left_gripper_names": list(left_grip_names),
    "right_gripper_names": list(right_grip_names),
    "aloha_16_mapping": {
        "description": "[left_arm(7), left_grip(1), right_arm(7), right_grip(1)]",
        "left_arm_indices": list(left_arm_ids),
        "left_grip_index": int(left_grip_ids[0]) if left_grip_ids else -1,
        "right_arm_indices": list(right_arm_ids),
        "right_grip_index": int(right_grip_ids[0]) if right_grip_ids else -1,
    },
}

print(f"\n16-DOF ALOHA-style mapping:")
print(f"  Left arm:    indices {list(left_arm_ids)} from joint_pos")
print(f"  Left grip:   index {left_grip_ids[0]} (first of {list(left_grip_ids)})")
print(f"  Right arm:   indices {list(right_arm_ids)} from joint_pos")
print(f"  Right grip:  index {right_grip_ids[0]} (first of {list(right_grip_ids)})")

if args.output:
    with open(args.output, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nMapping saved to: {args.output}")

print(f"{'='*60}\n")

env.close()
simulation_app.close()
