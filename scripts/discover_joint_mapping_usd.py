#!/usr/bin/env python3
"""Discover the joint index mapping by loading the USD stage in Isaac Sim.

Uses AppLauncher to start Kit (for USD APIs) but reads the stage directly
without physics simulation, avoiding CUDA compatibility issues.

Usage (inside container):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/discover_joint_mapping_usd.py
    /workspace/isaaclab/_isaac_sim/python.sh scripts/discover_joint_mapping_usd.py --output joint_mapping.json
"""

import argparse
import json
import os
import re
import sys

parser = argparse.ArgumentParser(description="Discover OpenArm joint mapping from USD file")
parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file path")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("[discover] App launched, importing omni.usd...", flush=True)

import omni.usd
from pxr import UsdPhysics

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
usd_path = os.path.join(
    repo_root,
    "source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/openarm_bimanual/openarm_bimanual.usd",
)
if not os.path.exists(usd_path):
    usd_path = os.path.join(
        repo_root,
        "source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/openarm_bimanual/openarm_bimanual_factory.usd",
    )

print(f"[discover] Loading USD: {usd_path}", flush=True)

stage_ctx = omni.usd.get_context()
result = stage_ctx.open_stage(usd_path)
print(f"[discover] open_stage result: {result}", flush=True)

for _ in range(10):
    simulation_app.update()

stage = stage_ctx.get_stage()
print(f"[discover] Got stage: {stage}", flush=True)

joint_prims = []
prim_count = 0
for prim in stage.Traverse():
    prim_count += 1
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        joint_prims.append(prim)

print(f"[discover] Traversed {prim_count} prims, found {len(joint_prims)} physics joints", flush=True)

if len(joint_prims) == 0:
    print("[discover] No UsdPhysics joints found. Trying name-based search...", flush=True)
    for prim in stage.Traverse():
        name = prim.GetName()
        type_name = prim.GetTypeName()
        if "joint" in name.lower():
            print(f"  Prim: {prim.GetPath()} type={type_name} name={name}", flush=True)
            joint_prims.append(prim)

joint_names_ordered = [p.GetName() for p in joint_prims]

print(f"\nFound {len(joint_names_ordered)} joints:", flush=True)
for i, name in enumerate(joint_names_ordered):
    print(f"  [{i:2d}] {name}", flush=True)

left_arm_ids = []
left_arm_names = []
right_arm_ids = []
right_arm_names = []
left_grip_ids = []
left_grip_names = []
right_grip_ids = []
right_grip_names = []

for i, name in enumerate(joint_names_ordered):
    if re.match(r"openarm_left_joint[1-7]$", name):
        left_arm_ids.append(i)
        left_arm_names.append(name)
    elif re.match(r"openarm_right_joint[1-7]$", name):
        right_arm_ids.append(i)
        right_arm_names.append(name)
    elif re.match(r"openarm_left_finger_joint", name):
        left_grip_ids.append(i)
        left_grip_names.append(name)
    elif re.match(r"openarm_right_finger_joint", name):
        right_grip_ids.append(i)
        right_grip_names.append(name)

print(f"\n{'='*60}", flush=True)
print(f"OpenArm Joint Mapping Discovery (from USD)", flush=True)
print(f"{'='*60}", flush=True)
print(f"Total joints: {len(joint_names_ordered)}", flush=True)
print(f"\nLeft arm IDs:       {left_arm_ids}", flush=True)
print(f"Left arm names:     {left_arm_names}", flush=True)
print(f"Right arm IDs:      {right_arm_ids}", flush=True)
print(f"Right arm names:    {right_arm_names}", flush=True)
print(f"Left gripper IDs:   {left_grip_ids}", flush=True)
print(f"Left gripper names: {left_grip_names}", flush=True)
print(f"Right gripper IDs:  {right_grip_ids}", flush=True)
print(f"Right gripper names:{right_grip_names}", flush=True)

if left_arm_ids:
    print(f"\n16-DOF ALOHA-style mapping:", flush=True)
    print(f"  Left arm:    indices {left_arm_ids} from joint_pos", flush=True)
    if left_grip_ids:
        print(f"  Left grip:   index {left_grip_ids[0]} (first of {left_grip_ids})", flush=True)
    if right_arm_ids:
        print(f"  Right arm:   indices {right_arm_ids} from joint_pos", flush=True)
    if right_grip_ids:
        print(f"  Right grip:  index {right_grip_ids[0]} (first of {right_grip_ids})", flush=True)

mapping = {
    "total_joints": len(joint_names_ordered),
    "all_joint_names": joint_names_ordered,
    "left_arm_ids": left_arm_ids,
    "right_arm_ids": right_arm_ids,
    "left_gripper_ids": left_grip_ids,
    "right_gripper_ids": right_grip_ids,
    "left_arm_names": left_arm_names,
    "right_arm_names": right_arm_names,
    "left_gripper_names": left_grip_names,
    "right_gripper_names": right_grip_names,
    "aloha_16_mapping": {
        "description": "[left_arm(7), left_grip(1), right_arm(7), right_grip(1)]",
        "left_arm_indices": left_arm_ids,
        "left_grip_index": int(left_grip_ids[0]) if left_grip_ids else -1,
        "right_arm_indices": right_arm_ids,
        "right_grip_index": int(right_grip_ids[0]) if right_grip_ids else -1,
    },
}

if args.output:
    with open(args.output, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nMapping saved to: {args.output}", flush=True)

print(f"{'='*60}\n", flush=True)

simulation_app.close()
