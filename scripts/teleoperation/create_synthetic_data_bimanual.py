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

"""
Synthetic Data Generation for Bimanual OpenArm

Based on play.py but uses the factory USD with cameras.

Usage:
    python create_synthetic_data_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 --checkpoint /path/to/model.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Synthetic data generation for bimanual OpenArm")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Reach-OpenArm-Bi-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run.")
parser.add_argument("--episode_length", type=int, default=200, help="Steps per episode.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import random

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401

# Import factory robot config (with cameras)
from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_FACTORY_HIGH_PD_CFG,
)

# USD imports for cube spawning
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics


def get_table_height(stage, table_name: str = "Danny") -> float:
    """Find table height from USD."""
    for prim in stage.Traverse():
        if table_name.lower() in prim.GetName().lower():
            bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(prim)
            if not bbox.GetRange().IsEmpty():
                return bbox.GetRange().GetMax()[2]
    return 0.0


def spawn_cube(stage, prim_path: str, position: tuple, size: float = 0.05):
    """Spawn a physics cube."""
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(size)
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(1, 0.2, 0.2)])
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr().Set(0.1)
    return cube


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Main entry point."""
    
    # Configure environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"
    
    # Use factory USD with cameras, start with arms raised above table
    from isaaclab.assets.articulation import ArticulationCfg
    env_cfg.scene.robot = OPEN_ARM_FACTORY_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # Left arm - raised pose (mirrored limits: j2 [-3.316, 0.175])
                "openarm_left_joint1": 0.0,
                "openarm_left_joint2": -0.8,  # shoulder pitch (negative for left arm)
                "openarm_left_joint3": 0.0,
                "openarm_left_joint4": 1.2,   # elbow bent
                "openarm_left_joint5": 0.0,
                "openarm_left_joint6": 0.0,
                "openarm_left_joint7": 0.0,
                # Right arm - raised pose (j2 [-0.175, 3.316])
                "openarm_right_joint1": 0.0,
                "openarm_right_joint2": 0.8,  # shoulder pitch (positive for right arm)
                "openarm_right_joint3": 0.0,
                "openarm_right_joint4": 1.2,  # elbow bent
                "openarm_right_joint5": 0.0,
                "openarm_right_joint6": 0.0,
                "openarm_right_joint7": 0.0,
                # Grippers open
                "openarm_left_finger_joint.*": 0.044,
                "openarm_right_finger_joint.*": 0.044,
            },
        ),
    )
    
    # Disable observation noise for clean data
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.enable_corruption = False
    
    print(f"\n[INFO] Creating environment: {args_cli.task}")
    print("[INFO] Using factory USD with cameras")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Print joint information for debugging initial pose
    robot = env.unwrapped.scene["robot"]
    joint_names = robot.joint_names
    joint_pos = robot.data.joint_pos[0].cpu().numpy()  # First env
    joint_limits = robot.data.soft_joint_pos_limits[0].cpu().numpy()  # [num_joints, 2]
    
    print("\n" + "="*80)
    print("JOINT INFORMATION")
    print("="*80)
    print(f"{'Joint Name':<30} {'Current Pos':>12} {'Lower Limit':>12} {'Upper Limit':>12}")
    print("-"*80)
    for i, name in enumerate(joint_names):
        pos = joint_pos[i]
        lower = joint_limits[i, 0]
        upper = joint_limits[i, 1]
        print(f"{name:<30} {pos:>12.4f} {lower:>12.4f} {upper:>12.4f}")
    print("="*80 + "\n")
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    resume_path = retrieve_file_path(args_cli.checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[INFO] Policy loaded!")
    
    # Get stage and table height
    stage = omni.usd.get_context().get_stage()
    table_height = get_table_height(stage, "Danny")
    print(f"[INFO] Table height (Danny): {table_height:.3f}")
    
    # Spawn cube
    cube_path = "/World/envs/env_0/Cube"
    if not stage.GetPrimAtPath(cube_path).IsValid():
        spawn_cube(stage, cube_path, (0.35, 0.0, table_height + 0.2))
        print(f"[INFO] Spawned cube at {cube_path}")
    
    print("\n" + "="*60)
    print("SYNTHETIC DATA GENERATION - BIMANUAL OPENARM")
    print("="*60)
    print(f"Episodes: {args_cli.num_episodes}")
    print(f"Steps per episode: {args_cli.episode_length}")
    print("="*60 + "\n")
    
    # Run episodes
    obs = env.get_observations()
    
    for episode in range(args_cli.num_episodes):
        print(f"\n--- Episode {episode + 1}/{args_cli.num_episodes} ---")
        
        # Randomize cube position
        cube = stage.GetPrimAtPath(cube_path)
        if cube.IsValid():
            x = random.uniform(0.2, 0.5)
            y = random.uniform(-0.2, 0.2)
            xform = UsdGeom.Xformable(cube)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, table_height + 0.1))
            print(f"[CUBE] Dropped at ({x:.2f}, {y:.2f})")
        
        # Run episode with policy
        for step in range(args_cli.episode_length):
            with torch.inference_mode():
                actions = policy(obs)
            
            obs, _, dones, _ = env.step(actions)
            
            # Print joint positions every step
            joint_pos = robot.data.joint_pos[0].cpu().numpy()
            pos_str = " ".join([f"{p:+.3f}" for p in joint_pos])
            print(f"[Step {step:3d}] Joints: {pos_str}")
            
            if not simulation_app.is_running():
                break
            
            if dones.any():
                print(f"  Episode ended at step {step + 1}")
                obs = env.get_observations()
                break
        
        print(f"  Episode {episode + 1} complete")
    
    print("\n[INFO] Synthetic data generation complete!")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
