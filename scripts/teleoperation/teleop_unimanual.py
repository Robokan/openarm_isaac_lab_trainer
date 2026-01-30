# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""Unimanual Teleoperation for OpenArm"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Unimanual Teleoperation")
parser.add_argument("--task", type=str, default="Isaac-Reach-OpenArm-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--input", type=str, default="keyboard", choices=["vive", "keyboard"])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--sensitivity", type=float, default=1.0)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
import openarm.tasks


class KeyboardDevice:
    def __init__(self, sensitivity=1.0):
        self.step = 0.02 * sensitivity
        self.pose = np.array([0.3, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0])
        print("\nKEYBOARD: W/S=X, A/D=Y, Q/E=Z, R=Reset\n")
    def get_pose(self): return self.pose.copy()
    def update(self, k):
        if k == "w": self.pose[0] += self.step
        elif k == "s": self.pose[0] -= self.step
        elif k == "a": self.pose[1] += self.step
        elif k == "d": self.pose[1] -= self.step
        elif k == "q": self.pose[2] += self.step
        elif k == "e": self.pose[2] -= self.step
        elif k == "r": self.pose = np.array([0.3, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0])


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    print("\n" + "="*50 + "\nOPENARM UNIMANUAL TELEOPERATION\n" + "="*50)
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.enable_corruption = False
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device="cuda:0")
    print("[INFO] Policy loaded!")
    
    device = KeyboardDevice(args_cli.sensitivity)
    obs, _ = env.get_observations()
    step = 0
    
    try:
        while simulation_app.is_running():
            pose = device.get_pose()
            unwrapped = env.unwrapped.unwrapped
            if hasattr(unwrapped, 'command_manager') and "ee_pose" in unwrapped.command_manager._terms:
                cmd = torch.tensor(pose, dtype=torch.float32, device="cuda:0")
                unwrapped.command_manager._terms["ee_pose"].command[:, :3] = cmd[:3].unsqueeze(0)
                unwrapped.command_manager._terms["ee_pose"].command[:, 3:7] = cmd[3:7].unsqueeze(0)
            
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            step += 1
            if step % 60 == 0:
                print(f"Step {step:5d} | Pose: [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]")
            if dones.any():
                obs, _ = env.get_observations()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped")
    finally:
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
