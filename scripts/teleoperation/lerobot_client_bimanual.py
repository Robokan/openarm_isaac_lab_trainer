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
LeRobot ACT Client for Bimanual OpenArm

Runs a trained LeRobot ACT (or Diffusion Policy) model in Isaac Lab simulation.
No server required - model runs directly in this process.

Usage:
    python lerobot_client_bimanual.py --checkpoint outputs/openarm_act/checkpoints/last/pretrained_model
    python lerobot_client_bimanual.py --checkpoint outputs/openarm_act/checkpoints/last/pretrained_model --num_episodes 5
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="LeRobot ACT Client for Bimanual OpenArm")
parser.add_argument("--task", type=str, default="Isaac-Reach-OpenArm-Bi-Teleop-v0", 
                    help="Task name (use Teleop variant to match training environment)")
parser.add_argument("--checkpoint", type=str, required=True, 
                    help="Path to LeRobot checkpoint directory")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
parser.add_argument("--max_episode_steps", type=int, default=1000, help="Max steps per episode")
parser.add_argument("--fps", type=float, default=50.0, help="Control frequency")

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after Isaac Sim is initialized."""

import gymnasium as gym
import torch
import numpy as np
import cv2
import time
import random
from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401


class LeRobotACTPolicy:
    """Wrapper for LeRobot ACT policy inference."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load the policy
        print(f"[INFO] Loading LeRobot policy from {checkpoint_path}")
        
        try:
            from lerobot.common.policies.act.modeling_act import ACTPolicy
            from lerobot.common.policies.act.configuration_act import ACTConfig
            from safetensors.torch import load_file
            import json
            
            # Load config
            config_path = self.checkpoint_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                self.config = ACTConfig(**config_dict)
            else:
                # Try to load from parent directory
                config_path = self.checkpoint_path.parent.parent / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config_dict = json.load(f)
                    self.config = ACTConfig(**config_dict)
                else:
                    raise FileNotFoundError(f"Config not found at {config_path}")
            
            # Create policy
            self.policy = ACTPolicy(self.config)
            
            # Load weights
            weights_path = self.checkpoint_path / "model.safetensors"
            if not weights_path.exists():
                weights_path = self.checkpoint_path / "pytorch_model.bin"
            
            if weights_path.suffix == ".safetensors":
                state_dict = load_file(weights_path)
            else:
                state_dict = torch.load(weights_path, map_location=device)
            
            self.policy.load_state_dict(state_dict)
            self.policy.to(device)
            self.policy.eval()
            
            print(f"[INFO] Policy loaded successfully!")
            print(f"[INFO] Action dim: {self.config.output_features['action']['shape'][0]}")
            print(f"[INFO] Chunk size: {self.config.chunk_size}")
            
        except ImportError as e:
            print(f"[ERROR] LeRobot not installed or import error: {e}")
            print("[INFO] Install with: pip install lerobot")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load policy: {e}")
            raise
        
        # Action chunking state
        self.action_queue = []
        self.chunk_size = getattr(self.config, 'chunk_size', 10)
        self.n_action_steps = getattr(self.config, 'n_action_steps', 1)
    
    def reset(self):
        """Reset action queue for new episode."""
        self.action_queue = []
    
    @torch.no_grad()
    def get_action(self, observation: dict) -> np.ndarray:
        """Get action from policy given observation.
        
        Args:
            observation: Dict with 'state' and 'images' keys
            
        Returns:
            Action array of shape (action_dim,)
        """
        # If we have queued actions, use them
        if self.action_queue:
            return self.action_queue.pop(0)
        
        # Prepare observation for policy
        obs_dict = {}
        
        # State
        state = observation['state']
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        obs_dict['observation.state'] = state.unsqueeze(0).to(self.device)
        
        # Images - convert from (C, H, W) to (B, C, H, W) and normalize
        for cam_name, img in observation['images'].items():
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            # Normalize to [0, 1]
            if img.max() > 1.0:
                img = img / 255.0
            # Add batch dimension
            img = img.unsqueeze(0).to(self.device)
            
            # Map camera names to LeRobot format
            if cam_name == 'cam_high' or cam_name == 'ego':
                obs_dict['observation.images.top'] = img
            elif cam_name == 'cam_left_wrist' or cam_name == 'left_wrist':
                obs_dict['observation.images.left_wrist'] = img
            elif cam_name == 'cam_right_wrist' or cam_name == 'right_wrist':
                obs_dict['observation.images.right_wrist'] = img
        
        # Run inference
        action_chunk = self.policy.select_action(obs_dict)
        
        # action_chunk shape: (chunk_size, action_dim)
        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.cpu().numpy()
        
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, -1)
        
        # Queue actions (use n_action_steps from the chunk)
        n_steps = min(self.n_action_steps, len(action_chunk))
        for i in range(n_steps):
            self.action_queue.append(action_chunk[i])
        
        # Return first action
        return self.action_queue.pop(0)


class OpenArmEnvironment:
    """Environment wrapper for OpenArm in Isaac Lab."""
    
    IMAGE_SIZE = 224
    
    def __init__(self, env):
        self._env = env
        self._unwrapped = env.unwrapped
        if hasattr(self._unwrapped, 'unwrapped'):
            self._unwrapped = self._unwrapped.unwrapped
        
        self._device = "cuda:0"
        self._obs = None
        self._done = True
        self._step_count = 0
        
        # Initialize object pool
        self._object_pool = {"cubes": [], "mugs": [], "fruits": []}
        self._active_objects = []
        self._init_object_pool()
    
    def _init_object_pool(self):
        """Initialize object pool from scene assets."""
        scene_keys = list(self._unwrapped.scene.keys())
        
        for i in range(5):
            cube_name = f"pool_cube_{i}"
            if cube_name in scene_keys:
                self._object_pool["cubes"].append({
                    "asset": self._unwrapped.scene[cube_name], 
                    "active": False, "idx": i
                })
        
        for i in range(4):
            mug_name = f"pool_mug_{i}"
            if mug_name in scene_keys:
                self._object_pool["mugs"].append({
                    "asset": self._unwrapped.scene[mug_name],
                    "active": False, "idx": i
                })
        
        for i in range(6):
            fruit_name = f"pool_fruit_{i}"
            if fruit_name in scene_keys:
                self._object_pool["fruits"].append({
                    "asset": self._unwrapped.scene[fruit_name],
                    "active": False, "idx": i
                })
        
        print(f"[Pool] Found {len(self._object_pool['cubes'])} cubes, "
              f"{len(self._object_pool['mugs'])} mugs, {len(self._object_pool['fruits'])} fruits")
    
    def spawn_random_object(self, position=None):
        """Spawn a random object from the pool."""
        if position is None:
            spawn_x = random.uniform(0.15, 0.35)
            spawn_y = random.uniform(-0.15, 0.15)
            spawn_z = 0.55
            position = (spawn_x, spawn_y, spawn_z)
        
        pool_types = ["cubes", "mugs", "fruits"]
        random.shuffle(pool_types)
        
        for pool_type in pool_types:
            for obj in self._object_pool[pool_type]:
                if not obj["active"]:
                    obj["active"] = True
                    asset = obj["asset"]
                    
                    pos = torch.tensor([[position[0], position[1], position[2]]], device=asset.device)
                    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=asset.device)
                    vel = torch.zeros((1, 6), device=asset.device)
                    
                    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
                    asset.write_root_velocity_to_sim(vel)
                    
                    self._active_objects.append({"asset": asset, "type": pool_type})
                    print(f"[Pool] Spawned {pool_type[:-1]} at ({position[0]:.2f}, {position[1]:.2f})")
                    return True
        
        print("[Pool] All pools exhausted!")
        return False
    
    def reset_all_objects(self):
        """Return all objects to pool."""
        for obj_info in self._active_objects:
            asset = obj_info["asset"]
            pos = torch.tensor([[-2.0, 0.0, 0.03]], device=asset.device)
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=asset.device)
            vel = torch.zeros((1, 6), device=asset.device)
            asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
            asset.write_root_velocity_to_sim(vel)
            
            for pool_type in ["cubes", "mugs", "fruits"]:
                for pool_obj in self._object_pool[pool_type]:
                    if pool_obj["asset"] is asset:
                        pool_obj["active"] = False
        
        self._active_objects.clear()
    
    def reset(self):
        """Reset environment."""
        self._obs, _ = self._env.reset()
        self._done = False
        self._step_count = 0
        return self._obs
    
    def get_observation(self) -> dict:
        """Get observation in LeRobot-compatible format."""
        robot = self._unwrapped.scene["robot"]
        joint_pos = robot.data.joint_pos[0].cpu().numpy()
        
        # Extract joint positions (16 DOF: 7+1 per arm)
        num_joints = len(joint_pos)
        if num_joints >= 16:
            left_arm = joint_pos[0:7]
            left_gripper = np.clip(joint_pos[7:8] / 0.044, 0.0, 1.0)
            right_arm = joint_pos[8:15]
            right_gripper = np.clip(joint_pos[15:16] / 0.044, 0.0, 1.0)
        else:
            half = num_joints // 2
            left_arm = joint_pos[0:min(7, half-1)]
            left_gripper = np.array([0.0])
            right_arm = joint_pos[half:half+7] if num_joints > half else np.zeros(7)
            right_gripper = np.array([0.0])
        
        left_arm = np.pad(left_arm, (0, max(0, 7 - len(left_arm))), mode='constant')
        right_arm = np.pad(right_arm, (0, max(0, 7 - len(right_arm))), mode='constant')
        
        state = np.concatenate([left_arm, left_gripper, right_arm, right_gripper]).astype(np.float32)
        
        # Get camera images (placeholder - would need actual camera capture)
        black_image = np.zeros((3, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float32)
        images = {
            "cam_high": black_image.copy(),
            "cam_left_wrist": black_image.copy(),
            "cam_right_wrist": black_image.copy(),
        }
        
        return {
            "state": state,
            "images": images,
        }
    
    def step(self, action: np.ndarray):
        """Apply action to environment."""
        # Convert action to tensor
        joint_targets = torch.tensor(
            action, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        
        # Denormalize grippers (indices 7 and 15)
        if joint_targets.shape[1] >= 16:
            joint_targets[0, 7] = joint_targets[0, 7] * 0.044
            joint_targets[0, 15] = joint_targets[0, 15] * 0.044
        
        self._obs, _, dones, _ = self._env.step(joint_targets)
        self._step_count += 1
        
        if hasattr(dones, 'any'):
            self._done = dones.any().item()
        else:
            self._done = bool(dones)
        
        return self._obs, self._done
    
    @property
    def is_done(self):
        return self._done
    
    @property
    def step_count(self):
        return self._step_count


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Main entry point."""
    
    # Load LeRobot policy
    policy = LeRobotACTPolicy(args_cli.checkpoint)
    
    # Configure environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"
    
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.enable_corruption = False
    
    # Create environment
    print(f"\n[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=False)
    
    # Create environment wrapper
    openarm_env = OpenArmEnvironment(env)
    
    print("\n" + "="*60)
    print("OPENARM BIMANUAL - LEROBOT ACT CLIENT")
    print("="*60)
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Episodes: {args_cli.num_episodes}")
    print(f"Max steps/episode: {args_cli.max_episode_steps}")
    print(f"FPS: {args_cli.fps}")
    print("="*60 + "\n")
    
    # Auto-spawn some objects
    print("[INFO] Spawning objects...")
    openarm_env.reset()
    for _ in range(10):
        env.step(torch.zeros((1, 16), device="cuda:0"))
    
    for _ in range(2):
        openarm_env.spawn_random_object()
    
    for _ in range(30):
        env.step(torch.zeros((1, 16), device="cuda:0"))
    
    dt = 1.0 / args_cli.fps
    
    try:
        for episode in range(args_cli.num_episodes):
            print(f"\n[Episode {episode + 1}/{args_cli.num_episodes}]")
            
            openarm_env.reset()
            policy.reset()
            
            for _ in range(10):
                env.step(torch.zeros((1, 16), device="cuda:0"))
            
            episode_start = time.time()
            
            while not openarm_env.is_done and openarm_env.step_count < args_cli.max_episode_steps:
                step_start = time.time()
                
                # Get observation
                obs = openarm_env.get_observation()
                
                # Get action from policy
                action = policy.get_action(obs)
                
                # Apply action
                openarm_env.step(action)
                
                # Maintain FPS
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                
                if openarm_env.step_count % 100 == 0:
                    print(f"  Step {openarm_env.step_count}")
            
            episode_time = time.time() - episode_start
            print(f"  Episode finished: {openarm_env.step_count} steps in {episode_time:.1f}s")
            
            # Reset objects for next episode
            openarm_env.reset_all_objects()
            for _ in range(2):
                openarm_env.spawn_random_object()
            for _ in range(30):
                env.step(torch.zeros((1, 16), device="cuda:0"))
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    print("\n[INFO] LeRobot client finished")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
