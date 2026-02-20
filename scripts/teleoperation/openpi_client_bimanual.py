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
OpenPI Client for Bimanual OpenArm

Connects to a π₀ policy server and executes VLA actions in Isaac Lab simulation.
This is a drop-in replacement for the OpenPI ALOHA simulator.

Cameras:
    Uses the openarm_bimanual_factory.usd which has cameras mounted on:
    - openarm_body_link (base/high camera)
    - openarm_left_link7 (left wrist camera)
    - openarm_right_link7 (right wrist camera)

Usage:
    python openpi_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0 --checkpoint /path/to/model.pt
    python openpi_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0 --checkpoint /path/to/model.pt --host localhost --port 8000
    python openpi_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0 --checkpoint /path/to/model.pt --prompt "pick up the cube"
    
    # Spawn objects before running (interactive mode)
    python openpi_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0 --checkpoint /path/to/model.pt --spawn-objects
    
    # Spawn specific number of random objects automatically
    python openpi_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0 --checkpoint /path/to/model.pt --auto-spawn 3
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="OpenPI Client for Bimanual OpenArm")
parser.add_argument("--task", type=str, default="Isaac-Reach-OpenArm-Bi-Teleop-v0", help="Task name (use Teleop variant to match training environment)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")

# OpenPI connection arguments
parser.add_argument("--host", type=str, default="localhost", help="OpenPI policy server host")
parser.add_argument("--port", type=int, default=8000, help="OpenPI policy server port")
parser.add_argument("--action_horizon", type=int, default=10, help="Action chunk size")
parser.add_argument("--prompt", type=str, default="perform the bimanual manipulation task", 
                    help="Task prompt for VLA")
parser.add_argument("--max_hz", type=float, default=50.0, help="Max control frequency")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
parser.add_argument("--max_episode_steps", type=int, default=1000, help="Max steps per episode")

# Camera options
parser.add_argument("--no_cameras", action="store_true", help="Disable camera capture (use black images)")

# Object spawning options
parser.add_argument("--spawn-objects", action="store_true", 
                    help="Interactive mode: spawn objects before running Pi")
parser.add_argument("--auto-spawn", type=int, default=0, metavar="N",
                    help="Automatically spawn N random objects before running Pi")
parser.add_argument("--interactive", action="store_true",
                    help="Interactive prompt mode: ask for prompt before each episode")

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
import random
import time

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401


# =============================================================================
# OpenPI Environment Wrapper for Bimanual OpenArm
# =============================================================================

class OpenArmBimanualEnvironment:
    """
    OpenPI-compatible Environment wrapper for Bimanual OpenArm in Isaac Lab.
    
    Provides the same interface as openpi_client.runtime.Environment,
    using ALOHA-style observation format for bimanual robots:
    - State: [left_arm_joints(7), left_gripper(1), right_arm_joints(7), right_gripper(1)] = 16 DOF
    - Images: cam_high, cam_left_wrist, cam_right_wrist
    """

    # Bimanual OpenArm: 7 arm joints + 1 gripper per arm = 16 total
    NUM_ARM_JOINTS = 7
    TOTAL_DOF = 16  # (7+1) * 2
    IMAGE_SIZE = 224

    def __init__(self, env, policy, prompt: str = "perform the task", use_cameras: bool = True):
        """
        Args:
            env: Isaac Lab wrapped environment
            policy: Trained RL policy for low-level control (fallback)
            prompt: Task prompt for VLA
            use_cameras: Whether to capture camera images (False = black images)
        """
        self._env = env
        self._policy = policy
        self._prompt = prompt
        self._device = "cuda:0"
        self._use_cameras = use_cameras
        
        # Get unwrapped env for direct access
        self._unwrapped = env.unwrapped
        if hasattr(self._unwrapped, 'unwrapped'):
            self._unwrapped = self._unwrapped.unwrapped
        
        # Initialize cameras if enabled
        self._cameras = {}
        if self._use_cameras:
            self._init_cameras()
        
        # Initialize object pool
        self._object_pool = {"cubes": [], "mugs": [], "fruits": []}
        self._active_objects = []
        self._init_object_pool()
        
        # State
        self._obs = None
        self._done = True
        self._step_count = 0
    
    def _init_object_pool(self):
        """Initialize object pool from scene assets (cubes, mugs, fruits)."""
        scene_keys = list(self._unwrapped.scene.keys())
        
        # Load cubes (5 total)
        for i in range(5):
            cube_name = f"pool_cube_{i}"
            if cube_name in scene_keys:
                cube_asset = self._unwrapped.scene[cube_name]
                self._object_pool["cubes"].append({"asset": cube_asset, "active": False, "idx": i})
        
        # Load mugs (4 total)
        for i in range(4):
            mug_name = f"pool_mug_{i}"
            if mug_name in scene_keys:
                mug_asset = self._unwrapped.scene[mug_name]
                self._object_pool["mugs"].append({"asset": mug_asset, "active": False, "idx": i})
        
        # Load fruits (6 total)
        for i in range(6):
            fruit_name = f"pool_fruit_{i}"
            if fruit_name in scene_keys:
                fruit_asset = self._unwrapped.scene[fruit_name]
                self._object_pool["fruits"].append({"asset": fruit_asset, "active": False, "idx": i})
        
        print(f"[Pool] Found {len(self._object_pool['cubes'])} cubes, "
              f"{len(self._object_pool['mugs'])} mugs, {len(self._object_pool['fruits'])} fruits")
    
    def spawn_random_object(self, position=None):
        """Spawn a random object from the pool at the given position.
        
        Args:
            position: (x, y, z) tuple or None for random position near table center
            
        Returns:
            (prim_path, pool_type) or (None, None) if all pools exhausted
        """
        if position is None:
            # Random position on table
            spawn_x = random.uniform(0.15, 0.35)
            spawn_y = random.uniform(-0.15, 0.15)
            spawn_z = 0.55  # Table height + small offset
            position = (spawn_x, spawn_y, spawn_z)
        
        # Try random pool type
        pool_types = ["cubes", "mugs", "fruits"]
        random.shuffle(pool_types)
        
        for pool_type in pool_types:
            result = self._activate_pool_object(pool_type, position)
            if result[0] is not None:
                return result[0], pool_type
        
        print("[Pool] All object pools exhausted!")
        return None, None
    
    def spawn_object_by_type(self, pool_type: str, position=None):
        """Spawn a specific type of object from the pool.
        
        Args:
            pool_type: "cubes", "mugs", or "fruits"
            position: (x, y, z) tuple or None for random position
            
        Returns:
            prim_path or None if pool exhausted
        """
        if position is None:
            spawn_x = random.uniform(0.15, 0.35)
            spawn_y = random.uniform(-0.15, 0.15)
            spawn_z = 0.55
            position = (spawn_x, spawn_y, spawn_z)
        
        result = self._activate_pool_object(pool_type, position)
        return result[0]
    
    def _activate_pool_object(self, pool_type: str, position: tuple):
        """Activate an object from the pool at the given position."""
        pool_list = self._object_pool.get(pool_type, [])
        
        for obj in pool_list:
            if not obj["active"]:
                obj["active"] = True
                asset = obj["asset"]
                
                # Teleport to position
                pos = torch.tensor([[position[0], position[1], position[2]]], device=asset.device)
                quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=asset.device)
                vel = torch.zeros((1, 6), device=asset.device)
                
                asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
                asset.write_root_velocity_to_sim(vel)
                
                # Get prim path
                pool_name = asset.cfg.prim_path.split("/")[-1]
                prim_path = f"/World/envs/env_0/{pool_name}"
                
                self._active_objects.append({"asset": asset, "prim_path": prim_path, "type": pool_type})
                print(f"[Pool] Spawned {pool_type[:-1]} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
                return prim_path, asset
        
        return None, None
    
    def reset_all_objects(self):
        """Return all active objects to the pool."""
        for obj_info in self._active_objects:
            asset = obj_info["asset"]
            
            # Move back to floor
            pos = torch.tensor([[-2.0, 0.0, 0.03]], device=asset.device)
            quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=asset.device)
            vel = torch.zeros((1, 6), device=asset.device)
            
            asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
            asset.write_root_velocity_to_sim(vel)
            
            # Mark as inactive
            for pool_type in ["cubes", "mugs", "fruits"]:
                for pool_obj in self._object_pool[pool_type]:
                    if pool_obj["asset"] is asset:
                        pool_obj["active"] = False
        
        count = len(self._active_objects)
        self._active_objects.clear()
        if count > 0:
            print(f"[Pool] Reset {count} objects to pool")
    
    def get_pool_status(self):
        """Get status of object pools."""
        status = {}
        for pool_type, pool_list in self._object_pool.items():
            total = len(pool_list)
            active = sum(1 for obj in pool_list if obj["active"])
            status[pool_type] = {"total": total, "active": active, "available": total - active}
        return status
    
    def set_prompt(self, prompt: str):
        """Update the task prompt."""
        self._prompt = prompt
        print(f"[INFO] Prompt set to: {prompt}")

    def _init_cameras(self):
        """Initialize camera sensors attached to the robot."""
        try:
            import omni.isaac.core.utils.prims as prim_utils
            
            # Camera prim paths (relative to robot)
            # These cameras should exist in the openarm_bimanual_factory.usd
            camera_configs = {
                "cam_high": "{ENV_REGEX_NS}/Robot/openarm_body_link/Camera",
                "cam_left_wrist": "{ENV_REGEX_NS}/Robot/openarm_left_link7/Camera", 
                "cam_right_wrist": "{ENV_REGEX_NS}/Robot/openarm_right_link7/Camera",
            }
            
            # Check if cameras exist in the USD
            for name, prim_path in camera_configs.items():
                # Replace ENV_REGEX_NS with env_0 for the first environment
                resolved_path = prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
                if prim_utils.is_prim_path_valid(resolved_path):
                    print(f"[INFO] Found camera: {name} at {resolved_path}")
                    self._cameras[name] = resolved_path
                else:
                    print(f"[WARN] Camera not found: {name} at {resolved_path}")
            
            if not self._cameras:
                print("[WARN] No cameras found in USD. Using black images.")
                print("[INFO] Make sure to use the factory USD with cameras:")
                print("       openarm_bimanual_factory.usd")
                self._use_cameras = False
                
        except Exception as e:
            print(f"[WARN] Failed to initialize cameras: {e}")
            self._use_cameras = False

    def _capture_camera(self, camera_name: str) -> np.ndarray:
        """Capture image from a camera sensor."""
        if not self._use_cameras or camera_name not in self._cameras:
            return np.zeros((3, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
        
        try:
            import omni.replicator.core as rep
            from omni.isaac.sensor import Camera as IsaacCamera
            
            prim_path = self._cameras[camera_name]
            
            # Get camera render product
            camera = IsaacCamera(prim_path)
            camera.initialize()
            
            # Get RGBA image
            rgba = camera.get_rgba()
            
            if rgba is not None:
                # Convert RGBA to RGB, resize, and transpose to [C, H, W]
                rgb = rgba[:, :, :3]
                rgb_resized = cv2.resize(rgb, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                rgb_chw = np.transpose(rgb_resized, (2, 0, 1))  # [H, W, C] -> [C, H, W]
                return rgb_chw.astype(np.uint8)
            
        except Exception as e:
            # Silently fall back to black image
            pass
        
        return np.zeros((3, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)

    def reset(self) -> None:
        """Reset the environment."""
        self._obs, _ = self._env.reset()
        self._done = False
        self._step_count = 0

    def is_episode_complete(self) -> bool:
        """Check if episode is complete."""
        return self._done

    def get_observation(self) -> dict:
        """
        Get observation in ALOHA-compatible format for OpenPI.
        
        Returns:
            dict with:
                - state: [16] array (left_arm(7), left_gripper(1), right_arm(7), right_gripper(1))
                - images: dict with camera images in [C, H, W] format
                - prompt: task instruction
        """
        if self._obs is None:
            raise RuntimeError("Observation not set. Call reset() first.")
        
        robot = self._unwrapped.scene["robot"]
        joint_pos = robot.data.joint_pos[0].cpu().numpy()
        
        # Extract left and right arm joints + grippers
        # Joint ordering based on openarm_bimanual.py:
        # left_joint1-7, left_finger_joint, right_joint1-7, right_finger_joint
        num_joints = len(joint_pos)
        
        if num_joints >= 16:
            # Full bimanual: 7 left arm + 1 left gripper + 7 right arm + 1 right gripper
            left_arm = joint_pos[0:7]
            left_gripper = np.clip(joint_pos[7:8] / 0.044, 0.0, 1.0)
            right_arm = joint_pos[8:15]
            right_gripper = np.clip(joint_pos[15:16] / 0.044, 0.0, 1.0)
        else:
            # Fallback: split evenly
            half = num_joints // 2
            left_arm = joint_pos[0:min(7, half-1)]
            left_gripper = np.clip(joint_pos[half-1:half] / 0.044, 0.0, 1.0) if half > 0 else np.array([0.0])
            right_arm = joint_pos[half:half+7] if num_joints > half else np.zeros(7)
            right_gripper = np.clip(joint_pos[-1:] / 0.044, 0.0, 1.0)
        
        # Pad to correct sizes if needed
        left_arm = np.pad(left_arm, (0, max(0, 7 - len(left_arm))), mode='constant')
        right_arm = np.pad(right_arm, (0, max(0, 7 - len(right_arm))), mode='constant')
        left_gripper = left_gripper if len(left_gripper) == 1 else np.array([0.0])
        right_gripper = right_gripper if len(right_gripper) == 1 else np.array([0.0])
        
        # Combine into ALOHA-style state: [left_arm, left_gripper, right_arm, right_gripper]
        state = np.concatenate([left_arm, left_gripper, right_arm, right_gripper]).astype(np.float32)
        
        # Capture camera images
        if self._use_cameras:
            images = {
                "cam_high": self._capture_camera("cam_high"),
                "cam_left_wrist": self._capture_camera("cam_left_wrist"),
                "cam_right_wrist": self._capture_camera("cam_right_wrist"),
            }
        else:
            # Placeholder black images
            black_image = np.zeros((3, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
            images = {
                "cam_high": black_image.copy(),
                "cam_left_wrist": black_image.copy(),
                "cam_right_wrist": black_image.copy(),
            }
        
        return {
            "state": state,
            "images": images,
            "prompt": self._prompt,
        }

    def apply_action(self, action: dict) -> None:
        """
        Apply action from OpenPI policy.
        
        The VLA returns actions in ALOHA format:
        [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)] = 16 DOF
        
        These are applied directly as joint position targets to the simulation.
        """
        vla_actions = action.get("actions")
        
        if vla_actions is not None:
            # Convert VLA actions to tensor
            vla_actions = np.asarray(vla_actions)
            
            # Handle action chunks - VLA may return [action_horizon, 16] or [16]
            if vla_actions.ndim == 2:
                # Action chunk - ActionChunkBroker already slices, but just in case
                vla_actions = vla_actions[0]
            
            # VLA actions are in ALOHA format: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
            # Convert to tensor for the environment
            joint_targets = torch.tensor(
                vla_actions, dtype=torch.float32, device=self._device
            ).unsqueeze(0)  # Add batch dimension
            
            # Denormalize gripper values (VLA uses 0-1, sim uses 0-0.044)
            # Indices 7 and 15 are the grippers
            if joint_targets.shape[1] >= 16:
                joint_targets[0, 7] = joint_targets[0, 7] * 0.044  # Left gripper
                joint_targets[0, 15] = joint_targets[0, 15] * 0.044  # Right gripper
            
            # Apply VLA joint targets directly to the simulation
            self._obs, _, dones, _ = self._env.step(joint_targets)
        else:
            # Fallback: use trained RL policy if no VLA actions
            with torch.inference_mode():
                actions = self._policy(self._obs)
            self._obs, _, dones, _ = self._env.step(actions)
        
        self._step_count += 1
        
        if hasattr(dones, 'any'):
            self._done = dones.any().item()
        else:
            self._done = bool(dones)
        
        if self._done:
            self._obs, _ = self._env.get_observations()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Main entry point for OpenPI client."""
    
    # Import OpenPI client components
    try:
        from openpi_client import action_chunk_broker
        from openpi_client import websocket_client_policy
        from openpi_client.runtime import runtime as openpi_runtime
        from openpi_client.runtime.agents import policy_agent
    except ImportError as e:
        print(f"\n[ERROR] OpenPI client not installed: {e}")
        print("[INFO] Install with: pip install -e /path/to/openpi/packages/openpi-client")
        print("[INFO] For example: pip install -e ../openpi/packages/openpi-client")
        return
    
    # Configure environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"
    
    # Disable randomization for stable control
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.enable_corruption = False
    
    # Create environment
    print(f"\n[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Load checkpoint (used as fallback policy)
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[INFO] Policy loaded successfully!")
    
    use_cameras = not args_cli.no_cameras
    
    print("\n" + "="*60)
    print("OPENARM BIMANUAL - OPENPI CLIENT")
    print("="*60)
    print(f"Connecting to π₀ server at {args_cli.host}:{args_cli.port}")
    print(f"Action horizon: {args_cli.action_horizon}")
    print(f"Max Hz: {args_cli.max_hz}")
    print(f"Episodes: {args_cli.num_episodes}")
    print(f"Max steps/episode: {args_cli.max_episode_steps}")
    print(f"Prompt: {args_cli.prompt}")
    print(f"Cameras: {'enabled' if use_cameras else 'disabled (black images)'}")
    print("="*60 + "\n")
    
    # Create OpenPI-compatible environment wrapper
    openpi_env = OpenArmBimanualEnvironment(
        env=env,
        policy=policy,
        prompt=args_cli.prompt,
        use_cameras=use_cameras,
    )
    
    # Reset environment to initialize
    openpi_env.reset()
    
    # Wait a moment for physics to settle
    for _ in range(10):
        env.step(torch.zeros((1, 16), device="cuda:0"))
    
    # Auto-spawn objects if requested
    if args_cli.auto_spawn > 0:
        print(f"\n[INFO] Auto-spawning {args_cli.auto_spawn} random objects...")
        for i in range(args_cli.auto_spawn):
            prim_path, obj_type = openpi_env.spawn_random_object()
            if prim_path is None:
                print(f"[WARN] Could not spawn object {i+1} - pools exhausted")
                break
        # Let physics settle
        for _ in range(30):
            env.step(torch.zeros((1, 16), device="cuda:0"))
        print()
    
    # Interactive object spawning mode
    if args_cli.spawn_objects:
        print("\n" + "="*60)
        print("INTERACTIVE OBJECT SPAWNING")
        print("="*60)
        print("Commands:")
        print("  c - Spawn cube")
        print("  m - Spawn mug")
        print("  f - Spawn fruit")
        print("  r - Spawn random object")
        print("  x - Reset all objects to pool")
        print("  s - Show pool status")
        print("  p - Set prompt")
        print("  ENTER - Done, start Pi")
        print("="*60 + "\n")
        
        while True:
            cmd = input("Spawn> ").strip().lower()
            
            if cmd == "" or cmd == "done" or cmd == "start":
                break
            elif cmd == "c" or cmd == "cube":
                openpi_env.spawn_object_by_type("cubes")
            elif cmd == "m" or cmd == "mug":
                openpi_env.spawn_object_by_type("mugs")
            elif cmd == "f" or cmd == "fruit":
                openpi_env.spawn_object_by_type("fruits")
            elif cmd == "r" or cmd == "random":
                openpi_env.spawn_random_object()
            elif cmd == "x" or cmd == "reset":
                openpi_env.reset_all_objects()
            elif cmd == "s" or cmd == "status":
                status = openpi_env.get_pool_status()
                for pool_type, info in status.items():
                    print(f"  {pool_type}: {info['active']}/{info['total']} active, {info['available']} available")
            elif cmd == "p" or cmd == "prompt":
                new_prompt = input("Enter prompt: ").strip()
                if new_prompt:
                    openpi_env.set_prompt(new_prompt)
            else:
                print(f"  Unknown command: {cmd}")
            
            # Step physics to let objects settle
            for _ in range(10):
                env.step(torch.zeros((1, 16), device="cuda:0"))
        
        print()
    
    # Interactive prompt mode
    if args_cli.interactive:
        print("\n[PROMPT] Enter task instruction for Pi (or press Enter to keep default):")
        print(f"[PROMPT] Current: {openpi_env._prompt}")
        new_prompt = input("[PROMPT] > ").strip()
        if new_prompt:
            openpi_env.set_prompt(new_prompt)
        print()
    
    # Create OpenPI runtime
    runtime = openpi_runtime.Runtime(
        environment=openpi_env,
        agent=policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=websocket_client_policy.WebsocketClientPolicy(
                    host=args_cli.host,
                    port=args_cli.port,
                ),
                action_horizon=args_cli.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args_cli.max_hz,
        num_episodes=args_cli.num_episodes,
        max_episode_steps=args_cli.max_episode_steps,
    )
    
    print("[INFO] Starting OpenPI client loop...")
    print(f"[INFO] Prompt: {openpi_env._prompt}")
    print("[INFO] Press Ctrl+C to stop\n")
    
    try:
        runtime.run()
    except KeyboardInterrupt:
        print("\n[INFO] Client stopped by user")
    
    print("[INFO] OpenPI client finished")
    env.close()
    print("[INFO] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
