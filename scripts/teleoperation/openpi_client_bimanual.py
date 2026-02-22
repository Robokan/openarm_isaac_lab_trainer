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

The Teleop environment config (Isaac-Reach-OpenArm-Bi-Teleop-v0) has been updated
to accept 16 DOF actions matching the VLA output format:
    [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]

Arms use scale=1.0, use_default_offset=False so VLA absolute joint positions
map directly.  Grippers use scale=0.044 so VLA [0,1] maps to sim [0, 0.044].

Cameras:
    Uses the openarm_bimanual_factory.usd which has cameras mounted on:
    - openarm_body_link (base/high camera) → cam_high
    - openarm_left_link7 (left wrist camera) → cam_left_wrist
    - openarm_right_link7 (right wrist camera) → cam_right_wrist

Usage:
    python openpi_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0
    python openpi_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0 --host localhost --port 8000
    python openpi_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0 --prompt "pick up the cube"
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OpenPI Client for Bimanual OpenArm")
parser.add_argument("--task", type=str, default="Isaac-Reach-OpenArm-Bi-Teleop-v0",
                    help="Task name (use Teleop variant for 16 DOF action space)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")

parser.add_argument("--host", type=str, default="localhost", help="OpenPI policy server host")
parser.add_argument("--port", type=int, default=8000, help="OpenPI policy server port")
parser.add_argument("--action_horizon", type=int, default=10, help="Action chunk size")
parser.add_argument("--prompt", type=str, default="perform the bimanual manipulation task",
                    help="Task prompt for VLA")
parser.add_argument("--max_hz", type=float, default=50.0, help="Max control frequency")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
parser.add_argument("--max_episode_steps", type=int, default=1000, help="Max steps per episode")

parser.add_argument("--no_cameras", action="store_true", help="Disable camera capture (use black images)")
parser.add_argument("--interactive", action="store_true",
                    help="Interactive prompt mode: ask for prompt before each episode")
parser.add_argument("--spawn-objects", action="store_true",
                    help="Spawn random object from pool on workspace at startup")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after Isaac Sim is initialized."""

import gymnasium as gym
import torch
import numpy as np
import cv2
import time
import random

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401


IMAGE_SIZE = 224
GRIPPER_OPEN_POS = 0.044
VLA_DOF = 16  # VLA output: [left_arm(7), left_grip(1), right_arm(7), right_grip(1)]
ENV_DOF = 18  # Env expects: [left_arm(7), left_grip(2), right_arm(7), right_grip(2)]


class OpenArmBimanualEnvironment:
    """
    OpenPI-compatible environment wrapper for Bimanual OpenArm in Isaac Lab.

    Matches OPENARM_MAPPING.md:
    - State: 16 DOF [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
    - Images: cam_high, cam_left_wrist, cam_right_wrist  ([C, H, W], uint8, 224x224)
    - Actions: 16 DOF passed directly to env.step()

    The Teleop env config (TeleopActionsCfg) has matching 16 DOF action space:
    - left_arm_action:     7 joints, scale=1.0, use_default_offset=False
    - left_gripper_action: 1 joint,  scale=0.044, use_default_offset=False
    - right_arm_action:    7 joints, scale=1.0, use_default_offset=False
    - right_gripper_action:1 joint,  scale=0.044, use_default_offset=False
    """

    def __init__(self, env, prompt: str = "perform the task", use_cameras: bool = True):
        self._env = env
        self._prompt = prompt
        self._use_cameras = use_cameras
        self._device = env.unwrapped.device

        self._unwrapped = env.unwrapped
        if hasattr(self._unwrapped, 'unwrapped'):
            self._unwrapped = self._unwrapped.unwrapped

        self._robot = self._unwrapped.scene["robot"]

        # Find joint IDs for building 16-dim state vector
        self._left_arm_ids, left_names = self._robot.find_joints([
            "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
            "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
            "openarm_left_joint7",
        ])
        self._right_arm_ids, right_names = self._robot.find_joints([
            "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
            "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
            "openarm_right_joint7",
        ])
        self._left_gripper_ids, _ = self._robot.find_joints("openarm_left_finger_joint.*")
        self._right_gripper_ids, _ = self._robot.find_joints("openarm_right_finger_joint.*")

        print(f"[INFO] Left arm joint IDs:  {list(self._left_arm_ids)} = {left_names}")
        print(f"[INFO] Right arm joint IDs: {list(self._right_arm_ids)} = {right_names}")
        print(f"[INFO] Left gripper IDs:    {list(self._left_gripper_ids)}")
        print(f"[INFO] Right gripper IDs:   {list(self._right_gripper_ids)}")

        # Verify action space matches VLA output
        num_actions = env.unwrapped.num_actions if hasattr(env.unwrapped, 'num_actions') else None
        print(f"[INFO] Environment action space: {num_actions} DOF (expected {ENV_DOF})")
        if num_actions is not None and num_actions != ENV_DOF:
            print(f"[WARN] Action space mismatch! Expected {ENV_DOF}, got {num_actions}")
            print(f"[WARN] Make sure to use Isaac-Reach-OpenArm-Bi-Teleop-v0 (TeleopActionsCfg)")

        # Camera render products (initialized lazily after first sim step)
        self._render_products = {}
        self._cameras_initialized = False

        self._obs = None
        self._done = True
        self._step_count = 0

        # Object pool for spawning
        self._object_pool = {"cubes": [], "mugs": [], "fruits": []}
        self._active_objects = []
        self._init_object_pool()

    def _init_object_pool(self):
        """Initialize object pool from scene assets (TeleopSceneCfg)."""
        scene_keys = list(self._unwrapped.scene.keys()) if hasattr(self._unwrapped.scene, 'keys') else []
        
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
        
        total = len(self._object_pool['cubes']) + len(self._object_pool['mugs']) + len(self._object_pool['fruits'])
        if total > 0:
            print(f"[Pool] Found {len(self._object_pool['cubes'])} cubes, "
                  f"{len(self._object_pool['mugs'])} mugs, {len(self._object_pool['fruits'])} fruits")
        else:
            print("[Pool] No pool objects found - ensure using TeleopSceneCfg")

    def spawn_random_object(self, position=None):
        """Spawn a random object from the pool onto the workspace."""
        if position is None:
            spawn_x = random.uniform(0.20, 0.35)
            spawn_y = random.uniform(-0.15, 0.15)
            spawn_z = 0.32  # Slightly above table surface
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
                    print(f"[Pool] Spawned {pool_type[:-1]} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
                    return True
        
        print("[Pool] All pools exhausted!")
        return False

    def reset_all_objects(self):
        """Return all active objects to their pool positions."""
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
        print("[Pool] All objects returned to pool")

    def _init_cameras_replicator(self):
        """Initialize cameras using omni.replicator render products (matches data collection)."""
        if self._cameras_initialized:
            return

        try:
            from pxr import UsdGeom
            import omni.replicator.core as rep
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            found = 0

            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Camera):
                    cam_path = str(prim.GetPath())
                    cam_name = prim.GetName()
                    if "env_0" not in cam_path:
                        continue
                    try:
                        render_product = rep.create.render_product(cam_path, (640, 480))
                        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
                        rgb_annot.attach([render_product])

                        if "ego" in cam_name.lower() or "high" in cam_name.lower() or "body" in cam_name.lower():
                            self._render_products["cam_high"] = rgb_annot
                        elif "left" in cam_name.lower():
                            self._render_products["cam_left_wrist"] = rgb_annot
                        elif "right" in cam_name.lower():
                            self._render_products["cam_right_wrist"] = rgb_annot

                        print(f"[CAM] {cam_name}: {cam_path} [READY]")
                        found += 1
                    except Exception as e:
                        print(f"[CAM] {cam_name}: {cam_path} [FAILED: {e}]")

            if found == 0:
                print("[WARN] No cameras found in USD stage. Using black images.")
                print("[WARN] Make sure to use the factory USD (openarm_bimanual_factory.usd)")
                self._use_cameras = False
            else:
                print(f"[CAM] Initialized {found} cameras via replicator render products")

        except Exception as e:
            print(f"[WARN] Failed to initialize cameras: {e}")
            self._use_cameras = False

        self._cameras_initialized = True

    def _capture_camera(self, cam_key: str) -> np.ndarray:
        """Capture and resize a camera image to [C, H, W] uint8 224x224."""
        black = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        if cam_key not in self._render_products:
            return black

        try:
            data = self._render_products[cam_key].get_data()
            if data is None:
                return black

            rgb = data[:, :, :3].astype(np.uint8)

            # Resize with padding to 224x224 (matches training preprocessing)
            h, w = rgb.shape[:2]
            ratio = max(w / IMAGE_SIZE, h / IMAGE_SIZE)
            new_h, new_w = int(h / ratio), int(w / ratio)
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            pad_y = (IMAGE_SIZE - new_h) // 2
            pad_x = (IMAGE_SIZE - new_w) // 2
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

            # [H, W, C] → [C, H, W]
            return np.transpose(canvas, (2, 0, 1))

        except Exception:
            return black

    def set_prompt(self, prompt: str):
        self._prompt = prompt
        print(f"[INFO] Prompt set to: {prompt}")

    def reset(self) -> None:
        print("[DEBUG] Environment reset() called")
        self._obs, _ = self._env.reset()
        self._done = False
        self._step_count = 0
        print(f"[DEBUG] After reset: _done={self._done}, _step_count={self._step_count}")

        if self._use_cameras and not self._cameras_initialized:
            self._init_cameras_replicator()

    def is_episode_complete(self) -> bool:
        if self._done:
            print(f"[DEBUG] is_episode_complete() returning True at step {self._step_count}")
        return self._done

    def get_observation(self) -> dict:
        """
        Build observation matching OPENARM_MAPPING.md:
        - state: 16-dim [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
        - images: cam_high, cam_left_wrist, cam_right_wrist  [C, H, W] uint8
        - prompt: task instruction string
        """
        joint_pos = self._robot.data.joint_pos[0].cpu().numpy()

        left_arm = joint_pos[list(self._left_arm_ids)]
        right_arm = joint_pos[list(self._right_arm_ids)]

        left_grip = np.array([joint_pos[self._left_gripper_ids[0]]], dtype=np.float32) if self._left_gripper_ids else np.array([0.0], dtype=np.float32)
        right_grip = np.array([joint_pos[self._right_gripper_ids[0]]], dtype=np.float32) if self._right_gripper_ids else np.array([0.0], dtype=np.float32)

        state = np.concatenate([left_arm, left_grip, right_arm, right_grip]).astype(np.float32)

        if self._use_cameras:
            images = {
                "cam_high": self._capture_camera("cam_high"),
                "cam_left_wrist": self._capture_camera("cam_left_wrist"),
                "cam_right_wrist": self._capture_camera("cam_right_wrist"),
            }
        else:
            black = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            images = {
                "cam_high": black.copy(),
                "cam_left_wrist": black.copy(),
                "cam_right_wrist": black.copy(),
            }

        return {
            "state": state,
            "images": images,
            "prompt": self._prompt,
        }

    def apply_action(self, action: dict) -> None:
        """
        Apply VLA action via env.step().

        VLA outputs 16 DOF: [left_arm(7), left_grip(1), right_arm(7), right_grip(1)]
        Env expects 18 DOF: [left_arm(7), left_grip(2), right_arm(7), right_grip(2)]
        
        Each gripper has 2 finger joints that move together (mimic), so we
        duplicate the single gripper value for both fingers.
        """
        vla_actions = action.get("actions")

        if vla_actions is not None:
            vla_actions = np.asarray(vla_actions, dtype=np.float32)
            if vla_actions.ndim == 2:
                vla_actions = vla_actions[0]

            # Expand 16 DOF to 18 DOF by duplicating gripper values
            # VLA: [left_arm(7), left_grip(1), right_arm(7), right_grip(1)]
            # Env: [left_arm(7), left_grip(2), right_arm(7), right_grip(2)]
            left_arm = vla_actions[0:7]
            left_grip = vla_actions[7]  # Single value, duplicate for 2 fingers
            right_arm = vla_actions[8:15]
            right_grip = vla_actions[15]  # Single value, duplicate for 2 fingers
            
            expanded_actions = np.concatenate([
                left_arm,
                [left_grip, left_grip],  # Duplicate for both finger joints
                right_arm,
                [right_grip, right_grip],  # Duplicate for both finger joints
            ])
            
            action_tensor = torch.tensor(expanded_actions, dtype=torch.float32, device=self._device).unsqueeze(0)
            self._obs, _, dones, _ = self._env.step(action_tensor)
        else:
            action_tensor = torch.zeros((1, ENV_DOF), device=self._device)
            self._obs, _, dones, _ = self._env.step(action_tensor)

        self._step_count += 1

        if hasattr(dones, 'any'):
            self._done = dones.any().item()
        else:
            self._done = bool(dones)
        
        if self._step_count <= 5 or self._done:
            print(f"[DEBUG] Step {self._step_count}: done={self._done}, dones_raw={dones}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Main entry point for OpenPI client."""

    try:
        from openpi_client import action_chunk_broker
        from openpi_client import websocket_client_policy
        from openpi_client.runtime import runtime as openpi_runtime
        from openpi_client.runtime.agents import policy_agent
    except ImportError as e:
        print(f"\n[ERROR] OpenPI client not installed: {e}")
        print("[INFO] Install with: pip install -e /path/to/openpi/packages/openpi-client")
        return

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = agent_cfg.device

    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.enable_corruption = False

    print(f"\n[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    use_cameras = not args_cli.no_cameras

    print("\n" + "=" * 60)
    print("OPENARM BIMANUAL - OPENPI CLIENT")
    print("=" * 60)
    print(f"Server:         {args_cli.host}:{args_cli.port}")
    print(f"Action horizon: {args_cli.action_horizon}")
    print(f"Max Hz:         {args_cli.max_hz}")
    print(f"Episodes:       {args_cli.num_episodes}")
    print(f"Max steps:      {args_cli.max_episode_steps}")
    print(f"Prompt:         {args_cli.prompt}")
    print(f"Cameras:        {'enabled' if use_cameras else 'disabled (black images)'}")
    print("=" * 60 + "\n")

    openpi_env = OpenArmBimanualEnvironment(
        env=env,
        prompt=args_cli.prompt,
        use_cameras=use_cameras,
    )

    # Spawn objects if requested
    if getattr(args_cli, 'spawn_objects', False):
        print("\n[INFO] Spawning object on workspace...")
        openpi_env.spawn_random_object()
        # Step simulation to let object settle
        for _ in range(30):
            openpi_env._env.step(torch.zeros((1, ENV_DOF), device=openpi_env._device))
        print("[INFO] Object spawned and settled\n")

    if args_cli.interactive:
        print("\n[PROMPT] Enter task instruction for Pi (or press Enter to keep default):")
        print(f"[PROMPT] Current: {openpi_env._prompt}")
        new_prompt = input("[PROMPT] > ").strip()
        if new_prompt:
            openpi_env.set_prompt(new_prompt)
        print()

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
