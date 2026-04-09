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
Diffusion Policy Client for Bimanual OpenArm in Isaac Lab

Connects to a Diffusion Policy WebSocket server and executes actions in Isaac Lab.
Uses the same WebSocket protocol as OpenPI, so the openpi_client library works.

Unlike Pi0.5, Diffusion Policy:
- Does not use language prompts (not a VLA)
- Uses action horizon of 8 (vs 25 for Pi0.5)
- Is trained on 16 DOF state/action space

Usage:
    python diffusion_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0
    python diffusion_client_bimanual.py --task Isaac-Reach-OpenArm-Bi-Teleop-v0 --host localhost --port 8001
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diffusion Policy Client for Bimanual OpenArm")
parser.add_argument("--task", type=str, default="Isaac-Reach-OpenArm-Bi-Teleop-v0",
                    help="Task name (use Teleop variant for 16 DOF action space)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")

parser.add_argument("--host", type=str, default="localhost", help="Diffusion Policy server host")
parser.add_argument("--port", type=int, default=8001, help="Diffusion Policy server port")
parser.add_argument("--action_horizon", type=int, default=8, help="Action chunk size (8 for Diffusion)")
parser.add_argument("--max_hz", type=float, default=50.0, help="Max control frequency")
parser.add_argument("--num_episodes", type=int, default=999999, help="Number of episodes to run")
parser.add_argument("--max_episode_steps", type=int, default=999999, help="Max steps per episode")

parser.add_argument("--no_cameras", action="store_true", help="Disable camera capture (use black images)")
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
VLA_DOF = 16
ENV_DOF = 18

# Arm raise trajectory - matches SparkJAX openpi_runner_node.py
# 16 DOF: [left_arm(7), left_grip(1), right_arm(7), right_grip(1)]
_LIFT_SPINE = np.array([
    [0.266, -0.049, -0.064, 0.345, -0.007, 0.102, -0.023, 0.004,
     -0.266, 0.049, 0.064, 0.345, 0.007, -0.102, 0.023, 0.004],
    [0.287, -0.049, -0.064, 0.364, -0.006, 0.102, -0.072, 0.004,
     -0.287, 0.049, 0.064, 0.364, 0.006, -0.102, 0.072, 0.004],
    [0.559, -0.049, -0.064, 0.611, -0.006, 0.102, -0.228, 0.004,
     -0.559, 0.049, 0.064, 0.611, 0.006, -0.102, 0.228, 0.004],
    [0.866, -0.049, -0.064, 1.022, -0.008, 0.102, -0.423, 0.004,
     -0.866, 0.049, 0.064, 1.022, 0.008, -0.102, 0.423, 0.004],
    [1.052, -0.049, -0.064, 1.170, -0.011, 0.102, -0.713, 0.004,
     -1.052, 0.049, 0.064, 1.170, 0.011, -0.102, 0.713, 0.004],
    [1.253, -0.049, -0.064, 1.377, -0.011, 0.102, -0.955, 0.004,
     -1.253, 0.049, 0.064, 1.377, 0.011, -0.102, 0.955, 0.004],
    [1.359, -0.049, -0.064, 1.442, -0.011, 0.102, -1.179, 0.004,
     -1.359, 0.049, 0.064, 1.442, 0.011, -0.102, 1.179, 0.004],
    [1.427, -0.050, -0.064, 1.530, -0.011, 0.102, -1.285, 0.004,
     -1.427, 0.050, 0.064, 1.530, 0.011, -0.102, 1.285, 0.004],
    [1.427, -0.050, -0.064, 1.530, -0.011, 0.102, -1.391, 0.004,
     -1.427, 0.050, 0.064, 1.530, 0.011, -0.102, 1.391, 0.004],
], dtype=np.float64)
_ARMS_UP = _LIFT_SPINE[-1].copy()
_RAISE_DURATION_S = 1.5
_RAISE_HZ = 50


class KeyboardListener:
    """Keyboard listener using Isaac Sim's input system."""
    
    def __init__(self):
        import carb.input
        import omni.appwindow
        
        self._carb_input = carb.input
        self._quit_requested = False
        self._spawn_requested = False
        self._reset_pool_requested = False
        self._reset_arms_requested = False
        
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = self._app_window.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )
        
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS (Diffusion Policy Client)")
        print("="*60)
        print("  Q: Quit current episode")
        print("  C: Spawn random object on workspace")
        print("  B: Reset all objects to pool")
        print("  R: Reset arms to initial position")
        print("  Ctrl+C: Exit completely")
        print("="*60 + "\n")
        
    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != self._carb_input.KeyboardEventType.KEY_PRESS:
            return True
            
        key = event.input
        
        if key == self._carb_input.KeyboardInput.Q:
            self._quit_requested = True
            print("[KEYBOARD] Quit requested")
            return False
        elif key == self._carb_input.KeyboardInput.C:
            self._spawn_requested = True
            print("[KEYBOARD] Spawn object requested")
            return False
        elif key == self._carb_input.KeyboardInput.B:
            self._reset_pool_requested = True
            print("[KEYBOARD] Reset pool requested")
            return False
        elif key == self._carb_input.KeyboardInput.R:
            self._reset_arms_requested = True
            print("[KEYBOARD] Reset arms requested")
            return False
            
        return True
        
    def start(self):
        pass
        
    def stop(self):
        if hasattr(self, '_sub_keyboard') and self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None
            
    def check_quit(self) -> bool:
        return self._quit_requested
    
    def check_spawn(self) -> bool:
        if self._spawn_requested:
            self._spawn_requested = False
            return True
        return False
    
    def check_reset_pool(self) -> bool:
        if self._reset_pool_requested:
            self._reset_pool_requested = False
            return True
        return False
    
    def check_reset_arms(self) -> bool:
        if self._reset_arms_requested:
            self._reset_arms_requested = False
            return True
        return False


class OpenArmBimanualEnvironment:
    """
    Diffusion Policy-compatible environment wrapper for Bimanual OpenArm in Isaac Lab.
    
    Similar to OpenPI wrapper but without prompt handling (Diffusion Policy is not a VLA).
    """

    def __init__(self, env, use_cameras: bool = True, keyboard_listener: KeyboardListener = None,
                 policy=None):
        self._env = env
        self._use_cameras = use_cameras
        self._device = env.unwrapped.device
        self._keyboard_listener = keyboard_listener
        self._policy = policy

        self._unwrapped = env.unwrapped
        if hasattr(self._unwrapped, 'unwrapped'):
            self._unwrapped = self._unwrapped.unwrapped

        self._robot = self._unwrapped.scene["robot"]

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

        self._render_products = {}
        self._cameras_initialized = False

        self._obs = None
        self._done = True
        self._step_count = 0

        self._object_pool = {"cubes": [], "mugs": [], "fruits": []}
        self._active_objects = []
        self._init_object_pool()

    def _init_object_pool(self):
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

    def spawn_random_object(self, position=None):
        if position is None:
            spawn_x = random.uniform(0.20, 0.35)
            spawn_y = random.uniform(-0.15, 0.15)
            spawn_z = 0.32
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

    def reset_arms(self):
        default_joint_pos = self._robot.data.default_joint_pos.clone()
        default_joint_vel = torch.zeros_like(self._robot.data.default_joint_vel)
        
        self._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        self._robot.set_joint_position_target(default_joint_pos)
        self._robot.write_data_to_sim()
        
        for _ in range(10):
            self._unwrapped.sim.step(render=False)
            self._robot.update(self._unwrapped.sim.get_physics_dt())
        
        print("[Robot] Arms reset to initial position")

    def raise_arms(self):
        """Raise arms to 'arms up' position - matches SparkJAX behavior.
        
        Interpolates through _LIFT_SPINE trajectory to reach _ARMS_UP position.
        This ensures the simulation starts from the same pose as real robot.
        """
        print("[Robot] Raising arms to start position...")
        
        num_frames = int(_RAISE_DURATION_S * _RAISE_HZ)
        n_wp = len(_LIFT_SPINE)
        t_wp = np.linspace(0.0, 1.0, n_wp)
        t_out = np.linspace(0.0, 1.0, num_frames)
        
        # Interpolate trajectory
        trajectory = np.zeros((num_frames, 16), dtype=np.float64)
        for j in range(16):
            trajectory[:, j] = np.interp(t_out, t_wp, _LIFT_SPINE[:, j])
        
        step_time = 1.0 / _RAISE_HZ
        
        for i, pos in enumerate(trajectory):
            # Expand 16 DOF to 18 DOF (duplicate gripper values)
            left_arm = pos[0:7]
            left_grip = pos[7]
            right_arm = pos[8:15]
            right_grip = pos[15]
            
            expanded = np.concatenate([
                left_arm,
                [left_grip, left_grip],
                right_arm,
                [right_grip, right_grip],
            ])
            
            action_tensor = torch.tensor(
                expanded, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            
            self._env.step(action_tensor)
            time.sleep(step_time)
        
        # Hold at final position for stability
        hold_frames = int(_RAISE_HZ * 0.5)
        final_pos = trajectory[-1]
        expanded_final = np.concatenate([
            final_pos[0:7],
            [final_pos[7], final_pos[7]],
            final_pos[8:15],
            [final_pos[15], final_pos[15]],
        ])
        final_tensor = torch.tensor(
            expanded_final, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        
        for _ in range(hold_frames):
            self._env.step(final_tensor)
            time.sleep(step_time)
        
        print("[Robot] Arms raised successfully")

    def reset_all_objects(self):
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
                self._use_cameras = False
            else:
                print(f"[CAM] Initialized {found} cameras via replicator render products")

        except Exception as e:
            print(f"[WARN] Failed to initialize cameras: {e}")
            self._use_cameras = False

        self._cameras_initialized = True

    def _capture_camera(self, cam_key: str) -> np.ndarray:
        black = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        if cam_key not in self._render_products:
            return black

        try:
            data = self._render_products[cam_key].get_data()
            if data is None:
                return black

            rgb = data[:, :, :3].astype(np.uint8)

            h, w = rgb.shape[:2]
            ratio = max(w / IMAGE_SIZE, h / IMAGE_SIZE)
            new_h, new_w = int(h / ratio), int(w / ratio)
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            pad_y = (IMAGE_SIZE - new_h) // 2
            pad_x = (IMAGE_SIZE - new_w) // 2
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

            return np.transpose(canvas, (2, 0, 1))

        except Exception:
            return black

    def reset(self) -> None:
        self._obs, _ = self._env.reset()
        self._done = False
        self._step_count = 0

        if self._use_cameras and not self._cameras_initialized:
            self._init_cameras_replicator()

    def is_episode_complete(self) -> bool:
        return self._done

    def get_observation(self) -> dict:
        """
        Build observation for Diffusion Policy:
        - state: 16-dim [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
        - images: cam_high, cam_left_wrist, cam_right_wrist [C, H, W] uint8
        - prompt: empty string (Diffusion Policy doesn't use prompts)
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
            "prompt": "",  # Diffusion Policy doesn't use prompts
        }

    def apply_action(self, action: dict) -> None:
        if self._keyboard_listener:
            if self._keyboard_listener.check_quit():
                print("\n[KEYBOARD] Quit requested. Ending episode...")
                self._done = True
                return
                
            if self._keyboard_listener.check_spawn():
                self.spawn_random_object()
                
            if self._keyboard_listener.check_reset_pool():
                self.reset_all_objects()
            
            if self._keyboard_listener.check_reset_arms():
                self.reset_arms()
        
        vla_actions = action.get("actions")

        if vla_actions is not None:
            vla_actions = np.asarray(vla_actions, dtype=np.float32)
            if vla_actions.ndim == 2:
                vla_actions = vla_actions[0]

            left_arm = vla_actions[0:7]
            left_grip = vla_actions[7]
            right_arm = vla_actions[8:15]
            right_grip = vla_actions[15]
            
            expanded_actions = np.concatenate([
                left_arm,
                [left_grip, left_grip],
                right_arm,
                [right_grip, right_grip],
            ])
            
            action_tensor = torch.tensor(expanded_actions, dtype=torch.float32, device=self._device).unsqueeze(0)
            self._obs, _, dones, _ = self._env.step(action_tensor)
        else:
            action_tensor = torch.zeros((1, ENV_DOF), device=self._device)
            self._obs, _, dones, _ = self._env.step(action_tensor)

        self._step_count += 1


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Main entry point for Diffusion Policy client."""

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
    print("OPENARM BIMANUAL - DIFFUSION POLICY CLIENT")
    print("=" * 60)
    print(f"Server:         {args_cli.host}:{args_cli.port}")
    print(f"Action horizon: {args_cli.action_horizon}")
    print(f"Max Hz:         {args_cli.max_hz}")
    print(f"Episodes:       {args_cli.num_episodes}")
    print(f"Max steps:      {args_cli.max_episode_steps}")
    print(f"Cameras:        {'enabled' if use_cameras else 'disabled (black images)'}")
    print("=" * 60 + "\n")

    keyboard_listener = KeyboardListener()
    keyboard_listener.start()

    broker = action_chunk_broker.ActionChunkBroker(
        policy=websocket_client_policy.WebsocketClientPolicy(
            host=args_cli.host,
            port=args_cli.port,
        ),
        action_horizon=args_cli.action_horizon,
    )

    openpi_env = OpenArmBimanualEnvironment(
        env=env,
        use_cameras=use_cameras,
        keyboard_listener=keyboard_listener,
        policy=broker,
    )

    # Reset environment and raise arms (matches SparkJAX real robot behavior)
    openpi_env.reset()
    openpi_env.raise_arms()

    if getattr(args_cli, 'spawn_objects', False):
        print("\n[INFO] Spawning object on workspace...")
        openpi_env.spawn_random_object()
        for _ in range(30):
            openpi_env._env.step(torch.zeros((1, ENV_DOF), device=openpi_env._device))
        print("[INFO] Object spawned and settled\n")

    runtime = openpi_runtime.Runtime(
        environment=openpi_env,
        agent=policy_agent.PolicyAgent(policy=broker),
        subscribers=[],
        max_hz=args_cli.max_hz,
        num_episodes=args_cli.num_episodes,
        max_episode_steps=args_cli.max_episode_steps,
    )

    print("[INFO] Starting Diffusion Policy client loop...")
    print("[INFO] Press Ctrl+C to stop\n")

    try:
        runtime.run()
    except KeyboardInterrupt:
        print("\n[INFO] Client stopped by user")
    finally:
        keyboard_listener.stop()

    print("[INFO] Diffusion Policy client finished")
    env.close()
    print("[INFO] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
