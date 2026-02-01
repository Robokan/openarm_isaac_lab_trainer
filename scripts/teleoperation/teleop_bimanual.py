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
Bimanual Teleoperation Script for OpenArm

Controls both robot arms using Vive controllers or keyboard.

Usage:
    python teleop_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 --checkpoint /path/to/model.pt
    python teleop_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 --checkpoint /path/to/model.pt --device keyboard
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Bimanual Teleoperation for OpenArm")
parser.add_argument("--task", type=str, default="Isaac-Reach-OpenArm-Bi-v0", help="Task name")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
parser.add_argument("--input", type=str, default="keyboard", choices=["vive", "keyboard", "gamepad", "xr"],
                    help="Input device for teleoperation (xr = VR handtracking)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Controller sensitivity")

# Add AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Enable XR mode if using xr input
app_launcher_args = vars(args_cli)
if args_cli.input == "xr":
    app_launcher_args["xr"] = True
    print("[INFO] XR mode enabled for VR handtracking")

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest follows after Isaac Sim is initialized."""

import gymnasium as gym
import os
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401


class KeyboardDevice:
    """Keyboard control using Isaac Sim's input system."""
    
    def __init__(self, sensitivity: float = 1.0):
        import carb.input
        import omni.appwindow
        
        self.sensitivity = sensitivity
        self.step_size = 0.01 * sensitivity
        # Initial poses (x, y, z, qw, qx, qy, qz)
        self.left_pose = np.array([0.2, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self.right_pose = np.array([0.2, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self.active_hand = "left"
        
        # Set up keyboard input
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = self._app_window.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        
        print("\n" + "="*60)
        print("KEYBOARD TELEOPERATION CONTROLS")
        print("="*60)
        print("Movement (active hand):")
        print("  W/S: Forward/Backward (X)")
        print("  A/D: Left/Right (Y)")  
        print("  Q/E: Up/Down (Z)")
        print("")
        print("Hand Selection:")
        print("  1: Select LEFT hand")
        print("  2: Select RIGHT hand")
        print("")
        print("Other:")
        print("  R: Reset poses to default")
        print("  Ctrl+C: Quit")
        print("="*60 + "\n")
    
    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events from Isaac Sim."""
        import carb.input
        
        # Only process key press events
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        
        # Get the key
        key = event.input
        pose = self.left_pose if self.active_hand == "left" else self.right_pose
        
        if key == carb.input.KeyboardInput.W:
            pose[0] += self.step_size
        elif key == carb.input.KeyboardInput.S:
            pose[0] -= self.step_size
        elif key == carb.input.KeyboardInput.A:
            pose[1] += self.step_size
        elif key == carb.input.KeyboardInput.D:
            pose[1] -= self.step_size
        elif key == carb.input.KeyboardInput.Q:
            pose[2] += self.step_size
        elif key == carb.input.KeyboardInput.E:
            pose[2] -= self.step_size
        elif key == carb.input.KeyboardInput.KEY_1:
            self.active_hand = "left"
            print("Active: LEFT hand")
        elif key == carb.input.KeyboardInput.KEY_2:
            self.active_hand = "right"
            print("Active: RIGHT hand")
        elif key == carb.input.KeyboardInput.R:
            self.left_pose = np.array([0.2, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
            self.right_pose = np.array([0.2, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
            print("Poses reset")
        
        return True
        
    def get_poses(self):
        return self.left_pose.copy(), self.right_pose.copy()
    
    def __del__(self):
        if hasattr(self, '_sub_keyboard') and self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)


class GamepadDevice:
    """Xbox/Gamepad controller using pygame for reliable input."""
    
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity
        self.step_size = 0.008 * sensitivity
        
        # Initial poses
        self.left_pose = np.array([0.2, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self.right_pose = np.array([0.2, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self._reset_pressed = False
        
        # Try to use pygame for gamepad
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            
            if pygame.joystick.get_count() > 0:
                self._joystick = pygame.joystick.Joystick(0)
                self._joystick.init()
                self._pygame = pygame
                print(f"[Gamepad] Found: {self._joystick.get_name()}")
            else:
                print("[Gamepad] No gamepad found. Connect Xbox controller and restart.")
                self._joystick = None
                self._pygame = None
        except ImportError:
            print("[Gamepad] pygame not installed. Installing...")
            import subprocess
            subprocess.run(["pip", "install", "pygame"], capture_output=True)
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._joystick = pygame.joystick.Joystick(0)
                self._joystick.init()
                self._pygame = pygame
                print(f"[Gamepad] Found: {self._joystick.get_name()}")
            else:
                self._joystick = None
                self._pygame = None
        
        print("\n" + "="*60)
        print("XBOX CONTROLLER TELEOPERATION")
        print("="*60)
        print("Left Stick:  Move LEFT arm (X/Y)")
        print("Right Stick: Move RIGHT arm (X/Y)")
        print("Triggers:    Up/Down (Z) - LT=Left down, RT=Right down")
        print("Bumpers:     Up (Z) - LB=Left up, RB=Right up")  
        print("A Button:    Reset poses")
        print("="*60 + "\n")
    
    def get_poses(self):
        if self._joystick is None or self._pygame is None:
            return self.left_pose.copy(), self.right_pose.copy()
        
        # Process pygame events (required for joystick to update)
        self._pygame.event.pump()
        
        # Read joystick axes (Xbox layout)
        # Axis 0: Left stick X
        # Axis 1: Left stick Y (inverted)
        # Axis 2: Right stick X
        # Axis 3: Right stick Y (inverted)
        # Axis 4: Left trigger (-1 to 1)
        # Axis 5: Right trigger (-1 to 1)
        
        try:
            num_axes = self._joystick.get_numaxes()
            
            # Left stick
            lx = self._joystick.get_axis(0) if num_axes > 0 else 0
            ly = -self._joystick.get_axis(1) if num_axes > 1 else 0  # Invert Y
            
            # Right stick  
            rx = self._joystick.get_axis(2) if num_axes > 2 else 0
            ry = -self._joystick.get_axis(3) if num_axes > 3 else 0  # Invert Y
            
            # Triggers (convert from -1..1 to 0..1)
            lt = (self._joystick.get_axis(4) + 1) / 2 if num_axes > 4 else 0
            rt = (self._joystick.get_axis(5) + 1) / 2 if num_axes > 5 else 0
            
            # Bumpers (buttons 4 and 5 on Xbox)
            num_buttons = self._joystick.get_numbuttons()
            lb = self._joystick.get_button(4) if num_buttons > 4 else 0
            rb = self._joystick.get_button(5) if num_buttons > 5 else 0
            
            # A button (button 0 on Xbox)
            a_button = self._joystick.get_button(0) if num_buttons > 0 else 0
            
        except Exception as e:
            return self.left_pose.copy(), self.right_pose.copy()
        
        # Apply deadzone
        deadzone = 0.15
        if abs(lx) < deadzone: lx = 0
        if abs(ly) < deadzone: ly = 0
        if abs(rx) < deadzone: rx = 0
        if abs(ry) < deadzone: ry = 0
        
        # Update left arm
        self.left_pose[0] += ly * self.step_size
        self.left_pose[1] += lx * self.step_size
        self.left_pose[2] += (lb - lt) * self.step_size
        
        # Update right arm
        self.right_pose[0] += ry * self.step_size
        self.right_pose[1] += rx * self.step_size
        self.right_pose[2] += (rb - rt) * self.step_size
        
        # Reset on A button
        if a_button and not self._reset_pressed:
            self.left_pose = np.array([0.2, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
            self.right_pose = np.array([0.2, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
            print("[Gamepad] Poses reset!")
        self._reset_pressed = a_button
        
        return self.left_pose.copy(), self.right_pose.copy()


class ViveDevice:
    """Vive controller input via OpenVR/SteamVR."""
    
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity
        self.left_pose = np.array([0.2, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self.right_pose = np.array([0.2, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self.left_id = None
        self.right_id = None
        
        try:
            import openvr
            self.openvr = openvr
            self.vr = openvr.init(openvr.VRApplication_Other)
            print("[ViveDevice] OpenVR initialized")
            self._find_controllers()
        except ImportError:
            raise RuntimeError("openvr not installed. Run: pip install openvr")
        except Exception as e:
            raise RuntimeError(f"Failed to init OpenVR: {e}. Is SteamVR running?")
    
    def _find_controllers(self):
        for i in range(self.openvr.k_unMaxTrackedDeviceCount):
            if self.vr.getTrackedDeviceClass(i) == self.openvr.TrackedDeviceClass_Controller:
                role = self.vr.getControllerRoleForTrackedDeviceIndex(i)
                if role == self.openvr.TrackedControllerRole_LeftHand:
                    self.left_id = i
                    print(f"[ViveDevice] Left controller: index {i}")
                elif role == self.openvr.TrackedControllerRole_RightHand:
                    self.right_id = i
                    print(f"[ViveDevice] Right controller: index {i}")
    
    def _get_pose(self, controller_id):
        if controller_id is None:
            return None
        poses = self.vr.getDeviceToAbsoluteTrackingPose(
            self.openvr.TrackingUniverseStanding, 0,
            self.openvr.k_unMaxTrackedDeviceCount
        )
        p = poses[controller_id]
        if not p.bPoseIsValid:
            return None
        m = p.mDeviceToAbsoluteTracking
        return np.array([
            m[0][3] * self.sensitivity,
            m[1][3] * self.sensitivity, 
            m[2][3] * self.sensitivity,
            1.0, 0.0, 0.0, 0.0  # Simplified quaternion
        ])
    
    def get_poses(self):
        left = self._get_pose(self.left_id)
        right = self._get_pose(self.right_id)
        if left is not None: self.left_pose = left
        if right is not None: self.right_pose = right
        return self.left_pose.copy(), self.right_pose.copy()
    
    def update(self, key: str):
        pass  # VR doesn't need keyboard updates
    
    def __del__(self):
        if hasattr(self, 'vr') and self.vr:
            self.openvr.shutdown()


class XRDevice:
    """OpenXR handtracking device using Isaac Lab's XR system (for WiVRn)."""
    
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity
        self.left_pose = np.array([0.2, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self.right_pose = np.array([0.2, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self._reported_session = False
        self._xr_log_path = "/tmp/xr_device.log"
        self._last_enable_attempt = 0.0
        self._enable_interval_s = 5.0
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
        try:
            with open(self._xr_log_path, "w", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass
        
        try:
            import time
            from omni.kit.xr.core import XRCore
            
            self._xr_core = XRCore.get_singleton()
            self._log("[XRDevice] OpenXR initialized via Isaac Lab")
            self._setup_keyboard_listener()
            try:
                self._enable_first_available_profile()
            except Exception:
                # Continue even if profile enabling isn't supported
                pass
            
            # Wait briefly for XR session to start
            for _ in range(50):
                if self._is_session_running():
                    break
                time.sleep(0.1)
            self._log(f"[XRDevice] Session running: {self._is_session_running()}")
            try:
                self._dump_status(prefix="[XRDevice]")
            except Exception as exc:
                self._log(f"[XRDevice] Warning: failed to query XR status: {exc}")
            
            self._log("\n" + "="*60)
            self._log("XR HANDTRACKING TELEOPERATION")
            self._log("="*60)
            self._log("Move your VR controllers to control the robot arms!")
            self._log("Left controller  -> Left arm")
            self._log("Right controller -> Right arm")
            self._log("="*60 + "\n")
            
        except ImportError as e:
            self._log(f"[XRDevice] Warning: OpenXR not available: {e}")
            self._log("[XRDevice] Falling back to static poses")
            self._xr_core = None
        except Exception as e:
            self._log(f"[XRDevice] Warning: XR init failed: {e}")
            self._xr_core = None
    
    def _log(self, message: str):
        print(message, flush=True)
        try:
            with open(self._xr_log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception:
            pass

    def _dump_status(self, prefix: str = "[XRDevice]"):
        try:
            profiles = self._xr_core.get_profile_name_list()
            systems = self._xr_core.get_system_names()
            self._log(f"{prefix} Profiles: {profiles}")
            self._log(f"{prefix} Systems: {systems}")
            if hasattr(self._xr_core, "is_xr_enabled"):
                self._log(f"{prefix} XR enabled: {self._xr_core.is_xr_enabled()}")
            if hasattr(self._xr_core, "is_xr_display_enabled"):
                self._log(f"{prefix} XR display enabled: {self._xr_core.is_xr_display_enabled()}")
            if hasattr(self._xr_core, "is_xr_viewport_enabled"):
                self._log(f"{prefix} XR viewport enabled: {self._xr_core.is_xr_viewport_enabled()}")
        except Exception as exc:
            self._log(f"{prefix} Warning: failed to query XR status: {exc}")

    def _setup_keyboard_listener(self):
        try:
            import carb.input
            import omni.appwindow

            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
            self._log("[XRDevice] Press 'X' in the Isaac Sim window to retry XR session start")
        except Exception as exc:
            self._log(f"[XRDevice] Keyboard listener not available: {exc}")

    def _on_keyboard_event(self, event, *args, **kwargs):
        try:
            import carb.input
            if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                return True
            if event.input in (carb.input.KeyboardInput.X, carb.input.KeyboardInput.KEY_X):
                self._log("[XRDevice] Manual XR enable requested")
                self._enable_first_available_profile()
                self._dump_status(prefix="[XRDevice][After X]")
        except Exception:
            pass
        return True

    def _enable_first_available_profile(self):
        # Prefer an available XR profile if provided by the runtime
        profile_names = []
        try:
            profile_names = [p for p in self._xr_core.get_profile_name_list() if p]
        except Exception:
            profile_names = []
        # Fall back to common names (both cases)
        if not profile_names:
            profile_names = ["vr", "ar", "VR", "AR"]

        for name in profile_names:
            try:
                # Try static request (some builds expose this as a static method)
                from omni.kit.xr.core import XRCore

                XRCore.request_enable_profile(name)
                self._log(f"[XRDevice] Requested XR profile (static): {name}")
            except Exception as exc:
                self._log(f"[XRDevice] Failed static enable for '{name}': {exc}")

            try:
                # Also try enabling via XRProfile
                profile = None
                if hasattr(self._xr_core, "ensure_profile"):
                    profile = self._xr_core.ensure_profile(name)
                elif hasattr(self._xr_core, "get_profile"):
                    profile = self._xr_core.get_profile(name)
                if profile is not None and hasattr(profile, "request_enable_profile"):
                    profile.request_enable_profile()
                    self._log(f"[XRDevice] Requested XR profile (profile): {name}")
                break
            except Exception as exc:
                self._log(f"[XRDevice] Failed profile enable for '{name}': {exc}")
        self._dump_status(prefix="[XRDevice][After enable]")

    def _is_session_running(self) -> bool:
        if self._xr_core is None:
            return False
        if hasattr(self._xr_core, "is_session_running"):
            return bool(self._xr_core.is_session_running())
        # Fallback: check if XR is enabled and a profile is enabled
        try:
            if hasattr(self._xr_core, "is_xr_enabled") and not self._xr_core.is_xr_enabled():
                return False
            profile = None
            if hasattr(self._xr_core, "get_current_xr_profile"):
                profile = self._xr_core.get_current_xr_profile()
            elif hasattr(self._xr_core, "get_current_profile"):
                profile = self._xr_core.get_current_profile()
            if profile is not None and hasattr(profile, "is_enabled"):
                return bool(profile.is_enabled())
        except Exception:
            return False
        return False

    def get_poses(self):
        if self._xr_core is None:
            return self.left_pose.copy(), self.right_pose.copy()
        
        try:
            # Get controller poses from XR system
            from omni.kit.xr.core import XRCore
            xr = XRCore.get_singleton()
            
            if xr and self._is_session_running():
                # Try controller pose helpers first
                left_hand = None
                right_hand = None
                if hasattr(xr, "get_controller_pose"):
                    left_hand = xr.get_controller_pose("left")
                    right_hand = xr.get_controller_pose("right")
                # Fallback to input devices if needed
                if left_hand is None and hasattr(xr, "get_input_device"):
                    left_dev = xr.get_input_device("/user/hand/left")
                    if left_dev is not None and hasattr(left_dev, "get_pose"):
                        left_hand = left_dev.get_pose()
                if right_hand is None and hasattr(xr, "get_input_device"):
                    right_dev = xr.get_input_device("/user/hand/right")
                    if right_dev is not None and hasattr(right_dev, "get_pose"):
                        right_hand = right_dev.get_pose()

                if left_hand is not None and hasattr(left_hand, "GetTranslation"):
                    pos = left_hand.GetTranslation()
                    self.left_pose[:3] = [pos[0] * self.sensitivity,
                                          pos[1] * self.sensitivity,
                                          pos[2] * self.sensitivity]
                if right_hand is not None and hasattr(right_hand, "GetTranslation"):
                    pos = right_hand.GetTranslation()
                    self.right_pose[:3] = [pos[0] * self.sensitivity,
                                           pos[1] * self.sensitivity,
                                           pos[2] * self.sensitivity]
            else:
                import time
                now = time.time()
                if now - self._last_enable_attempt > self._enable_interval_s:
                    self._last_enable_attempt = now
                    self._log("[XRDevice] Session not running; attempting to enable profile")
                    try:
                        self._enable_first_available_profile()
                    except Exception as exc:
                        self._log(f"[XRDevice] Profile enable attempt failed: {exc}")
                if not self._reported_session:
                    self._log("[XRDevice] Session not running; using static poses")
                    self._reported_session = True
        except Exception as e:
            pass  # Silently continue with last known poses
        
        return self.left_pose.copy(), self.right_pose.copy()

    def __del__(self):
        if self._input and self._keyboard and self._sub_keyboard:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            except Exception:
                pass


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Main teleoperation loop."""
    
    print("\n" + "="*60)
    print("OPENARM BIMANUAL TELEOPERATION")
    print("="*60)
    
    # Configure environment for teleoperation
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"
    
    # Disable randomization for stable teleop
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.enable_corruption = False
    
    # Create environment
    print(f"\n[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    
    # Create policy runner to load the model
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[INFO] Policy loaded successfully!")
    
    # Initialize input device
    print(f"\n[INFO] Initializing {args_cli.input} input...")
    if args_cli.input == "xr":
        input_device = XRDevice(args_cli.sensitivity)
    elif args_cli.input == "vive":
        try:
            input_device = ViveDevice(args_cli.sensitivity)
        except RuntimeError as e:
            print(f"[WARN] {e}")
            print("[INFO] Falling back to keyboard control")
            input_device = KeyboardDevice(args_cli.sensitivity)
    elif args_cli.input == "gamepad":
        input_device = GamepadDevice(args_cli.sensitivity)
    else:
        input_device = KeyboardDevice(args_cli.sensitivity)
    
    # Get initial observations
    obs = env.get_observations()
    
    print("\n[INFO] Starting teleoperation loop...")
    print("[INFO] Press Ctrl+C to stop\n")
    
    step_count = 0
    
    try:
        while simulation_app.is_running() and step_count < 100000:
            # Get controller poses
            left_pose, right_pose = input_device.get_poses()
            
            # Update environment commands with teleop poses
            unwrapped = env.unwrapped.unwrapped  # Get through wrappers
            if hasattr(unwrapped, 'command_manager'):
                # Convert to tensors
                left_cmd = torch.tensor(left_pose, dtype=torch.float32, device="cuda:0")
                right_cmd = torch.tensor(right_pose, dtype=torch.float32, device="cuda:0")
                
                # Set left arm target
                if "left_ee_pose" in unwrapped.command_manager._terms:
                    unwrapped.command_manager._terms["left_ee_pose"].command[:, :3] = left_cmd[:3].unsqueeze(0)
                    unwrapped.command_manager._terms["left_ee_pose"].command[:, 3:7] = left_cmd[3:7].unsqueeze(0)
                
                # Set right arm target
                if "right_ee_pose" in unwrapped.command_manager._terms:
                    unwrapped.command_manager._terms["right_ee_pose"].command[:, :3] = right_cmd[:3].unsqueeze(0)
                    unwrapped.command_manager._terms["right_ee_pose"].command[:, 3:7] = right_cmd[3:7].unsqueeze(0)
            
            # Run policy
            with torch.inference_mode():
                actions = policy(obs)
            
            # Step environment
            obs, _, dones, _ = env.step(actions)
            
            # Print status periodically
            step_count += 1
            if step_count % 60 == 0:
                print(f"Step {step_count:5d} | "
                      f"L:[{left_pose[0]:.2f},{left_pose[1]:.2f},{left_pose[2]:.2f}] | "
                      f"R:[{right_pose[0]:.2f},{right_pose[1]:.2f},{right_pose[2]:.2f}]")
            
            # Handle reset if episode ends
            if dones.any():
                obs = env.get_observations()
                
    except KeyboardInterrupt:
        print("\n[INFO] Teleoperation stopped by user")
    finally:
        env.close()
        print("[INFO] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
