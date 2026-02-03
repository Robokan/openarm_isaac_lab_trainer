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
Bimanual IK Teleoperation for OpenArm

Uses inverse kinematics to control bimanual robot arms via keyboard or VR controllers.

Usage:
    python teleop_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 --input keyboard
    python teleop_bimanual.py --task Isaac-Reach-OpenArm-Bi-v0 --input xr
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Bimanual IK Teleoperation for OpenArm")
parser.add_argument("--task", type=str, default="Isaac-Reach-OpenArm-Bi-v0", help="Task name")
parser.add_argument("--input", type=str, default="keyboard", choices=["vive", "keyboard", "gamepad", "xr"],
                    help="Input device for teleoperation (xr = VR handtracking)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Controller sensitivity")
parser.add_argument("--viewer-eye", type=float, nargs=3, default=None,
                    help="Initial viewer eye position (x y z)")
parser.add_argument("--viewer-lookat", type=float, nargs=3, default=None,
                    help="Initial viewer look-at position (x y z)")
parser.add_argument("--viewport-camera", type=str, default=None,
                    help="Viewport camera prim path to render from (overrides auto selection)")

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
import time
import torch
import numpy as np
import math

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import isaaclab.utils.math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401


# =============================================================================
# Teleoperation Input Devices
# =============================================================================

class KeyboardDevice:
    """Keyboard control using Isaac Sim's input system."""
    
    def __init__(self, sensitivity: float = 1.0):
        import carb.input
        import omni.appwindow
        
        self.sensitivity = sensitivity
        self.step_size = 0.01 * sensitivity
        self.rot_step = 0.05  # radians per update (~3 degrees)
        # Initial poses (x, y, z, qw, qx, qy, qz)
        self.left_pose = np.array([0.2, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self.right_pose = np.array([0.2, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        # Track euler angles for easier rotation (roll, pitch, yaw)
        self.left_euler = np.array([0.0, 0.0, 0.0])
        self.right_euler = np.array([0.0, 0.0, 0.0])
        # Gripper values: 0.0 = open, 1.0 = closed
        self.left_gripper = 0.0
        self.right_gripper = 0.0
        self.gripper_step = 0.05  # How fast gripper opens/closes
        self.markers_visible = True  # Toggle for marker visibility
        self.active_hand = "left"
        self._carb_input = carb.input
        self._key_states = {}  # Track held keys for polling
        
        # Set up keyboard input
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = self._app_window.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        
        print("[KeyboardDevice] Keyboard input initialized")
        
        print("\n" + "="*60)
        print("KEYBOARD TELEOPERATION CONTROLS")
        print("="*60)
        print("Position (active hand):")
        print("  W/S: Forward/Backward (X)")
        print("  A/D: Left/Right (Y)")  
        print("  Q/E: Up/Down (Z)")
        print("")
        print("Rotation (active hand):")
        print("  I/K: Pitch up/down")
        print("  J/L: Yaw left/right")
        print("  U/O: Roll left/right")
        print("")
        print("Gripper (active hand):")
        print("  ; : Close gripper")
        print("  ' : Open gripper")
        print("")
        print("Hand Selection:")
        print("  1: Select LEFT hand")
        print("  2: Select RIGHT hand")
        print("")
        print("Other:")
        print("  M: Toggle marker visibility")
        print("  R: Reset poses to default")
        print("  Ctrl+C: Quit")
        print("="*60 + "\n")
    
    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events from Isaac Sim.
        
        Returns False for keys we handle to prevent UI from consuming them.
        Returns True for other keys to let them propagate.
        """
        key = event.input
        
        # Track key states for all events
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            self._key_states[key] = True
        elif event.type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self._key_states[key] = False
            return True  # Let release events propagate
        elif event.type != self._carb_input.KeyboardEventType.KEY_REPEAT:
            return True
        
        # Handle one-shot keys (hand selection, reset)
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            if key == self._carb_input.KeyboardInput.KEY_1:
                self.active_hand = "left"
                print("[Keyboard] Active: LEFT hand")
                return False
            elif key == self._carb_input.KeyboardInput.KEY_2:
                self.active_hand = "right"
                print("[Keyboard] Active: RIGHT hand")
                return False
            elif key == self._carb_input.KeyboardInput.R:
                self.left_pose = np.array([0.2, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
                self.right_pose = np.array([0.2, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
                self.left_euler = np.array([0.0, 0.0, 0.0])
                self.right_euler = np.array([0.0, 0.0, 0.0])
                self.left_gripper = 0.0
                self.right_gripper = 0.0
                print("[Keyboard] Poses reset")
                return False
            elif key == self._carb_input.KeyboardInput.M:
                self.markers_visible = not self.markers_visible
                print(f"[Keyboard] Markers {'visible' if self.markers_visible else 'hidden'}")
                return False
        
        # Check if it's a movement/rotation/gripper key we handle
        movement_keys = {
            self._carb_input.KeyboardInput.W,
            self._carb_input.KeyboardInput.S,
            self._carb_input.KeyboardInput.A,
            self._carb_input.KeyboardInput.D,
            self._carb_input.KeyboardInput.Q,
            self._carb_input.KeyboardInput.E,
            self._carb_input.KeyboardInput.I,
            self._carb_input.KeyboardInput.K,
            self._carb_input.KeyboardInput.J,
            self._carb_input.KeyboardInput.L,
            self._carb_input.KeyboardInput.U,
            self._carb_input.KeyboardInput.O,
            self._carb_input.KeyboardInput.SEMICOLON,
            self._carb_input.KeyboardInput.APOSTROPHE,
        }
        
        if key in movement_keys:
            return False  # Consume movement keys
        
        return True  # Let other keys propagate
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """Convert euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])
    
    def update(self):
        """Poll key states and update poses. Call this each frame."""
        pose = self.left_pose if self.active_hand == "left" else self.right_pose
        euler = self.left_euler if self.active_hand == "left" else self.right_euler
        
        # Position updates
        if self._key_states.get(self._carb_input.KeyboardInput.W, False):
            pose[0] += self.step_size
        if self._key_states.get(self._carb_input.KeyboardInput.S, False):
            pose[0] -= self.step_size
        if self._key_states.get(self._carb_input.KeyboardInput.A, False):
            pose[1] += self.step_size
        if self._key_states.get(self._carb_input.KeyboardInput.D, False):
            pose[1] -= self.step_size
        if self._key_states.get(self._carb_input.KeyboardInput.Q, False):
            pose[2] += self.step_size
        if self._key_states.get(self._carb_input.KeyboardInput.E, False):
            pose[2] -= self.step_size
        
        # Rotation updates (euler angles: roll, pitch, yaw)
        rot_changed = False
        if self._key_states.get(self._carb_input.KeyboardInput.U, False):
            euler[0] -= self.rot_step  # Roll left
            rot_changed = True
        if self._key_states.get(self._carb_input.KeyboardInput.O, False):
            euler[0] += self.rot_step  # Roll right
            rot_changed = True
        if self._key_states.get(self._carb_input.KeyboardInput.I, False):
            euler[1] += self.rot_step  # Pitch up
            rot_changed = True
        if self._key_states.get(self._carb_input.KeyboardInput.K, False):
            euler[1] -= self.rot_step  # Pitch down
            rot_changed = True
        if self._key_states.get(self._carb_input.KeyboardInput.J, False):
            euler[2] += self.rot_step  # Yaw left
            rot_changed = True
        if self._key_states.get(self._carb_input.KeyboardInput.L, False):
            euler[2] -= self.rot_step  # Yaw right
            rot_changed = True
        
        # Update quaternion if rotation changed
        if rot_changed:
            pose[3:7] = self._euler_to_quat(euler[0], euler[1], euler[2])
        
        # Gripper updates (for active hand)
        if self.active_hand == "left":
            if self._key_states.get(self._carb_input.KeyboardInput.SEMICOLON, False):
                self.left_gripper = min(1.0, self.left_gripper + self.gripper_step)
            if self._key_states.get(self._carb_input.KeyboardInput.APOSTROPHE, False):
                self.left_gripper = max(0.0, self.left_gripper - self.gripper_step)
        else:
            if self._key_states.get(self._carb_input.KeyboardInput.SEMICOLON, False):
                self.right_gripper = min(1.0, self.right_gripper + self.gripper_step)
            if self._key_states.get(self._carb_input.KeyboardInput.APOSTROPHE, False):
                self.right_gripper = max(0.0, self.right_gripper - self.gripper_step)
        
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

    def get_gripper_targets(self):
        return None, None


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
        quat = _quat_from_rot_matrix(
            m[0][0], m[0][1], m[0][2],
            m[1][0], m[1][1], m[1][2],
            m[2][0], m[2][1], m[2][2],
        )
        return np.array([
            m[0][3] * self.sensitivity,
            m[1][3] * self.sensitivity, 
            m[2][3] * self.sensitivity,
            quat[0], quat[1], quat[2], quat[3]
        ])

    def _get_trigger(self, controller_id):
        if controller_id is None:
            return None
        try:
            _, state = self.vr.getControllerState(controller_id)
            # OpenVR: trigger is typically rAxis[1].x in [0, 1]
            if hasattr(state, "rAxis") and len(state.rAxis) > 1:
                return float(state.rAxis[1].x)
        except Exception:
            return None
        return None
    
    def get_poses(self):
        left = self._get_pose(self.left_id)
        right = self._get_pose(self.right_id)
        if left is not None: self.left_pose = left
        if right is not None: self.right_pose = right
        return self.left_pose.copy(), self.right_pose.copy()

    def get_gripper_targets(self):
        left = self._get_trigger(self.left_id)
        right = self._get_trigger(self.right_id)
        return left, right
    
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

                if left_hand is not None:
                    pos = _pos_from_pose(left_hand)
                    if pos is not None:
                        pos = _map_xr_position(pos)
                        self.left_pose[:3] = [
                            pos[0] * self.sensitivity,
                            pos[1] * self.sensitivity,
                            pos[2] * self.sensitivity,
                        ]
                if left_hand is not None:
                    quat = _quat_from_pose(left_hand)
                    if quat is not None:
                        self.left_pose[3:7] = _map_xr_quat(quat)
                if right_hand is not None:
                    pos = _pos_from_pose(right_hand)
                    if pos is not None:
                        pos = _map_xr_position(pos)
                        self.right_pose[:3] = [
                            pos[0] * self.sensitivity,
                            pos[1] * self.sensitivity,
                            pos[2] * self.sensitivity,
                        ]
                if right_hand is not None:
                    quat = _quat_from_pose(right_hand)
                    if quat is not None:
                        self.right_pose[3:7] = _map_xr_quat(quat)
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

    def _get_trigger_value_from_device(self, device):
        """Try various methods to get trigger value from an input device."""
        if device is None:
            return None
        try:
            # Try get_input_value
            if hasattr(device, "get_input_value"):
                for key in ("trigger", "trigger/value", "select", "squeeze"):
                    try:
                        val = device.get_input_value(key)
                        if val is not None:
                            return float(val)
                    except Exception:
                        pass
            # Try get_float
            if hasattr(device, "get_float"):
                for key in ("trigger", "trigger/value", "select", "squeeze"):
                    try:
                        val = device.get_float(key)
                        if val is not None:
                            return float(val)
                    except Exception:
                        pass
            # Try get_axis
            if hasattr(device, "get_axis"):
                for key in ("trigger", "select", "squeeze"):
                    try:
                        val = device.get_axis(key)
                        if val is not None:
                            return float(val)
                    except Exception:
                        pass
            # Try direct attribute access
            for attr in ("trigger", "trigger_value", "squeeze", "select"):
                if hasattr(device, attr):
                    val = getattr(device, attr)
                    if val is not None and not callable(val):
                        return float(val)
        except Exception:
            pass
        return None

    def get_gripper_targets(self):
        if self._xr_core is None:
            return None, None
        try:
            from omni.kit.xr.core import XRCore
            xr = XRCore.get_singleton()
            if xr is None:
                return None, None
            
            left_val = None
            right_val = None
            
            # Try method 1: get_input_device with various input names
            if hasattr(xr, "get_input_device"):
                left_dev = xr.get_input_device("/user/hand/left")
                right_dev = xr.get_input_device("/user/hand/right")
                left_val = self._get_trigger_value_from_device(left_dev)
                right_val = self._get_trigger_value_from_device(right_dev)
            
            # Try method 2: get_controller_inputs
            if (left_val is None or right_val is None) and hasattr(xr, "get_controller_inputs"):
                try:
                    inputs = xr.get_controller_inputs()
                    if inputs:
                        if left_val is None:
                            for key in ("left_trigger", "left_squeeze", "left_select"):
                                if key in inputs:
                                    left_val = float(inputs[key])
                                    break
                        if right_val is None:
                            for key in ("right_trigger", "right_squeeze", "right_select"):
                                if key in inputs:
                                    right_val = float(inputs[key])
                                    break
                except Exception:
                    pass
            
            # Try method 3: get_action_float for action-based input
            if (left_val is None or right_val is None) and hasattr(xr, "get_action_float"):
                try:
                    if left_val is None:
                        left_val = xr.get_action_float("/user/hand/left/input/trigger/value")
                    if right_val is None:
                        right_val = xr.get_action_float("/user/hand/right/input/trigger/value")
                except Exception:
                    pass
            
            # Try method 4: Access active profile's inputs
            if (left_val is None or right_val is None) and self._profile is not None:
                if hasattr(self._profile, "get_inputs"):
                    try:
                        inputs = self._profile.get_inputs()
                        if inputs:
                            if left_val is None and hasattr(inputs, "left_trigger"):
                                left_val = float(inputs.left_trigger)
                            if right_val is None and hasattr(inputs, "right_trigger"):
                                right_val = float(inputs.right_trigger)
                    except Exception:
                        pass
            
            return left_val, right_val
        except Exception:
            return None, None

    def __del__(self):
        if self._input and self._keyboard and self._sub_keyboard:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            except Exception:
                pass


def _quat_from_rot_matrix(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion."""
    t = m00 + m11 + m22
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def _quat_from_pose(pose):
    """Best-effort quaternion extraction from an XR pose object."""
    try:
        if hasattr(pose, "ExtractRotationQuat"):
            q = pose.ExtractRotationQuat()
            if hasattr(q, "GetReal") and hasattr(q, "GetImaginary"):
                im = q.GetImaginary()
                return np.array([q.GetReal(), im[0], im[1], im[2]], dtype=np.float32)
        if hasattr(pose, "GetRotation"):
            rot = pose.GetRotation()
            if hasattr(rot, "GetQuaternion"):
                q = rot.GetQuaternion()
                return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
            if hasattr(rot, "GetQuat"):
                q = rot.GetQuat()
                if hasattr(q, "GetReal") and hasattr(q, "GetImaginary"):
                    im = q.GetImaginary()
                    return np.array([q.GetReal(), im[0], im[1], im[2]], dtype=np.float32)
        if hasattr(pose, "GetQuaternion"):
            q = pose.GetQuaternion()
            if len(q) == 4:
                return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        if hasattr(pose, "GetRow"):
            r0 = pose.GetRow(0)
            r1 = pose.GetRow(1)
            r2 = pose.GetRow(2)
            return _quat_from_rot_matrix(r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2])
    except Exception:
        return None
    return None


def _pos_from_pose(pose):
    """Best-effort translation extraction from an XR pose object."""
    try:
        if hasattr(pose, "GetTranslation"):
            pos = pose.GetTranslation()
            if pos is not None:
                return np.array([pos[0], pos[1], pos[2]], dtype=np.float32)
        if hasattr(pose, "GetPosition"):
            pos = pose.GetPosition()
            if pos is not None:
                return np.array([pos[0], pos[1], pos[2]], dtype=np.float32)
        if hasattr(pose, "GetRow"):
            r3 = pose.GetRow(3)
            return np.array([r3[0], r3[1], r3[2]], dtype=np.float32)
        if hasattr(pose, "GetMatrix"):
            m = pose.GetMatrix()
            return np.array([m[0][3], m[1][3], m[2][3]], dtype=np.float32)
    except Exception:
        return None
    return None


def _map_xr_position(pos):
    """Map OpenXR position to robot space (swap Y/Z by default)."""
    try:
        m = _get_xr_map_matrix()
        mapped = m @ np.array([pos[0], pos[1], pos[2]], dtype=np.float32)
        return mapped
    except Exception:
        pass
    return np.array([pos[0], pos[1], pos[2]], dtype=np.float32)


def _get_xr_map_matrix():
    swap_yz = os.environ.get("XR_SWAP_YZ", "1").lower() not in ("0", "false", "no")
    flip_x = os.environ.get("XR_FLIP_X", "0").lower() not in ("0", "false", "no")
    flip_z = os.environ.get("XR_FLIP_Z", "1").lower() not in ("0", "false", "no")
    fx = -1.0 if flip_x else 1.0
    fz = -1.0 if flip_z else 1.0
    if swap_yz:
        # [x, y, z] -> [fx*x, fz*z, y]
        return np.array([[fx, 0.0, 0.0], [0.0, 0.0, fz], [0.0, 1.0, 0.0]], dtype=np.float32)
    return np.array([[fx, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, fz]], dtype=np.float32)


def _rot_matrix_from_quat(quat):
    """Convert (w, x, y, z) quaternion to 3x3 rotation matrix."""
    w, x, y, z = quat
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )


def _map_xr_quat(quat):
    """Map OpenXR orientation into robot space using the same axis mapping."""
    try:
        m = _get_xr_map_matrix()
        r = _rot_matrix_from_quat(quat)
        r_mapped = m @ r @ m.T
        # Apply -90° rotation around X so controller Y-forward maps to gripper Z-forward
        rot_x_deg = float(os.environ.get("XR_GRIPPER_ROT_X_DEG", "-90"))
        rot_x = math.radians(rot_x_deg)
        c = math.cos(rot_x)
        s = math.sin(rot_x)
        r_fix = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
        r_mapped = r_mapped @ r_fix
        return _quat_from_rot_matrix(
            r_mapped[0, 0], r_mapped[0, 1], r_mapped[0, 2],
            r_mapped[1, 0], r_mapped[1, 1], r_mapped[1, 2],
            r_mapped[2, 0], r_mapped[2, 1], r_mapped[2, 2],
        )
    except Exception:
        return quat


def _find_body_camera_prim(stage, body_name: str, camera_names: list[str]) -> str | None:
    """Find a camera prim under a body link by name."""
    try:
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            if prim.GetTypeName() != "Camera":
                continue
            if prim.GetName() not in camera_names:
                continue
            path = prim.GetPath().pathString
            if body_name in path:
                return path
    except Exception:
        return None
    return None


def _set_viewport_camera(camera_path: str) -> bool:
    """Set active viewport camera to the given prim path."""
    try:
        import omni.kit.viewport.utility as viewport_utils
        viewport = viewport_utils.get_active_viewport()
        if viewport is None:
            return False
        if hasattr(viewport, "set_active_camera"):
            viewport.set_active_camera(camera_path)
        elif hasattr(viewport, "set_camera_path"):
            viewport.set_camera_path(camera_path)
        elif hasattr(viewport, "camera_path"):
            viewport.camera_path = camera_path
        else:
            viewport_utils.set_active_viewport_camera(viewport, camera_path)
        return True
    except Exception:
        return False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg=None):
    """Main entry point for IK-based teleoperation."""
    # Note: agent_cfg is passed by hydra but not used for IK control
    
    # Configure environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"

    if hasattr(env_cfg, "viewer"):
        if args_cli.viewer_eye is not None:
            env_cfg.viewer.eye = tuple(args_cli.viewer_eye)
        if args_cli.viewer_lookat is not None:
            env_cfg.viewer.lookat = tuple(args_cli.viewer_lookat)
    
    # Disable randomization for stable control
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.enable_corruption = False
    
    # Create environment
    print(f"\n[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)

    # If in XR mode, set viewport to EgoVR camera by default (or user override)
    if args_cli.input == "xr":
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            camera_path = args_cli.viewport_camera
            if camera_path is None:
                camera_path = _find_body_camera_prim(stage, "openarm_body_link", ["EgoVR", "Ego", "Camera"])
            if camera_path:
                if _set_viewport_camera(camera_path):
                    print(f"[INFO] Viewport camera set to: {camera_path}")
                else:
                    print(f"[WARN] Failed to set viewport camera: {camera_path}")
            else:
                print("[WARN] EgoVR camera not found under openarm_body_link")
        except Exception as exc:
            print(f"[WARN] Unable to set viewport camera: {exc}")
    
    print("[INFO] Using IK-based control")
    
    # Run IK teleoperation
    run_teleop(env, args_cli)
    
    env.close()
    print("[INFO] Environment closed")


def run_teleop(env, args):
    """Run IK-based teleoperation with manual input devices."""
    
    print("\n" + "="*60)
    print("OPENARM BIMANUAL IK TELEOPERATION")
    print("="*60)
    
    # Initialize input device
    print(f"\n[INFO] Initializing {args.input} input...")
    if args.input == "xr":
        input_device = XRDevice(args.sensitivity)
    elif args.input == "vive":
        try:
            input_device = ViveDevice(args.sensitivity)
        except RuntimeError as e:
            print(f"[WARN] {e}")
            print("[INFO] Falling back to keyboard control")
            input_device = KeyboardDevice(args.sensitivity)
    elif args.input == "gamepad":
        input_device = GamepadDevice(args.sensitivity)
    else:
        input_device = KeyboardDevice(args.sensitivity)
    
    # Get unwrapped env
    unwrapped = env.unwrapped
    if hasattr(unwrapped, 'unwrapped'):
        unwrapped = unwrapped.unwrapped

    # Gripper joints (optional trigger control)
    gripper_open_pos = 0.044
    robot = unwrapped.scene["robot"]
    left_gripper_ids, _ = robot.find_joints("openarm_left_finger_joint.*")
    right_gripper_ids, _ = robot.find_joints("openarm_right_finger_joint.*")
    sim_device = unwrapped.device if hasattr(unwrapped, "device") else "cuda:0"
    
    # IK setup - SPLIT CONTROL:
    # - Joints 1-4: IK for position only (shoulder/elbow)
    # - Joints 5-7: Direct control from euler angles (wrist)
    print("[INFO] Setting up split IK controllers (position-only for joints 1-4)...")
    
    # Create IK controllers for position-only control (first 4 joints)
    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",  # Only 3-DOF position control
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.1},
    )
    left_ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=sim_device)
    right_ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=sim_device)
    
    # Get joint IDs - split into position joints (1-4) and wrist joints (5-7)
    left_pos_joint_ids, left_pos_joint_names = robot.find_joints([
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4"
    ])
    left_wrist_joint_ids, left_wrist_joint_names = robot.find_joints([
        "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7"
    ])
    right_pos_joint_ids, right_pos_joint_names = robot.find_joints([
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4"
    ])
    right_wrist_joint_ids, right_wrist_joint_names = robot.find_joints([
        "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7"
    ])
    
    # Get body IDs for end-effectors (still use hand for position tracking)
    left_body_ids, _ = robot.find_bodies("openarm_left_hand")
    right_body_ids, _ = robot.find_bodies("openarm_right_hand")
    left_body_idx = left_body_ids[0]
    right_body_idx = right_body_ids[0]
    
    # For fixed-base robots, jacobian body index is offset by 1
    # Only use first 4 joints for position IK
    if robot.is_fixed_base:
        left_jacobi_body_idx = left_body_idx - 1
        right_jacobi_body_idx = right_body_idx - 1
        left_jacobi_joint_ids = left_pos_joint_ids  # Only joints 1-4
        right_jacobi_joint_ids = right_pos_joint_ids  # Only joints 1-4
    else:
        left_jacobi_body_idx = left_body_idx
        right_jacobi_body_idx = right_body_idx
        left_jacobi_joint_ids = [i + 6 for i in left_pos_joint_ids]
        right_jacobi_joint_ids = [i + 6 for i in right_pos_joint_ids]
    
    print(f"[INFO] Left position joints (IK): {left_pos_joint_names}")
    print(f"[INFO] Left wrist joints (direct): {left_wrist_joint_names}")
    print(f"[INFO] Right position joints (IK): {right_pos_joint_names}")
    print(f"[INFO] Right wrist joints (direct): {right_wrist_joint_names}")
    
    print("\n[INFO] Starting teleoperation loop...")
    print("[INFO] Press Ctrl+C to stop\n")
    
    # Debug: check what XR input methods are available
    if hasattr(input_device, "_xr_core") and input_device._xr_core is not None:
        try:
            from omni.kit.xr.core import XRCore
            xr = XRCore.get_singleton()
            if xr:
                xr_methods = [m for m in dir(xr) if not m.startswith("_") and callable(getattr(xr, m, None))]
                print(f"[DEBUG] XR Core methods: {xr_methods[:20]}...")  # First 20 methods
                if hasattr(xr, "get_input_device"):
                    left_dev = xr.get_input_device("/user/hand/left")
                    if left_dev:
                        dev_methods = [m for m in dir(left_dev) if not m.startswith("_")]
                        print(f"[DEBUG] Input device methods: {dev_methods[:15]}...")
        except Exception as e:
            print(f"[DEBUG] XR introspection failed: {e}")
    
    # Set grippers to open position at startup
    if left_gripper_ids:
        left_open = torch.full((1, len(left_gripper_ids)), gripper_open_pos, device=sim_device)
        robot.write_joint_position_to_sim(left_open, joint_ids=left_gripper_ids)
    if right_gripper_ids:
        right_open = torch.full((1, len(right_gripper_ids)), gripper_open_pos, device=sim_device)
        robot.write_joint_position_to_sim(right_open, joint_ids=right_gripper_ids)
    
    step_count = 0
    prev_markers_visible = True  # Track previous visibility state
    
    try:
        while simulation_app.is_running() and step_count < 100000:
            # Update input device (poll key states for keyboard)
            if hasattr(input_device, 'update'):
                input_device.update()
            
            # Toggle marker visibility if changed
            if hasattr(input_device, 'markers_visible'):
                if input_device.markers_visible != prev_markers_visible:
                    prev_markers_visible = input_device.markers_visible
                    # Toggle visibility of all marker prims
                    try:
                        import omni.usd
                        from pxr import UsdGeom
                        stage = omni.usd.get_context().get_stage()
                        count = 0
                        # Find all prims with marker-related names
                        for prim in stage.Traverse():
                            prim_path = str(prim.GetPath())
                            # Match goal_pose, body_pose markers
                            if ("goal_pose" in prim_path or "body_pose" in prim_path or 
                                ("Visuals" in prim_path and "Command" in prim_path)):
                                try:
                                    imageable = UsdGeom.Imageable(prim)
                                    if imageable:
                                        if input_device.markers_visible:
                                            imageable.MakeVisible()
                                        else:
                                            imageable.MakeInvisible()
                                        count += 1
                                except:
                                    pass
                        if count > 0:
                            print(f"[Markers] Toggled {count} prims {'visible' if input_device.markers_visible else 'hidden'}")
                    except Exception as e:
                        print(f"[WARN] Could not toggle markers: {e}")
            
            # Get controller poses
            left_pose, right_pose = input_device.get_poses()
            
            # Map trigger/keyboard to gripper positions (open by default, close when triggered)
            left_trigger = 0.0
            right_trigger = 0.0
            if hasattr(input_device, "get_gripper_targets"):
                lt, rt = input_device.get_gripper_targets()
                if lt is not None:
                    left_trigger = float(np.clip(lt, 0.0, 1.0))
                if rt is not None:
                    right_trigger = float(np.clip(rt, 0.0, 1.0))
            # Use keyboard gripper values if available
            if hasattr(input_device, "left_gripper"):
                left_trigger = input_device.left_gripper
                right_trigger = input_device.right_gripper
            
            # Left gripper: open (0.044) when trigger=0, closed (0) when trigger=1
            if left_gripper_ids:
                left_pos = gripper_open_pos * (1.0 - left_trigger)
                left_targets = torch.full(
                    (1, len(left_gripper_ids)),
                    left_pos,
                    device=sim_device,
                )
                robot.write_joint_position_to_sim(left_targets, joint_ids=left_gripper_ids)
            
            # Right gripper: open (0.044) when trigger=0, closed (0) when trigger=1
            if right_gripper_ids:
                right_pos = gripper_open_pos * (1.0 - right_trigger)
                right_targets = torch.full(
                    (1, len(right_gripper_ids)),
                    right_pos,
                    device=sim_device,
                )
                robot.write_joint_position_to_sim(right_targets, joint_ids=right_gripper_ids)
            
            # ===== SPLIT CONTROL =====
            # Position IK (joints 1-4) + Direct wrist control (joints 5-7)
            
            # Convert position targets to tensors
            left_target_pos = torch.tensor(left_pose[:3], dtype=torch.float32, device=sim_device).unsqueeze(0)
            right_target_pos = torch.tensor(right_pose[:3], dtype=torch.float32, device=sim_device).unsqueeze(0)
            
            # Get euler angles from input device for direct wrist control
            if hasattr(input_device, 'left_euler'):
                left_euler = input_device.left_euler
                right_euler = input_device.right_euler
            else:
                # For VR/gamepad, extract euler from quaternion
                left_quat = left_pose[3:7]  # qw, qx, qy, qz
                right_quat = right_pose[3:7]
                # Convert to euler (simplified - may need adjustment for VR)
                left_euler = np.array([0.0, 0.0, 0.0])
                right_euler = np.array([0.0, 0.0, 0.0])
            
            # Get current end-effector position in robot base frame
            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            
            # Left arm EE pose
            left_ee_pos_w = robot.data.body_pos_w[:, left_body_idx]
            left_ee_quat_w = robot.data.body_quat_w[:, left_body_idx]
            left_ee_pos_b, left_ee_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, left_ee_pos_w, left_ee_quat_w
            )
            
            # Right arm EE pose
            right_ee_pos_w = robot.data.body_pos_w[:, right_body_idx]
            right_ee_quat_w = robot.data.body_quat_w[:, right_body_idx]
            right_ee_pos_b, right_ee_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, right_ee_pos_w, right_ee_quat_w
            )
            
            # Get Jacobians - only for position joints (1-4)
            jacobians = robot.root_physx_view.get_jacobians()
            
            # Left arm Jacobian (only position joints, only position rows)
            left_jacobian_w = jacobians[:, left_jacobi_body_idx, :, left_jacobi_joint_ids]
            base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
            left_jacobian_b = left_jacobian_w.clone()
            left_jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, left_jacobian_w[:, :3, :])
            left_jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, left_jacobian_w[:, 3:, :])
            
            # Right arm Jacobian
            right_jacobian_w = jacobians[:, right_jacobi_body_idx, :, right_jacobi_joint_ids]
            right_jacobian_b = right_jacobian_w.clone()
            right_jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, right_jacobian_w[:, :3, :])
            right_jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, right_jacobian_w[:, 3:, :])
            
            # Get current position joint values (joints 1-4 only)
            left_pos_joint_vals = robot.data.joint_pos[:, left_pos_joint_ids]
            right_pos_joint_vals = robot.data.joint_pos[:, right_pos_joint_ids]
            
            # ===== POSITION IK (joints 1-4) =====
            # Set position-only target (ee_quat is required for display but not used in IK)
            left_ik_controller.set_command(left_target_pos, ee_quat=left_ee_quat_b)
            right_ik_controller.set_command(right_target_pos, ee_quat=right_ee_quat_b)
            
            # Compute IK to get desired joint positions for joints 1-4
            left_pos_joints_des = left_ik_controller.compute(
                left_ee_pos_b, left_ee_quat_b, left_jacobian_b, left_pos_joint_vals
            )
            right_pos_joints_des = right_ik_controller.compute(
                right_ee_pos_b, right_ee_quat_b, right_jacobian_b, right_pos_joint_vals
            )
            
            # ===== DIRECT WRIST CONTROL (joints 5-7) =====
            # Convert euler angles directly to wrist joint positions
            # Joint axes: 5=Z, 6=X, 7=Y
            # Euler: roll=X, pitch=Y, yaw=Z
            # Mapping: joint5=yaw(Z), joint6=roll(X), joint7=pitch(Y)
            left_wrist_joints_des = torch.tensor(
                [[left_euler[2], left_euler[0], left_euler[1]]],  # yaw, roll, pitch
                dtype=torch.float32, device=sim_device
            )
            right_wrist_joints_des = torch.tensor(
                [[right_euler[2], right_euler[0], right_euler[1]]],  # yaw, roll, pitch
                dtype=torch.float32, device=sim_device
            )
            
            # Apply joint position targets
            robot.set_joint_position_target(left_pos_joints_des, joint_ids=left_pos_joint_ids)
            robot.set_joint_position_target(left_wrist_joints_des, joint_ids=left_wrist_joint_ids)
            robot.set_joint_position_target(right_pos_joints_des, joint_ids=right_pos_joint_ids)
            robot.set_joint_position_target(right_wrist_joints_des, joint_ids=right_wrist_joint_ids)
            
            # Write the articulation data to simulation
            robot.write_data_to_sim()
            
            # Step the simulation directly (bypass env.step to avoid command manager resampling)
            unwrapped.sim.step(render=True)
            
            # Update robot data from simulation
            robot.update(unwrapped.sim.get_physics_dt())
            
            # Print status periodically
            step_count += 1
            if step_count % 60 == 0:
                print(f"Step {step_count:5d} | "
                      f"L:[{left_pose[0]:.2f},{left_pose[1]:.2f},{left_pose[2]:.2f}] | "
                      f"R:[{right_pose[0]:.2f},{right_pose[1]:.2f},{right_pose[2]:.2f}] | "
                      f"Grip L:{left_trigger:.2f} R:{right_trigger:.2f}")
                
    except KeyboardInterrupt:
        print("\n[INFO] Teleoperation stopped by user")


if __name__ == "__main__":
    main()
    simulation_app.close()
