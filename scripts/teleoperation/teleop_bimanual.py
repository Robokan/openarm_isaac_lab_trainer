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
        print("  C: Spawn cube")
        print("  M: Toggle marker visibility")
        print("  R: Reset poses to default")
        print("  Ctrl+C: Quit")
        print("="*60 + "\n")
        
        self.spawn_cube_requested = False  # Flag for cube spawning
    
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
            elif key == self._carb_input.KeyboardInput.C:
                self.spawn_cube_requested = True
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
            # Try get_input_base (XRInputDevice method)
            if hasattr(device, "get_input_base"):
                for key in ("select", "trigger", "squeeze", "grip"):
                    try:
                        val = device.get_input_base(key)
                        if val is not None:
                            return float(val)
                    except Exception:
                        pass
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
        """Get gripper values from controller trigger/squeeze buttons.
        
        Uses Isaac Lab's approach: get_input_gesture_value("trigger", "value")
        """
        if self._xr_core is None:
            return None, None
        try:
            from omni.kit.xr.core import XRCore
            xr = XRCore.get_singleton()
            if xr is None:
                return None, None
            
            left_val = None
            right_val = None
            
            if hasattr(xr, 'get_input_device'):
                left_dev = xr.get_input_device("/user/hand/left")
                right_dev = xr.get_input_device("/user/hand/right")
                
                # Debug once: print device methods and available inputs
                if not hasattr(self, '_debug_trigger_printed'):
                    self._debug_trigger_printed = True
                    if left_dev:
                        dev_methods = [m for m in dir(left_dev) if not m.startswith('_')]
                        print(f"[DEBUG] Left device methods: {dev_methods[:20]}...")
                        # Check for gesture methods (Isaac Lab approach)
                        for method in ['has_input_gesture', 'get_input_gesture_value', 
                                       'get_input_names', 'get_all_virtual_world_poses']:
                            if hasattr(left_dev, method):
                                print(f"[DEBUG]   Has: {method}")
                        # List available input gestures
                        if hasattr(left_dev, 'has_input_gesture'):
                            for gesture in ['trigger', 'squeeze', 'grip', 'select']:
                                for subtype in ['value', 'click', 'touch']:
                                    try:
                                        has = left_dev.has_input_gesture(gesture, subtype)
                                        if has:
                                            print(f"[DEBUG]   has_input_gesture('{gesture}', '{subtype}'): {has}")
                                    except:
                                        pass
                
                left_val = self._get_trigger_from_gesture(left_dev)
                right_val = self._get_trigger_from_gesture(right_dev)
            
            return left_val, right_val
        except Exception as e:
            if not hasattr(self, '_debug_gripper_err'):
                self._debug_gripper_err = True
                print(f"[DEBUG] get_gripper_targets error: {e}")
            return None, None
    
    def _get_trigger_from_gesture(self, device):
        """Get trigger value using Isaac Lab's input gesture API.
        
        Uses has_input_gesture() and get_input_gesture_value() methods.
        """
        if device is None:
            return None
        
        try:
            # Method 1: Isaac Lab approach - get_input_gesture_value
            if hasattr(device, 'has_input_gesture') and hasattr(device, 'get_input_gesture_value'):
                # Try trigger first
                if device.has_input_gesture("trigger", "value"):
                    return float(device.get_input_gesture_value("trigger", "value"))
                # Try squeeze/grip
                if device.has_input_gesture("squeeze", "value"):
                    return float(device.get_input_gesture_value("squeeze", "value"))
                if device.has_input_gesture("grip", "value"):
                    return float(device.get_input_gesture_value("grip", "value"))
                # Try select (often mapped to trigger)
                if device.has_input_gesture("select", "value"):
                    return float(device.get_input_gesture_value("select", "value"))
                if device.has_input_gesture("select", "click"):
                    val = device.get_input_gesture_value("select", "click")
                    return 1.0 if val else 0.0
            
            # Method 2: Fallback to get_input_base
            if hasattr(device, 'get_input_base'):
                for key in ('trigger', 'squeeze', 'grip', 'select'):
                    try:
                        val = device.get_input_base(key)
                        if val is not None:
                            return float(val) if isinstance(val, (int, float)) else (1.0 if val else 0.0)
                    except:
                        pass
        except Exception:
            pass
        
        return None
    
    def get_button_states(self):
        """Get button states (A, B, X, Y) from controllers.
        
        Returns dict with button states: {'right_a': bool, 'right_b': bool, 'left_x': bool, 'left_y': bool}
        """
        states = {'right_a': False, 'right_b': False, 'left_x': False, 'left_y': False}
        
        if self._xr_core is None:
            return states
        
        try:
            from omni.kit.xr.core import XRCore
            xr = XRCore.get_singleton()
            if xr is None:
                return states
            
            if hasattr(xr, 'get_input_device'):
                right_dev = xr.get_input_device("/user/hand/right")
                left_dev = xr.get_input_device("/user/hand/left")
                
                # Right controller: A and B buttons
                if right_dev and hasattr(right_dev, 'has_input_gesture'):
                    if right_dev.has_input_gesture("a", "click"):
                        states['right_a'] = bool(right_dev.get_input_gesture_value("a", "click"))
                    if right_dev.has_input_gesture("b", "click"):
                        states['right_b'] = bool(right_dev.get_input_gesture_value("b", "click"))
                
                # Left controller: X and Y buttons
                if left_dev and hasattr(left_dev, 'has_input_gesture'):
                    if left_dev.has_input_gesture("x", "click"):
                        states['left_x'] = bool(left_dev.get_input_gesture_value("x", "click"))
                    if left_dev.has_input_gesture("y", "click"):
                        states['left_y'] = bool(left_dev.get_input_gesture_value("y", "click"))
        except Exception:
            pass
        
        return states

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
        
        # Optional rotation corrections
        rot_x_deg = float(os.environ.get("XR_GRIPPER_ROT_X_DEG", "-180"))
        rot_y_deg = float(os.environ.get("XR_GRIPPER_ROT_Y_DEG", "0"))
        
        if rot_x_deg != 0:
            rot_x = math.radians(rot_x_deg)
            cx = math.cos(rot_x)
            sx = math.sin(rot_x)
            r_fix_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
            r_mapped = r_mapped @ r_fix_x
        
        if rot_y_deg != 0:
            rot_y = math.radians(rot_y_deg)
            cy = math.cos(rot_y)
            sy = math.sin(rot_y)
            r_fix_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
            r_mapped = r_mapped @ r_fix_y
        
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


def spawn_cube(stage, position=(0.4, 0.0, 0.5), size=0.025, color=(0.2, 0.6, 1.0), cube_counter=[0]):
    """Spawn a physics-enabled cube at the given position.
    
    Args:
        stage: USD stage
        position: (x, y, z) spawn position in meters
        size: cube size in meters
        color: (r, g, b) color values 0-1
        cube_counter: mutable counter for unique naming
    
    Returns:
        Path to the spawned cube prim
    """
    from pxr import UsdGeom, UsdPhysics, Gf, Sdf
    
    cube_counter[0] += 1
    cube_path = f"/World/spawned_cube_{cube_counter[0]}"
    
    # Create the cube
    cube_prim = UsdGeom.Cube.Define(stage, cube_path)
    cube_prim.GetSizeAttr().Set(size * 2)  # USD cube size is full width
    
    # Set position
    xform = UsdGeom.Xformable(cube_prim.GetPrim())
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
    
    # Set color
    cube_prim.GetDisplayColorAttr().Set([Gf.Vec3f(color[0], color[1], color[2])])
    
    # Add rigid body physics
    prim = cube_prim.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    
    # Set mass
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.GetMassAttr().Set(0.1)  # 100 grams
    
    print(f"[Cube] Spawned cube at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
    return cube_path


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
    
    # IK setup - Full pose IK for all 7 joints
    print("[INFO] Setting up full-pose IK controllers...")
    
    # ===== SPLIT IK: Position (joints 1-4) + Direct wrist control (joints 5-7) =====
    print("[INFO] Setting up split IK: Position (joints 1-4) + Wrist (joints 5-7)...")
    
    # Position-only IK for joints 1-4
    ik_cfg = DifferentialIKControllerCfg(
        command_type="position",  # Position-only control
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.1},
    )
    left_ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=sim_device)
    right_ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=sim_device)
    
    # Get joint IDs for position control (joints 1-4)
    left_pos_joint_ids, left_pos_joint_names = robot.find_joints([
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3", "openarm_left_joint4"
    ])
    right_pos_joint_ids, right_pos_joint_names = robot.find_joints([
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3", "openarm_right_joint4"
    ])
    
    # Get joint IDs for wrist control (joints 5-7)
    left_wrist_joint_ids, left_wrist_joint_names = robot.find_joints([
        "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7"
    ])
    right_wrist_joint_ids, right_wrist_joint_names = robot.find_joints([
        "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7"
    ])
    
    # Combined joint IDs for reference
    left_arm_joint_ids = list(left_pos_joint_ids) + list(left_wrist_joint_ids)
    right_arm_joint_ids = list(right_pos_joint_ids) + list(right_wrist_joint_ids)
    
    # Get body IDs for end-effectors (hand for position target)
    left_body_ids, _ = robot.find_bodies("openarm_left_hand")
    right_body_ids, _ = robot.find_bodies("openarm_right_hand")
    left_body_idx = left_body_ids[0]
    right_body_idx = right_body_ids[0]
    
    # Get body IDs for link4 (forearm - for wrist relative orientation)
    left_link4_ids, _ = robot.find_bodies("openarm_left_link4")
    right_link4_ids, _ = robot.find_bodies("openarm_right_link4")
    left_link4_idx = left_link4_ids[0]
    right_link4_idx = right_link4_ids[0]
    print(f"[INFO] Link4 body idx: L={left_link4_idx}, R={right_link4_idx}")
    
    # For fixed-base robots, jacobian body index is offset by 1
    # Only use joints 1-4 for position IK Jacobian
    if robot.is_fixed_base:
        left_jacobi_body_idx = left_body_idx - 1
        right_jacobi_body_idx = right_body_idx - 1
        left_jacobi_joint_ids = list(left_pos_joint_ids)  # Only joints 1-4
        right_jacobi_joint_ids = list(right_pos_joint_ids)  # Only joints 1-4
    else:
        left_jacobi_body_idx = left_body_idx
        right_jacobi_body_idx = right_body_idx
        left_jacobi_joint_ids = [i + 6 for i in left_pos_joint_ids]
        right_jacobi_joint_ids = [i + 6 for i in right_pos_joint_ids]
    
    print(f"[INFO] Position joints (1-4): L={left_pos_joint_names}, R={right_pos_joint_names}")
    print(f"[INFO] Wrist joints (5-7): L={left_wrist_joint_names}, R={right_wrist_joint_names}")
    print(f"[INFO] Position joint IDs: L={left_pos_joint_ids}, R={right_pos_joint_ids}")
    print(f"[INFO] Wrist joint IDs: L={left_wrist_joint_ids}, R={right_wrist_joint_ids}")
    print(f"[INFO] EE body index: L={left_body_idx}, R={right_body_idx}")
    print(f"[INFO] Jacobi body idx: L={left_jacobi_body_idx}, R={right_jacobi_body_idx}")
    
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
    prev_a_button = False  # Track A button for edge detection
    
    # Get USD stage for cube spawning
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    except Exception:
        stage = None
        print("[WARN] Could not get USD stage for cube spawning")
    
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
            # Try XR triggers
            if hasattr(input_device, "get_gripper_targets"):
                lt, rt = input_device.get_gripper_targets()
                # Debug print every 120 steps
                if step_count % 120 == 0 and (lt is not None or rt is not None):
                    print(f"[DEBUG] Triggers: L={lt}, R={rt}")
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
            
            # Check for A button press (VR) or C key (keyboard) to spawn cube
            spawn_requested = False
            
            # VR: Check A button on right controller
            if hasattr(input_device, "get_button_states"):
                button_states = input_device.get_button_states()
                a_pressed = button_states.get('right_a', False)
                
                # Edge detection: spawn only on button press (not hold)
                if a_pressed and not prev_a_button:
                    spawn_requested = True
                prev_a_button = a_pressed
            
            # Keyboard: Check C key flag
            if hasattr(input_device, "spawn_cube_requested") and input_device.spawn_cube_requested:
                spawn_requested = True
                input_device.spawn_cube_requested = False  # Reset flag
            
            # Spawn cube if requested
            if spawn_requested and stage is not None:
                # Spawn cube in front of robot arms (center, slightly forward and up)
                spawn_x = 0.4  # Forward
                spawn_y = 0.0  # Center
                spawn_z = 0.6  # Height to drop from
                spawn_cube(stage, position=(spawn_x, spawn_y, spawn_z))
            
            # ===== FULL POSE IK =====
            # Convert poses to tensors (pose format: x, y, z, qw, qx, qy, qz)
            left_target_pos = torch.tensor(left_pose[:3], dtype=torch.float32, device=sim_device).unsqueeze(0)
            left_target_quat = torch.tensor(left_pose[3:7], dtype=torch.float32, device=sim_device).unsqueeze(0)
            right_target_pos = torch.tensor(right_pose[:3], dtype=torch.float32, device=sim_device).unsqueeze(0)
            right_target_quat = torch.tensor(right_pose[3:7], dtype=torch.float32, device=sim_device).unsqueeze(0)
            
            # Get current end-effector poses in robot base frame
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
            
            # Get Jacobians for position IK (only joints 1-4)
            jacobians = robot.root_physx_view.get_jacobians()
            
            # Left arm Jacobian (only for joints 1-4)
            left_jacobian_w = jacobians[:, left_jacobi_body_idx, :, left_jacobi_joint_ids]
            base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
            left_jacobian_b = left_jacobian_w.clone()
            left_jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, left_jacobian_w[:, :3, :])
            left_jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, left_jacobian_w[:, 3:, :])
            
            # Right arm Jacobian (only for joints 1-4)
            right_jacobian_w = jacobians[:, right_jacobi_body_idx, :, right_jacobi_joint_ids]
            right_jacobian_b = right_jacobian_w.clone()
            right_jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, right_jacobian_w[:, :3, :])
            right_jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, right_jacobian_w[:, 3:, :])
            
            # Get current joint positions for position IK (joints 1-4 only)
            left_pos_joint_pos = robot.data.joint_pos[:, left_pos_joint_ids]
            right_pos_joint_pos = robot.data.joint_pos[:, right_pos_joint_ids]
            
            # ===== POSITION IK (joints 1-4) =====
            # Set position target (position-only IK needs ee_quat for reference)
            left_ik_controller.set_command(left_target_pos, ee_quat=left_ee_quat_b)
            right_ik_controller.set_command(right_target_pos, ee_quat=right_ee_quat_b)
            
            # Compute position IK for joints 1-4
            left_pos_des = left_ik_controller.compute(
                left_ee_pos_b, left_ee_quat_b, left_jacobian_b, left_pos_joint_pos
            )
            right_pos_des = right_ik_controller.compute(
                right_ee_pos_b, right_ee_quat_b, right_jacobian_b, right_pos_joint_pos
            )
            
            # ===== WRIST DIRECT CONTROL (joints 5-7) =====
            # Joint 5: Z-axis (forearm twist/yaw)
            # Joint 6: X-axis (wrist flex/roll)
            # Joint 7: Y-axis (wrist deviation/pitch)
            
            # Extract Euler angles from VR target quaternion
            left_q = left_target_quat[0].cpu().numpy()
            right_q = right_target_quat[0].cpu().numpy()
            
            def extract_euler(q):
                """Extract roll (X), pitch (Y), yaw (Z) from quaternion (w,x,y,z)."""
                w, x, y, z = q[0], q[1], q[2], q[3]
                # Roll (X-axis)
                sinr_cosp = 2.0 * (w * x + y * z)
                cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
                roll = np.arctan2(sinr_cosp, cosr_cosp)
                # Pitch (Y-axis)
                sinp = 2.0 * (w * y - z * x)
                pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
                # Yaw (Z-axis)
                siny_cosp = 2.0 * (w * z + x * y)
                cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
                yaw = np.arctan2(siny_cosp, cosy_cosp)
                return roll, pitch, yaw
            
            left_roll, left_pitch, left_yaw = extract_euler(left_q)
            right_roll, right_pitch, right_yaw = extract_euler(right_q)
            
            # Track initial Euler offsets (set on first frame or reset)
            if not hasattr(run_teleop, '_init_left_euler'):
                run_teleop._init_left_euler = (left_roll, left_pitch, left_yaw)
                run_teleop._init_right_euler = (right_roll, right_pitch, right_yaw)
            
            # Compute deltas from initial orientation
            left_j5 = left_yaw - run_teleop._init_left_euler[2]  # Z-axis
            left_j6 = left_roll - run_teleop._init_left_euler[0]  # X-axis
            left_j7 = left_pitch - run_teleop._init_left_euler[1]  # Y-axis
            
            # Right arm is mirrored, negate rotations
            right_j5 = -(right_yaw - run_teleop._init_right_euler[2])
            right_j6 = -(right_roll - run_teleop._init_right_euler[0])
            right_j7 = -(right_pitch - run_teleop._init_right_euler[1])
            
            # Wrap to [-π, π]
            def wrap_angle(a):
                return np.arctan2(np.sin(a), np.cos(a))
            left_j5, left_j6, left_j7 = wrap_angle(left_j5), wrap_angle(left_j6), wrap_angle(left_j7)
            right_j5, right_j6, right_j7 = wrap_angle(right_j5), wrap_angle(right_j6), wrap_angle(right_j7)
            
            # Debug print occasionally
            if step_count % 60 == 0:
                print(f"[WRIST] L: j5={np.degrees(left_j5):.1f} j6={np.degrees(left_j6):.1f} j7={np.degrees(left_j7):.1f} deg")
                print(f"        R: j5={np.degrees(right_j5):.1f} j6={np.degrees(right_j6):.1f} j7={np.degrees(right_j7):.1f} deg")
            
            # Apply to joints 5, 6, 7
            left_wrist_des = torch.tensor([[left_j5, left_j6, left_j7]], dtype=torch.float32, device=sim_device)
            right_wrist_des = torch.tensor([[right_j5, right_j6, right_j7]], dtype=torch.float32, device=sim_device)
            
            # Apply joint position targets
            robot.set_joint_position_target(left_pos_des, joint_ids=left_pos_joint_ids)
            robot.set_joint_position_target(left_wrist_des, joint_ids=left_wrist_joint_ids)
            robot.set_joint_position_target(right_pos_des, joint_ids=right_pos_joint_ids)
            robot.set_joint_position_target(right_wrist_des, joint_ids=right_wrist_joint_ids)
            
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
