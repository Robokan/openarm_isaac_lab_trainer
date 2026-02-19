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
parser.add_argument("--task", type=str, default="Isaac-Reach-OpenArm-Bi-Teleop-v0", help="Task name")
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
parser.add_argument("--script", type=str, default=None,
                    help="Path to YAML script file for automated command sequences")

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
import yaml
import random
import json
from concurrent.futures import ThreadPoolExecutor

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
        self.markers_visible = False  # Toggle for marker visibility (hidden by default, M to show)
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
        print("Position (active hand, world coords):")
        print("  W/S: Forward/Backward (X)")
        print("  A/D: Left/Right (Y)")  
        print("  Q/E: Up/Down (Z)")
        print("")
        print("Orientation (active hand, world coords):")
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
        print("  P: Print current pose of active hand")
        print("  M: Toggle marker visibility")
        print("  R: Reset poses to default")
        print("  Y: Start LeRobot recording")
        print("  T: Stop LeRobot recording")
        print("  Ctrl+C: Quit")
        print("="*60 + "\n")
        
        self.spawn_cube_requested = False  # Flag for cube spawning
        self.reset_requested = False  # Flag for reset + script restart
        self.print_pose_requested = False  # Flag for printing current pose
        self.start_recording_requested = False  # Flag for starting LeRobot recording
        self.stop_recording_requested = False  # Flag for stopping LeRobot recording
    
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
                self.reset_requested = True
                print("[Keyboard] Reset requested")
                return False
            elif key == self._carb_input.KeyboardInput.M:
                self.markers_visible = not self.markers_visible
                print(f"[Keyboard] Markers {'visible' if self.markers_visible else 'hidden'}")
                return False
            elif key == self._carb_input.KeyboardInput.C:
                self.spawn_cube_requested = True
                return False
            elif key == self._carb_input.KeyboardInput.P:
                self.print_pose_requested = True
                return False
            elif key == self._carb_input.KeyboardInput.Y:
                self.start_recording_requested = True
                print("[Keyboard] Start recording requested")
                return False
            elif key == self._carb_input.KeyboardInput.T:
                self.stop_recording_requested = True
                print("[Keyboard] Stop recording requested")
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
        
        # Orientation updates (euler angles: roll, pitch, yaw)
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
        self.markers_visible = False  # Hidden by default, M to show
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
            elif event.input == carb.input.KeyboardInput.M:
                self.markers_visible = not self.markers_visible
                self._log(f"[XRDevice] Markers {'visible' if self.markers_visible else 'hidden'}")
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
                        self.left_pose[3:7] = _map_xr_quat(quat, hand="left")
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
                        self.right_pose[3:7] = _map_xr_quat(quat, hand="right")
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

    def get_thumbstick_values(self):
        """Get thumbstick X/Y values from both controllers.
        
        Returns dict with thumbstick values in range [-1, 1]:
        {'left_x': float, 'left_y': float, 'right_x': float, 'right_y': float}
        """
        values = {'left_x': 0.0, 'left_y': 0.0, 'right_x': 0.0, 'right_y': 0.0}
        
        if self._xr_core is None:
            return values
        
        try:
            from omni.kit.xr.core import XRCore
            xr = XRCore.get_singleton()
            if xr is None:
                return values
            
            if hasattr(xr, 'get_input_device'):
                left_dev = xr.get_input_device("/user/hand/left")
                right_dev = xr.get_input_device("/user/hand/right")
                
                # Debug: print available gestures once
                if not hasattr(self, '_debug_thumbstick_printed'):
                    self._debug_thumbstick_printed = True
                    if left_dev and hasattr(left_dev, 'has_input_gesture'):
                        for gesture in ['thumbstick', 'trackpad', 'joystick', 'primary2d']:
                            for component in ['x', 'y', 'click', 'touch']:
                                if left_dev.has_input_gesture(gesture, component):
                                    print(f"[DEBUG] Left has: {gesture}/{component}")
                
                # Try various gesture names for thumbstick
                for dev, prefix in [(left_dev, 'left'), (right_dev, 'right')]:
                    if dev and hasattr(dev, 'has_input_gesture'):
                        # Try common OpenXR thumbstick gesture names
                        for gesture_name in ['thumbstick', 'primary2d', 'trackpad', 'joystick']:
                            if dev.has_input_gesture(gesture_name, 'x'):
                                values[f'{prefix}_x'] = float(dev.get_input_gesture_value(gesture_name, 'x'))
                            if dev.has_input_gesture(gesture_name, 'y'):
                                values[f'{prefix}_y'] = float(dev.get_input_gesture_value(gesture_name, 'y'))
                            # Break if we found values
                            if values[f'{prefix}_x'] != 0.0 or values[f'{prefix}_y'] != 0.0:
                                break
        except Exception as e:
            pass
        
        return values

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


def _map_xr_quat(quat, hand="left"):
    """Map OpenXR orientation into robot space using the same axis mapping.
    
    Args:
        quat: Input quaternion from XR controller
        hand: "left" or "right" - applies different Z rotations per hand
    """
    try:
        m = _get_xr_map_matrix()
        r = _rot_matrix_from_quat(quat)
        r_mapped = m @ r @ m.T
        
        # Optional rotation corrections
        # X rotation: -180 flips gripper, -135 tilts 45° down toward table
        rot_x_deg = float(os.environ.get("XR_GRIPPER_ROT_X_DEG", "-135"))
        rot_y_deg = float(os.environ.get("XR_GRIPPER_ROT_Y_DEG", "0"))
        
        # Per-hand Z rotation to align VR controllers with robot wrists
        if hand == "left":
            rot_z_deg = -90.0  # Left wrist: -90
        else:
            rot_z_deg = -90.0  # Right wrist: -90 (confirmed working)
        
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
        
        if rot_z_deg != 0:
            rot_z = math.radians(rot_z_deg)
            cz = math.cos(rot_z)
            sz = math.sin(rot_z)
            r_fix_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            r_mapped = r_mapped @ r_fix_z
        
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


# =============================================================================
# YAML Script Executor
# =============================================================================

class ScriptExecutor:
    """Executes a YAML-based command script to automate robot actions.
    
    All positions and orientations are in world (global) coordinates.
    
    Commands:
        spawn_cube:      {name: str, position: [x,y,z] | random_area: {x:[lo,hi], y:[lo,hi], z:[lo,hi]},
                          size: float, color: [r,g,b]}
        move_to:         {arm: left|right, to: <name> | position: [x,y,z], above: float,
                          rotation: {roll, pitch, yaw}, duration: float}
        rotate:          {arm: left|right, roll: float, pitch: float, yaw: float, duration: float}
        close_gripper:   {arm: left|right, force: float (N, default 5.0), duration: float (s, default 0.5)}
        open_gripper:    {arm: left|right}
        wait:            float (seconds)
        wait_until_reached: {arm: left|right|both}
        print:           str
        parallel:        [list of sub-commands]
    """
    
    POSITION_TOLERANCE = 0.02  # 2cm tolerance for "reached" (matches orientation freeze threshold)
    GRIPPER_FORCE_THRESHOLD = 5.0  # Newtons
    GRIPPER_CLOSE_STEP = 0.002  # How much to close per frame
    GRIPPER_OPEN_POS = 0.044
    GRIPPER_Z_OFFSET = -0.06  # Distance from hand body to fingertip grasp point (m)
    
    def __init__(self, yaml_path, stage):
        self.yaml_path = yaml_path
        with open(yaml_path, "r") as f:
            self.commands = yaml.safe_load(f)
        if not isinstance(self.commands, list):
            raise ValueError(f"YAML script must be a list of commands, got {type(self.commands)}")
        
        self.stage = stage
        self.cmd_index = 0
        self.cmd_state = {}  # Per-command state (start_time, etc.)
        self.finished = False
        
        # Named object registry: name -> [x, y, z] position
        self.object_registry = {}
        
        # Output targets (read by the teleop loop each frame)
        self.left_target_pos = None   # [x, y, z] or None (world coords)
        self.right_target_pos = None
        self.left_target_quat = None  # [w, x, y, z] or None (world coords)
        self.right_target_quat = None
        self.left_gripper_target = None   # float 0..1 (0=open, 1=closed) or None
        self.right_gripper_target = None
        self.spawn_request = None  # (position, size, color, name) or None
        
        # Force-gripper state
        self.left_gripper_closing = False
        self.right_gripper_closing = False
        self.left_gripper_pos = 0.0  # 0=open, 1=fully closed
        self.right_gripper_pos = 0.0
        
        # Motion interpolation state
        self._left_start_pos = None
        self._left_end_pos = None
        self._right_start_pos = None
        self._right_end_pos = None
        self._left_start_quat = None
        self._left_end_quat = None
        self._right_start_quat = None
        self._right_end_quat = None
        
        print(f"[Script] Loaded {len(self.commands)} commands from {yaml_path}")
    
    def reset(self):
        """Reset the script executor to the beginning, reloading the YAML file from disk."""
        # Reload commands from file to pick up any edits
        with open(self.yaml_path, "r") as f:
            self.commands = yaml.safe_load(f)
        if not isinstance(self.commands, list):
            print(f"[Script] WARNING: YAML reload failed, got {type(self.commands)}")
            self.commands = []
        print(f"[Script] Reloaded {len(self.commands)} commands from {self.yaml_path}")
        
        self.cmd_index = 0
        self.cmd_state = {}
        self.finished = False
        self.object_registry.clear()
        
        self.left_target_pos = None
        self.right_target_pos = None
        self.left_target_quat = None
        self.right_target_quat = None
        self.left_gripper_target = None
        self.right_gripper_target = None
        self.spawn_request = None
        
        self.left_gripper_closing = False
        self.right_gripper_closing = False
        self.left_gripper_pos = 0.0
        self.right_gripper_pos = 0.0
        
        self._left_start_pos = None
        self._left_end_pos = None
        self._right_start_pos = None
        self._right_end_pos = None
        self._left_start_quat = None
        self._left_end_quat = None
        self._right_start_quat = None
        self._right_end_quat = None
        
        print("[Script] Reset - restarting from beginning")
    
    def _smoothstep(self, t):
        """Ease-in interpolation curve -- smooth start, linear finish (no crawl at end)."""
        t = np.clip(t, 0.0, 1.0)
        # Quadratic ease-in: starts slow, reaches full speed, no deceleration
        # Blend: 70% linear + 30% ease-in for a gentle start without a slow end
        ease_in = t * t
        return 0.3 * ease_in + 0.7 * t
    
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
    
    def _slerp(self, q0, q1, t):
        """Spherical linear interpolation between two quaternions (w,x,y,z)."""
        q0 = np.array(q0, dtype=np.float64)
        q1 = np.array(q1, dtype=np.float64)
        dot = np.dot(q0, q1)
        if dot < 0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        result = s0 * q0 + s1 * q1
        return result / np.linalg.norm(result)
    
    def _point_at_quat(self, from_pos, to_pos, current_quat=None):
        """Compute quaternion that orients gripper to point at a target position.
        
        Based on empirical measurement: when gripper points at object, it has
        roll≈180° (π), pitch based on vertical angle, yaw based on horizontal angle.
        
        Returns quaternion (w, x, y, z).
        """
        from_pos = np.array(from_pos, dtype=np.float64)
        to_pos = np.array(to_pos, dtype=np.float64)
        
        # Direction from gripper to target
        direction = to_pos - from_pos
        horizontal_dist = np.sqrt(direction[0]**2 + direction[1]**2)
        
        # Cache yaw when we have good horizontal distance
        if horizontal_dist > 0.02:
            yaw = np.arctan2(direction[1], direction[0])
            self._last_point_at_yaw = yaw
        else:
            # When very close horizontally, use cached yaw to prevent flip
            yaw = getattr(self, '_last_point_at_yaw', 0.0)
        
        # Pitch: vertical angle - negative means tilting down
        # When object is at same height, pitch ≈ -90° to point gripper forward-down
        # When object is below, pitch more negative
        total_dist = np.linalg.norm(direction)
        if total_dist > 0.01:
            pitch = np.arctan2(direction[2], horizontal_dist) - np.pi/2
        else:
            # Very close - use a sensible default (pointing down)
            pitch = -np.pi/2
        
        # Roll: gripper needs ~180° roll for proper finger orientation
        roll = np.pi
        
        # Convert euler (roll, pitch, yaw) to quaternion
        # Using ZYX convention (yaw first, then pitch, then roll)
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
        
        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)
    
    def _resolve_position(self, params):
        """Resolve target position from command params (named object or explicit coords).
        
        When targeting a named object ('to'), applies GRIPPER_Z_OFFSET so that
        above: 0.0 places the gripper fingers at the object center.
        Explicit 'position' values are used as-is (no offset).
        """
        if "to" in params:
            name = params["to"]
            if name not in self.object_registry:
                print(f"[Script] WARNING: Unknown object '{name}', skipping")
                return None
            pos = list(self.object_registry[name])
            # Apply gripper offset so fingers are at the object, not the hand body
            pos[2] += self.GRIPPER_Z_OFFSET
            if "above" in params:
                pos[2] += params["above"]
            return pos
        elif "position" in params:
            pos = list(params["position"])
            if "above" in params:
                pos[2] += params["above"]
            return pos
        return None
    
    def step(self, sim_time, left_ee_pos=None, right_ee_pos=None,
             left_contact_force=0.0, right_contact_force=0.0,
             current_left_pos=None, current_right_pos=None,
             current_left_quat=None, current_right_quat=None):
        """Advance the script by one frame. Returns True if script is still running."""
        if self.finished or self.cmd_index >= len(self.commands):
            self.finished = True
            return False
        
        # Handle force-gripper closing each frame
        # With PD-controlled gripper (set_joint_position_target), we keep the target
        # past the contact point so the PD controller maintains grip force.
        # Contact detection just marks the command as "done" (grip achieved).
        
        if self.left_gripper_closing:
            threshold = getattr(self, '_left_gripper_force', self.GRIPPER_FORCE_THRESHOLD)
            speed = getattr(self, '_left_gripper_speed', self.GRIPPER_CLOSE_STEP)
            if left_contact_force > threshold:
                self.left_gripper_closing = False
                # Keep gripper target where it is (or fully closed) - PD controller
                # will maintain grip force against the contact
                self.left_gripper_target = self.left_gripper_pos
                print(f"[Script] Left gripper contact detected ({left_contact_force:.1f}N >= {threshold:.1f}N), holding at {self.left_gripper_pos:.3f}")
            else:
                self.left_gripper_pos = min(1.0, self.left_gripper_pos + speed)
                self.left_gripper_target = self.left_gripper_pos
        
        if self.right_gripper_closing:
            threshold = getattr(self, '_right_gripper_force', self.GRIPPER_FORCE_THRESHOLD)
            speed = getattr(self, '_right_gripper_speed', self.GRIPPER_CLOSE_STEP)
            if right_contact_force > threshold:
                self.right_gripper_closing = False
                self.right_gripper_target = self.right_gripper_pos
                print(f"[Script] Right gripper contact detected ({right_contact_force:.1f}N >= {threshold:.1f}N), holding at {self.right_gripper_pos:.3f}")
            else:
                self.right_gripper_pos = min(1.0, self.right_gripper_pos + speed)
                self.right_gripper_target = self.right_gripper_pos
        
        cmd_entry = self.commands[self.cmd_index]
        if not isinstance(cmd_entry, dict) or len(cmd_entry) != 1:
            print(f"[Script] WARNING: Malformed command at index {self.cmd_index}: {cmd_entry}")
            self.cmd_index += 1
            return True
        
        cmd_name = list(cmd_entry.keys())[0]
        params = cmd_entry[cmd_name]
        
        # Initialize command state on first call
        if "started" not in self.cmd_state:
            self.cmd_state = {"started": True, "start_time": sim_time}
            self._on_command_start(cmd_name, params, sim_time,
                                   current_left_pos, current_right_pos,
                                   current_left_quat, current_right_quat)
        
        # Check if command is complete
        done = self._check_command(cmd_name, params, sim_time,
                                    left_ee_pos, right_ee_pos,
                                    left_contact_force, right_contact_force)
        
        if done:
            self.cmd_state = {}
            self.cmd_index += 1
            if self.cmd_index >= len(self.commands):
                self.finished = True
                print("[Script] All commands completed!")
        
        return True
    
    def _on_command_start(self, cmd_name, params, sim_time,
                          current_left_pos, current_right_pos,
                          current_left_quat=None, current_right_quat=None):
        """Called once when a new command starts."""
        if cmd_name == "print":
            print(f"[Script] {params}")
        
        elif cmd_name == "spawn_cube":
            name = params.get("name", f"cube_{self.cmd_index}")
            size = params.get("size", 0.025)
            color = params.get("color", [0.2, 0.6, 1.0])
            
            if "random_area" in params:
                area = params["random_area"]
                x = random.uniform(area["x"][0], area["x"][1])
                y = random.uniform(area["y"][0], area["y"][1])
                z = random.uniform(area["z"][0], area["z"][1])
                position = [x, y, z]
            elif "position" in params:
                position = list(params["position"])
            else:
                position = [0.4, 0.0, 0.5]
            
            self.object_registry[name] = position
            self.spawn_request = (position, size, color, name)
            print(f"[Script] Registered object '{name}' at ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
        
        elif cmd_name == "move_to":
            arm = params.get("arm", "left")
            target_pos = self._resolve_position(params)
            if target_pos is None:
                print(f"[Script] ERROR: move_to failed - could not resolve position for {params}")
                print(f"[Script]   Known objects: {list(self.object_registry.keys())}")
            if target_pos is not None:
                duration = params.get("duration", 2.0)
                self.cmd_state["duration"] = duration
                self.cmd_state["arm"] = arm
                self.cmd_state["target_pos"] = target_pos
                
                # Optional rotation for EE orientation (roll, pitch, yaw -> quaternion)
                # Empty rotation dict {} means "keep current orientation"
                # When targeting a named object ('to:') with no rotation, auto-compute
                # an approach orientation that points the gripper at the object.
                rot = params.get("rotation", None)
                has_explicit_rotation = rot is not None and len(rot) > 0
                auto_orient = "to" in params and not has_explicit_rotation
                target_quat = None
                
                if has_explicit_rotation:
                    r = rot.get("roll", 0.0)
                    p = rot.get("pitch", 0.0)
                    y = rot.get("yaw", 0.0)
                    target_quat = self._euler_to_quat(r, p, y)
                elif auto_orient:
                    # Auto-orient: compute quaternion to point gripper at the object
                    # Use target_pos (where gripper will be) not start_pos
                    obj_name = params["to"]
                    if obj_name in self.object_registry:
                        obj_pos = self.object_registry[obj_name]
                        # Point from where we'll END UP toward the object
                        target_quat = self._point_at_quat(target_pos, obj_pos)
                
                has_rotation = has_explicit_rotation or (auto_orient and target_quat is not None)
                self.cmd_state["has_rotation"] = has_rotation
                self.cmd_state["auto_orient"] = auto_orient
                if "to" in params:
                    self.cmd_state["obj_name"] = params["to"]
                
                if arm == "left":
                    self._left_start_pos = current_left_pos
                    self._left_end_pos = target_pos
                    self.left_target_pos = list(current_left_pos) if current_left_pos is not None else target_pos
                    # Reset cached auto-orient quat for new command
                    self._left_auto_orient_quat = None
                    if has_rotation or auto_orient:
                        self._left_start_quat = np.array(self.left_target_quat) if self.left_target_quat is not None else (current_left_quat if current_left_quat is not None else np.array([1.0, 0.0, 0.0, 0.0]))
                        self._left_end_quat = target_quat
                        self.left_target_quat = list(self._left_start_quat)
                else:
                    self._right_start_pos = current_right_pos
                    self._right_end_pos = target_pos
                    self.right_target_pos = list(current_right_pos) if current_right_pos is not None else target_pos
                    # Reset cached auto-orient quat for new command
                    self._right_auto_orient_quat = None
                    if has_rotation or auto_orient:
                        self._right_start_quat = np.array(self.right_target_quat) if self.right_target_quat is not None else (current_right_quat if current_right_quat is not None else np.array([1.0, 0.0, 0.0, 0.0]))
                        self._right_end_quat = target_quat
                        self.right_target_quat = list(self._right_start_quat)
                
                rot_msg = ""
                if has_explicit_rotation and "rotation" in params:
                    rot = params["rotation"]
                    rot_msg = f" rot=({np.degrees(rot.get('roll',0)):.0f}, {np.degrees(rot.get('pitch',0)):.0f}, {np.degrees(rot.get('yaw',0)):.0f})°"
                elif auto_orient and target_quat is not None:
                    rot_msg = " (auto-orient toward object)"
                start_p = current_left_pos if arm == "left" else current_right_pos
                if start_p is not None:
                    print(f"[Script] Moving {arm} arm: ({start_p[0]:.3f}, {start_p[1]:.3f}, {start_p[2]:.3f}) -> ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}){rot_msg} [{duration:.1f}s]")
                else:
                    print(f"[Script] Moving {arm} arm: (unknown start) -> ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}){rot_msg} [{duration:.1f}s]")
        
        elif cmd_name == "rotate":
            arm = params.get("arm", "left")
            r = params.get("roll", 0.0)
            p = params.get("pitch", 0.0)
            y = params.get("yaw", 0.0)
            duration = params.get("duration", 1.0)
            target_quat = self._euler_to_quat(r, p, y)
            
            self.cmd_state["duration"] = duration
            self.cmd_state["arm"] = arm
            
            if arm == "left":
                self._left_start_quat = np.array(self.left_target_quat) if self.left_target_quat is not None else (current_left_quat if current_left_quat is not None else np.array([1.0, 0.0, 0.0, 0.0]))
                self._left_end_quat = target_quat
                self.left_target_quat = list(self._left_start_quat)
            else:
                self._right_start_quat = np.array(self.right_target_quat) if self.right_target_quat is not None else (current_right_quat if current_right_quat is not None else np.array([1.0, 0.0, 0.0, 0.0]))
                self._right_end_quat = target_quat
                self.right_target_quat = list(self._right_start_quat)
            
            print(f"[Script] Rotating {arm} EE to (roll={np.degrees(r):.0f}, pitch={np.degrees(p):.0f}, yaw={np.degrees(y):.0f})° over {duration:.1f}s")
        
        elif cmd_name == "close_gripper":
            arm = params.get("arm", "left") if isinstance(params, dict) else "left"
            force = params.get("force", self.GRIPPER_FORCE_THRESHOLD) if isinstance(params, dict) else self.GRIPPER_FORCE_THRESHOLD
            duration = params.get("duration", 0.5) if isinstance(params, dict) else 0.5
            # Convert duration to per-frame step (assuming ~100Hz sim rate)
            speed = 1.0 / max(duration * 100.0, 1.0)
            if arm == "left":
                self.left_gripper_closing = True
                self.left_gripper_pos = 0.0
                self.left_gripper_target = 0.0
                self._left_gripper_force = force
                self._left_gripper_speed = speed
            else:
                self.right_gripper_closing = True
                self.right_gripper_pos = 0.0
                self.right_gripper_target = 0.0
                self._right_gripper_force = force
                self._right_gripper_speed = speed
            self.cmd_state["arm"] = arm
            print(f"[Script] Closing {arm} gripper (force: {force:.1f}N, duration: {duration:.1f}s)")
        
        elif cmd_name == "open_gripper":
            arm = params.get("arm", "left") if isinstance(params, dict) else "left"
            if arm == "left":
                self.left_gripper_closing = False
                self.left_gripper_pos = 0.0
                self.left_gripper_target = 0.0
            else:
                self.right_gripper_closing = False
                self.right_gripper_pos = 0.0
                self.right_gripper_target = 0.0
            self.cmd_state["arm"] = arm
            print(f"[Script] Opening {arm} gripper")
        
        elif cmd_name == "wait":
            self.cmd_state["wait_duration"] = float(params)
            print(f"[Script] Waiting {params}s")
        
        elif cmd_name == "wait_until_reached":
            arm = params.get("arm", "both") if isinstance(params, dict) else str(params)
            self.cmd_state["arm"] = arm
            target = self.left_target_pos if arm in ("left", "both") else self.right_target_pos
            if target is not None:
                print(f"[Script] Waiting for {arm} arm to reach ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
            else:
                print(f"[Script] Waiting for {arm} arm (no target set!)")
        
        elif cmd_name == "parallel":
            sub_cmds = params if isinstance(params, list) else []
            sub_states = [{"started": True, "start_time": sim_time} for _ in sub_cmds]
            self.cmd_state["sub_cmds"] = sub_cmds
            self.cmd_state["sub_states"] = sub_states
            print(f"[Script] Running {len(sub_cmds)} commands in parallel: {sub_cmds}")
            for i, sub_cmd in enumerate(sub_cmds):
                if isinstance(sub_cmd, dict) and len(sub_cmd) == 1:
                    sub_name = list(sub_cmd.keys())[0]
                    sub_params = sub_cmd[sub_name]
                    saved_state = self.cmd_state
                    self.cmd_state = sub_states[i]
                    self._on_command_start(sub_name, sub_params, sim_time,
                                           current_left_pos, current_right_pos,
                                           current_left_quat, current_right_quat)
                    sub_states[i] = self.cmd_state
                    self.cmd_state = saved_state
    
    def _check_command(self, cmd_name, params, sim_time,
                       left_ee_pos, right_ee_pos,
                       left_contact_force, right_contact_force):
        """Check if the current command is complete. Returns True if done."""
        elapsed = sim_time - self.cmd_state["start_time"]
        
        if cmd_name in ("print", "spawn_cube"):
            return True  # Instant commands
        
        elif cmd_name == "move_to":
            duration = self.cmd_state.get("duration", 2.0)
            arm = self.cmd_state.get("arm", "left")
            has_rotation = self.cmd_state.get("has_rotation", False)
            t = self._smoothstep(elapsed / duration) if duration > 0 else 1.0
            
            auto_orient = self.cmd_state.get("auto_orient", False)
            obj_name = self.cmd_state.get("obj_name", None)
            
            if arm == "left" and self._left_start_pos is not None and self._left_end_pos is not None:
                interp = [
                    self._left_start_pos[i] + t * (self._left_end_pos[i] - self._left_start_pos[i])
                    for i in range(3)
                ]
                self.left_target_pos = interp
                
                # Auto-orient: continuously point at the object, but freeze when very close
                if auto_orient and obj_name and obj_name in self.object_registry:
                    obj_pos = self.object_registry[obj_name]
                    dist_to_obj = np.linalg.norm(np.array(interp) - np.array(obj_pos))
                    # Only update orientation when far enough to compute a stable direction
                    if dist_to_obj > 0.02:  # > 2cm: compute fresh orientation
                        new_quat = self._point_at_quat(interp, obj_pos)
                        self.left_target_quat = list(new_quat)
                        self._left_auto_orient_quat = new_quat  # Cache last good orientation
                    elif hasattr(self, '_left_auto_orient_quat') and self._left_auto_orient_quat is not None:
                        # Close to object: keep the last computed orientation
                        self.left_target_quat = list(self._left_auto_orient_quat)
                elif has_rotation and self._left_start_quat is not None and self._left_end_quat is not None:
                    self.left_target_quat = list(self._slerp(self._left_start_quat, self._left_end_quat, t))
                    
            elif arm == "right" and self._right_start_pos is not None and self._right_end_pos is not None:
                interp = [
                    self._right_start_pos[i] + t * (self._right_end_pos[i] - self._right_start_pos[i])
                    for i in range(3)
                ]
                self.right_target_pos = interp
                
                # Auto-orient: continuously point at the object, but freeze when very close
                if auto_orient and obj_name and obj_name in self.object_registry:
                    obj_pos = self.object_registry[obj_name]
                    dist_to_obj = np.linalg.norm(np.array(interp) - np.array(obj_pos))
                    # Only update orientation when far enough to compute a stable direction
                    if dist_to_obj > 0.02:  # > 2cm: compute fresh orientation
                        new_quat = self._point_at_quat(interp, obj_pos)
                        self.right_target_quat = list(new_quat)
                        self._right_auto_orient_quat = new_quat  # Cache last good orientation
                    elif hasattr(self, '_right_auto_orient_quat') and self._right_auto_orient_quat is not None:
                        # Close to object: keep the last computed orientation
                        self.right_target_quat = list(self._right_auto_orient_quat)
                elif has_rotation and self._right_start_quat is not None and self._right_end_quat is not None:
                    self.right_target_quat = list(self._slerp(self._right_start_quat, self._right_end_quat, t))
            
            # When move completes, lock targets to exact end values
            done = elapsed >= duration
            if done:
                if arm == "left":
                    self.left_target_pos = list(self._left_end_pos)
                    if hasattr(self, '_left_auto_orient_quat') and self._left_auto_orient_quat is not None:
                        self.left_target_quat = list(self._left_auto_orient_quat)
                else:
                    self.right_target_pos = list(self._right_end_pos)
                    if hasattr(self, '_right_auto_orient_quat') and self._right_auto_orient_quat is not None:
                        self.right_target_quat = list(self._right_auto_orient_quat)
            return done
        
        elif cmd_name == "rotate":
            duration = self.cmd_state.get("duration", 1.0)
            arm = self.cmd_state.get("arm", "left")
            t = self._smoothstep(elapsed / duration) if duration > 0 else 1.0
            
            if arm == "left" and self._left_start_quat is not None and self._left_end_quat is not None:
                self.left_target_quat = list(self._slerp(self._left_start_quat, self._left_end_quat, t))
            elif arm == "right" and self._right_start_quat is not None and self._right_end_quat is not None:
                self.right_target_quat = list(self._slerp(self._right_start_quat, self._right_end_quat, t))
            
            return elapsed >= duration
        
        elif cmd_name == "close_gripper":
            arm = self.cmd_state.get("arm", "left")
            if arm == "left":
                done = not self.left_gripper_closing or self.left_gripper_pos >= 1.0
                if done and self.left_gripper_closing:
                    self.left_gripper_closing = False
                    print(f"[Script] Left gripper fully closed (pos={self.left_gripper_pos:.3f})")
                return done
            else:
                done = not self.right_gripper_closing or self.right_gripper_pos >= 1.0
                if done and self.right_gripper_closing:
                    self.right_gripper_closing = False
                    print(f"[Script] Right gripper fully closed (pos={self.right_gripper_pos:.3f})")
                return done
        
        elif cmd_name == "open_gripper":
            return True  # Instant
        
        elif cmd_name == "wait":
            wait_duration = self.cmd_state.get("wait_duration", 1.0)
            return elapsed >= wait_duration
        
        elif cmd_name == "wait_until_reached":
            arm = self.cmd_state.get("arm", "both")
            left_ok = True
            right_ok = True
            
            dist = 0.0
            if arm in ("left", "both") and self.left_target_pos is not None and left_ee_pos is not None:
                dist = np.linalg.norm(np.array(self.left_target_pos) - np.array(left_ee_pos))
                left_ok = dist < self.POSITION_TOLERANCE
                # Print status every 2 seconds
                if int(elapsed * 5) % 10 == 0 and elapsed > 0.1:
                    print(f"[Script] wait: pos=({left_ee_pos[0]:.3f}, {left_ee_pos[1]:.3f}, {left_ee_pos[2]:.3f}) dist={dist:.3f}m")
            if arm in ("right", "both") and self.right_target_pos is not None and right_ee_pos is not None:
                dist = np.linalg.norm(np.array(self.right_target_pos) - np.array(right_ee_pos))
                right_ok = dist < self.POSITION_TOLERANCE
            
            # Also timeout after 10 seconds to prevent hangs
            if elapsed > 10.0:
                target = self.left_target_pos if arm in ("left", "both") else self.right_target_pos
                actual = left_ee_pos if arm in ("left", "both") else right_ee_pos
                print(f"[Script] wait_until_reached TIMEOUT after 10s")
                if actual is not None:
                    print(f"[Script]   current: ({actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f})")
                if target is not None:
                    print(f"[Script]   target:  ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
                print(f"[Script]   dist: {dist:.3f}m (tolerance: {self.POSITION_TOLERANCE}m)")
                return True
            
            if left_ok and right_ok:
                print(f"[Script] {arm} arm reached target (dist={dist:.3f}m)")
            return left_ok and right_ok
        
        elif cmd_name == "parallel":
            sub_cmds = self.cmd_state.get("sub_cmds", [])
            sub_states = self.cmd_state.get("sub_states", [])
            all_done = True
            for i, sub_cmd in enumerate(sub_cmds):
                if not isinstance(sub_cmd, dict) or len(sub_cmd) != 1:
                    continue
                sub_name = list(sub_cmd.keys())[0]
                sub_params = sub_cmd[sub_name]
                # Temporarily swap cmd_state so _check_command sees the sub-state
                saved_state = self.cmd_state
                self.cmd_state = sub_states[i]
                done = self._check_command(sub_name, sub_params, sim_time,
                                           left_ee_pos, right_ee_pos,
                                           left_contact_force, right_contact_force)
                sub_states[i] = self.cmd_state
                self.cmd_state = saved_state
                if not done:
                    all_done = False
            return all_done
        
        return True  # Unknown commands complete immediately
    


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
    if args_cli.script:
        print(f"[INFO] Script mode: {args_cli.script}")
    
    # Run IK teleoperation
    run_teleop(env, args_cli)
    
    env.close()
    print("[INFO] Environment closed")


# Mug USD assets for random spawning
import os as _os
_OPENARM_ROOT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_OPENARM_ASSETS_DIR = _os.path.join(
    _os.path.dirname(_OPENARM_ROOT_DIR),  # scripts/
    "..", "source", "openarm", "openarm", "tasks", "manager_based", "openarm_manipulation"
)
_OPENARM_ASSETS_DIR = _os.path.normpath(_OPENARM_ASSETS_DIR)

MUG_ASSETS = [
    f"{_OPENARM_ASSETS_DIR}/usds/mugs/1.usd",
    f"{_OPENARM_ASSETS_DIR}/usds/mugs/2.usd",
    f"{_OPENARM_ASSETS_DIR}/usds/mugs/3.usd",
    f"{_OPENARM_ASSETS_DIR}/usds/mugs/4.usd",
]

# SimReady fruit assets from Omniverse (have physics baked in)
SIMREADY_ASSETS_URL = "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props"
FRUIT_ASSETS = [
    f"{SIMREADY_ASSETS_URL}/pomegranate01/pomegranate01.usd",
    f"{SIMREADY_ASSETS_URL}/orange_02/orange_02.usd",
    f"{SIMREADY_ASSETS_URL}/lemon_02/lemon_02.usd",
    f"{SIMREADY_ASSETS_URL}/lime01/lime01.usd",
    f"{SIMREADY_ASSETS_URL}/avocado01/avocado01.usd",
]

# All spawnable objects (mugs + fruits)
ALL_SPAWN_ASSETS = MUG_ASSETS + FRUIT_ASSETS


def spawn_object(stage, position=(0.4, 0.0, 0.5), object_counter=[0], scale=0.01):
    """Spawn a random object (mug or fruit) at the given position.
    
    SimReady fruit assets have physics baked in and should work immediately.
    Local mug assets need physics added manually.
    
    Args:
        stage: USD stage
        position: (x, y, z) spawn position in meters
        object_counter: mutable counter for unique naming
        scale: Scale factor (default 0.01 for mugs, 1.0 for fruits)
    
    Returns:
        Path to the spawned object prim
    """
    from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
    
    # Pick a random object (mug or fruit)
    usd_path = random.choice(ALL_SPAWN_ASSETS)
    obj_name = os.path.basename(usd_path).replace('.usd', '')
    is_fruit = usd_path.startswith(SIMREADY_ASSETS_URL)
    
    # Fruits are already correctly scaled, mugs need scaling
    if is_fruit:
        scale = 1.0  # SimReady assets are in meters
    
    obj_type = "fruit" if is_fruit else "mug"
    print(f"[Object] Loading {obj_type} from: {usd_path}", flush=True)
    
    object_counter[0] += 1
    object_path = f"/World/spawned_object_{object_counter[0]}"
    
    # Create xform and add USD reference
    xform = UsdGeom.Xform.Define(stage, object_path)
    prim = xform.GetPrim()
    prim.GetReferences().AddReference(usd_path)
    
    # Set transform
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
    scale_op = xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3d(scale, scale, scale))
    
    # For both fruits and mugs, we need to set up physics properly for dynamic bodies
    # Remove any existing RigidBodyAPI from children (causes hierarchy conflicts)
    for child in prim.GetAllChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            child.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if child.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            child.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
    
    # Add rigid body physics to the root prim
    UsdPhysics.RigidBodyAPI.Apply(prim)
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateEnableGyroscopicForcesAttr().Set(True)
    
    # Add mass
    mass = 0.1 if is_fruit else 0.15  # 100g fruit, 150g mug
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(mass)
    
    # Add collision with convex decomposition for dynamic bodies
    # Find all mesh prims and set collision approximation
    from pxr import UsdGeom as UsdGeomModule, Usd
    
    # Use Usd.PrimRange to traverse ALL descendants (including nested references)
    for descendant in Usd.PrimRange(prim):
        # Check if this prim is a mesh or has collision
        is_mesh = descendant.IsA(UsdGeomModule.Mesh)
        has_collision = descendant.HasAPI(UsdPhysics.CollisionAPI)
        
        if is_mesh or has_collision:
            # Ensure collision API is applied
            if not has_collision:
                UsdPhysics.CollisionAPI.Apply(descendant)
            
            # Apply PhysX collision API and set approximation directly
            physx_col = PhysxSchema.PhysxCollisionAPI.Apply(descendant)
            
            # Remove triangle mesh collision API if present
            if descendant.HasAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI):
                descendant.RemoveAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI)
            
            # Set the approximation attribute directly on the mesh
            # This tells PhysX to use convex decomposition instead of triangle mesh
            mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(descendant)
            mesh_col_api.CreateApproximationAttr().Set("convexDecomposition")
    
    # Try to force physics to recognize the new object
    try:
        import omni.physx
        physx_interface = omni.physx.get_physx_interface()
        # Force a simulation step update to pick up new objects
        physx_interface.force_load_physics_from_usd()
        print(f"[Object] Physics reloaded for new object", flush=True)
    except Exception as e:
        print(f"[Object] Could not reload physics: {e}", flush=True)
    
    print(f"[Object] Spawned {obj_type} '{obj_name}' at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), scale={scale}", flush=True)
    return object_path, usd_path  # Return both path and USD asset path


def spawn_cube(stage, position=(0.4, 0.0, 0.5), size=0.025, color=None, object_counter=[0]):
    """Spawn a physics-enabled cube at the given position.
    
    Args:
        stage: USD stage
        position: (x, y, z) spawn position in meters
        size: cube size in meters (half-extent)
        color: (r, g, b) color values 0-1, or None for random color
        object_counter: mutable counter for unique naming
    
    Returns:
        Path to the spawned cube prim
    """
    from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema
    
    # Random color if not specified
    if color is None:
        color = (random.random(), random.random(), random.random())
    
    object_counter[0] += 1
    cube_path = f"/World/spawned_cube_{object_counter[0]}"
    
    # Create the cube geometry
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
    
    # Add PhysX properties (don't set velocities - GPU physics doesn't allow it)
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateEnableGyroscopicForcesAttr().Set(True)
    
    # Set mass
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(0.05)  # 50 grams
    
    # Try to force physics to recognize the new object
    try:
        import omni.physx
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.force_load_physics_from_usd()
        print(f"[Object] Physics reloaded for new cube", flush=True)
    except Exception as e:
        print(f"[Object] Could not reload physics: {e}", flush=True)
    
    print(f"[Object] Spawned cube at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})", flush=True)
    return cube_path, None  # Return tuple with None for usd_path


def spawn_random_object(stage, position=(0.4, 0.0, 0.5), object_counter=[0]):
    """Spawn either a random mug or a cube at the given position.
    
    Args:
        stage: USD stage
        position: (x, y, z) spawn position in meters
        object_counter: mutable counter for unique naming
    
    Returns:
        Tuple of (prim_path, usd_asset_path) where usd_asset_path is None for cubes
    """
    # 50% chance of mug/fruit, 50% chance of cube
    if random.random() < 0.5:
        return spawn_object(stage, position, object_counter)
    else:
        return spawn_cube(stage, position, object_counter=object_counter)


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
    
    # Print available scene entities (for debugging cameras)
    print(f"[INFO] Scene entities: {list(unwrapped.scene.keys())}")
    
    # ===== FULL 7-DOF POSE IK =====
    print("[INFO] Setting up full 7-DOF pose IK (all joints, targeting hand)...")
    
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.05},  # Lower = more responsive, but less stable near singularities
    )
    left_ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=sim_device)
    right_ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=sim_device)
    
    # Get all 7 joint IDs per arm
    left_arm_joint_ids, left_arm_joint_names = robot.find_joints([
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3", "openarm_left_joint4",
        "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7"
    ])
    right_arm_joint_ids, right_arm_joint_names = robot.find_joints([
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3", "openarm_right_joint4",
        "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7"
    ])
    
    # IK target body: the hand (end-effector after all 7 joints)
    left_body_ids, _ = robot.find_bodies("openarm_left_hand")
    right_body_ids, _ = robot.find_bodies("openarm_right_hand")
    left_body_idx = left_body_ids[0]
    right_body_idx = right_body_ids[0]
    
    # Jacobian indices: for fixed-base robots, body index is offset by 1
    if robot.is_fixed_base:
        left_jacobi_body_idx = left_body_idx - 1
        right_jacobi_body_idx = right_body_idx - 1
        left_jacobi_joint_ids = list(left_arm_joint_ids)
        right_jacobi_joint_ids = list(right_arm_joint_ids)
    else:
        left_jacobi_body_idx = left_body_idx
        right_jacobi_body_idx = right_body_idx
        left_jacobi_joint_ids = [i + 6 for i in left_arm_joint_ids]
        right_jacobi_joint_ids = [i + 6 for i in right_arm_joint_ids]
    
    print(f"[INFO] IK joints (1-7): L={left_arm_joint_names}, R={right_arm_joint_names}")
    print(f"[INFO] IK target body (hand): L={left_body_idx}, R={right_body_idx}")
    print(f"[INFO] Jacobi body idx: L={left_jacobi_body_idx}, R={right_jacobi_body_idx}")
    
    # Get joint limits for clamping IK output
    joint_limits_low = robot.data.soft_joint_pos_limits[0, :, 0]
    joint_limits_high = robot.data.soft_joint_pos_limits[0, :, 1]
    left_limits_low = joint_limits_low[left_arm_joint_ids]
    left_limits_high = joint_limits_high[left_arm_joint_ids]
    right_limits_low = joint_limits_low[right_arm_joint_ids]
    right_limits_high = joint_limits_high[right_arm_joint_ids]
    print(f"[INFO] Left arm joint limits: low={left_limits_low.cpu().numpy()}, high={left_limits_high.cpu().numpy()}")
    
    # Rest pose for null-space bias (bent elbow configuration)
    # This helps escape extended-arm singularities by pulling toward a "comfortable" pose
    # Joint order: j1, j2, j3, j4, j5, j6, j7
    # j3 is typically the elbow - we want it slightly bent (negative = bent inward for most arms)
    rest_pose_left = torch.tensor([[0.0, 0.3, -0.8, 0.5, 0.0, 0.0, 0.0]], device=sim_device)
    rest_pose_right = torch.tensor([[0.0, -0.3, 0.8, 0.5, 0.0, 0.0, 0.0]], device=sim_device)
    rest_pose_gain = 0.02  # How strongly to pull toward rest pose (0 = none, 1 = strong)
    print(f"[INFO] Rest pose bias enabled with gain={rest_pose_gain}")
    
    # Add high-friction physics material to gripper finger links
    try:
        from pxr import UsdShade, UsdPhysics as UsdPhysicsAPI
        stage = omni.usd.get_context().get_stage()
        finger_link_patterns = [
            "/World/envs/env_0/Robot/openarm_left_finger_link_1",
            "/World/envs/env_0/Robot/openarm_left_finger_link_2",
            "/World/envs/env_0/Robot/openarm_right_finger_link_1",
            "/World/envs/env_0/Robot/openarm_right_finger_link_2",
        ]
        for fp in finger_link_patterns:
            finger_prim = stage.GetPrimAtPath(fp)
            if finger_prim.IsValid():
                mat_path = f"{fp}/GripFrictionMaterial"
                UsdShade.Material.Define(stage, mat_path)
                mat_prim = stage.GetPrimAtPath(mat_path)
                mat_api = UsdPhysicsAPI.MaterialAPI.Apply(mat_prim)
                mat_api.CreateStaticFrictionAttr().Set(1.0)
                mat_api.CreateDynamicFrictionAttr().Set(1.0)
                mat_api.CreateRestitutionAttr().Set(0.0)
                mat_binding = UsdShade.MaterialBindingAPI.Apply(finger_prim)
                mat_binding.Bind(UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")
                print(f"[INFO] Added friction material to {fp}")
            else:
                print(f"[WARN] Finger link not found: {fp}")
    except Exception as e:
        print(f"[WARN] Could not add friction to gripper fingers: {e}")
    
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
    prev_markers_visible = False  # Track previous visibility state (hidden by default)
    prev_a_button = False  # Track A button for edge detection
    prev_y_button = False  # Track Y button for recording start
    prev_x_button = False  # Track X button for recording stop
    
    # LeRobot capture state
    lerobot_recording = False
    lerobot_current_episode = None
    spawned_objects_cache = []  # Cache of spawned object prim paths
    lerobot_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vla_teleop_data")
    lerobot_task_text = "bimanual teleoperation"
    
    # Check for existing episodes and set counter to next available
    lerobot_episode_count = 0
    episodes_dir = os.path.join(lerobot_output_dir, "episodes")
    if os.path.exists(episodes_dir):
        existing_episodes = [d for d in os.listdir(episodes_dir) if d.startswith("episode_")]
        if existing_episodes:
            # Extract episode numbers and find the max
            episode_nums = []
            for ep in existing_episodes:
                try:
                    num = int(ep.replace("episode_", ""))
                    episode_nums.append(num)
                except ValueError:
                    pass
            if episode_nums:
                lerobot_episode_count = max(episode_nums) + 1
                print(f"[LeRobot] Found {len(existing_episodes)} existing episodes, starting from episode_{lerobot_episode_count}")
    
    # LeRobot-compatible format - convert using openpi/examples/openarm/convert_to_lerobot.py
    print("[LeRobot] Will save in LeRobot-compatible format (parquet + images)")
    print("[LeRobot] Convert to native format using: openpi/examples/openarm/convert_to_lerobot.py")
    
    # Get USD stage for cube spawning and markers
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    except Exception:
        stage = None
        print("[WARN] Could not get USD stage for cube spawning")
    
    # Create target markers (3-axis arrows showing position + orientation)
    left_marker_path = "/World/ik_target_left"
    right_marker_path = "/World/ik_target_right"
    try:
        from pxr import UsdGeom, Gf
        
        arrow_length = 0.06
        arrow_radius = 0.003
        
        # Each axis: cylinder along Z, rotated to point along X/Y/Z
        # axis_defs: (sub_name, color, rotate_euler_deg)
        axis_defs = [
            ("x_axis", (1.0, 0.0, 0.0), (0, 90, 0)),   # Red = X
            ("y_axis", (0.0, 1.0, 0.0), (-90, 0, 0)),   # Green = Y
            ("z_axis", (0.0, 0.4, 1.0), (0, 0, 0)),     # Blue = Z (up)
        ]
        
        for marker_path in [left_marker_path, right_marker_path]:
            # Parent xform for the whole marker
            parent_xform = UsdGeom.Xform.Define(stage, marker_path)
            pxf = UsdGeom.Xformable(parent_xform.GetPrim())
            pxf.ClearXformOpOrder()
            pxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
            pxf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
            
            for sub_name, color, (rx, ry, rz) in axis_defs:
                cyl_path = f"{marker_path}/{sub_name}"
                cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
                cyl.GetRadiusAttr().Set(arrow_radius)
                cyl.GetHeightAttr().Set(arrow_length)
                cyl.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
                
                cxf = UsdGeom.Xformable(cyl.GetPrim())
                cxf.ClearXformOpOrder()
                # Offset along cylinder's local Z so base is at origin
                cxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, arrow_length / 2.0))
                # Rotate to point along the correct world axis
                cxf.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))
        
        print("[INFO] Created IK target markers (3-axis arrows, left + right)")
    except Exception as e:
        print(f"[WARN] Could not create target markers: {e}")
    
    # Initialize script executor if script file is provided
    script_executor = None
    if args.script and stage is not None:
        script_executor = ScriptExecutor(args.script, stage)
    
    # Get contact sensors if available
    left_contact_sensor = None
    right_contact_sensor = None
    try:
        left_contact_sensor = unwrapped.scene.get("left_contact", None)
        right_contact_sensor = unwrapped.scene.get("right_contact", None)
        if left_contact_sensor:
            print("[INFO] Left contact sensor found")
        if right_contact_sensor:
            print("[INFO] Right contact sensor found")
    except Exception:
        pass
    
    # Sync input device poses to actual robot EE position at startup (world coords)
    # so IK targets match where the robot is (no unwanted movement)
    try:
        unwrapped.sim.step(render=True)
        robot.update(unwrapped.sim.get_physics_dt())
        _left_ee_w = robot.data.body_pos_w[:, left_body_idx]
        _left_eq_w = robot.data.body_quat_w[:, left_body_idx]
        _right_ee_w = robot.data.body_pos_w[:, right_body_idx]
        _right_eq_w = robot.data.body_quat_w[:, right_body_idx]
        if hasattr(input_device, 'left_pose'):
            input_device.left_pose[:3] = _left_ee_w[0].cpu().numpy()
            input_device.left_pose[3:7] = _left_eq_w[0].cpu().numpy()
            input_device.right_pose[:3] = _right_ee_w[0].cpu().numpy()
            input_device.right_pose[3:7] = _right_eq_w[0].cpu().numpy()
            # Sync euler angles from world-frame quaternion
            if hasattr(input_device, 'left_euler'):
                def _quat_to_euler_init(q):
                    w, x, y, z = q[0], q[1], q[2], q[3]
                    roll = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
                    pitch = np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0))
                    yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
                    return np.array([roll, pitch, yaw])
                input_device.left_euler[:] = _quat_to_euler_init(input_device.left_pose[3:7])
                input_device.right_euler[:] = _quat_to_euler_init(input_device.right_pose[3:7])
        print(f"[INFO] Initial EE sync (world): L=({input_device.left_pose[0]:.3f}, {input_device.left_pose[1]:.3f}, {input_device.left_pose[2]:.3f})"
              f" R=({input_device.right_pose[0]:.3f}, {input_device.right_pose[1]:.3f}, {input_device.right_pose[2]:.3f})")
    except Exception as e:
        print(f"[WARN] Could not sync initial EE poses: {e}")
    
    # =====================================================================
    # OBJECT POOL - Using Isaac Lab RigidObject assets from scene config
    # Objects are pre-spawned on floor away from robot, teleported to table on activation
    # =====================================================================
    object_pool = {
        "cubes": [],    # List of {"asset": RigidObject, "active": bool, "idx": int}
        "mugs": [],     # List of {"asset": RigidObject, "active": bool, "idx": int}
    }
    
    # Get pool objects from scene (defined in joint_pos_env_cfg.py)
    print(f"\n[Pool] Loading pool objects from scene...")
    scene_keys = list(unwrapped.scene.keys()) if hasattr(unwrapped.scene, 'keys') else []
    
    # Load cubes (5 total)
    for i in range(5):
        cube_name = f"pool_cube_{i}"
        if cube_name in scene_keys:
            cube_asset = unwrapped.scene[cube_name]
            object_pool["cubes"].append({"asset": cube_asset, "active": False, "idx": i})
    
    # Load mugs (4 total)
    for i in range(4):
        mug_name = f"pool_mug_{i}"
        if mug_name in scene_keys:
            mug_asset = unwrapped.scene[mug_name]
            object_pool["mugs"].append({"asset": mug_asset, "active": False, "idx": i})
    
    print(f"[Pool] Found {len(object_pool['cubes'])} cubes, {len(object_pool['mugs'])} mugs")
    
    def activate_pool_object(pool_type, position):
        """Activate an object from the pool at the given position.
        Returns (prim_path, asset) or (None, None) if pool exhausted."""
        import torch
        
        pool_list = object_pool[pool_type]
        for obj in pool_list:
            if not obj["active"]:
                obj["active"] = True
                asset = obj["asset"]
                
                # Teleport to position using Isaac Lab API
                pos = torch.tensor([[position[0], position[1], position[2]]], device=asset.device)
                quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=asset.device)  # w,x,y,z upright
                vel = torch.zeros((1, 6), device=asset.device)
                
                # Write pose and velocity to simulation (properly syncs with physics)
                asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
                asset.write_root_velocity_to_sim(vel)
                
                # Get actual prim path (not the config regex pattern)
                # For single env, it's env_0
                pool_name = asset.cfg.prim_path.split("/")[-1]  # e.g., "PoolCube_0"
                prim_path = f"/World/envs/env_0/{pool_name}"
                return prim_path, asset
        
        print(f"[Pool] WARNING: {pool_type} pool exhausted!")
        return None, None
    
    def deactivate_pool_object(asset):
        """Return an object to the pool (move back to floor away from robot)."""
        import torch
        
        # Move back to floor away from robot
        pos = torch.tensor([[-2.0, 0.0, 0.03]], device=asset.device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=asset.device)
        vel = torch.zeros((1, 6), device=asset.device)
        
        asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        asset.write_root_velocity_to_sim(vel)
        
        # Mark as inactive
        for pool_type in ["cubes", "mugs"]:
            for obj in object_pool[pool_type]:
                if obj["asset"] is asset:
                    obj["active"] = False
                    return True
        return False
    
    # Track active objects for recording
    active_pool_objects = []  # List of {"asset": RigidObject, "prim_path": str}
    
    # =====================================================================
    
    # VR tracking active flag - wait for user input before tracking starts
    vr_tracking_active = False
    if args.input != "xr":
        vr_tracking_active = True  # Always active for keyboard/gamepad
    else:
        print(f"\n{'='*60}")
        print("[VR] WAITING FOR INPUT - Position yourself in VR")
        print("[VR] Press any button or move joystick to start tracking")
        print(f"{'='*60}\n")
    
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
            
            # Check for A button press (VR) or C key (keyboard) to spawn cube
            spawn_requested = False
            
            # VR: Check A button on right controller
            if hasattr(input_device, "get_button_states"):
                button_states = input_device.get_button_states()
                a_pressed = button_states.get('right_a', False)
                b_pressed = button_states.get('right_b', False)
                y_pressed = button_states.get('left_y', False)
                x_pressed = button_states.get('left_x', False)
                
                # Check for any VR input to activate tracking
                if not vr_tracking_active:
                    # Check buttons
                    any_button = a_pressed or b_pressed or y_pressed or x_pressed
                    # Check triggers
                    any_trigger = left_trigger > 0.1 or right_trigger > 0.1
                    # Check thumbsticks
                    any_thumbstick = False
                    if hasattr(input_device, 'get_thumbstick_values'):
                        ts = input_device.get_thumbstick_values()
                        any_thumbstick = (abs(ts.get('left_x', 0)) > 0.2 or abs(ts.get('left_y', 0)) > 0.2 or
                                         abs(ts.get('right_x', 0)) > 0.2 or abs(ts.get('right_y', 0)) > 0.2)
                    
                    if any_button or any_trigger or any_thumbstick:
                        vr_tracking_active = True
                        # Sync VR controller poses to current robot EE positions
                        # so arms start from controller position without jumping
                        try:
                            _left_ee_w = robot.data.body_pos_w[:, left_body_idx]
                            _left_eq_w = robot.data.body_quat_w[:, left_body_idx]
                            _right_ee_w = robot.data.body_pos_w[:, right_body_idx]
                            _right_eq_w = robot.data.body_quat_w[:, right_body_idx]
                            if hasattr(input_device, 'left_pose'):
                                input_device.left_pose[:3] = _left_ee_w[0].cpu().numpy()
                                input_device.left_pose[3:7] = _left_eq_w[0].cpu().numpy()
                                input_device.right_pose[:3] = _right_ee_w[0].cpu().numpy()
                                input_device.right_pose[3:7] = _right_eq_w[0].cpu().numpy()
                        except Exception:
                            pass
                        print(f"\n{'='*60}")
                        print("[VR] TRACKING ACTIVATED - Arms will now follow controllers")
                        print(f"{'='*60}\n")
                
                # Edge detection: spawn only on button press (not hold)
                if a_pressed and not prev_a_button:
                    spawn_requested = True
                prev_a_button = a_pressed
                
                # Y button (left controller): Start recording
                if y_pressed and not prev_y_button:
                    if not lerobot_recording:
                        lerobot_recording = True
                        os.makedirs(lerobot_output_dir, exist_ok=True)
                        
                        # Initialize episode frame buffer
                        num_joints = robot.data.joint_pos.shape[1]
                        fps = int(1.0 / unwrapped.sim.get_physics_dt())
                        
                        # Capture initial conditions
                        initial_conditions = {
                            # Robot initial state
                            "robot_joint_pos": robot.data.joint_pos[0].cpu().numpy().tolist(),
                            "robot_joint_vel": robot.data.joint_vel[0].cpu().numpy().tolist(),
                            # End-effector positions and orientations
                            "left_ee_pos": robot.data.body_pos_w[:, left_body_idx][0].cpu().numpy().tolist(),
                            "left_ee_quat": robot.data.body_quat_w[:, left_body_idx][0].cpu().numpy().tolist(),
                            "right_ee_pos": robot.data.body_pos_w[:, right_body_idx][0].cpu().numpy().tolist(),
                            "right_ee_quat": robot.data.body_quat_w[:, right_body_idx][0].cpu().numpy().tolist(),
                            # Objects on table
                            "objects": [],
                        }
                        
                        # Find all spawned objects and record their positions
                        if stage is not None:
                            from pxr import UsdGeom, Gf, Sdf
                            for prim in stage.Traverse():
                                prim_path = str(prim.GetPath())
                                if prim_path.startswith("/World/spawned_"):
                                    try:
                                        xformable = UsdGeom.Xformable(prim)
                                        xform_ops = xformable.GetOrderedXformOps()
                                        pos = [0.0, 0.0, 0.0]
                                        rot = [1.0, 0.0, 0.0, 0.0]  # quat w,x,y,z
                                        scale = [1.0, 1.0, 1.0]
                                        
                                        for op in xform_ops:
                                            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                                t = op.Get()
                                                pos = [float(t[0]), float(t[1]), float(t[2])]
                                            elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                                                s = op.Get()
                                                scale = [float(s[0]), float(s[1]), float(s[2])]
                                            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                                                q = op.Get()
                                                rot = [float(q.GetReal()), float(q.GetImaginary()[0]), 
                                                       float(q.GetImaginary()[1]), float(q.GetImaginary()[2])]
                                        
                                        # Detect object type from prim path and get USD reference if any
                                        obj_type = "cube"  # default
                                        usd_path = None
                                        
                                        if "spawned_cube_" in prim_path:
                                            obj_type = "cube"
                                        elif "spawned_object_" in prim_path:
                                            obj_type = "usd_reference"
                                            # Get the USD reference path
                                            refs = prim.GetReferences()
                                            ref_list = refs.GetAddedOrExplicitItems()
                                            if ref_list:
                                                usd_path = str(ref_list[0].assetPath)
                                        
                                        obj_data = {
                                            "prim_path": prim_path,
                                            "type": obj_type,
                                            "position": pos,
                                            "orientation": rot,
                                            "scale": scale,
                                        }
                                        if usd_path:
                                            obj_data["usd_path"] = usd_path
                                        
                                        initial_conditions["objects"].append(obj_data)
                                    except Exception as e:
                                        print(f"[LeRobot] Could not get transform for {prim_path}: {e}")
                        
                        lerobot_current_episode = {
                            "frames": [],
                            "start_time": time.time(),
                            "fps": fps,
                            "num_joints": num_joints,
                            "initial_conditions": initial_conditions,
                            "render_products": {},  # Store render products for reuse
                        }
                        
                        # Find all cameras and create render products ONCE
                        camera_status = []
                        found_cameras = 0
                        try:
                            from pxr import UsdGeom
                            import omni.replicator.core as rep
                            
                            for prim in stage.Traverse():
                                if prim.IsA(UsdGeom.Camera):
                                    cam_path = str(prim.GetPath())
                                    cam_name = prim.GetName()
                                    # Only use env_0 cameras
                                    if "env_0" in cam_path:
                                        try:
                                            render_product = rep.create.render_product(cam_path, (640, 480))
                                            rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
                                            rgb_annot.attach([render_product])
                                            # Map to LeRobot key
                                            if "ego" in cam_name.lower() or "high" in cam_name.lower():
                                                lerobot_current_episode["render_products"]["observation.images.ego"] = rgb_annot
                                            elif "left" in cam_name.lower():
                                                lerobot_current_episode["render_products"]["observation.images.left_wrist"] = rgb_annot
                                            elif "right" in cam_name.lower():
                                                lerobot_current_episode["render_products"]["observation.images.right_wrist"] = rgb_annot
                                            camera_status.append(f"  {cam_name}: {cam_path} [READY]")
                                            found_cameras += 1
                                        except Exception as e:
                                            camera_status.append(f"  {cam_name}: {cam_path} [FAILED: {e}]")
                        except Exception as e:
                            camera_status.append(f"  Error setting up cameras: {e}")
                        
                        if found_cameras == 0:
                            camera_status.append("  No cameras found in stage")
                        
                        print(f"\n{'='*60}")
                        print(f"[LeRobot] RECORDING STARTED - Episode {lerobot_episode_count + 1}")
                        print(f"[LeRobot] Output: {lerobot_output_dir}")
                        print(f"[LeRobot] Task: {lerobot_task_text}")
                        print(f"[LeRobot] Initial objects: {len(initial_conditions['objects'])}")
                        print(f"[LeRobot] Cameras ({found_cameras} active):")
                        for status in camera_status:
                            print(status)
                        print(f"[LeRobot] Press X on left controller to stop recording")
                        print(f"{'='*60}\n")
                    else:
                        print("[LeRobot] Already recording!")
                prev_y_button = y_pressed
                
                # X button (left controller): Stop recording and save
                if x_pressed and not prev_x_button:
                    if lerobot_recording and lerobot_current_episode is not None:
                        lerobot_recording = False
                        num_frames = len(lerobot_current_episode["frames"])
                        
                        if num_frames > 0:
                            # Save episode in LeRobot-compatible format (parquet + images)
                            import pandas as pd
                            from PIL import Image
                            
                            fps = lerobot_current_episode["fps"]
                            ep_dir = os.path.join(lerobot_output_dir, "episodes", f"episode_{lerobot_episode_count}")
                            os.makedirs(ep_dir, exist_ok=True)
                            
                            # Extract tabular data
                            states = np.array([f["observation.state"] for f in lerobot_current_episode["frames"]])
                            actions = np.array([f["action"] for f in lerobot_current_episode["frames"]])
                            timestamps = np.arange(num_frames) / fps
                            
                            # Save tabular data as parquet
                            df = pd.DataFrame({
                                "timestamp": timestamps,
                                "task": [lerobot_task_text] * num_frames,
                            })
                            for i in range(states.shape[1]):
                                df[f"observation.state.{i}"] = states[:, i]
                            for i in range(actions.shape[1]):
                                df[f"action.{i}"] = actions[:, i]
                            
                            df.to_parquet(os.path.join(ep_dir, "data.parquet"))
                            
                            # Save object states per frame (for kinematic replay)
                            objects_per_frame = [f.get("objects_state", []) for f in lerobot_current_episode["frames"]]
                            if any(objects_per_frame):
                                with open(os.path.join(ep_dir, "objects_state.json"), "w") as f:
                                    json.dump(objects_per_frame, f)
                                print(f"[LeRobot] Saved object states for {sum(1 for o in objects_per_frame if o)} frames")
                            
                            # Save images in parallel for speed
                            def save_image(args):
                                img, path = args
                                img.save(path)
                            
                            save_tasks = []
                            for cam_key in ["observation.images.ego", "observation.images.left_wrist", "observation.images.right_wrist"]:
                                cam_name = cam_key.split(".")[-1]
                                cam_dir = os.path.join(ep_dir, cam_name)
                                os.makedirs(cam_dir, exist_ok=True)
                                
                                for idx, frame in enumerate(lerobot_current_episode["frames"]):
                                    if cam_key in frame and frame[cam_key] is not None:
                                        img = frame[cam_key]
                                        path = os.path.join(cam_dir, f"frame_{idx:06d}.png")
                                        save_tasks.append((img, path))
                            
                            with ThreadPoolExecutor(max_workers=8) as executor:
                                list(executor.map(save_image, save_tasks))
                            
                            # Save metadata
                            episode_metadata = {
                                "task": "Isaac-Reach-OpenArm-Bi-Teleop-v0",
                                "task_text": lerobot_task_text,
                                "fps": fps,
                                "num_frames": num_frames,
                                "duration_seconds": num_frames / fps,
                                "num_joints": lerobot_current_episode["num_joints"],
                            }
                            with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
                                json.dump(episode_metadata, f, indent=2)
                            
                            # Save initial conditions (objects, robot state at start)
                            if "initial_conditions" in lerobot_current_episode:
                                init_cond = lerobot_current_episode["initial_conditions"]
                                if "spawn_events" in lerobot_current_episode:
                                    init_cond["spawn_events"] = lerobot_current_episode["spawn_events"]
                                with open(os.path.join(ep_dir, "initial_conditions.json"), "w") as f:
                                    json.dump(init_cond, f, indent=2)
                                num_initial = len(init_cond.get('objects', []))
                                num_spawned = len(init_cond.get('spawn_events', []))
                                print(f"[LeRobot] Saved initial conditions: {num_initial} initial objects, {num_spawned} spawn events")
                            
                            # Also create/update top-level metadata for play script
                            top_metadata_path = os.path.join(lerobot_output_dir, "metadata.json")
                            top_metadata = {
                                "task": "Isaac-Reach-OpenArm-Bi-Teleop-v0",
                                "task_text": lerobot_task_text,
                                "fps": fps,
                                "num_joints": lerobot_current_episode["num_joints"],
                                "robot_type": "openarm_bimanual",
                                "episode_count": lerobot_episode_count + 1,
                            }
                            with open(top_metadata_path, "w") as f:
                                json.dump(top_metadata, f, indent=2)
                            
                            duration = num_frames / fps
                            
                            print(f"\n{'='*60}")
                            print(f"[LeRobot] RECORDING SAVED - Episode {lerobot_episode_count}")
                            print(f"[LeRobot] Frames: {num_frames}, Duration: {duration:.1f}s")
                            print(f"[LeRobot] Saved to: {ep_dir}")
                            print(f"{'='*60}\n")
                            
                            lerobot_episode_count += 1
                        else:
                            print("[LeRobot] No frames recorded, discarding episode")
                        
                        lerobot_current_episode = None
                    else:
                        print("[LeRobot] Not currently recording")
                prev_x_button = x_pressed
            
            # Keyboard: Check C key flag
            if hasattr(input_device, "spawn_cube_requested") and input_device.spawn_cube_requested:
                spawn_requested = True
                input_device.spawn_cube_requested = False  # Reset flag
            
            # Keyboard: Y key to start recording
            if hasattr(input_device, "start_recording_requested") and input_device.start_recording_requested:
                input_device.start_recording_requested = False  # Reset flag
                if not lerobot_recording:
                    lerobot_recording = True
                    os.makedirs(lerobot_output_dir, exist_ok=True)
                    
                    # Initialize episode frame buffer
                    num_joints = robot.data.joint_pos.shape[1]
                    fps = int(1.0 / unwrapped.sim.get_physics_dt())
                    
                    # Capture initial conditions
                    initial_conditions = {
                        "robot_joint_pos": robot.data.joint_pos[0].cpu().numpy().tolist(),
                        "robot_joint_vel": robot.data.joint_vel[0].cpu().numpy().tolist(),
                        "left_ee_pos": robot.data.body_pos_w[:, left_body_idx][0].cpu().numpy().tolist(),
                        "left_ee_quat": robot.data.body_quat_w[:, left_body_idx][0].cpu().numpy().tolist(),
                        "right_ee_pos": robot.data.body_pos_w[:, right_body_idx][0].cpu().numpy().tolist(),
                        "right_ee_quat": robot.data.body_quat_w[:, right_body_idx][0].cpu().numpy().tolist(),
                        "objects": [],
                    }
                    
                    # Find all spawned objects
                    if stage is not None:
                        from pxr import UsdGeom, Gf, Sdf
                        for prim in stage.Traverse():
                            prim_path = str(prim.GetPath())
                            if prim_path.startswith("/World/spawned_"):
                                try:
                                    xformable = UsdGeom.Xformable(prim)
                                    xform_ops = xformable.GetOrderedXformOps()
                                    pos = [0.0, 0.0, 0.0]
                                    rot = [1.0, 0.0, 0.0, 0.0]
                                    scale = [1.0, 1.0, 1.0]
                                    for op in xform_ops:
                                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                            t = op.Get()
                                            pos = [float(t[0]), float(t[1]), float(t[2])]
                                        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                                            s = op.Get()
                                            scale = [float(s[0]), float(s[1]), float(s[2])]
                                        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                                            q = op.Get()
                                            rot = [float(q.GetReal()), float(q.GetImaginary()[0]),
                                                   float(q.GetImaginary()[1]), float(q.GetImaginary()[2])]
                                    
                                    # Detect object type from prim path and get USD reference if any
                                    obj_type = "cube"  # default
                                    usd_path = None
                                    
                                    if "spawned_cube_" in prim_path:
                                        obj_type = "cube"
                                    elif "spawned_object_" in prim_path:
                                        obj_type = "usd_reference"
                                        # Get the USD reference path
                                        refs = prim.GetReferences()
                                        ref_list = refs.GetAddedOrExplicitItems()
                                        if ref_list:
                                            usd_path = str(ref_list[0].assetPath)
                                    
                                    obj_data = {
                                        "prim_path": prim_path,
                                        "type": obj_type,
                                        "position": pos,
                                        "orientation": rot,
                                        "scale": scale,
                                    }
                                    if usd_path:
                                        obj_data["usd_path"] = usd_path
                                    
                                    initial_conditions["objects"].append(obj_data)
                                except Exception:
                                    pass
                    
                    lerobot_current_episode = {
                        "frames": [],
                        "start_time": time.time(),
                        "fps": fps,
                        "num_joints": num_joints,
                        "initial_conditions": initial_conditions,
                        "render_products": {},  # Store render products for reuse
                    }
                    
                    # Find all cameras and create render products ONCE
                    camera_status = []
                    found_cameras = 0
                    try:
                        from pxr import UsdGeom
                        import omni.replicator.core as rep
                        
                        for prim in stage.Traverse():
                            if prim.IsA(UsdGeom.Camera):
                                cam_path = str(prim.GetPath())
                                cam_name = prim.GetName()
                                if "env_0" in cam_path:
                                    try:
                                        render_product = rep.create.render_product(cam_path, (640, 480))
                                        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
                                        rgb_annot.attach([render_product])
                                        # Map to LeRobot key
                                        if "ego" in cam_name.lower() or "high" in cam_name.lower():
                                            lerobot_current_episode["render_products"]["observation.images.ego"] = rgb_annot
                                        elif "left" in cam_name.lower():
                                            lerobot_current_episode["render_products"]["observation.images.left_wrist"] = rgb_annot
                                        elif "right" in cam_name.lower():
                                            lerobot_current_episode["render_products"]["observation.images.right_wrist"] = rgb_annot
                                        camera_status.append(f"  {cam_name}: {cam_path} [READY]")
                                        found_cameras += 1
                                    except Exception as e:
                                        camera_status.append(f"  {cam_name}: {cam_path} [FAILED: {e}]")
                    except Exception as e:
                        camera_status.append(f"  Error setting up cameras: {e}")
                    
                    if found_cameras == 0:
                        camera_status.append("  No cameras found in stage")
                    
                    print(f"\n{'='*60}")
                    print(f"[LeRobot] RECORDING STARTED - Episode {lerobot_episode_count + 1}")
                    print(f"[LeRobot] Output: {lerobot_output_dir}")
                    print(f"[LeRobot] Task: {lerobot_task_text}")
                    print(f"[LeRobot] Initial objects: {len(initial_conditions['objects'])}")
                    print(f"[LeRobot] Cameras ({found_cameras} active):")
                    for status in camera_status:
                        print(status)
                    print(f"[LeRobot] Press T to stop recording")
                    print(f"{'='*60}\n")
                else:
                    print("[LeRobot] Already recording!")
            
            # Keyboard: T key to stop recording
            if hasattr(input_device, "stop_recording_requested") and input_device.stop_recording_requested:
                input_device.stop_recording_requested = False  # Reset flag
                if lerobot_recording and lerobot_current_episode is not None:
                    lerobot_recording = False
                    num_frames = len(lerobot_current_episode["frames"])
                    
                    if num_frames > 0:
                        # Save in LeRobot-compatible format (parquet + images)
                        import pandas as pd
                        from PIL import Image
                        
                        fps = lerobot_current_episode["fps"]
                        ep_dir = os.path.join(lerobot_output_dir, "episodes", f"episode_{lerobot_episode_count}")
                        os.makedirs(ep_dir, exist_ok=True)
                        
                        states = np.array([f["observation.state"] for f in lerobot_current_episode["frames"]])
                        actions = np.array([f["action"] for f in lerobot_current_episode["frames"]])
                        timestamps = np.arange(num_frames) / fps
                        
                        df = pd.DataFrame({
                            "timestamp": timestamps,
                            "episode_index": [lerobot_episode_count] * num_frames,
                            "frame_index": np.arange(num_frames),
                            "task": [lerobot_task_text] * num_frames,
                        })
                        for i in range(states.shape[1]):
                            df[f"observation.state.{i}"] = states[:, i]
                        for i in range(actions.shape[1]):
                            df[f"action.{i}"] = actions[:, i]
                        
                        df.to_parquet(os.path.join(ep_dir, "data.parquet"))
                        
                        # Save object states per frame (for kinematic replay)
                        objects_per_frame = [f.get("objects_state", []) for f in lerobot_current_episode["frames"]]
                        if any(objects_per_frame):
                            with open(os.path.join(ep_dir, "objects_state.json"), "w") as f:
                                json.dump(objects_per_frame, f)
                            print(f"[LeRobot] Saved object states for {sum(1 for o in objects_per_frame if o)} frames")
                        
                        def save_image(args):
                            img, path = args
                            img.save(path)
                        
                        save_tasks = []
                        for cam_key in ["observation.images.ego", "observation.images.left_wrist", "observation.images.right_wrist"]:
                            cam_name = cam_key.split(".")[-1]
                            cam_dir = os.path.join(ep_dir, cam_name)
                            os.makedirs(cam_dir, exist_ok=True)
                            
                            for idx, frame in enumerate(lerobot_current_episode["frames"]):
                                if cam_key in frame and frame[cam_key] is not None:
                                    img = frame[cam_key]
                                    path = os.path.join(cam_dir, f"frame_{idx:06d}.png")
                                    save_tasks.append((img, path))
                        
                        with ThreadPoolExecutor(max_workers=8) as executor:
                            list(executor.map(save_image, save_tasks))
                        
                        episode_metadata = {
                            "task": "Isaac-Reach-OpenArm-Bi-Teleop-v0",
                            "task_text": lerobot_task_text,
                            "fps": fps,
                            "num_frames": num_frames,
                            "duration_seconds": num_frames / fps,
                            "num_joints": lerobot_current_episode["num_joints"],
                        }
                        with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
                            json.dump(episode_metadata, f, indent=2)
                        
                        if "initial_conditions" in lerobot_current_episode:
                            init_cond = lerobot_current_episode["initial_conditions"]
                            if "spawn_events" in lerobot_current_episode:
                                init_cond["spawn_events"] = lerobot_current_episode["spawn_events"]
                            with open(os.path.join(ep_dir, "initial_conditions.json"), "w") as f:
                                json.dump(init_cond, f, indent=2)
                        
                        top_metadata_path = os.path.join(lerobot_output_dir, "metadata.json")
                        top_metadata = {
                            "task": "Isaac-Reach-OpenArm-Bi-Teleop-v0",
                            "task_text": lerobot_task_text,
                            "fps": fps,
                            "num_joints": lerobot_current_episode["num_joints"],
                            "robot_type": "openarm_bimanual",
                            "episode_count": lerobot_episode_count + 1,
                        }
                        with open(top_metadata_path, "w") as f:
                            json.dump(top_metadata, f, indent=2)
                        
                        print(f"\n{'='*60}")
                        print(f"[LeRobot] RECORDING SAVED - Episode {lerobot_episode_count}")
                        print(f"[LeRobot] Frames: {num_frames}")
                        print(f"[LeRobot] Duration: {num_frames / fps:.1f}s")
                        print(f"[LeRobot] Location: {ep_dir}")
                        print(f"{'='*60}\n")
                        
                        lerobot_episode_count += 1
                    else:
                        print("[LeRobot] No frames recorded, discarding episode")
                    
                    lerobot_current_episode = None
                else:
                    print("[LeRobot] Not currently recording")
            
            # Spawn random object from pool if requested
            if spawn_requested:
                print("[Object] Spawn requested from pool...", flush=True)
                # Random position on the table
                spawn_x = random.uniform(0.25, 0.50)  # Forward range on table
                spawn_y = random.uniform(-0.20, 0.20)  # Left/right range on table
                spawn_z = random.uniform(0.45, 0.55)  # Drop height above table
                try:
                    # Randomly pick from cubes or mugs pool, try other if first exhausted
                    pool_types = ["cubes", "mugs"]
                    random.shuffle(pool_types)
                    
                    spawned_path, spawned_asset = None, None
                    for pool_type in pool_types:
                        spawned_path, spawned_asset = activate_pool_object(pool_type, (spawn_x, spawn_y, spawn_z))
                        if spawned_path:
                            break
                    
                    # Track active object for recording (using Isaac Lab RigidObject)
                    if spawned_path and spawned_asset:
                        active_pool_objects.append({"asset": spawned_asset, "prim_path": spawned_path})
                        print(f"[Pool] Activated {pool_type} at ({spawn_x:.2f}, {spawn_y:.2f}, {spawn_z:.2f})", flush=True)
                    else:
                        print(f"[Pool] All pools exhausted!", flush=True)
                    
                    # If recording, add spawn event
                    if lerobot_current_episode is not None and spawned_path and spawned_asset:
                        # Get position from Isaac Lab RigidObject
                        pos = spawned_asset.data.root_pos_w[0].cpu().numpy().tolist()
                        rot_quat = spawned_asset.data.root_quat_w[0].cpu().numpy().tolist()  # w,x,y,z
                        rot = [float(rot_quat[0]), float(rot_quat[1]), float(rot_quat[2]), float(rot_quat[3])]
                        scale = [1.0, 1.0, 1.0]  # Default scale
                            
                        obj_type = f"pool_{pool_type[:-1]}"  # cubes->pool_cube, mugs->pool_mug, fruits->pool_fruit
                        obj_data = {
                            "prim_path": spawned_path,
                            "type": obj_type,
                            "position": pos,
                            "orientation": rot,
                            "scale": scale,
                        }
                        
                        # Add as spawn event with frame number (for timed spawning during playback)
                        current_frame = len(lerobot_current_episode.get("frames", []))
                        spawn_event = {
                            "frame": current_frame,
                            **obj_data
                        }
                        
                        if "spawn_events" not in lerobot_current_episode:
                            lerobot_current_episode["spawn_events"] = []
                        lerobot_current_episode["spawn_events"].append(spawn_event)
                        print(f"[LeRobot] Added spawn event at frame {current_frame}: {spawned_path}")
                except Exception as e:
                    import traceback
                    print(f"[Object] Spawn failed: {e}", flush=True)
                    traceback.print_exc()
            
            # Handle P key: print current pose of active hand (world coords)
            if hasattr(input_device, "print_pose_requested") and input_device.print_pose_requested:
                input_device.print_pose_requested = False
                hand = input_device.active_hand if hasattr(input_device, 'active_hand') else "left"
                ee_idx = left_body_idx if hand == "left" else right_body_idx
                ee_pos_w = robot.data.body_pos_w[:, ee_idx][0].cpu().numpy()
                ee_quat_w = robot.data.body_quat_w[:, ee_idx][0].cpu().numpy()
                w, x, y, z = ee_quat_w[0], ee_quat_w[1], ee_quat_w[2], ee_quat_w[3]
                _roll = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
                _pitch = np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0))
                _yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
                _pose = input_device.left_pose if hand == "left" else input_device.right_pose
                print(f"\n{'='*60}")
                print(f"  {hand.upper()} arm (world coords):")
                print(f"  EE position:   x={ee_pos_w[0]:.4f}, y={ee_pos_w[1]:.4f}, z={ee_pos_w[2]:.4f}")
                print(f"  EE quaternion: w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f}")
                print(f"  EE rotation:   roll={np.degrees(_roll):.1f}°, pitch={np.degrees(_pitch):.1f}°, yaw={np.degrees(_yaw):.1f}°")
                print(f"  Target pos:    x={_pose[0]:.4f}, y={_pose[1]:.4f}, z={_pose[2]:.4f}")
                print(f"{'='*60}\n")
            
            # Handle reset: reset scene, clear cubes, restart script
            if hasattr(input_device, "reset_requested") and input_device.reset_requested:
                input_device.reset_requested = False
                print("\n[Reset] ========== RESETTING SCENE ==========")
                
                # 1. Remove all spawned cubes from the stage
                if stage is not None:
                    removed = 0
                    for prim in list(stage.Traverse()):
                        if prim.GetPath().pathString.startswith("/World/spawned_cube_"):
                            stage.RemovePrim(prim.GetPath())
                            removed += 1
                    if removed > 0:
                        print(f"[Reset] Removed {removed} spawned cube(s)")
                    spawn_cube.__defaults__[3][0] = 0
                
                # 2. Reset robot joints to default positions
                all_joint_pos = robot.data.default_joint_pos.clone()
                all_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
                robot.write_joint_state_to_sim(all_joint_pos, all_joint_vel)
                # Also set joint position TARGETS to defaults so the PD controller
                # doesn't drive the joints back to the old targets
                robot.set_joint_position_target(all_joint_pos)
                robot.write_data_to_sim()
                
                # Open grippers
                if left_gripper_ids:
                    left_open = torch.full((1, len(left_gripper_ids)), gripper_open_pos, device=sim_device)
                    robot.write_joint_position_to_sim(left_open, joint_ids=left_gripper_ids)
                    robot.set_joint_position_target(left_open, joint_ids=left_gripper_ids)
                if right_gripper_ids:
                    right_open = torch.full((1, len(right_gripper_ids)), gripper_open_pos, device=sim_device)
                    robot.write_joint_position_to_sim(right_open, joint_ids=right_gripper_ids)
                    robot.set_joint_position_target(right_open, joint_ids=right_gripper_ids)
                
                # Step sim a few times to let reset settle
                for _ in range(20):
                    unwrapped.sim.step(render=False)
                    robot.update(unwrapped.sim.get_physics_dt())
                unwrapped.sim.step(render=True)
                robot.update(unwrapped.sim.get_physics_dt())
                
                # 3. Read back actual EE positions in world frame and sync input device
                _left_ee_w = robot.data.body_pos_w[:, left_body_idx]
                _left_eq_w = robot.data.body_quat_w[:, left_body_idx]
                _right_ee_w = robot.data.body_pos_w[:, right_body_idx]
                _right_eq_w = robot.data.body_quat_w[:, right_body_idx]
                
                if hasattr(input_device, 'left_pose'):
                    input_device.left_pose[:3] = _left_ee_w[0].cpu().numpy()
                    input_device.left_pose[3:7] = _left_eq_w[0].cpu().numpy()
                    input_device.right_pose[:3] = _right_ee_w[0].cpu().numpy()
                    input_device.right_pose[3:7] = _right_eq_w[0].cpu().numpy()
                    if hasattr(input_device, 'left_euler'):
                        def _quat_to_euler_reset(q):
                            w, x, y, z = q[0], q[1], q[2], q[3]
                            roll = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
                            pitch = np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0))
                            yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
                            return np.array([roll, pitch, yaw])
                        input_device.left_euler[:] = _quat_to_euler_reset(input_device.left_pose[3:7])
                        input_device.right_euler[:] = _quat_to_euler_reset(input_device.right_pose[3:7])
                    input_device.left_gripper = 0.0
                    input_device.right_gripper = 0.0
                
                # 4. Reset IK controllers
                left_ik_controller.reset()
                right_ik_controller.reset()
                
                # 6. Restart script executor (reloads YAML from disk)
                if script_executor is not None:
                    script_executor.reset()
                
                # 7. Reset step counter for script timing
                step_count = 0
                
                print("[Reset] Scene reset complete, script restarting\n")
                continue  # Restart loop so get_poses() reads the fresh synced values
            
            # ===== SCRIPT EXECUTOR =====
            if script_executor is not None and not script_executor.finished:
                # Get current EE positions in world frame for the executor
                _left_ee_w = robot.data.body_pos_w[:, left_body_idx]
                _left_eq_w = robot.data.body_quat_w[:, left_body_idx]
                _right_ee_w = robot.data.body_pos_w[:, right_body_idx]
                _right_eq_w = robot.data.body_quat_w[:, right_body_idx]
                
                _left_ee_np = _left_ee_w[0].cpu().numpy()
                _right_ee_np = _right_ee_w[0].cpu().numpy()
                _left_eq_np = _left_eq_w[0].cpu().numpy()
                _right_eq_np = _right_eq_w[0].cpu().numpy()
                
                # Read contact forces
                _left_force = 0.0
                _right_force = 0.0
                if left_contact_sensor is not None:
                    try:
                        forces = left_contact_sensor.data.net_forces_w
                        _left_force = float(torch.norm(forces, dim=-1).max().cpu())
                    except Exception:
                        pass
                if right_contact_sensor is not None:
                    try:
                        forces = right_contact_sensor.data.net_forces_w
                        _right_force = float(torch.norm(forces, dim=-1).max().cpu())
                    except Exception:
                        pass
                
                sim_time = step_count * 0.01  # Approximate sim time at 100Hz
                script_executor.step(
                    sim_time,
                    left_ee_pos=_left_ee_np,
                    right_ee_pos=_right_ee_np,
                    left_contact_force=_left_force,
                    right_contact_force=_right_force,
                    current_left_pos=_left_ee_np,
                    current_right_pos=_right_ee_np,
                    current_left_quat=_left_eq_np,
                    current_right_quat=_right_eq_np,
                )
                
                # Handle spawn requests from script
                if script_executor.spawn_request is not None:
                    pos, size, color, name = script_executor.spawn_request
                    spawn_cube(stage, position=tuple(pos), size=size, color=tuple(color))
                    script_executor.spawn_request = None
                
                # Override input poses with script targets (world coords)
                if script_executor.left_target_pos is not None:
                    left_pose[:3] = script_executor.left_target_pos
                    if hasattr(input_device, 'left_pose'):
                        input_device.left_pose[:3] = script_executor.left_target_pos
                if script_executor.right_target_pos is not None:
                    right_pose[:3] = script_executor.right_target_pos
                    if hasattr(input_device, 'right_pose'):
                        input_device.right_pose[:3] = script_executor.right_target_pos
                
                if script_executor.left_target_quat is not None:
                    left_pose[3:7] = script_executor.left_target_quat
                    if hasattr(input_device, 'left_pose'):
                        input_device.left_pose[3:7] = script_executor.left_target_quat
                if script_executor.right_target_quat is not None:
                    right_pose[3:7] = script_executor.right_target_quat
                    if hasattr(input_device, 'right_pose'):
                        input_device.right_pose[3:7] = script_executor.right_target_quat
                
                # Override gripper targets from script
                if script_executor.left_gripper_target is not None:
                    left_trigger = script_executor.left_gripper_target
                    if hasattr(input_device, "left_gripper"):
                        input_device.left_gripper = left_trigger
                if script_executor.right_gripper_target is not None:
                    right_trigger = script_executor.right_gripper_target
                    if hasattr(input_device, "right_gripper"):
                        input_device.right_gripper = right_trigger
                
                # When script finishes, sync keyboard euler from current pose quat
                if script_executor.finished and hasattr(input_device, 'left_euler'):
                    def _quat_to_euler(q):
                        w, x, y, z = q[0], q[1], q[2], q[3]
                        roll = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
                        pitch = np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0))
                        yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
                        return roll, pitch, yaw
                    
                    lr, lp, ly = _quat_to_euler(input_device.left_pose[3:7])
                    input_device.left_euler[:] = [lr, lp, ly]
                    rr, rp, ry = _quat_to_euler(input_device.right_pose[3:7])
                    input_device.right_euler[:] = [rr, rp, ry]
                
                # Stop loop when script is done and no manual override
                if script_executor.finished and args.input != "keyboard":
                    print("[Script] Script finished, exiting...")
                    break
            
            # ===== WRITE GRIPPERS =====
            # Only apply gripper targets when tracking is active
            # Use set_joint_position_target (PD controller) instead of write_joint_position_to_sim
            # so that finger contact forces are respected and don't penetrate objects.
            if vr_tracking_active:
                # Left gripper: open (0.044) when trigger=0, closed (0) when trigger=1
                if left_gripper_ids:
                    left_pos = gripper_open_pos * (1.0 - left_trigger)
                    left_targets = torch.full(
                        (1, len(left_gripper_ids)),
                        left_pos,
                        device=sim_device,
                    )
                    robot.set_joint_position_target(left_targets, joint_ids=left_gripper_ids)
                
                # Right gripper: open (0.044) when trigger=0, closed (0) when trigger=1
                if right_gripper_ids:
                    right_pos = gripper_open_pos * (1.0 - right_trigger)
                    right_targets = torch.full(
                        (1, len(right_gripper_ids)),
                        right_pos,
                        device=sim_device,
                    )
                    robot.set_joint_position_target(right_targets, joint_ids=right_gripper_ids)
            
            # ===== FULL 7-DOF POSE IK =====
            # Targets are in world coordinates; transform to base frame for IK
            left_target_pos = torch.tensor(left_pose[:3], dtype=torch.float32, device=sim_device).unsqueeze(0)
            left_target_quat = torch.tensor(left_pose[3:7], dtype=torch.float32, device=sim_device).unsqueeze(0)
            
            # Debug: check for NaN or invalid quaternion
            if step_count % 60 == 0:
                quat_norm = np.linalg.norm(left_pose[3:7])
                if abs(quat_norm - 1.0) > 0.01 or np.isnan(quat_norm):
                    print(f"[WARN] Invalid left quaternion! norm={quat_norm:.3f}, quat={left_pose[3:7]}")
            right_target_pos = torch.tensor(right_pose[:3], dtype=torch.float32, device=sim_device).unsqueeze(0)
            right_target_quat = torch.tensor(right_pose[3:7], dtype=torch.float32, device=sim_device).unsqueeze(0)
            
            # ===== THUMBSTICK WRIST ROTATION OFFSET =====
            # Add up to ±45 degrees of extra wrist rotation based on thumbstick
            # Also track offset quaternions for marker visualization
            left_offset_quat = None
            right_offset_quat = None
            
            if hasattr(input_device, 'get_thumbstick_values'):
                thumbstick = input_device.get_thumbstick_values()
                max_angle_deg = 90.0
                
                # Left controller thumbstick -> left arm wrist rotation
                left_stick_x = thumbstick.get('left_x', 0.0)
                left_stick_y = thumbstick.get('left_y', 0.0)
                
                # Right controller thumbstick -> right arm wrist rotation  
                right_stick_x = thumbstick.get('right_x', 0.0)
                right_stick_y = thumbstick.get('right_y', 0.0)
                
                # Apply rotation offset if thumbstick is moved
                # Y (forward/back) controls pitch, X (left/right) controls roll/twist
                # Both axes inverted per user request
                if abs(left_stick_x) > 0.1 or abs(left_stick_y) > 0.1:
                    pitch_rad = math.radians(-left_stick_y * max_angle_deg)  # Forward/back -> pitch (inverted)
                    roll_rad = math.radians(-left_stick_x * max_angle_deg)   # Left/right -> roll/twist (inverted)
                    left_offset_quat = math_utils.quat_from_euler_xyz(
                        torch.tensor([roll_rad], device=sim_device),   # Roll (X)
                        torch.tensor([pitch_rad], device=sim_device),  # Pitch (Y)
                        torch.tensor([0.0], device=sim_device)         # Yaw (Z)
                    )  # Shape: [1, 4]
                    left_target_quat = math_utils.quat_mul(left_target_quat, left_offset_quat)
                
                if abs(right_stick_x) > 0.1 or abs(right_stick_y) > 0.1:
                    pitch_rad = math.radians(-right_stick_y * max_angle_deg)  # Inverted
                    roll_rad = math.radians(-right_stick_x * max_angle_deg)   # Inverted
                    right_offset_quat = math_utils.quat_from_euler_xyz(
                        torch.tensor([roll_rad], device=sim_device),
                        torch.tensor([pitch_rad], device=sim_device),
                        torch.tensor([0.0], device=sim_device)
                    )  # Shape: [1, 4]
                    right_target_quat = math_utils.quat_mul(right_target_quat, right_offset_quat)
            
            # Update target markers (position + orientation, including thumbstick offset)
            if stage is not None:
                try:
                    from pxr import UsdGeom, Gf
                    markers_vis = input_device.markers_visible if hasattr(input_device, 'markers_visible') else True
                    # Use the modified quaternions (with thumbstick offset) for markers
                    left_marker_quat = left_target_quat[0].cpu().numpy()
                    right_marker_quat = right_target_quat[0].cpu().numpy()
                    for m_path, pos, quat in [
                        (left_marker_path, left_pose[:3], left_marker_quat),
                        (right_marker_path, right_pose[:3], right_marker_quat),
                    ]:
                        m_prim = stage.GetPrimAtPath(m_path)
                        if m_prim.IsValid():
                            xform = UsdGeom.Xformable(m_prim)
                            ops = xform.GetOrderedXformOps()
                            if len(ops) >= 2:
                                ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                                ops[1].Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
                            img = UsdGeom.Imageable(m_prim)
                            if markers_vis:
                                img.MakeVisible()
                            else:
                                img.MakeInvisible()
                except Exception:
                    pass
            
            # Robot base frame
            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            
            # Transform targets from world to base frame
            left_target_pos_b, left_target_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, left_target_pos, left_target_quat
            )
            right_target_pos_b, right_target_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, right_target_pos, right_target_quat
            )
            
            # Current EE poses in base frame
            left_ee_pos_w = robot.data.body_pos_w[:, left_body_idx]
            left_ee_quat_w = robot.data.body_quat_w[:, left_body_idx]
            left_ee_pos_b, left_ee_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, left_ee_pos_w, left_ee_quat_w
            )
            right_ee_pos_w = robot.data.body_pos_w[:, right_body_idx]
            right_ee_quat_w = robot.data.body_quat_w[:, right_body_idx]
            right_ee_pos_b, right_ee_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, right_ee_pos_w, right_ee_quat_w
            )
            
            # Jacobians (full 7 joints for hand body)
            jacobians = robot.root_physx_view.get_jacobians()
            base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
            
            left_jacobian_w = jacobians[:, left_jacobi_body_idx, :, left_jacobi_joint_ids]
            left_jacobian_b = left_jacobian_w.clone()
            left_jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, left_jacobian_w[:, :3, :])
            left_jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, left_jacobian_w[:, 3:, :])
            
            right_jacobian_w = jacobians[:, right_jacobi_body_idx, :, right_jacobi_joint_ids]
            right_jacobian_b = right_jacobian_w.clone()
            right_jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, right_jacobian_w[:, :3, :])
            right_jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, right_jacobian_w[:, 3:, :])
            
            # Current joint positions (all 7)
            left_joint_pos = robot.data.joint_pos[:, left_arm_joint_ids]
            right_joint_pos = robot.data.joint_pos[:, right_arm_joint_ids]
            
            # Only compute and apply IK when tracking is active
            # This allows user to position in VR before arms start following
            if vr_tracking_active:
                # Set IK commands (position + orientation in base frame)
                left_pose_cmd = torch.cat([left_target_pos_b, left_target_quat_b], dim=-1)
                right_pose_cmd = torch.cat([right_target_pos_b, right_target_quat_b], dim=-1)
                
                left_ik_controller.set_command(left_pose_cmd)
                right_ik_controller.set_command(right_pose_cmd)
                
                # Compute IK -> 7 joint targets per arm
                left_joint_des = left_ik_controller.compute(
                    left_ee_pos_b, left_ee_quat_b, left_jacobian_b, left_joint_pos
                )
                right_joint_des = right_ik_controller.compute(
                    right_ee_pos_b, right_ee_quat_b, right_jacobian_b, right_joint_pos
                )
                
                # Add rest pose bias to help escape singularities (like extended arm)
                # This gently pulls joints toward a "comfortable" bent-elbow configuration
                left_rest_pull = rest_pose_gain * (rest_pose_left - left_joint_pos)
                right_rest_pull = rest_pose_gain * (rest_pose_right - right_joint_pos)
                left_joint_des = left_joint_des + left_rest_pull
                right_joint_des = right_joint_des + right_rest_pull
                
                # Clamp IK output to joint limits to prevent getting stuck
                left_joint_des = torch.clamp(left_joint_des, left_limits_low, left_limits_high)
                right_joint_des = torch.clamp(right_joint_des, right_limits_low, right_limits_high)
                
                # Apply all 7 joint targets per arm
                robot.set_joint_position_target(left_joint_des, joint_ids=left_arm_joint_ids)
                robot.set_joint_position_target(right_joint_des, joint_ids=right_arm_joint_ids)
            
            # Write the articulation data to simulation
            robot.write_data_to_sim()
            
            # Step the simulation directly (bypass env.step to avoid command manager resampling)
            unwrapped.sim.step(render=True)
            
            # Update robot data from simulation
            robot.update(unwrapped.sim.get_physics_dt())
            
            # LeRobot: Capture frame if recording (only when tracking is active)
            if lerobot_recording and lerobot_current_episode is not None and vr_tracking_active:
                from PIL import Image
                
                # Get current joint positions (observation state)
                joint_pos = robot.data.joint_pos[0].cpu().numpy().astype(np.float32)
                
                # Action = commanded joint targets from IK + gripper targets
                # Build full action array with the same shape as observation
                action = joint_pos.copy()  # Start with current state
                
                # Overwrite arm joints with the IK-computed targets (what we commanded)
                action[left_arm_joint_ids] = left_joint_des[0].cpu().numpy().astype(np.float32)
                action[right_arm_joint_ids] = right_joint_des[0].cpu().numpy().astype(np.float32)
                
                # Overwrite gripper joints with gripper targets
                if left_gripper_ids:
                    left_gripper_target = gripper_open_pos * (1.0 - left_trigger)
                    for gid in left_gripper_ids:
                        action[gid] = left_gripper_target
                if right_gripper_ids:
                    right_gripper_target = gripper_open_pos * (1.0 - right_trigger)
                    for gid in right_gripper_ids:
                        action[gid] = right_gripper_target
                
                # Build frame dict
                frame = {
                    "observation.state": joint_pos.copy(),
                    "action": action.copy(),
                }
                
                # Record object positions/orientations using Isaac Lab RigidObject API
                # This reads directly from physics simulation (accurate transforms)
                if active_pool_objects:
                    objects_state = []
                    dt = unwrapped.sim.get_physics_dt()
                    
                    for obj_info in active_pool_objects:
                        asset = obj_info["asset"]
                        prim_path = obj_info["prim_path"]
                        try:
                            # Update asset data from simulation (critical!)
                            asset.update(dt)
                            
                            # Read physics state from Isaac Lab RigidObject
                            pos = asset.data.root_pos_w[0].cpu().numpy().tolist()
                            quat = asset.data.root_quat_w[0].cpu().numpy().tolist()  # w,x,y,z
                            
                            objects_state.append({
                                "prim_path": prim_path,
                                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                                "orientation": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                                "scale": [1.0, 1.0, 1.0],
                            })
                        except Exception as e:
                            print(f"[Recording] Error getting transform for {prim_path}: {e}")
                    
                    if objects_state:
                        frame["objects_state"] = objects_state
                
                # Capture camera images using pre-created render products (fast)
                render_products = lerobot_current_episode.get("render_products", {})
                for cam_key, rgb_annot in render_products.items():
                    try:
                        data = rgb_annot.get_data()
                        if data is not None:
                            img_np = data[:, :, :3].astype(np.uint8)
                            frame[cam_key] = Image.fromarray(img_np)
                    except Exception:
                        pass
                
                lerobot_current_episode["frames"].append(frame)
                
                # Print recording indicator periodically
                if len(lerobot_current_episode["frames"]) % 60 == 0:
                    elapsed = time.time() - lerobot_current_episode["start_time"]
                    print(f"[LeRobot] Recording... {len(lerobot_current_episode['frames'])} frames ({elapsed:.1f}s)")
            
            # Print status periodically
            step_count += 1
            if step_count % 60 == 0:
                # Compute position errors (target - current)
                left_pos_err = (left_target_pos_b - left_ee_pos_b)[0].cpu().numpy()
                right_pos_err = (right_target_pos_b - right_ee_pos_b)[0].cpu().numpy()
                left_err_mag = np.linalg.norm(left_pos_err)
                right_err_mag = np.linalg.norm(right_pos_err)
                
                print(f"Step {step_count:5d} | "
                      f"L:[{left_pose[0]:.2f},{left_pose[1]:.2f},{left_pose[2]:.2f}] | "
                      f"R:[{right_pose[0]:.2f},{right_pose[1]:.2f},{right_pose[2]:.2f}] | "
                      f"Grip L:{left_trigger:.2f} R:{right_trigger:.2f}")
                
                # Show IK errors if significant (only when IK is active)
                if vr_tracking_active and (left_err_mag > 0.02 or right_err_mag > 0.02):
                    # Compute joint deltas (how much IK wants to move)
                    left_joint_delta = (left_joint_des - left_joint_pos)[0].cpu().numpy()
                    right_joint_delta = (right_joint_des - right_joint_pos)[0].cpu().numpy()
                    print(f"  [IK] Pos err: L={left_err_mag:.3f}m R={right_err_mag:.3f}m")
                    print(f"  [IK] L delta: [{', '.join(f'{d:.3f}' for d in left_joint_delta)}]")
                    print(f"  [IK] R delta: [{', '.join(f'{d:.3f}' for d in right_joint_delta)}]")
                
    except KeyboardInterrupt:
        print("\n[INFO] Teleoperation stopped by user")


if __name__ == "__main__":
    main()
    simulation_app.close()
