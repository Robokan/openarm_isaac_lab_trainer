"""SparkJAX Digital Twin Station Manager.

Loads the SparkPackDigialTwin USD scene in Isaac Sim, discovers all bimanual
robot articulations under the "robots" XForm, and bridges them to SparkJAX
via ROS2.

Modes:
  - Idle (default): no physics stepping, advertises virtual CAN interfaces,
    publishes static joint states for registered arms.
  - Teleop (--keyboard): physics stepping enabled, keyboard IK control of
    leader arm, follower mirrors leader joints, camera images published.

Uses only std_msgs and sensor_msgs (no custom messages needed), compatible
with Isaac Sim's bundled ROS2 Humble bridge.

Usage:
    python.sh station_manager.py              # idle, windowed
    python.sh station_manager.py --headless   # idle, headless
    python.sh station_manager.py --keyboard   # teleop with keyboard
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(
    description="SparkJAX Digital Twin Station Manager")
parser.add_argument(
    "--usd-path", type=str, default=None,
    help="Override USD scene path (default: auto-detect SparkPackDigialTwin)")
parser.add_argument(
    "--robots-xform", type=str, default="/World/robots",
    help="Prim path of the XForm containing bimanual robot pairs")
parser.add_argument(
    "--pub-rate", type=float, default=50.0,
    help="Joint state publish rate in Hz")
parser.add_argument(
    "--headless", action="store_true",
    help="Run without GUI")
parser.add_argument(
    "--keyboard", action="store_true",
    help="Enable keyboard teleoperation input")
args_cli, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args_cli.headless})

# ---------------------------------------------------------------------------
# Reduce rendering cost for off-screen camera render products.
# These mirror settings from isaaclab.python.rendering.kit that make
# camera captures lightweight without affecting the main viewport much.
# ---------------------------------------------------------------------------
import carb  # available after SimulationApp init

_cs = carb.settings.get_settings()
_cs.set_bool("/rtx/translucency/enabled", False)
_cs.set_bool("/rtx/reflections/enabled", False)
_cs.set_bool("/rtx/indirectDiffuse/enabled", False)
_cs.set_bool("/rtx/raytracing/cached/enabled", False)
_cs.set_bool("/rtx/ambientOcclusion/enabled", False)
_cs.set_bool("/rtx/directLighting/sampledLighting/enabled", True)
_cs.set_int("/rtx/directLighting/sampledLighting/samplesPerPixel", 1)
_cs.set_int("/rtx/post/dlss/execMode", 0)  # Performance mode
_cs.set_bool("/omni/replicator/asyncRendering", False)
del _cs

# ---------------------------------------------------------------------------
# All heavy imports AFTER Isaac Sim has initialised
# ---------------------------------------------------------------------------
import json
import math
import os
import threading
import time

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from sensor_msgs.msg import JointState, Image
    from std_msgs.msg import String
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False
    print("[WARN] rclpy not found -- ROS2 bridge will be disabled. "
          "Source ROS2 and the SparkJAX workspace before running.")

import omni.usd
from pxr import Usd, UsdPhysics, UsdGeom

try:
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.prims import SingleArticulation as Articulation
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.articulations import Articulation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LEFT_PREFIX = "openarm_left_"
_RIGHT_PREFIX = "openarm_right_"
_CANONICAL_PREFIX = "openarm_"

_LEFT_ARM_JOINTS = [
    "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
    "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
    "openarm_left_joint7",
]
_RIGHT_ARM_JOINTS = [
    "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
    "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
    "openarm_right_joint7",
]

_CANONICAL_JOINTS = [
    "openarm_joint1", "openarm_joint2", "openarm_joint3",
    "openarm_joint4", "openarm_joint5", "openarm_joint6",
    "openarm_joint7", "openarm_finger_joint1",
]


# ---------------------------------------------------------------------------
# Keyboard Device (ported from teleop_bimanual.py)
# ---------------------------------------------------------------------------

class KeyboardDevice:
    """Keyboard control using Isaac Sim's carb.input system."""

    def __init__(self, sensitivity: float = 1.0):
        import carb.input
        import omni.appwindow

        self.sensitivity = sensitivity
        self.step_size = 0.01 * sensitivity
        self.rot_step = 0.05
        self.left_pose = np.array([0.2, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self.right_pose = np.array([0.2, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0])
        self.left_euler = np.array([0.0, 0.0, 0.0])
        self.right_euler = np.array([0.0, 0.0, 0.0])
        self.left_gripper = 0.0
        self.right_gripper = 0.0
        self.gripper_step = 0.05
        self.active_hand = "left"
        self._carb_input = carb.input
        self._key_states = {}

        self._input = carb.input.acquire_input_interface()
        self._app_window = None
        for _ in range(50):
            self._app_window = omni.appwindow.get_default_app_window()
            if self._app_window is not None:
                break
            time.sleep(0.1)
        if self._app_window is None:
            raise RuntimeError("App window not available for keyboard input")
        self._keyboard = self._app_window.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event)

        self.markers_visible = False
        self.start_recording_requested = False
        self.stop_recording_requested = False
        self.reset_requested = False

        print("\n" + "=" * 60)
        print("KEYBOARD TELEOPERATION (leader-follower)")
        print("=" * 60)
        print("Position:  W/S (X)  A/D (Y)  Q/E (Z)")
        print("Rotation:  I/K (pitch)  J/L (yaw)  U/O (roll)")
        print("Gripper:   ; (close)  ' (open)")
        print("Hand:      1 (left)  2 (right)")
        print("Recording: Y (start)  T (stop)")
        print("Other:     M (toggle markers)  R (reset poses)")
        print("Follower station mirrors leader station.")
        print("=" * 60 + "\n")

    def _on_keyboard_event(self, event, *args, **kwargs):
        key = event.input
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            self._key_states[key] = True
        elif event.type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self._key_states[key] = False
            return True
        elif event.type != self._carb_input.KeyboardEventType.KEY_REPEAT:
            return True

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
                return False
            elif key == self._carb_input.KeyboardInput.M:
                self.markers_visible = not self.markers_visible
                print(f"[Keyboard] Markers {'visible' if self.markers_visible else 'hidden'}")
                return False
            elif key == self._carb_input.KeyboardInput.Y:
                self.start_recording_requested = True
                return False
            elif key == self._carb_input.KeyboardInput.T:
                self.stop_recording_requested = True
                return False

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
            self._carb_input.KeyboardInput.M,
        }
        if key in movement_keys:
            return False
        return True

    def update(self):
        """Poll key states and update EE target poses."""
        pose = self.left_pose if self.active_hand == "left" else self.right_pose
        euler = self.left_euler if self.active_hand == "left" else self.right_euler

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

        rot_changed = False
        if self._key_states.get(self._carb_input.KeyboardInput.U, False):
            euler[0] -= self.rot_step
            rot_changed = True
        if self._key_states.get(self._carb_input.KeyboardInput.O, False):
            euler[0] += self.rot_step
            rot_changed = True
        if self._key_states.get(self._carb_input.KeyboardInput.I, False):
            euler[1] += self.rot_step
            rot_changed = True
        if self._key_states.get(self._carb_input.KeyboardInput.K, False):
            euler[1] -= self.rot_step
            rot_changed = True
        if self._key_states.get(self._carb_input.KeyboardInput.J, False):
            euler[2] += self.rot_step
            rot_changed = True
        if self._key_states.get(self._carb_input.KeyboardInput.L, False):
            euler[2] -= self.rot_step
            rot_changed = True

        if rot_changed:
            pose[3:7] = _euler_to_quat(euler[0], euler[1], euler[2])

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

    def reset_poses(self):
        self.left_pose[:] = [0.2, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0]
        self.right_pose[:] = [0.2, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0]
        self.left_euler[:] = 0.0
        self.right_euler[:] = 0.0
        self.left_gripper = 0.0
        self.right_gripper = 0.0

    def __del__(self):
        if hasattr(self, '_sub_keyboard') and self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._sub_keyboard)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _euler_to_quat(roll, pitch, yaw):
    """Euler (roll, pitch, yaw) -> quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _dls_ik(jacobian: np.ndarray, delta_pose: np.ndarray,
            lambda_val: float = 0.05) -> np.ndarray:
    """Damped Least Squares IK: J^T (J J^T + λ²I)^{-1} δx."""
    JJT = jacobian @ jacobian.T + lambda_val ** 2 * np.eye(6)
    return jacobian.T @ np.linalg.solve(JJT, delta_pose)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _quat_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _pose_error(target_pos, target_quat, current_pos, current_quat):
    """Compute 6D pose error (position + orientation) in world frame."""
    pos_err = target_pos - current_pos
    quat_err = _quat_multiply(target_quat, _quat_conjugate(current_quat))
    if quat_err[0] < 0:
        quat_err = -quat_err
    rot_err = 2.0 * quat_err[1:4]
    return np.concatenate([pos_err, rot_err])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ArmBridge:
    """Tracks one simulated arm (half of a bimanual pair)."""

    def __init__(self, can_interface: str, side: str,
                 joint_indices: list[int]):
        self.can_interface = can_interface
        self.side = side
        self.joint_indices = joint_indices
        self.namespace: str | None = None
        self.publisher = None
        self.active = False


class StationPair:
    """A bimanual pair discovered in the USD scene."""

    def __init__(self, prim_name: str, prim_path: str,
                 articulation: Articulation,
                 left: ArmBridge, right: ArmBridge):
        self.prim_name = prim_name
        self.prim_path = prim_path
        self.articulation = articulation
        self.left = left
        self.right = right


class CameraInfo:
    """A discovered camera in the USD scene."""

    def __init__(self, name: str, prim_path: str, role: str,
                 pair_name: str):
        self.name = name
        self.prim_path = prim_path
        self.role = role  # "ego", "left_wrist", "right_wrist"
        self.pair_name = pair_name
        self.render_product = None
        self.annotator = None
        self.publisher = None


class TeleopState:
    """Tracks active teleop session state for one pair."""

    def __init__(self, pair: StationPair):
        self.pair = pair
        self.active = False


# ---------------------------------------------------------------------------
# USD Discovery
# ---------------------------------------------------------------------------

def _find_usd_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "..", "..", "source", "openarm", "openarm",
                     "tasks", "manager_based", "openarm_manipulation",
                     "usds", "SparkPackDigialTwin.usd"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Cannot find SparkPackDigialTwin.usd. Use --usd-path to specify.")


def _discover_pairs(world: World, robots_xform_path: str,
                    ) -> list[StationPair]:
    """Walk the robots XForm and build StationPair objects."""
    stage = omni.usd.get_context().get_stage()
    xform_prim = stage.GetPrimAtPath(robots_xform_path)
    if not xform_prim.IsValid():
        print(f"[WARN] No prim at {robots_xform_path}")
        return []

    pending: list[tuple[str, str, Articulation]] = []
    for child in xform_prim.GetChildren():
        if not child.IsA(UsdGeom.Xform):
            continue
        prim_path = str(child.GetPath())
        prim_name = child.GetName()
        art = Articulation(prim_path=prim_path, name=prim_name)
        world.scene.add(art)
        pending.append((prim_name, prim_path, art))

    if not pending:
        print("[WARN] No articulations found under", robots_xform_path)
        return []

    world.reset()
    simulation_app.update()
    print(f"[INFO] Physics initialized, querying DOFs for "
          f"{len(pending)} prims")

    pairs: list[StationPair] = []
    can_idx = 0

    for prim_name, prim_path, art in pending:
        dof_names = list(art.dof_names) if art.dof_names else []
        if not dof_names:
            print(f"[WARN] No DOFs found for {prim_name}, skipping")
            continue

        left_indices, right_indices = [], []
        for i, name in enumerate(dof_names):
            if name.startswith(_LEFT_PREFIX):
                left_indices.append(i)
            elif name.startswith(_RIGHT_PREFIX):
                right_indices.append(i)

        if not left_indices or not right_indices:
            print(f"[WARN] {prim_name}: cannot split left/right joints, "
                  f"skipping (dofs={dof_names})")
            continue

        left_arm = ArmBridge(can_interface=f"can{can_idx}",
                             side="left", joint_indices=left_indices)
        right_arm = ArmBridge(can_interface=f"can{can_idx + 1}",
                              side="right", joint_indices=right_indices)
        can_idx += 2

        pairs.append(StationPair(
            prim_name=prim_name, prim_path=prim_path,
            articulation=art, left=left_arm, right=right_arm))
        print(f"[INFO] Station '{prim_name}': "
              f"left={left_arm.can_interface}, "
              f"right={right_arm.can_interface}, "
              f"left_dofs={len(left_indices)}, "
              f"right_dofs={len(right_indices)}")

    return pairs


def _discover_cameras(pairs: list[StationPair]) -> list[CameraInfo]:
    """Find Camera prims by walking each station's USD subtree."""
    stage = omni.usd.get_context().get_stage()
    cameras: list[CameraInfo] = []

    for pair in pairs:
        root = stage.GetPrimAtPath(pair.prim_path)
        if not root or not root.IsValid():
            continue

        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Camera):
                continue
            cam_path = str(prim.GetPath())
            cam_name = prim.GetName()
            name_lower = cam_name.lower()

            if "ego" in name_lower or "high" in name_lower or "body" in name_lower:
                role = "ego"
            elif "left" in name_lower:
                role = "left_wrist"
            elif "right" in name_lower:
                role = "right_wrist"
            else:
                continue

            cameras.append(CameraInfo(
                name=cam_name, prim_path=cam_path,
                role=role, pair_name=pair.prim_name))
            print(f"[INFO] Camera: {cam_name} ({role}) "
                  f"station={pair.prim_name} at {cam_path}")

    return cameras


def _setup_camera_annotators(cameras: list[CameraInfo],
                             pair_name: str,
                             resolution: tuple[int, int] = (320, 240)):
    """Create replicator render products for cameras belonging to one pair.

    Only call this when teleop starts for a specific pair, to avoid
    creating dozens of off-screen render passes that tank performance.
    """
    try:
        import omni.replicator.core as rep
    except ImportError:
        print("[WARN] omni.replicator.core not available, cameras disabled")
        return

    count = 0
    for cam in cameras:
        if cam.pair_name != pair_name:
            continue
        if cam.annotator is not None:
            continue
        try:
            rp = rep.create.render_product(cam.prim_path, resolution)
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach([rp])
            cam.render_product = rp
            cam.annotator = annot
            count += 1
            print(f"[INFO] Camera {cam.name} render product ready")
        except Exception as e:
            print(f"[WARN] Camera {cam.name} setup failed: {e}")
    print(f"[INFO] Activated {count} camera render products for {pair_name}")


def _snapshot_cameras(cameras: list[CameraInfo], world,
                      pair_name: str = "",
                      resolution: tuple[int, int] = (320, 240)
                      ) -> dict[str, str]:
    """Take a one-shot JPEG snapshot of each camera, return {name: base64_jpeg}.

    Uses _setup_camera_annotators to create persistent render products
    (stored on the CameraInfo objects). They are NOT destroyed so they
    remain valid for teleop camera publishing later.
    """
    import base64 as b64mod

    target_cams = [c for c in cameras
                   if not pair_name or c.pair_name == pair_name]
    if not target_cams:
        return {}

    _setup_camera_annotators(cameras, pair_name or target_cams[0].pair_name,
                             resolution)

    for _ in range(5):
        world.step(render=True)

    snapshots: dict[str, str] = {}
    for cam in target_cams:
        if cam.annotator is None:
            continue
        try:
            data = cam.annotator.get_data()
            if data is not None and hasattr(data, 'shape') and data.size > 0:
                rgb = data[:, :, :3].astype(np.uint8).copy()
                bgr = rgb[:, :, ::-1]
                import cv2
                _, buf = cv2.imencode('.jpg', bgr,
                                      [cv2.IMWRITE_JPEG_QUALITY, 70])
                snapshots[cam.name] = b64mod.b64encode(
                    buf.tobytes()).decode()
        except Exception:
            pass

    print(f"[INFO] Captured {len(snapshots)} camera snapshots "
          f"for {pair_name or 'all'}")
    return snapshots


def _teardown_camera_annotators(cameras: list[CameraInfo]):
    """Destroy render products so they stop consuming GPU each frame."""
    try:
        import omni.replicator.core as rep
    except ImportError:
        pass

    count = 0
    for cam in cameras:
        if cam.render_product is not None:
            try:
                cam.render_product.destroy()
            except Exception:
                pass
            cam.render_product = None
            count += 1
        cam.annotator = None
    if count:
        print(f"[INFO] Destroyed {count} camera render products")


# ---------------------------------------------------------------------------
# Joint-state helpers
# ---------------------------------------------------------------------------

def _remap_joint_name(name: str, side_prefix: str) -> str:
    if name.startswith(side_prefix):
        return _CANONICAL_PREFIX + name[len(side_prefix):]
    return name


def _build_joint_state_msg(stamp, positions, dof_names: list[str],
                           indices: list[int],
                           side_prefix: str) -> JointState:
    msg = JointState()
    msg.header.stamp = stamp
    names, pos = [], []
    for idx in indices:
        cname = _remap_joint_name(dof_names[idx], side_prefix)
        if cname in _CANONICAL_JOINTS:
            names.append(cname)
            val = positions[idx]
            pos.append(float(val.item()) if hasattr(val, 'item') else float(val))
    msg.name = names
    msg.position = pos
    msg.velocity = [0.0] * len(pos)
    msg.effort = [0.0] * len(pos)
    return msg


def _build_image_msg(stamp, rgb_data: np.ndarray) -> Image:
    """Build a sensor_msgs/Image from an HxWx3 uint8 numpy array."""
    msg = Image()
    msg.header.stamp = stamp
    msg.height = rgb_data.shape[0]
    msg.width = rgb_data.shape[1]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = rgb_data.shape[1] * 3
    msg.data = rgb_data.tobytes()
    return msg


# ---------------------------------------------------------------------------
# ROS2 Bridge Node
# ---------------------------------------------------------------------------

class StationManagerNode(Node):
    """ROS2 node for hardware advertising, teleop signalling, and publishing."""

    def __init__(self, pairs: list[StationPair],
                 cameras: list[CameraInfo],
                 world=None):
        super().__init__("isaac_station_manager")
        self._pairs = pairs
        self._cameras = cameras
        self._world = world
        self._all_arms: dict[str, ArmBridge] = {}
        for pair in pairs:
            self._all_arms[pair.left.can_interface] = pair.left
            self._all_arms[pair.right.can_interface] = pair.right

        latched_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._hw_pub = self.create_publisher(
            String, "/isaac_sim/available_hardware", latched_qos)
        self._cam_pub = self.create_publisher(
            String, "/isaac_sim/available_cameras", latched_qos)

        self.create_subscription(
            String, "/jax/registration_events",
            self._on_registration_event, 10)
        self.create_subscription(
            String, "/jax/start_teleop_sim",
            self._on_start_teleop, 10)
        self.create_subscription(
            String, "/jax/stop_teleop_sim",
            self._on_stop_teleop, 10)

        self._teleop_pair_name: str | None = None
        self._teleop_follower_name: str = ""
        self._teleop_requested = False
        self._stop_teleop_requested = False

        self._camera_publishers: dict[str, object] = {}

        self._publish_available_hardware()
        self._publish_available_cameras()

        # Auto-activate all arms with default namespaces so joint states
        # are always published, even before SparkJAX registration.
        ns_idx = 1
        for pair in pairs:
            for arm in (pair.left, pair.right):
                if not arm.active:
                    arm.namespace = f"/openarm{ns_idx}"
                    topic = f"{arm.namespace}/joint_states"
                    arm.publisher = self.create_publisher(JointState, topic, 10)
                    arm.active = True
                ns_idx += 1

        self.get_logger().info(
            f"Station manager ready: {len(pairs)} pair(s), "
            f"{len(self._all_arms)} virtual CAN(s), "
            f"{len(cameras)} camera(s)")

    def _publish_available_hardware(self):
        cans = sorted(self._all_arms.keys())
        msg = String()
        msg.data = json.dumps(cans)
        self._hw_pub.publish(msg)
        self.get_logger().info(f"Advertised virtual CAN: {cans}")

    def _publish_available_cameras(self):
        snapshots: dict[str, str] = {}
        if self._world is not None and self._cameras:
            self.get_logger().info("Taking camera snapshots for wizard...")
            for pair in self._pairs:
                s = _snapshot_cameras(
                    self._cameras, self._world, pair_name=pair.prim_name)
                snapshots.update(s)

        cam_info = []
        for cam in self._cameras:
            entry = {
                "name": cam.name, "role": cam.role,
                "prim_path": cam.prim_path,
                "pair_name": cam.pair_name,
            }
            if cam.name in snapshots:
                entry["snapshot_b64"] = snapshots[cam.name]
            cam_info.append(entry)
        msg = String()
        msg.data = json.dumps(cam_info)
        self._cam_pub.publish(msg)

    def _on_registration_event(self, msg: String):
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return

        can = data.get("can_interface", "")
        if can not in self._all_arms:
            return
        arm = self._all_arms[can]
        if arm.active:
            return

        ns = data.get("robot_namespace", "")
        if not ns:
            return

        arm.namespace = ns
        topic = f"{ns}/joint_states"
        arm.publisher = self.create_publisher(JointState, topic, 10)
        arm.active = True
        self.get_logger().info(
            f"Activated {can} -> {ns} (publishing on {topic})")

    def _on_start_teleop(self, msg: String):
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        self._teleop_pair_name = data.get("pair_name", "")
        leader_can = data.get("leader_can", "")
        follower_can = data.get("follower_can", "")
        self._teleop_follower_name = ""

        leader_station = None
        follower_station = None
        for pair in self._pairs:
            if (pair.left.can_interface == leader_can
                    or pair.right.can_interface == leader_can):
                leader_station = pair.prim_name
            if (pair.left.can_interface == follower_can
                    or pair.right.can_interface == follower_can):
                follower_station = pair.prim_name

        if leader_station:
            self._teleop_pair_name = leader_station
        elif not self._teleop_pair_name:
            self._teleop_pair_name = self._pairs[0].prim_name if self._pairs else ""

        if follower_station and follower_station != self._teleop_pair_name:
            self._teleop_follower_name = follower_station
        elif len(self._pairs) > 1:
            for pair in self._pairs:
                if pair.prim_name != self._teleop_pair_name:
                    self._teleop_follower_name = pair.prim_name
                    break

        self._teleop_requested = True
        self.get_logger().info(
            f"Teleop start requested: leader={self._teleop_pair_name} "
            f"follower={self._teleop_follower_name}")

    def _on_stop_teleop(self, msg: String):
        self._stop_teleop_requested = True
        self.get_logger().info("Teleop stop requested")


    def get_or_create_camera_pub(self, topic: str):
        if topic not in self._camera_publishers:
            self._camera_publishers[topic] = self.create_publisher(
                Image, topic, 10)
        return self._camera_publishers[topic]

    def get_ros_stamp(self):
        return self.get_clock().now().to_msg()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_ee_pose(art_view, body_idx):
    """Get end-effector world pose from physics view. Returns (pos[3], quat_wxyz[4])."""
    link_tf = art_view._physics_view.get_link_transforms()
    # link_tf shape: (1, num_bodies, 7) -- pos(3) + quat_xyzw(4)
    tf = link_tf[0, body_idx]
    if hasattr(tf, 'cpu'):
        tf = tf.cpu().numpy()
    pos = tf[:3].copy()
    quat_xyzw = tf[3:7].copy()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz


def _extract_jacobian(art_view, body_idx, joint_ids):
    """Extract 6xN Jacobian for specific body and joints."""
    jac_full = art_view.get_jacobians()
    # shape: (1, num_bodies-1, 6, num_dofs) for fixed base
    if hasattr(jac_full, 'cpu'):
        jac_full = jac_full.cpu().numpy()
    jac = jac_full[0, body_idx - 1, :, :]  # body_idx-1 for fixed base
    jac_sub = jac[:, joint_ids]
    return jac_sub


def main():
    if not _HAS_RCLPY:
        print("[ERROR] rclpy is required.")
        simulation_app.close()
        return

    usd_path = args_cli.usd_path or _find_usd_path()
    print(f"[INFO] Loading USD: {usd_path}")

    world = World(stage_units_in_meters=1.0)
    add_reference_to_stage(usd_path=usd_path, prim_path="/World")

    print(f"[INFO] Discovering robots under {args_cli.robots_xform}...")
    pairs = _discover_pairs(world, args_cli.robots_xform)
    if not pairs:
        print("[ERROR] No bimanual pairs found.")
        simulation_app.close()
        return

    total_arms = len(pairs) * 2
    print(f"[INFO] Found {len(pairs)} pair(s) ({total_arms} arms)")

    print("[INFO] Discovering cameras...")
    cameras = _discover_cameras(pairs)

    keyboard = None
    if args_cli.keyboard:
        simulation_app.update()
        keyboard = KeyboardDevice()

    rclpy.init()
    ros_node = StationManagerNode(pairs, cameras, world=world)
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    pair_dof_names: dict[str, list[str]] = {}
    pair_positions: dict[str, np.ndarray] = {}
    for pair in pairs:
        pair_dof_names[pair.prim_name] = list(pair.articulation.dof_names)
        pos = pair.articulation.get_joint_positions()
        if pos is None:
            pos = np.zeros(len(pair.articulation.dof_names))
        elif isinstance(pos, (list, tuple)):
            pos = np.array(pos)
        elif hasattr(pos, 'cpu'):
            pos = pos.cpu().numpy()
        if pos.ndim > 1:
            pos = pos[0]
        pair_positions[pair.prim_name] = pos

    # Find EE body indices for IK
    pair_ik_info: dict[str, dict] = {}
    if args_cli.keyboard:
        for pair in pairs:
            art_view = pair.articulation._articulation_view
            body_names = art_view.body_names
            kps, kds = art_view.get_gains()
            print(f"[INFO] {pair.prim_name} PD gains: "
                  f"kp={kps[0][:3]}... kd={kds[0][:3]}...")
            print(f"[INFO] {pair.prim_name} bodies: {body_names}")

            left_ee_idx = None
            right_ee_idx = None
            for i, name in enumerate(body_names):
                if "left_hand" in name or "left_link7" in name:
                    left_ee_idx = i
                elif "right_hand" in name or "right_link7" in name:
                    right_ee_idx = i

            if left_ee_idx is None:
                for i, name in enumerate(body_names):
                    if "left" in name:
                        left_ee_idx = i
            if right_ee_idx is None:
                for i, name in enumerate(body_names):
                    if "right" in name:
                        right_ee_idx = i

            dof_names = list(pair.articulation.dof_names)
            left_arm_ids = []
            right_arm_ids = []
            for i, n in enumerate(dof_names):
                if n.startswith("openarm_left_joint"):
                    left_arm_ids.append(i)
                elif n.startswith("openarm_right_joint"):
                    right_arm_ids.append(i)

            pair_ik_info[pair.prim_name] = {
                "left_ee_idx": left_ee_idx,
                "right_ee_idx": right_ee_idx,
                "left_arm_ids": left_arm_ids,
                "right_arm_ids": right_arm_ids,
            }
            print(f"[INFO] {pair.prim_name} IK: "
                  f"left_ee={left_ee_idx} ({body_names[left_ee_idx] if left_ee_idx else '?'}), "
                  f"right_ee={right_ee_idx} ({body_names[right_ee_idx] if right_ee_idx else '?'}), "
                  f"left_joints={left_arm_ids}, right_joints={right_arm_ids}")

    pub_period = 1.0 / args_cli.pub_rate
    cam_pub_period = 1.0 / 10.0  # 10 Hz for camera images
    idle_frame_period = 1.0 / 30.0
    last_pub = 0.0
    last_cam_pub = 0.0
    teleop_active = False
    teleop_pair: StationPair | None = None
    step_count = 0

    follower_pair: StationPair | None = None
    follower_mirroring = False
    cameras_active = False

    if keyboard:
        teleop_pair = pairs[0]

        # Initialize keyboard poses to current EE positions on leader station
        ik_info = pair_ik_info.get(teleop_pair.prim_name)
        if ik_info and ik_info["left_ee_idx"] is not None:
            art_view = teleop_pair.articulation._articulation_view
            world.step(render=True)
            left_pos, left_quat = _get_ee_pose(
                art_view, ik_info["left_ee_idx"])
            right_pos, right_quat = _get_ee_pose(
                art_view, ik_info["right_ee_idx"])
            keyboard.left_pose[:3] = left_pos
            keyboard.left_pose[3:7] = left_quat
            keyboard.right_pose[:3] = right_pos
            keyboard.right_pose[3:7] = right_quat
            print(f"[INFO] Leader EE poses: "
                  f"L={left_pos} R={right_pos}")

        initial_joint_positions = {
            name: pos.copy()
            for name, pos in pair_positions.items()
        }
        initial_left_pose = keyboard.left_pose.copy()
        initial_right_pose = keyboard.right_pose.copy()
        initial_left_euler = keyboard.left_euler.copy()
        initial_right_euler = keyboard.right_euler.copy()

        print(f"[INFO] Keyboard controls leader station: "
              f"{teleop_pair.prim_name}")
        print("[INFO] Follower mirroring will activate when "
              "SparkJAX starts teleoperation")

    # Create IK target marker for leader arm
    left_marker_path = "/ik_target_leader"
    stage = omni.usd.get_context().get_stage()
    try:
        from pxr import Gf
        arrow_length = 0.06
        arrow_radius = 0.003
        axis_defs = [
            ("x_axis", (1.0, 0.0, 0.0), (0, 90, 0)),
            ("y_axis", (0.0, 1.0, 0.0), (-90, 0, 0)),
            ("z_axis", (0.0, 0.4, 1.0), (0, 0, 0)),
        ]
        parent_xform = UsdGeom.Xform.Define(stage, left_marker_path)
        pxf = UsdGeom.Xformable(parent_xform.GetPrim())
        pxf.ClearXformOpOrder()
        pxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        pxf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
        for sub_name, color, (rx, ry, rz) in axis_defs:
            cyl_path = f"{left_marker_path}/{sub_name}"
            cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
            cyl.GetRadiusAttr().Set(arrow_radius)
            cyl.GetHeightAttr().Set(arrow_length)
            cyl.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            cxf = UsdGeom.Xformable(cyl.GetPrim())
            cxf.ClearXformOpOrder()
            cxf.AddTranslateOp().Set(Gf.Vec3d(0, 0, arrow_length / 2.0))
            cxf.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))
        print("[INFO] Created leader IK target marker")
    except Exception as e:
        print(f"[WARN] Could not create target marker: {e}")

    print(f"[INFO] Publishing joint states at {args_cli.pub_rate} Hz.")

    try:
        while simulation_app.is_running():
            if ros_node._teleop_requested:
                ros_node._teleop_requested = False
                f_name = ros_node._teleop_follower_name
                follower_pair = None
                for p in pairs:
                    if p.prim_name == f_name:
                        follower_pair = p
                        break
                if follower_pair is None and len(pairs) > 1:
                    for p in pairs:
                        if p.prim_name != teleop_pair.prim_name:
                            follower_pair = p
                            break
                follower_mirroring = follower_pair is not None
                teleop_active = True

                if follower_pair is not None:
                    _setup_camera_annotators(
                        cameras, follower_pair.prim_name)
                    for _ in range(5):
                        world.step(render=True)
                    cameras_active = True

                print(f"[INFO] Teleop activated: leader={teleop_pair.prim_name}"
                      f" follower={follower_pair.prim_name if follower_pair else 'none'}"
                      f" mirroring={follower_mirroring}")

            if ros_node._stop_teleop_requested:
                ros_node._stop_teleop_requested = False
                teleop_active = False
                follower_mirroring = False
                follower_pair = None
                _teardown_camera_annotators(cameras)
                cameras_active = False
                print("[INFO] Teleop stopped, follower mirroring disabled")

            if keyboard and teleop_pair:
                teleop_frame_start = time.monotonic()
                keyboard.update()

                if keyboard.reset_requested:
                    keyboard.reset_requested = False
                    keyboard.left_pose[:] = initial_left_pose
                    keyboard.right_pose[:] = initial_right_pose
                    keyboard.left_euler[:] = initial_left_euler
                    keyboard.right_euler[:] = initial_right_euler
                    keyboard.left_gripper = 0.0
                    keyboard.right_gripper = 0.0
                    for name, init_pos in initial_joint_positions.items():
                        pair_positions[name] = init_pos.copy()
                        for p in pairs:
                            if p.prim_name == name:
                                pos = init_pos.reshape(1, -1)
                                if _HAS_TORCH:
                                    pos = torch.tensor(
                                        pos, dtype=torch.float32,
                                        device="cpu")
                                view = p.articulation._articulation_view
                                view.set_joint_positions(pos)
                                view.set_joint_velocities(
                                    torch.zeros_like(pos) if _HAS_TORCH
                                    else np.zeros_like(pos))
                                view.set_joint_position_targets(pos)
                                break
                    for _ in range(5):
                        world.step(render=False)
                    world.step(render=True)
                    print("[INFO] Reset to initial pose")

                art = teleop_pair.articulation
                art_view = art._articulation_view
                ik_info = pair_ik_info.get(teleop_pair.prim_name)

                if ik_info and ik_info["left_ee_idx"] is not None:
                    left_ee_idx = ik_info["left_ee_idx"]
                    right_ee_idx = ik_info["right_ee_idx"]
                    left_arm_ids = ik_info["left_arm_ids"]
                    right_arm_ids = ik_info["right_arm_ids"]

                    # IK for left arm
                    left_pos_cur, left_quat_cur = _get_ee_pose(
                        art_view, left_ee_idx)
                    left_err = _pose_error(
                        keyboard.left_pose[:3], keyboard.left_pose[3:7],
                        left_pos_cur, left_quat_cur)
                    left_jac = _extract_jacobian(
                        art_view, left_ee_idx, left_arm_ids)
                    left_dq = _dls_ik(left_jac, left_err, lambda_val=0.05)

                    # IK for right arm
                    right_pos_cur, right_quat_cur = _get_ee_pose(
                        art_view, right_ee_idx)
                    right_err = _pose_error(
                        keyboard.right_pose[:3], keyboard.right_pose[3:7],
                        right_pos_cur, right_quat_cur)
                    right_jac = _extract_jacobian(
                        art_view, right_ee_idx, right_arm_ids)
                    right_dq = _dls_ik(right_jac, right_err, lambda_val=0.05)

                    # Apply IK deltas to leader station
                    positions = pair_positions[teleop_pair.prim_name].copy()
                    for i, jid in enumerate(left_arm_ids):
                        positions[jid] += float(left_dq[i])
                    for i, jid in enumerate(right_arm_ids):
                        positions[jid] += float(right_dq[i])

                    targets = positions.reshape(1, -1)
                    if _HAS_TORCH:
                        targets = torch.tensor(
                            targets, dtype=torch.float32, device="cpu")
                    art_view.set_joint_position_targets(targets)

                    # Mirror leader joints to follower station (only when
                    # SparkJAX has started a teleop session)
                    if follower_mirroring and follower_pair is not None:
                        f_view = follower_pair.articulation._articulation_view
                        f_targets = positions.reshape(1, -1)
                        if _HAS_TORCH:
                            f_targets = torch.tensor(
                                f_targets, dtype=torch.float32, device="cpu")
                        f_view.set_joint_position_targets(f_targets)

                world.step(render=True)

                # Update IK target marker (active hand)
                try:
                    from pxr import Gf
                    pose = (keyboard.left_pose
                            if keyboard.active_hand == "left"
                            else keyboard.right_pose)
                    m_prim = stage.GetPrimAtPath(left_marker_path)
                    if m_prim.IsValid():
                        xform = UsdGeom.Xformable(m_prim)
                        ops = xform.GetOrderedXformOps()
                        if len(ops) >= 2:
                            ops[0].Set(Gf.Vec3d(
                                float(pose[0]), float(pose[1]),
                                float(pose[2])))
                            ops[1].Set(Gf.Quatf(
                                float(pose[3]), float(pose[4]),
                                float(pose[5]), float(pose[6])))
                        img = UsdGeom.Imageable(m_prim)
                        if keyboard.markers_visible:
                            img.MakeVisible()
                        else:
                            img.MakeInvisible()
                except Exception:
                    pass

                # Update leader station actual positions
                actual = art.get_joint_positions()
                if actual is not None:
                    if hasattr(actual, 'cpu'):
                        actual = actual.cpu().numpy()
                    if actual.ndim > 1:
                        actual = actual[0]
                    pair_positions[teleop_pair.prim_name] = actual
                else:
                    pair_positions[teleop_pair.prim_name] = positions

                if follower_mirroring and follower_pair is not None:
                    f_actual = follower_pair.articulation.get_joint_positions()
                    if f_actual is not None:
                        if hasattr(f_actual, 'cpu'):
                            f_actual = f_actual.cpu().numpy()
                        if f_actual.ndim > 1:
                            f_actual = f_actual[0]
                        pair_positions[follower_pair.prim_name] = f_actual

                step_count += 1
                if step_count % 120 == 0:
                    lp = keyboard.left_pose[:3]
                    rp = keyboard.right_pose[:3]
                    print(f"[TELEOP] step={step_count} "
                          f"L=[{lp[0]:.3f},{lp[1]:.3f},{lp[2]:.3f}] "
                          f"R=[{rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f}] "
                          f"hand={keyboard.active_hand} "
                          f"mirror={'ON' if follower_mirroring else 'off'}"
                          f" cam={'ON' if cameras_active else 'off'}")

                # Cap teleop loop at 30 fps
                teleop_elapsed = time.monotonic() - teleop_frame_start
                teleop_sleep = idle_frame_period - teleop_elapsed
                if teleop_sleep > 0:
                    time.sleep(teleop_sleep)
            else:
                frame_start = time.monotonic()
                simulation_app.update()
                elapsed = time.monotonic() - frame_start
                sleep_time = idle_frame_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            now = time.monotonic()
            if now - last_pub < pub_period:
                continue
            last_pub = now

            stamp = ros_node.get_ros_stamp()

            for pair in pairs:
                positions = pair_positions[pair.prim_name]
                dof_names = pair_dof_names[pair.prim_name]

                if pair.left.active and pair.left.publisher:
                    msg = _build_joint_state_msg(
                        stamp, positions, dof_names,
                        pair.left.joint_indices, _LEFT_PREFIX)
                    pair.left.publisher.publish(msg)

                if pair.right.active and pair.right.publisher:
                    msg = _build_joint_state_msg(
                        stamp, positions, dof_names,
                        pair.right.joint_indices, _RIGHT_PREFIX)
                    pair.right.publisher.publish(msg)

            # Publish camera images from the follower pair (throttled)
            if cameras_active and follower_pair is not None and now - last_cam_pub >= cam_pub_period:
                last_cam_pub = now
                cam_pub_count = 0
                for cam in cameras:
                    if cam.pair_name != follower_pair.prim_name:
                        continue
                    if cam.annotator is None:
                        continue
                    try:
                        data = cam.annotator.get_data()
                        if (data is not None and hasattr(data, 'shape')
                                and len(data.shape) == 3
                                and data.shape[2] >= 3
                                and data.size > 0):
                            if data.dtype in (np.float32, np.float64):
                                rgb = (data[:, :, :3] * 255).clip(0, 255).astype(np.uint8).copy()
                            else:
                                rgb = data[:, :, :3].astype(np.uint8).copy()
                            topic = f'/isaac_sim/camera/{cam.role}'
                            pub = ros_node.get_or_create_camera_pub(topic)
                            pub.publish(_build_image_msg(stamp, rgb))
                            cam_pub_count += 1
                    except Exception:
                        pass
                if cam_pub_count > 0 and not hasattr(ros_node, '_cam_pub_logged'):
                    ros_node._cam_pub_logged = True
                    print(f"[INFO] Publishing {cam_pub_count} camera feed(s) "
                          f"from {follower_pair.prim_name}")

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
