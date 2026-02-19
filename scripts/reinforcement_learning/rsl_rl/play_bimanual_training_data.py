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

"""Script to verify and playback VLA training data captured for Pi 0.5.

This script reads VLA training data (LeRobot v3.0 format or legacy HDF5) and can:
1. Display camera images and stats (--verify mode, no simulation required)
2. Replay actions in simulation (--replay mode)

Usage:
    # Verify data (show images and stats, no simulation):
    python play_bimanual_training_data.py <data_dir> --verify
    
    # Replay in simulation:
    python play_bimanual_training_data.py <data_dir> --replay [--episode N] [--loop]
"""

import argparse
import os
import re
import sys


def detect_format(data_dir: str) -> str:
    """Detect the dataset format (lerobot, lerobot_fallback, or hdf5)."""
    # Check for LeRobot native format (meta/info.json)
    if os.path.exists(os.path.join(data_dir, "meta", "info.json")):
        return "lerobot"
    # Check for fallback format (episodes/ directory with parquet files)
    if os.path.exists(os.path.join(data_dir, "episodes")):
        return "lerobot_fallback"
    # Check for legacy HDF5 format
    hdf5_files = [f for f in os.listdir(data_dir) if f.endswith(".hdf5")]
    if hdf5_files:
        return "hdf5"
    return "unknown"


def verify_data(data_dir: str, episode_idx: int = None):
    """Verify VLA training data by displaying images and stats."""
    format_type = detect_format(data_dir)
    print(f"[INFO] Detected format: {format_type}")
    
    if format_type == "lerobot":
        verify_lerobot_data(data_dir, episode_idx)
        return
    elif format_type == "lerobot_fallback":
        verify_lerobot_fallback_data(data_dir, episode_idx)
        return
    elif format_type == "hdf5":
        verify_hdf5_data(data_dir, episode_idx)
        return
    else:
        print(f"[ERROR] Unknown data format in {data_dir}")
        return


def verify_lerobot_data(data_dir: str, episode_idx: int = None):
    """Verify LeRobot v3.0 native format data."""
    import json
    
    # Load LeRobot metadata
    info_path = os.path.join(data_dir, "meta", "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)
    
    print(f"[INFO] LeRobot Dataset:")
    print(f"  Robot type: {info.get('robot_type', 'N/A')}")
    print(f"  FPS: {info.get('fps', 'N/A')}")
    print(f"  Features: {list(info.get('features', {}).keys())}")
    
    # Load tasks
    tasks_path = os.path.join(data_dir, "meta", "tasks.jsonl")
    if os.path.exists(tasks_path):
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        print(f"  Tasks: {[t.get('task', 'N/A') for t in tasks]}")
    
    # Try to use LeRobot to load and inspect
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(repo_id=f"local/{os.path.basename(data_dir)}", root=os.path.dirname(data_dir))
        print(f"  Total frames: {len(dataset)}")
        print(f"  Episodes: {dataset.meta.total_episodes}")
        
        # Show sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n[INFO] Sample frame keys: {list(sample.keys())}")
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value).__name__}")
    except Exception as e:
        print(f"[WARNING] Could not load with LeRobot API: {e}")
        print("[INFO] Install LeRobot for full dataset inspection: pip install lerobot")


def verify_lerobot_fallback_data(data_dir: str, episode_idx: int = None):
    """Verify LeRobot fallback format (parquet + images)."""
    import json
    import numpy as np
    
    # Load metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"[INFO] Dataset metadata:")
        print(f"  Task: {metadata.get('task_text', 'N/A')}")
        print(f"  Robot: {metadata.get('robot_type', 'N/A')}")
        print(f"  FPS: {metadata.get('fps', 'N/A')}")
        print(f"  Joints: {metadata.get('num_joints', 'N/A')}")
    
    # Find episodes
    episodes_dir = os.path.join(data_dir, "episodes")
    episode_dirs = sorted([d for d in os.listdir(episodes_dir) if d.startswith("episode_")])
    print(f"[INFO] Found {len(episode_dirs)} episodes")
    
    if episode_idx is not None:
        # Find episode by name (episode_N) rather than list index
        target_name = f"episode_{episode_idx}"
        if target_name in episode_dirs:
            episode_dirs = [target_name]
        elif episode_idx < len(episode_dirs):
            episode_dirs = [episode_dirs[episode_idx]]
        else:
            print(f"[ERROR] Episode {episode_idx} not found. Available: {episode_dirs}")
            return
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        has_plt = True
    except ImportError:
        has_plt = False
    
    for ep_dir in episode_dirs[:3]:  # Show first 3
        ep_path = os.path.join(episodes_dir, ep_dir)
        parquet_path = os.path.join(ep_path, "data.parquet")
        
        if os.path.exists(parquet_path):
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            print(f"\n[INFO] {ep_dir}:")
            print(f"  Frames: {len(df)}")
            print(f"  Columns: {list(df.columns)[:10]}...")
            
            # Show images if available
            for cam_name in ["ego", "left_wrist", "right_wrist"]:
                cam_dir = os.path.join(ep_path, cam_name)
                if os.path.exists(cam_dir):
                    frames = sorted(os.listdir(cam_dir))
                    print(f"  {cam_name}: {len(frames)} frames")
                    
                    if has_plt and len(frames) > 0:
                        from PIL import Image
                        img = Image.open(os.path.join(cam_dir, frames[0]))
                        plt.figure(figsize=(6, 6))
                        plt.imshow(img)
                        plt.title(f"{ep_dir} - {cam_name} (frame 0)")
                        plt.axis('off')
                        plt.show()


def verify_hdf5_data(data_dir: str, episode_idx: int = None):
    """Verify legacy HDF5 format data."""
    import h5py
    import json
    import numpy as np
    
    # Load metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"[INFO] Dataset metadata:")
        print(f"  Task: {metadata.get('task_text', 'N/A')}")
        print(f"  Robot: {metadata.get('robot_type', 'N/A')}")
        print(f"  FPS: {metadata.get('fps', 'N/A')}")
        print(f"  Cameras: {metadata.get('cameras', [])}")
        print()
    
    # Find all episode files
    episode_files = sorted([f for f in os.listdir(data_dir) if f.startswith("episode_") and f.endswith(".hdf5")])
    print(f"[INFO] Found {len(episode_files)} episodes")
    
    if len(episode_files) == 0:
        print("[ERROR] No episode files found!")
        return
    
    # Select episodes to verify
    if episode_idx is not None:
        if episode_idx >= len(episode_files):
            print(f"[ERROR] Episode {episode_idx} not found. Max: {len(episode_files) - 1}")
            return
        episode_files = [episode_files[episode_idx]]
    
    # Try to import matplotlib for image display
    try:
        import matplotlib.pyplot as plt
        has_plt = True
    except ImportError:
        has_plt = False
        print("[WARNING] matplotlib not available, skipping image display")
    
    for ep_file in episode_files:
        ep_path = os.path.join(data_dir, ep_file)
        print(f"\n[INFO] Verifying: {ep_file}")
        
        with h5py.File(ep_path, "r") as f:
            # Print structure
            print(f"  Task: {f.attrs.get('task', 'N/A')}")
            print(f"  Num steps: {f.attrs.get('num_steps', 'N/A')}")
            
            # Check cube initial positions
            if "left_cube_start_pos" in f:
                left_pos = f["left_cube_start_pos"][:]
                print(f"  Left cube start pos: [{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}]")
            if "right_cube_start_pos" in f:
                right_pos = f["right_cube_start_pos"][:]
                print(f"  Right cube start pos: [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]")
            
            # Check robot initial positions
            if "robot_start_qpos" in f:
                robot_qpos = f["robot_start_qpos"][:]
                print(f"  Robot start qpos: {robot_qpos.shape} joints")
                print(f"    range: [{robot_qpos.min():.3f}, {robot_qpos.max():.3f}]")
            
            # Check observations
            if "observations" in f:
                obs = f["observations"]
                if "qpos" in obs:
                    qpos = obs["qpos"][:]
                    print(f"  qpos shape: {qpos.shape}")
                    print(f"    range: [{qpos.min():.3f}, {qpos.max():.3f}]")
                
                if "qvel" in obs:
                    qvel = obs["qvel"][:]
                    print(f"  qvel shape: {qvel.shape}")
                
                if "images" in obs:
                    img_grp = obs["images"]
                    print(f"  Cameras: {list(img_grp.keys())}")
                    
                    for cam_name in img_grp.keys():
                        imgs = img_grp[cam_name][:]
                        print(f"    {cam_name}: {imgs.shape} (dtype: {imgs.dtype})")
                        
                        # Display first frame
                        if has_plt and len(imgs) > 0:
                            # Convert (C, H, W) to (H, W, C) for display
                            img = imgs[0]
                            if img.shape[0] == 3:  # (C, H, W)
                                img = np.transpose(img, (1, 2, 0))
                            
                            plt.figure(figsize=(6, 6))
                            plt.imshow(img)
                            plt.title(f"{ep_file} - {cam_name} (frame 0)")
                            plt.axis('off')
                            plt.show()
            
            # Check actions
            if "action" in f:
                actions = f["action"][:]
                print(f"  actions shape: {actions.shape}")
                print(f"    range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    print("\n[INFO] Verification complete!")


def replay_data(data_dir: str, episode_idx: int = None, loop: bool = False, real_time: bool = False):
    """Replay VLA training data in simulation."""
    # Detect format
    format_type = detect_format(data_dir)
    print(f"[INFO] Detected format: {format_type}")
    
    if format_type == "lerobot":
        print("[INFO] LeRobot native format detected.")
        print("[INFO] For LeRobot datasets, use the built-in visualization:")
        print(f"       python -m lerobot.scripts.visualize_dataset --repo-id local/{os.path.basename(data_dir)}")
        print("[INFO] Or to replay in Isaac Sim, convert using the fallback format.")
        return
    elif format_type == "lerobot_fallback":
        replay_lerobot_fallback_data(data_dir, episode_idx, loop, real_time)
        return
    elif format_type == "hdf5":
        replay_hdf5_data(data_dir, episode_idx, loop, real_time)
        return
    else:
        print(f"[ERROR] Unknown data format in {data_dir}")
        return


def replay_lerobot_fallback_data(data_dir: str, episode_idx: int = None, loop: bool = False, real_time: bool = False):
    """Replay LeRobot fallback format data in simulation."""
    # Delayed imports for simulation
    from isaaclab.app import AppLauncher
    import argparse
    
    args = argparse.Namespace(
        headless=False,
        enable_cameras=False,
        device="cuda:0",
        livestream=-1,
        experience="",
    )
    
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    import gymnasium as gym
    import json
    import numpy as np
    import pandas as pd
    import signal
    import threading
    import time
    import torch
    
    import isaaclab_tasks  # noqa: F401
    import openarm.tasks  # noqa: F401
    
    # Camera view window disabled (Isaac Sim bundles headless OpenCV)
    HAS_CV2 = False
    
    # Load metadata (try top-level first, then per-episode)
    metadata_path = os.path.join(data_dir, "metadata.json")
    metadata = {}
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        # Try to load from first episode's metadata
        episodes_dir = os.path.join(data_dir, "episodes")
        if os.path.exists(episodes_dir):
            ep_dirs = sorted([d for d in os.listdir(episodes_dir) if d.startswith("episode_")])
            if ep_dirs:
                ep_meta_path = os.path.join(episodes_dir, ep_dirs[0], "metadata.json")
                if os.path.exists(ep_meta_path):
                    with open(ep_meta_path, "r") as f:
                        metadata = json.load(f)
                    print(f"[INFO] Using metadata from {ep_dirs[0]}")
    
    task_name = metadata.get("task", "Isaac-Reach-OpenArm-Bi-Teleop-v0")
    # Convert to Play variant if needed (but Teleop tasks don't have Play variants)
    if "-Play" not in task_name and "-Teleop" not in task_name:
        task_name = task_name.replace("-v0", "-Play-v0")
    
    fps = metadata.get("fps", 50)
    dt = 1.0 / fps
    task_text = metadata.get("task_text", metadata.get("task", ""))
    
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Task text: {task_text}")
    print(f"[INFO] FPS: {fps}, dt: {dt}")
    
    # Find episode directories
    episodes_dir = os.path.join(data_dir, "episodes")
    episode_dirs = sorted([d for d in os.listdir(episodes_dir) if d.startswith("episode_")])
    
    if episode_idx is not None:
        # Find episode by name (episode_N) rather than list index
        target_name = f"episode_{episode_idx}"
        if target_name in episode_dirs:
            episode_dirs = [target_name]
        elif episode_idx < len(episode_dirs):
            # Fall back to list index if name not found
            episode_dirs = [episode_dirs[episode_idx]]
        else:
            print(f"[ERROR] Episode {episode_idx} not found. Available: {episode_dirs}")
            simulation_app.close()
            return
    
    # Load environment config
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=1)
    env_cfg.scene.num_envs = 1
    
    # Disable randomization for replay
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.enable_corruption = False
    
    # Create environment
    env = gym.make(task_name, cfg=env_cfg)
    unwrapped = env.unwrapped
    
    # Load pool objects from scene (for kinematic playback of recorded pool objects)
    pool_objects = {}  # Maps pool name (e.g., "pool_cube_0") to scene asset
    scene_keys = list(unwrapped.scene.keys()) if hasattr(unwrapped.scene, 'keys') else []
    for key in scene_keys:
        if key.startswith("pool_cube_") or key.startswith("pool_mug_") or key.startswith("pool_fruit_"):
            pool_objects[key] = unwrapped.scene[key]
    print(f"[INFO] Found {len(pool_objects)} pool objects in scene: {list(pool_objects.keys())}")
    
    # Keyboard controls
    import carb.input
    import omni.appwindow
    
    input_interface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    
    stop_playback = [False]
    skip_episode = [False]
    pause_playback = [False]
    
    def on_keyboard_event(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.Q:
                stop_playback[0] = True
                return True
            elif event.input == carb.input.KeyboardInput.N:
                skip_episode[0] = True
                return True
            elif event.input == carb.input.KeyboardInput.SPACE:
                pause_playback[0] = not pause_playback[0]
                print(f"[INFO] {'Paused' if pause_playback[0] else 'Resumed'}")
                return True
        return False
    
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    print("[INFO] Controls: Q=quit, N=next episode, SPACE=pause/resume")
    
    stop_requested = threading.Event()
    
    def monitor_app():
        while not stop_requested.is_set():
            if not simulation_app.is_running():
                os.kill(os.getpid(), signal.SIGINT)
                break
            time.sleep(0.5)
    
    monitor_thread = threading.Thread(target=monitor_app, daemon=True)
    monitor_thread.start()
    
    try:
        while True:
            for ep_dir_name in episode_dirs:
              try:
                if stop_playback[0]:
                    break
                
                ep_path = os.path.join(episodes_dir, ep_dir_name)
                parquet_path = os.path.join(ep_path, "data.parquet")
                
                if not os.path.exists(parquet_path):
                    print(f"[WARNING] No data.parquet in {ep_dir_name}, skipping")
                    continue
                
                df = pd.read_parquet(parquet_path)
                
                # Extract actions from columns (sort numerically, not alphabetically)
                action_cols = [c for c in df.columns if c.startswith("action.")]
                action_cols = sorted(action_cols, key=lambda x: int(x.split(".")[-1]))
                actions = df[action_cols].values
                
                # Load initial conditions if available
                init_cond_path = os.path.join(ep_path, "initial_conditions.json")
                init_cond = None
                if os.path.exists(init_cond_path):
                    with open(init_cond_path, "r") as f:
                        init_cond = json.load(f)
                
                # Load object states per frame for kinematic replay
                objects_state_path = os.path.join(ep_path, "objects_state.json")
                objects_per_frame = None
                if os.path.exists(objects_state_path):
                    with open(objects_state_path, "r") as f:
                        objects_per_frame = json.load(f)
                    print(f"  [INFO] Loaded object states for {len(objects_per_frame)} frames (kinematic replay)")
                
                # Extract observation states for kinematic replay (robot joint states)
                state_cols = [c for c in df.columns if c.startswith("observation.state.")]
                state_cols = sorted(state_cols, key=lambda x: int(x.split(".")[-1]))
                states = df[state_cols].values if state_cols else None
                
                print(f"\n[INFO] Playing: {ep_dir_name} ({len(actions)} steps)", flush=True)
                
                # Check simulation still running
                if not simulation_app.is_running():
                    print("  [WARNING] Simulation stopped after loading episode", flush=True)
                    break
                
                # Load camera images for this episode if available
                camera_images = {"left_wrist": [], "ego": [], "right_wrist": []}
                if HAS_CV2:
                    for cam_name in camera_images.keys():
                        cam_dir = os.path.join(ep_path, cam_name)
                        if os.path.exists(cam_dir):
                            frame_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(".png")])
                            for ff in frame_files:
                                img = cv2.imread(os.path.join(cam_dir, ff))
                                if img is not None:
                                    camera_images[cam_name].append(img)
                    
                    # Report camera availability
                    for cam_name, imgs in camera_images.items():
                        if imgs:
                            print(f"  Camera {cam_name}: {len(imgs)} frames ({imgs[0].shape[1]}x{imgs[0].shape[0]})")
                        else:
                            print(f"  Camera {cam_name}: not found")
                    
                    # Create camera view window
                    cv2.namedWindow("Camera Views", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Camera Views", 1920, 480)
                
                # Reset environment
                print("  [INFO] Resetting environment...", flush=True)
                if not simulation_app.is_running():
                    print("  [WARNING] Simulation stopped before reset", flush=True)
                    break
                try:
                    obs, _ = env.reset()
                except Exception as e:
                    print(f"  [ERROR] Reset failed: {e}", flush=True)
                    break
                if not simulation_app.is_running():
                    print("  [WARNING] Simulation stopped after reset", flush=True)
                    break
                print("  [INFO] Environment reset complete", flush=True)
                skip_episode[0] = False
                print("  [DEBUG] Step 1: skip_episode set", flush=True)
                
                # Check simulation still running
                sim_running = simulation_app.is_running()
                print(f"  [DEBUG] Step 2: sim_running={sim_running}", flush=True)
                if not sim_running:
                    print("  [WARNING] Simulation stopped before initial conditions", flush=True)
                    break
                
                print("  [DEBUG] Step 3: about to apply initial conditions", flush=True)
                # Apply initial conditions if available
                print(f"  [INFO] Applying initial conditions (init_cond={'present' if init_cond else 'None'})...", flush=True)
                if init_cond is not None:
                    num_envs = unwrapped.num_envs
                    device = unwrapped.device
                    
                    # Set cube positions (old format: left_cube_pos/right_cube_pos)
                    if "left_cube_pos" in init_cond and "right_cube_pos" in init_cond:
                        try:
                            left_obj = unwrapped.scene["object_left"]
                            right_obj = unwrapped.scene["object_right"]
                            
                            left_pos = torch.tensor(init_cond["left_cube_pos"], device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                            left_quat = torch.tensor(init_cond.get("left_cube_quat", [1, 0, 0, 0]), device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                            right_pos = torch.tensor(init_cond["right_cube_pos"], device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                            right_quat = torch.tensor(init_cond.get("right_cube_quat", [1, 0, 0, 0]), device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                            
                            left_obj.write_root_pose_to_sim(torch.cat([left_pos, left_quat], dim=-1))
                            right_obj.write_root_pose_to_sim(torch.cat([right_pos, right_quat], dim=-1))
                            
                            zeros_vel = torch.zeros((num_envs, 6), device=device, dtype=torch.float32)
                            left_obj.write_root_velocity_to_sim(zeros_vel)
                            right_obj.write_root_velocity_to_sim(zeros_vel)
                            print("  [INFO] Cube positions set")
                        except KeyError:
                            pass
                    
                    # Spawn objects from teleop recording (new format: objects array)
                    if "objects" in init_cond and init_cond["objects"]:
                        import omni.usd
                        from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Usd
                        stage = omni.usd.get_context().get_stage()
                        
                        print(f"  [INFO] Found {len(init_cond['objects'])} objects to spawn", flush=True)
                        spawned_count = 0
                        updated_count = 0
                        
                        for obj in init_cond["objects"]:
                            print(f"  [DEBUG] Processing object: {obj.get('prim_path')} type={obj.get('type')}", flush=True)
                            prim_path = obj.get("prim_path", "")
                            pos = obj.get("position", [0, 0, 0])
                            quat = obj.get("orientation", [1, 0, 0, 0])  # [w, x, y, z]
                            scale = obj.get("scale", [1, 1, 1])
                            obj_type = obj.get("type", "cube")
                            usd_path = obj.get("usd_path", None)
                            
                            prim = stage.GetPrimAtPath(prim_path)
                            
                            if not prim.IsValid():
                                print(f"  [DEBUG] Prim doesn't exist, creating: {prim_path}", flush=True)
                                if obj_type == "cube":
                                    # Spawn a visual cube (no physics - kinematic playback)
                                    cube_prim = UsdGeom.Cube.Define(stage, prim_path)
                                    cube_prim.GetSizeAttr().Set(0.05)  # 5cm cube
                                    prim = cube_prim.GetPrim()
                                    
                                    # Set transform (no physics)
                                    xformable = UsdGeom.Xformable(prim)
                                    xformable.ClearXformOpOrder()
                                    translate_op = xformable.AddTranslateOp()
                                    translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                                    orient_op = xformable.AddOrientOp()
                                    orient_op.Set(Gf.Quatf(1, 0, 0, 0))
                                    scale_op = xformable.AddScaleOp()
                                    scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
                                    
                                    print(f"  [DEBUG] Created cube at {prim_path}", flush=True)
                                    spawned_count += 1
                                    
                                elif obj_type == "usd_reference" and usd_path:
                                    # Spawn USD reference (no physics - kinematic playback)
                                    xform = UsdGeom.Xform.Define(stage, prim_path)
                                    prim = xform.GetPrim()
                                    prim.GetReferences().AddReference(usd_path)
                                    
                                    # Set transform (no physics)
                                    xformable = UsdGeom.Xformable(prim)
                                    xformable.ClearXformOpOrder()
                                    translate_op = xformable.AddTranslateOp()
                                    translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                                    orient_op = xformable.AddOrientOp()
                                    orient_op.Set(Gf.Quatf(1, 0, 0, 0))
                                    scale_val = scale[0] if isinstance(scale, list) else scale
                                    scale_op = xformable.AddScaleOp()
                                    scale_op.Set(Gf.Vec3d(scale_val, scale_val, scale_val))
                                    
                                    spawned_count += 1
                            else:
                                updated_count += 1
                        
                        # Force physics to recognize new objects
                        try:
                            import omni.physx
                            physx_interface = omni.physx.get_physx_interface()
                            physx_interface.force_load_physics_from_usd()
                        except Exception:
                            pass
                        
                        print(f"  [INFO] Objects: {spawned_count} spawned, {updated_count} updated", flush=True)
                        
                        # Step physics a few times to let objects settle
                        for _ in range(5):
                            unwrapped.sim.step(render=True)
                    
                    # Set robot positions (supports both old 'robot_qpos' and new 'robot_joint_pos')
                    robot_qpos = init_cond.get("robot_qpos") or init_cond.get("robot_joint_pos")
                    robot_qvel = init_cond.get("robot_qvel") or init_cond.get("robot_joint_vel")
                    if robot_qpos is not None:
                        robot = unwrapped.scene["robot"]
                        qpos = torch.tensor(robot_qpos, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                        if robot_qvel is not None:
                            qvel = torch.tensor(robot_qvel, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                        else:
                            qvel = torch.zeros_like(qpos)
                        # Set both joint state AND position targets (so PD controller doesn't fight)
                        robot.write_joint_state_to_sim(qpos, qvel)
                        robot.set_joint_position_target(qpos)
                        robot.write_data_to_sim()
                        print("  [INFO] Robot positions set")
                    
                    # Step physics a few times to let initial conditions settle
                    for _ in range(10):
                        unwrapped.sim.step(render=True)
                
                robot = unwrapped.scene["robot"]
                num_robot_joints = robot.num_joints
                print(f"  [INFO] Action dim: {actions.shape[1]}, Robot joints: {num_robot_joints}")
                
                # Load spawn events (objects spawned during recording)
                spawn_events = init_cond.get("spawn_events", []) if init_cond else []
                if spawn_events:
                    print(f"  [INFO] {len(spawn_events)} spawn events to trigger during playback")
                
                # Build scale cache from initial conditions + spawn events
                # This is used for older recordings that don't have per-frame scale
                object_scales = {}
                if init_cond:
                    for obj in init_cond.get("objects", []):
                        prim_path = obj.get("prim_path", "")
                        scale = obj.get("scale", [0.01, 0.01, 0.01])  # Default small scale for cups/mugs
                        if prim_path:
                            object_scales[prim_path] = scale
                    for event in spawn_events:
                        prim_path = event.get("prim_path", "")
                        scale = event.get("scale", [0.01, 0.01, 0.01])
                        if prim_path:
                            object_scales[prim_path] = scale
                
                
                # Get stage for spawning
                import omni.usd
                from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Usd
                stage = omni.usd.get_context().get_stage()
                
                for step_idx, action in enumerate(actions):
                    if stop_playback[0] or skip_episode[0]:
                        break
                    
                    while pause_playback[0] and simulation_app.is_running():
                        time.sleep(0.1)
                    
                    if not simulation_app.is_running():
                        break
                    
                    # Check for spawn events at this frame
                    for event in spawn_events:
                        if event.get("frame") == step_idx:
                            prim_path = event.get("prim_path", "")
                            pos = event.get("position", [0, 0, 0])
                            quat = event.get("orientation", [1, 0, 0, 0])
                            scale = event.get("scale", [1, 1, 1])
                            obj_type = event.get("type", "cube")
                            usd_path = event.get("usd_path", None)
                            
                            # Check if this is a pool object
                            # Extract pool name from prim_path like /World/envs/env_0/PoolCube_0
                            pool_name = None
                            prim_name = prim_path.split("/")[-1] if "/" in prim_path else ""
                            if prim_name.startswith("Pool"):
                                import re
                                parts = re.findall(r'[A-Z][a-z]*|[0-9]+', prim_name)
                                if len(parts) >= 2:
                                    pool_name = f"{parts[0].lower()}_{parts[1].lower()}_{prim_name.split('_')[-1]}"
                            
                            if pool_name and pool_name in pool_objects:
                                # Use Isaac Lab API to teleport pool object
                                print(f"  [SPAWN] Frame {step_idx}: {pool_name} (pool object)")
                                try:
                                    asset = pool_objects[pool_name]
                                    pos_tensor = torch.tensor([[pos[0], pos[1], pos[2]]], device=asset.device)
                                    quat_tensor = torch.tensor([[quat[0], quat[1], quat[2], quat[3]]], device=asset.device)
                                    vel = torch.zeros((1, 6), device=asset.device)
                                    asset.write_root_pose_to_sim(torch.cat([pos_tensor, quat_tensor], dim=-1))
                                    asset.write_root_velocity_to_sim(vel)
                                except Exception as e:
                                    print(f"  [WARN] Failed to teleport pool object {pool_name}: {e}")
                            else:
                                # Fallback: create new prim dynamically
                                print(f"  [SPAWN] Frame {step_idx}: {prim_path} ({obj_type})")
                                
                                prim = stage.GetPrimAtPath(prim_path)
                                if not prim.IsValid():
                                    if obj_type == "cube":
                                        cube_prim = UsdGeom.Cube.Define(stage, prim_path)
                                        cube_prim.GetSizeAttr().Set(0.05)
                                        prim = cube_prim.GetPrim()
                                        xformable = UsdGeom.Xformable(prim)
                                        xformable.ClearXformOpOrder()
                                        translate_op = xformable.AddTranslateOp()
                                        translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                                        orient_op = xformable.AddOrientOp()
                                        orient_op.Set(Gf.Quatf(1, 0, 0, 0))
                                        scale_op = xformable.AddScaleOp()
                                        scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
                                    elif obj_type == "usd_reference" and usd_path:
                                        xform = UsdGeom.Xform.Define(stage, prim_path)
                                        prim = xform.GetPrim()
                                        prim.GetReferences().AddReference(usd_path)
                                        xformable = UsdGeom.Xformable(prim)
                                        xformable.ClearXformOpOrder()
                                        translate_op = xformable.AddTranslateOp()
                                        translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                                        orient_op = xformable.AddOrientOp()
                                        orient_op.Set(Gf.Quatf(1, 0, 0, 0))
                                        scale_val = scale[0] if isinstance(scale, list) else scale
                                        scale_op = xformable.AddScaleOp()
                                        scale_op.Set(Gf.Vec3d(scale_val, scale_val, scale_val))
                    
                    start_time = time.time()
                    
                    # Kinematic replay: set states directly instead of targets
                    # Use observation.state (recorded joint positions) for exact replay
                    if states is not None and step_idx < len(states):
                        state_tensor = torch.tensor(
                            states[step_idx], device=unwrapped.device, dtype=torch.float32
                        ).unsqueeze(0).expand(unwrapped.num_envs, -1)
                        qvel = torch.zeros_like(state_tensor)
                        robot.write_joint_state_to_sim(state_tensor, qvel)
                        robot.set_joint_position_target(state_tensor)  # Prevent PD fight
                    else:
                        # Fallback to action-based
                        action_tensor = torch.tensor(
                            action, device=unwrapped.device, dtype=torch.float32
                        ).unsqueeze(0).expand(unwrapped.num_envs, -1)
                        robot.set_joint_position_target(action_tensor)
                    
                    # Set object positions for this frame (kinematic replay)
                    if objects_per_frame is not None and step_idx < len(objects_per_frame):
                        frame_objects = objects_per_frame[step_idx]
                        for obj in frame_objects:
                            prim_path = obj.get("prim_path", "")
                            pos = obj.get("position", [0, 0, 0])
                            quat = obj.get("orientation", [1, 0, 0, 0])
                            
                            # Check if this is a pool object
                            # Extract pool name from paths like:
                            #   /World/envs/env_0/PoolCube_0 (new format)
                            #   /World/envs/env_.*/PoolCube_0 (old format with regex)
                            pool_name = None
                            # Get the last component (e.g., "PoolCube_0") and convert to pool key
                            prim_name = prim_path.split("/")[-1] if "/" in prim_path else ""
                            if prim_name.startswith("Pool"):
                                # Convert PoolCube_0 -> pool_cube_0
                                # Split on capital letters: Pool, Cube, _0
                                import re
                                parts = re.findall(r'[A-Z][a-z]*|[0-9]+', prim_name)
                                if len(parts) >= 2:
                                    pool_name = f"{parts[0].lower()}_{parts[1].lower()}_{prim_name.split('_')[-1]}"
                            
                            if pool_name and pool_name in pool_objects:
                                # Use Isaac Lab API to set position (kinematic)
                                try:
                                    asset = pool_objects[pool_name]
                                    pos_tensor = torch.tensor([[pos[0], pos[1], pos[2]]], device=asset.device)
                                    quat_tensor = torch.tensor([[quat[0], quat[1], quat[2], quat[3]]], device=asset.device)
                                    vel = torch.zeros((1, 6), device=asset.device)
                                    asset.write_root_pose_to_sim(torch.cat([pos_tensor, quat_tensor], dim=-1))
                                    asset.write_root_velocity_to_sim(vel)
                                except Exception as e:
                                    pass  # Silently skip errors
                            else:
                                # Fallback to USD xform for non-pool objects
                                # Convert regex pattern to actual path if needed
                                actual_path = prim_path.replace("env_.*", "env_0").replace("{ENV_REGEX_NS}", "/World/envs/env_0")
                                prim = stage.GetPrimAtPath(actual_path)
                                if prim.IsValid():
                                    try:
                                        xformable = UsdGeom.Xformable(prim)
                                        for op in xformable.GetOrderedXformOps():
                                            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                                op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                                            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                                                op.Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))
                                    except Exception:
                                        pass
                    
                    robot.write_data_to_sim()
                    unwrapped.sim.step(render=True)
                    
                    # Update camera view window
                    if HAS_CV2:
                        # Get images for this frame (or last available if fewer frames than actions)
                        left_img = camera_images["left_wrist"][min(step_idx, len(camera_images["left_wrist"]) - 1)] if camera_images["left_wrist"] else None
                        ego_img = camera_images["ego"][min(step_idx, len(camera_images["ego"]) - 1)] if camera_images["ego"] else None
                        right_img = camera_images["right_wrist"][min(step_idx, len(camera_images["right_wrist"]) - 1)] if camera_images["right_wrist"] else None
                        
                        # Create composite image: [left_wrist | ego | right_wrist]
                        target_height = 480
                        panels = []
                        
                        for img, label in [(left_img, "Left Wrist"), (ego_img, "Ego"), (right_img, "Right Wrist")]:
                            if img is not None:
                                # Resize to target height while maintaining aspect ratio
                                h, w = img.shape[:2]
                                scale = target_height / h
                                new_w = int(w * scale)
                                resized = cv2.resize(img, (new_w, target_height))
                                # Add label
                                cv2.putText(resized, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                cv2.putText(resized, f"Frame {step_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                                panels.append(resized)
                            else:
                                # Create placeholder
                                placeholder = np.zeros((target_height, 640, 3), dtype=np.uint8)
                                cv2.putText(placeholder, f"{label} - No Data", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                                panels.append(placeholder)
                        
                        # Concatenate horizontally
                        if panels:
                            composite = np.hstack(panels)
                            cv2.imshow("Camera Views", composite)
                            cv2.waitKey(1)  # Required to update window
                    
                    if step_idx % 50 == 0:
                        print(f"  Step {step_idx}/{len(actions)}", end="\r")
                    
                    if real_time:
                        sleep_time = dt - (time.time() - start_time)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                
                print(f"  {ep_dir_name} complete.          ")
                
                # Close camera window after episode
                if HAS_CV2:
                    cv2.destroyWindow("Camera Views")
              except Exception as e:
                print(f"  [ERROR] Episode processing failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
            
            if not loop or stop_playback[0]:
                break
            print("\n[INFO] Looping...")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_requested.set()
        if HAS_CV2:
            cv2.destroyAllWindows()
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
        env.close()
        simulation_app.close()


def replay_hdf5_data(data_dir: str, episode_idx: int = None, loop: bool = False, real_time: bool = False):
    """Replay legacy HDF5 format data in simulation."""
    # Delayed imports for simulation
    from isaaclab.app import AppLauncher
    import argparse
    
    args = argparse.Namespace(
        headless=False,
        enable_cameras=False,
        device="cuda:0",
        livestream=-1,
        experience="",
    )
    
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    import gymnasium as gym
    import h5py
    import json
    import numpy as np
    import signal
    import threading
    import time
    import torch
    
    import isaaclab_tasks  # noqa: F401
    import openarm.tasks  # noqa: F401
    
    # Load metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    task_name = metadata.get("task", "Isaac-Reach-OpenArm-Bi-Teleop-v0")
    # Convert to Play variant if needed (but Teleop tasks don't have Play variants)
    if "-Play" not in task_name and "-Teleop" not in task_name:
        task_name = task_name.replace("-v0", "-Play-v0")
    
    dt = metadata.get("dt", 0.02)
    task_text = metadata.get("task_text", metadata.get("task", ""))
    
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Task text: {task_text}")
    print(f"[INFO] dt: {dt}")
    
    # Find episode files
    episode_files = sorted([f for f in os.listdir(data_dir) if f.startswith("episode_") and f.endswith(".hdf5")])
    
    if episode_idx is not None:
        # Find episode by name (episode_N.hdf5) rather than list index
        target_name = f"episode_{episode_idx}.hdf5"
        if target_name in episode_files:
            episode_files = [target_name]
        elif episode_idx < len(episode_files):
            episode_files = [episode_files[episode_idx]]
        else:
            print(f"[ERROR] Episode {episode_idx} not found. Available: {episode_files}")
            simulation_app.close()
            return
    
    # Create environment
    env = gym.make(task_name)
    unwrapped = env.unwrapped
    
    # Keyboard controls
    import carb.input
    import omni.appwindow
    
    input_interface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    
    stop_playback = [False]
    skip_episode = [False]
    pause_playback = [False]
    
    def on_keyboard_event(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.Q:
                stop_playback[0] = True
                return True
            elif event.input == carb.input.KeyboardInput.N:
                skip_episode[0] = True
                return True
            elif event.input == carb.input.KeyboardInput.SPACE:
                pause_playback[0] = not pause_playback[0]
                print(f"[INFO] {'Paused' if pause_playback[0] else 'Resumed'}")
                return True
        return False
    
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    print("[INFO] Controls: Q=quit, N=next episode, SPACE=pause/resume")
    
    # Monitor thread
    stop_requested = threading.Event()
    
    def monitor_app():
        while not stop_requested.is_set():
            if not simulation_app.is_running():
                os.kill(os.getpid(), signal.SIGINT)
                break
            time.sleep(0.5)
    
    monitor_thread = threading.Thread(target=monitor_app, daemon=True)
    monitor_thread.start()
    
    try:
        while True:
            for ep_file in episode_files:
                if stop_playback[0]:
                    break
                
                ep_path = os.path.join(data_dir, ep_file)
                
                with h5py.File(ep_path, "r") as f:
                    actions = f["action"][:]
                    qpos = f["observations/qpos"][:] if "observations/qpos" in f else None
                    left_cube_pos = f["left_cube_start_pos"][:] if "left_cube_start_pos" in f else None
                    right_cube_pos = f["right_cube_start_pos"][:] if "right_cube_start_pos" in f else None
                    left_cube_quat = f["left_cube_start_quat"][:] if "left_cube_start_quat" in f else None
                    right_cube_quat = f["right_cube_start_quat"][:] if "right_cube_start_quat" in f else None
                    robot_start_qpos = f["robot_start_qpos"][:] if "robot_start_qpos" in f else None
                    robot_start_qvel = f["robot_start_qvel"][:] if "robot_start_qvel" in f else None
                
                print(f"\n[INFO] Playing: {ep_file} ({len(actions)} steps)")
                if left_cube_pos is not None:
                    print(f"  Left cube: [{left_cube_pos[0]:.3f}, {left_cube_pos[1]:.3f}, {left_cube_pos[2]:.3f}]")
                if right_cube_pos is not None:
                    print(f"  Right cube: [{right_cube_pos[0]:.3f}, {right_cube_pos[1]:.3f}, {right_cube_pos[2]:.3f}]")
                
                # Reset environment
                obs, _ = env.reset()
                skip_episode[0] = False
                
                # Set cube positions to match the recorded episode
                if left_cube_pos is not None and right_cube_pos is not None:
                    # Get objects from scene
                    left_obj = unwrapped.scene["object_left"]
                    right_obj = unwrapped.scene["object_right"]
                    
                    # Prepare pose tensors - broadcast to all envs
                    num_envs = unwrapped.num_envs
                    device = unwrapped.device
                    
                    # Left cube pose
                    left_pos_tensor = torch.tensor(left_cube_pos, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                    left_quat_tensor = torch.tensor(left_cube_quat if left_cube_quat is not None else [1, 0, 0, 0], device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                    
                    # Right cube pose
                    right_pos_tensor = torch.tensor(right_cube_pos, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                    right_quat_tensor = torch.tensor(right_cube_quat if right_cube_quat is not None else [1, 0, 0, 0], device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                    
                    # Write poses to simulation
                    left_obj.write_root_pose_to_sim(torch.cat([left_pos_tensor, left_quat_tensor], dim=-1))
                    right_obj.write_root_pose_to_sim(torch.cat([right_pos_tensor, right_quat_tensor], dim=-1))
                    
                    # Zero out velocities
                    zeros_vel = torch.zeros((num_envs, 6), device=device, dtype=torch.float32)
                    left_obj.write_root_velocity_to_sim(zeros_vel)
                    right_obj.write_root_velocity_to_sim(zeros_vel)
                    
                    print("  [INFO] Cube positions set to recorded values")
                
                # Set robot initial joint positions
                if robot_start_qpos is not None:
                    robot = unwrapped.scene["robot"]
                    num_envs = unwrapped.num_envs
                    device = unwrapped.device
                    
                    # Prepare joint position tensor - broadcast to all envs
                    qpos_tensor = torch.tensor(robot_start_qpos, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                    
                    # Prepare joint velocity tensor (zero or from recording)
                    if robot_start_qvel is not None:
                        qvel_tensor = torch.tensor(robot_start_qvel, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
                    else:
                        qvel_tensor = torch.zeros_like(qpos_tensor)
                    
                    # Write joint state to simulation
                    robot.write_joint_state_to_sim(qpos_tensor, qvel_tensor)
                    
                    print("  [INFO] Robot joint positions set to recorded values")
                
                for step_idx, action in enumerate(actions):
                    if stop_playback[0] or skip_episode[0]:
                        break
                    
                    while pause_playback[0] and simulation_app.is_running():
                        time.sleep(0.1)
                    
                    if not simulation_app.is_running():
                        break
                    
                    start_time = time.time()
                    
                    # Apply joint positions directly (like teleop does)
                    robot = unwrapped.scene["robot"]
                    action_tensor = torch.tensor(
                        action, device=unwrapped.device, dtype=torch.float32
                    ).unsqueeze(0).expand(unwrapped.num_envs, -1)
                    
                    # Set joint position targets and step physics
                    robot.set_joint_position_target(action_tensor)
                    robot.write_data_to_sim()
                    unwrapped.sim.step(render=True)
                    
                    if step_idx % 50 == 0:
                        print(f"  Step {step_idx}/{len(actions)}", end="\r")
                    
                    if real_time:
                        sleep_time = dt - (time.time() - start_time)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                
                print(f"  {ep_file} complete.          ")
            
            if not loop or stop_playback[0]:
                break
            print("\n[INFO] Looping...")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_requested.set()
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
        env.close()
        simulation_app.close()


def main():
    parser = argparse.ArgumentParser(description="Verify or replay VLA training data for Pi 0.5")
    parser.add_argument("data_dir", type=str, help="Path to VLA training data directory")
    parser.add_argument("--verify", action="store_true", help="Verify data (show images and stats)")
    parser.add_argument("--replay", action="store_true", help="Replay data in simulation")
    parser.add_argument("--episode", type=int, default=None, help="Specific episode index")
    parser.add_argument("--loop", action="store_true", help="Loop playback (replay mode)")
    parser.add_argument("--real-time", action="store_true", help="Real-time playback (replay mode)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Not a directory: {args.data_dir}")
        sys.exit(1)
    
    if args.verify:
        verify_data(args.data_dir, args.episode)
    elif args.replay:
        replay_data(args.data_dir, args.episode, args.loop, args.real_time)
    else:
        # Default to verify
        print("[INFO] No mode specified, running --verify")
        verify_data(args.data_dir, args.episode)


if __name__ == "__main__":
    main()
