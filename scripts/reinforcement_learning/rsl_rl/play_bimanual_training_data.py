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

This script reads the HDF5 episode files created by --create-vla-training-data
and can either:
1. Display camera images and stats (--verify mode, no simulation required)
2. Replay joint positions in simulation (--replay mode)

Usage:
    # Verify data (show images and stats, no simulation):
    python play_bimanual_training_data.py <data_dir> --verify
    
    # Replay in simulation:
    python play_bimanual_training_data.py <data_dir> --replay [--episode N] [--loop]
"""

import argparse
import os
import sys


def verify_data(data_dir: str, episode_idx: int = None):
    """Verify VLA training data by displaying images and stats."""
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
    # Delayed imports for simulation
    from isaaclab.app import AppLauncher
    
    # Create a minimal arg namespace for AppLauncher
    class Args:
        headless = False
        enable_cameras = False
        device = "cuda:0"
    
    app_launcher = AppLauncher(Args())
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
    
    task_name = metadata.get("task", "Isaac-Lift-Cube-OpenArm-Bi-Play-v0")
    if "-Play" not in task_name:
        task_name = task_name.replace("-v0", "-Play-v0")
    
    dt = metadata.get("dt", 0.02)
    
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] dt: {dt}")
    
    # Find episode files
    episode_files = sorted([f for f in os.listdir(data_dir) if f.startswith("episode_") and f.endswith(".hdf5")])
    
    if episode_idx is not None:
        episode_files = [episode_files[episode_idx]]
    
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
                    
                    # Convert to tensor
                    action_tensor = torch.tensor(
                        [action] * unwrapped.num_envs,
                        device=unwrapped.device,
                        dtype=torch.float32
                    )
                    
                    obs, _, _, _, _ = env.step(action_tensor)
                    
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
