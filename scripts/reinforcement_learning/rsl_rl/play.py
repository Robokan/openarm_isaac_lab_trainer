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

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--no-self-collisions", action="store_true", default=False,
    help="Disable self-collision detection for the robot (faster but less realistic)."
)
parser.add_argument(
    "--create-vla-training-data", type=str, default=None, metavar="FILENAME",
    help="Capture VLA training data (cube positions and motor commands) to the specified file."
)
parser.add_argument(
    "--vla-max-episodes", type=int, default=100,
    help="Maximum number of episodes to capture for VLA training data (default: 100)."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import signal
import threading
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Disable self-collisions if requested
    if args_cli.no_self_collisions:
        if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'robot'):
            if hasattr(env_cfg.scene.robot, 'spawn') and hasattr(env_cfg.scene.robot.spawn, 'articulation_props'):
                env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
                print("[INFO] Self-collisions disabled via --no-self-collisions flag")

    # create isaac environment
    # Enable rgb_array render mode for video recording (--video flag or interactive V key recording)
    # When not headless, always enable render_mode to support interactive video capture
    use_render_mode = args_cli.video or not args_cli.headless
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if use_render_mode else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    def load_checkpoint(path):
        """Load checkpoint and return policy."""
        print(f"[INFO]: Loading model checkpoint from: {path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        return runner, policy

    # Initial checkpoint load
    runner, policy = load_checkpoint(resume_path)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Set up keyboard input for checkpoint reload
    import carb.input
    import omni.appwindow
    
    input_interface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    reload_requested = [False]  # Use list to allow modification in callback
    reset_requested = [False]
    toggle_markers_requested = [False]
    markers_visible = [True]  # Track marker visibility state
    toggle_video_requested = [False]  # Video recording toggle
    
    def on_keyboard_event(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.L:
                reload_requested[0] = True
                return True  # Consume event to prevent UI from handling it
            elif event.input == carb.input.KeyboardInput.R:
                reset_requested[0] = True
                return True  # Consume event to prevent UI from handling it
            elif event.input == carb.input.KeyboardInput.M:
                toggle_markers_requested[0] = True
                return True  # Consume event to prevent Isaac Sim UI from handling it
            elif event.input == carb.input.KeyboardInput.V or event.input == carb.input.KeyboardInput.KEY_0:
                toggle_video_requested[0] = True
                print("[DEBUG] Video key pressed - video toggle requested")
                return True  # Consume event
        return False
    
    # Also subscribe to KEY_RELEASE to ensure we catch the event
    def on_keyboard_event_release(event, *args, **kwargs):
        # Some systems need release event handling
        return False
    
    def toggle_all_markers(env, visible: bool):
        """Toggle visibility of all debug markers in the environment."""
        unwrapped = env.unwrapped
        
        # Toggle command visualizers (e.g., object_pose target marker)
        if hasattr(unwrapped, 'command_manager'):
            for term_name, term in unwrapped.command_manager._terms.items():
                if hasattr(term, 'cfg') and hasattr(term.cfg, 'debug_vis'):
                    if hasattr(term, 'set_debug_vis'):
                        term.set_debug_vis(visible)
                    elif hasattr(term, '_debug_vis_handle'):
                        term._debug_vis_handle.set_visibility(visible)
        
        # Toggle frame transformer visualizers (EE frames)
        if hasattr(unwrapped, 'scene'):
            for name in ['left_ee_frame', 'right_ee_frame', 'ee_frame']:
                if name in unwrapped.scene._sensors:
                    sensor = unwrapped.scene._sensors[name]
                    if hasattr(sensor, 'set_debug_vis'):
                        sensor.set_debug_vis(visible)
                    elif hasattr(sensor, 'visualizer') and sensor.visualizer is not None:
                        sensor.visualizer.set_visibility(visible)
        
        print(f"[INFO] Markers {'shown' if visible else 'hidden'}")
    
    # Video recording state
    video_recording = [False]
    video_frames = []
    video_start_time = [None]
    # Save videos to top-level 'video' directory in the repository
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    video_output_dir = os.path.join(repo_root, "video")
    os.makedirs(video_output_dir, exist_ok=True)
    
    def start_video_recording():
        """Start capturing video frames."""
        video_frames.clear()
        video_start_time[0] = time.strftime("%Y%m%d_%H%M%S")
        video_recording[0] = True
        print(f"[VIDEO] Recording started...")
    
    def stop_video_recording():
        """Stop recording and save video to MP4."""
        video_recording[0] = False
        if len(video_frames) == 0:
            print("[VIDEO] No frames captured.")
            return
        
        # Save as MP4 using imageio
        try:
            import imageio
            output_path = os.path.join(video_output_dir, f"recording_{video_start_time[0]}.mp4")
            fps = int(1.0 / dt) if dt > 0 else 30
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
            for frame in video_frames:
                writer.append_data(frame)
            writer.close()
            print(f"[VIDEO] Saved {len(video_frames)} frames to {output_path}")
        except ImportError:
            # Fallback: save as individual frames
            print("[VIDEO] imageio not available, saving as PNG frames...")
            frames_dir = os.path.join(video_output_dir, f"frames_{video_start_time[0]}")
            os.makedirs(frames_dir, exist_ok=True)
            import numpy as np
            from PIL import Image
            for i, frame in enumerate(video_frames):
                img = Image.fromarray(frame)
                img.save(os.path.join(frames_dir, f"frame_{i:06d}.png"))
            print(f"[VIDEO] Saved {len(video_frames)} frames to {frames_dir}/")
        except Exception as e:
            print(f"[VIDEO] Error saving video: {e}")
        finally:
            video_frames.clear()
    
    def capture_frame():
        """Capture current frame using env.render() - the Isaac Lab way."""
        try:
            import numpy as np
            
            # Get the underlying gym env (unwrap from RslRlVecEnvWrapper)
            gym_env = env.unwrapped
            
            # Use the render method which returns RGB array
            frame = gym_env.render()
            
            if frame is not None:
                # Ensure it's a numpy array
                if hasattr(frame, 'cpu'):
                    frame = frame.cpu().numpy()
                elif not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                
                # Handle different frame formats
                if len(frame.shape) == 4:
                    # Batched frames (num_envs, H, W, C) - take first env
                    frame = frame[0]
                
                # Ensure uint8
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                # Convert RGBA to RGB if needed
                if len(frame.shape) == 3 and frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                
                video_frames.append(frame.copy())
                
                # Print progress every 100 frames
                if len(video_frames) % 100 == 0:
                    print(f"[VIDEO] Captured {len(video_frames)} frames...")
            elif len(video_frames) == 0:
                print("[VIDEO] Warning: render() returned None - ensure env was created with render_mode='rgb_array'")
                
        except Exception as e:
            if len(video_frames) == 0:
                print(f"[VIDEO] Error capturing frame: {e}")
    
    # Subscribe to keyboard events with high priority (order=0) to capture before Isaac Sim UI
    # Lower order values have higher priority
    try:
        keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event, order=0)
    except TypeError:
        # Fallback if order parameter not supported
        keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    # Track previous key states for edge detection (for keys that might not trigger events)
    prev_key_v = [False]
    prev_key_0 = [False]
    
    print("[INFO] Press 'L' to reload checkpoint, 'R' to reset environment, 'M' to toggle markers, 'V' or '0' to toggle video recording, close window to exit")

    # Background thread to monitor for window close (non-headless mode)
    # This allows graceful exit when the user closes the visualization window
    stop_requested = threading.Event()
    
    def monitor_app():
        """Monitor simulation app and signal stop when window is closed."""
        while not stop_requested.is_set():
            if not simulation_app.is_running():
                print("\n[INFO] Window closed. Exiting...")
                os.kill(os.getpid(), signal.SIGINT)
                break
            time.sleep(0.5)
    
    if not args_cli.headless:
        monitor_thread = threading.Thread(target=monitor_app, daemon=True)
        monitor_thread.start()

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # VLA training data capture for Pi 0.5 (LeRobot-compatible format)
    vla_data = None
    vla_current_episode = None
    vla_episode_count = 0
    vla_prev_episode_lengths = None
    vla_output_dir = None
    vla_task_text = "pick up cubes"
    
    if args_cli.create_vla_training_data:
        import h5py
        import numpy as np
        
        vla_output_dir = args_cli.create_vla_training_data
        os.makedirs(vla_output_dir, exist_ok=True)
        
        print(f"[VLA] Pi 0.5 training data capture enabled.")
        print(f"[VLA] Output directory: {vla_output_dir}")
        print(f"[VLA] Task text: '{vla_task_text}'")
        print(f"[VLA] Will capture up to {args_cli.vla_max_episodes} episodes.")
        print(f"[VLA] Cameras: ego_cam, left_wrist_cam, right_wrist_cam")
        
        # Initialize camera access (cameras exist in USD file)
        unwrapped = env.unwrapped
        print("[VLA] Cameras will be accessed via Isaac Sim camera API (Ego, LeftArm, RightArm)")
        
        vla_data = {
            "metadata": {
                "task": args_cli.task,
                "task_text": vla_task_text,
                "checkpoint": resume_path,
                "dt": dt,
                "fps": int(1.0 / dt) if dt > 0 else 50,
                "num_envs": env_cfg.scene.num_envs,
                "robot_type": "openarm_bimanual",
                "cameras": ["ego_cam", "left_wrist_cam", "right_wrist_cam"],
            },
            "episode_count": 0,
        }
        
        # Save metadata
        with open(os.path.join(vla_output_dir, "metadata.json"), "w") as f:
            json.dump(vla_data["metadata"], f, indent=2)
        
        # Initialize current episode data with cube starting positions
        import numpy as np
        robot = unwrapped.scene["robot"]
        vla_current_episode = {
            "left_cube_start_pos": unwrapped.scene["object_left"].data.root_pos_w[0].cpu().numpy().copy(),
            "right_cube_start_pos": unwrapped.scene["object_right"].data.root_pos_w[0].cpu().numpy().copy(),
            "left_cube_start_quat": unwrapped.scene["object_left"].data.root_quat_w[0].cpu().numpy().copy(),
            "right_cube_start_quat": unwrapped.scene["object_right"].data.root_quat_w[0].cpu().numpy().copy(),
            "robot_start_qpos": robot.data.joint_pos[0].cpu().numpy().copy(),
            "robot_start_qvel": robot.data.joint_vel[0].cpu().numpy().copy(),
            "observations": {
                "qpos": [],  # Joint positions
                "qvel": [],  # Joint velocities
                "images": {
                    "ego_cam": [],
                    "left_wrist_cam": [],
                    "right_wrist_cam": [],
                },
            },
            "actions": [],
        }
        vla_prev_episode_lengths = unwrapped.episode_length_buf.clone()
    
    # simulate environment with proper cleanup on exit
    try:
        while simulation_app.is_running():
            start_time = time.time()
            
            # Check for checkpoint reload request
            if reload_requested[0]:
                reload_requested[0] = False
                print("\n[INFO] Reloading checkpoint...")
                runner, policy = load_checkpoint(resume_path)
                print("[INFO] Checkpoint reloaded successfully!")
            
            # Check for marker toggle request
            if toggle_markers_requested[0]:
                toggle_markers_requested[0] = False
                markers_visible[0] = not markers_visible[0]
                toggle_all_markers(env, markers_visible[0])
            
            # Poll keyboard state directly as backup for keys that might not trigger events
            try:
                key_v_down = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.V)
                key_0_down = input_interface.get_keyboard_value(keyboard, carb.input.KeyboardInput.KEY_0)
                
                # Detect rising edge (key just pressed)
                if key_v_down and not prev_key_v[0]:
                    toggle_video_requested[0] = True
                    print("[DEBUG] V key detected via polling")
                if key_0_down and not prev_key_0[0]:
                    toggle_video_requested[0] = True
                    print("[DEBUG] 0 key detected via polling")
                
                prev_key_v[0] = key_v_down
                prev_key_0[0] = key_0_down
            except Exception:
                pass  # Polling not supported, rely on event subscription
            
            # Check for video recording toggle
            if toggle_video_requested[0]:
                print("[DEBUG] Processing video toggle request...")
                toggle_video_requested[0] = False
                if video_recording[0]:
                    stop_video_recording()
                else:
                    start_video_recording()
            
            # Capture video frame if recording
            if video_recording[0]:
                capture_frame()
            
            # Check for environment reset request (set flag for next step)
            force_reset = reset_requested[0]
            if force_reset:
                reset_requested[0] = False
                print("\n[INFO] Resetting environment...")
                # Set episode length high to trigger timeout on next step
                with torch.no_grad():
                    env.unwrapped.episode_length_buf[:] = 999999
            
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
            
            # VLA: Capture observations and actions for Pi 0.5 training
            if vla_data is not None and vla_current_episode is not None:
                import numpy as np
                unwrapped = env.unwrapped
                
                # Get current joint positions and velocities
                robot = unwrapped.scene["robot"]
                joint_pos = robot.data.joint_pos[0].cpu().numpy()
                joint_vel = robot.data.joint_vel[0].cpu().numpy()
                
                # Store proprioceptive state
                vla_current_episode["observations"]["qpos"].append(joint_pos.copy())
                vla_current_episode["observations"]["qvel"].append(joint_vel.copy())
                
                # Store actions
                vla_current_episode["actions"].append(actions[0].cpu().numpy().copy())
                
                # Capture camera images via Isaac Sim camera API
                try:
                    from omni.isaac.sensor import Camera
                    import omni.isaac.core.utils.prims as prim_utils
                    
                    # Camera name mapping: internal name -> USD camera name
                    camera_mapping = {
                        "ego_cam": "Ego",
                        "left_wrist_cam": "LeftArm",
                        "right_wrist_cam": "RightArm",
                    }
                    
                    for internal_name, usd_cam_name in camera_mapping.items():
                        try:
                            # Get camera prim path for env 0
                            cam_prim_path = f"/World/envs/env_0/Robot/{usd_cam_name}"
                            
                            # Get camera and capture frame
                            camera = Camera(prim_path=cam_prim_path)
                            camera.initialize()
                            img = camera.get_rgba()
                            
                            if img is not None:
                                # Convert to (C, H, W) format for LeRobot compatibility
                                img = img[:, :, :3]  # Remove alpha
                                img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                                vla_current_episode["observations"]["images"][internal_name].append(img.astype(np.uint8))
                        except Exception as cam_err:
                            if len(vla_current_episode["observations"]["qpos"]) == 1:
                                print(f"[VLA] Camera {usd_cam_name} error: {cam_err}")
                except Exception as e:
                    if len(vla_current_episode["observations"]["qpos"]) == 1:
                        print(f"[VLA] Camera capture error: {e}")
            
            # env stepping (outside inference mode to allow reset)
            obs, _, _, _ = env.step(actions)
            
            # VLA: Check for episode reset (episode_length_buf goes to 0)
            if vla_data is not None:
                import numpy as np
                unwrapped = env.unwrapped
                current_episode_lengths = unwrapped.episode_length_buf
                # Detect reset: current length is less than previous (wrapped around)
                reset_mask = current_episode_lengths < vla_prev_episode_lengths
                
                if reset_mask[0].item():  # Episode 0 reset
                    # Save completed episode as HDF5 file (LeRobot/Aloha compatible format)
                    num_steps = len(vla_current_episode["observations"]["qpos"])
                    if num_steps > 0:
                        import h5py
                        ep_path = os.path.join(vla_output_dir, f"episode_{vla_episode_count}.hdf5")
                        
                        with h5py.File(ep_path, "w") as f:
                            # Store initial cube positions
                            f.create_dataset("left_cube_start_pos", data=vla_current_episode["left_cube_start_pos"])
                            f.create_dataset("right_cube_start_pos", data=vla_current_episode["right_cube_start_pos"])
                            f.create_dataset("left_cube_start_quat", data=vla_current_episode["left_cube_start_quat"])
                            f.create_dataset("right_cube_start_quat", data=vla_current_episode["right_cube_start_quat"])
                            
                            # Store initial robot joint positions
                            f.create_dataset("robot_start_qpos", data=vla_current_episode["robot_start_qpos"])
                            f.create_dataset("robot_start_qvel", data=vla_current_episode["robot_start_qvel"])
                            
                            # Store observations
                            obs_grp = f.create_group("observations")
                            obs_grp.create_dataset("qpos", data=np.array(vla_current_episode["observations"]["qpos"]))
                            obs_grp.create_dataset("qvel", data=np.array(vla_current_episode["observations"]["qvel"]))
                            
                            # Store camera images
                            img_grp = obs_grp.create_group("images")
                            for cam_name, images in vla_current_episode["observations"]["images"].items():
                                if len(images) > 0:
                                    img_grp.create_dataset(cam_name, data=np.array(images), compression="gzip")
                            
                            # Store actions
                            f.create_dataset("action", data=np.array(vla_current_episode["actions"]))
                            
                            # Store metadata
                            f.attrs["task"] = vla_task_text
                            f.attrs["num_steps"] = num_steps
                        
                        vla_episode_count += 1
                        print(f"[VLA] Episode {vla_episode_count} saved: {ep_path} ({num_steps} steps)")
                        
                        # Check if we've captured enough episodes
                        if vla_episode_count >= args_cli.vla_max_episodes:
                            print(f"[VLA] Captured {vla_episode_count} episodes. Stopping...")
                            break
                    
                    # Start new episode with cube starting positions
                    vla_current_episode = {
                        "left_cube_start_pos": unwrapped.scene["object_left"].data.root_pos_w[0].cpu().numpy().copy(),
                        "right_cube_start_pos": unwrapped.scene["object_right"].data.root_pos_w[0].cpu().numpy().copy(),
                        "left_cube_start_quat": unwrapped.scene["object_left"].data.root_quat_w[0].cpu().numpy().copy(),
                        "right_cube_start_quat": unwrapped.scene["object_right"].data.root_quat_w[0].cpu().numpy().copy(),
                        "robot_start_qpos": robot.data.joint_pos[0].cpu().numpy().copy(),
                        "robot_start_qvel": robot.data.joint_vel[0].cpu().numpy().copy(),
                        "observations": {
                            "qpos": [],
                            "qvel": [],
                            "images": {
                                "ego_cam": [],
                                "left_wrist_cam": [],
                                "right_wrist_cam": [],
                            },
                        },
                        "actions": [],
                    }
                
                vla_prev_episode_lengths = current_episode_lengths.clone()
            
            if force_reset:
                obs = env.get_observations()
                print("[INFO] Environment reset!")
            
            # Debug: Print phase and distance to lift targets (every 50 steps to avoid spam)
            if timestep % 50 == 0:
                import sys
                print(f"[DEBUG] Step {timestep}, headless={args_cli.headless}", flush=True)
                unwrapped = env.unwrapped
                # Get lift target distances and phase info
                left_target = torch.tensor([0.0, 0.2, 0.355], device=unwrapped.device)
                right_target = torch.tensor([0.0, -0.2, 0.355], device=unwrapped.device)
                
                try:
                    left_ee = unwrapped.scene["left_ee_frame"].data.target_pos_w[:, 0, :]
                    right_ee = unwrapped.scene["right_ee_frame"].data.target_pos_w[:, 0, :]
                    
                    left_dist = torch.norm(left_ee - left_target, dim=-1)[0].item()
                    right_dist = torch.norm(right_ee - right_target, dim=-1)[0].item()
                    
                    # Check phase (one-way gate)
                    if hasattr(unwrapped, '_arm_reached_lift_target'):
                        left_phase2 = unwrapped._arm_reached_lift_target[0, 0].item()
                        right_phase2 = unwrapped._arm_reached_lift_target[0, 1].item()
                    else:
                        left_phase2 = False
                        right_phase2 = False
                    
                    phase = "PHASE 2 (reaching)" if (left_phase2 and right_phase2) else "PHASE 1 (lifting)"
                    print(f"[DEBUG] {phase} | Left dist: {left_dist:.3f}m ({'✓' if left_phase2 else '○'}) | Right dist: {right_dist:.3f}m ({'✓' if right_phase2 else '○'})", flush=True)
                except Exception as e:
                    print(f"[DEBUG ERROR] {type(e).__name__}: {e}", flush=True)
            
            timestep += 1
            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Closing...")
    finally:
        # Save video recording if in progress
        if video_recording[0]:
            stop_video_recording()
        
        # Save VLA training data if capturing
        if vla_data is not None and vla_output_dir is not None:
            import numpy as np
            import h5py
            
            # Save any in-progress episode
            num_steps = len(vla_current_episode["observations"]["qpos"]) if vla_current_episode else 0
            if num_steps > 0:
                ep_path = os.path.join(vla_output_dir, f"episode_{vla_episode_count}.hdf5")
                
                with h5py.File(ep_path, "w") as f:
                    # Store initial cube positions
                    f.create_dataset("left_cube_start_pos", data=vla_current_episode["left_cube_start_pos"])
                    f.create_dataset("right_cube_start_pos", data=vla_current_episode["right_cube_start_pos"])
                    f.create_dataset("left_cube_start_quat", data=vla_current_episode["left_cube_start_quat"])
                    f.create_dataset("right_cube_start_quat", data=vla_current_episode["right_cube_start_quat"])
                    
                    # Store initial robot joint positions
                    f.create_dataset("robot_start_qpos", data=vla_current_episode["robot_start_qpos"])
                    f.create_dataset("robot_start_qvel", data=vla_current_episode["robot_start_qvel"])
                    
                    # Store observations
                    obs_grp = f.create_group("observations")
                    obs_grp.create_dataset("qpos", data=np.array(vla_current_episode["observations"]["qpos"]))
                    obs_grp.create_dataset("qvel", data=np.array(vla_current_episode["observations"]["qvel"]))
                    
                    # Store camera images
                    img_grp = obs_grp.create_group("images")
                    for cam_name, images in vla_current_episode["observations"]["images"].items():
                        if len(images) > 0:
                            img_grp.create_dataset(cam_name, data=np.array(images), compression="gzip")
                    
                    # Store actions
                    f.create_dataset("action", data=np.array(vla_current_episode["actions"]))
                    
                    # Store metadata
                    f.attrs["task"] = vla_task_text
                    f.attrs["num_steps"] = num_steps
                
                vla_episode_count += 1
                print(f"[VLA] Final episode saved: {ep_path} ({num_steps} steps)")
            
            # Update metadata with final episode count
            if vla_episode_count > 0:
                vla_data["metadata"]["total_episodes"] = vla_episode_count
                with open(os.path.join(vla_output_dir, "metadata.json"), "w") as f:
                    json.dump(vla_data["metadata"], f, indent=2)
                print(f"[VLA] Saved {vla_episode_count} episodes to {vla_output_dir}/")
            else:
                print("[VLA] No episodes captured.")
        
        # Signal monitor thread to stop
        stop_requested.set()
        # Unsubscribe from keyboard events
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
