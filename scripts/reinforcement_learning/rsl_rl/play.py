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
        import numpy as np
        from PIL import Image
        
        # Try to import LeRobot for native format support
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            HAS_LEROBOT = True
            print("[VLA] LeRobot found - will save in native LeRobot v3.0 format")
        except ImportError:
            HAS_LEROBOT = False
            print("[VLA] LeRobot not found - will save in LeRobot-compatible format")
            print("[VLA] Install LeRobot to enable native format: pip install lerobot")
        
        vla_output_dir = args_cli.create_vla_training_data
        os.makedirs(vla_output_dir, exist_ok=True)
        
        # Get robot info for feature definitions
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        num_joints = robot.data.joint_pos.shape[1]
        fps = int(1.0 / dt) if dt > 0 else 50
        
        print(f"[VLA] Pi 0.5 training data capture enabled.")
        print(f"[VLA] Output directory: {vla_output_dir}")
        print(f"[VLA] Task text: '{vla_task_text}'")
        print(f"[VLA] Will capture up to {args_cli.vla_max_episodes} episodes.")
        print(f"[VLA] Robot joints: {num_joints}, FPS: {fps}")
        print(f"[VLA] Cameras: observation.images.ego, observation.images.left_wrist, observation.images.right_wrist")
        
        if HAS_LEROBOT:
            # Create LeRobot dataset with proper feature definitions
            vla_dataset = LeRobotDataset.create(
                repo_id=f"local/{os.path.basename(vla_output_dir)}",
                root=vla_output_dir,
                robot_type="openarm_bimanual",
                fps=fps,
                features={
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (num_joints,),
                        "names": None,
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (num_joints,),
                        "names": None,
                    },
                    "observation.images.ego": {
                        "dtype": "video",
                        "shape": (3, 480, 640),  # Will be determined from actual images
                        "names": ["channel", "height", "width"],
                    },
                    "observation.images.left_wrist": {
                        "dtype": "video",
                        "shape": (3, 480, 640),
                        "names": ["channel", "height", "width"],
                    },
                    "observation.images.right_wrist": {
                        "dtype": "video",
                        "shape": (3, 480, 640),
                        "names": ["channel", "height", "width"],
                    },
                },
            )
            vla_data = {"dataset": vla_dataset, "episode_count": 0}
        else:
            # Fallback: Save in a format that can be converted later
            vla_data = {
                "metadata": {
                    "task": args_cli.task,
                    "task_text": vla_task_text,
                    "checkpoint": resume_path,
                    "dt": dt,
                    "fps": fps,
                    "num_envs": env_cfg.scene.num_envs,
                    "robot_type": "openarm_bimanual",
                    "num_joints": num_joints,
                    "cameras": ["ego", "left_wrist", "right_wrist"],
                },
                "episode_count": 0,
            }
            # Save metadata
            with open(os.path.join(vla_output_dir, "metadata.json"), "w") as f:
                json.dump(vla_data["metadata"], f, indent=2)
            
            # Create directories for episode data
            os.makedirs(os.path.join(vla_output_dir, "episodes"), exist_ok=True)
        
        # Initialize current episode frame buffer
        # Capture initial conditions for this episode
        robot = unwrapped.scene["robot"]
        vla_current_episode = {
            "frames": [],  # List of frame dicts for LeRobot
            # Initial conditions for replay
            "initial_conditions": {
                "left_cube_pos": unwrapped.scene["object_left"].data.root_pos_w[0].cpu().numpy().copy(),
                "right_cube_pos": unwrapped.scene["object_right"].data.root_pos_w[0].cpu().numpy().copy(),
                "left_cube_quat": unwrapped.scene["object_left"].data.root_quat_w[0].cpu().numpy().copy(),
                "right_cube_quat": unwrapped.scene["object_right"].data.root_quat_w[0].cpu().numpy().copy(),
                "robot_qpos": robot.data.joint_pos[0].cpu().numpy().copy(),
                "robot_qvel": robot.data.joint_vel[0].cpu().numpy().copy(),
            },
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
                from PIL import Image
                unwrapped = env.unwrapped
                
                # Get current joint positions (state)
                robot = unwrapped.scene["robot"]
                joint_pos = robot.data.joint_pos[0].cpu().numpy().astype(np.float32)
                action_np = actions[0].cpu().numpy().astype(np.float32)
                
                # Build frame dict for LeRobot format
                frame = {
                    "observation.state": joint_pos.copy(),
                    "action": action_np.copy(),
                }
                
                # Capture camera images via Isaac Sim camera API
                try:
                    from omni.isaac.sensor import Camera
                    
                    # Camera name mapping: LeRobot name -> USD camera name
                    camera_mapping = {
                        "observation.images.ego": "Ego",
                        "observation.images.left_wrist": "LeftArm",
                        "observation.images.right_wrist": "RightArm",
                    }
                    
                    for lerobot_name, usd_cam_name in camera_mapping.items():
                        try:
                            cam_prim_path = f"/World/envs/env_0/Robot/{usd_cam_name}"
                            camera = Camera(prim_path=cam_prim_path)
                            camera.initialize()
                            img = camera.get_rgba()
                            
                            if img is not None:
                                # Convert to PIL Image for LeRobot (expects PIL or path)
                                img_rgb = img[:, :, :3].astype(np.uint8)
                                frame[lerobot_name] = Image.fromarray(img_rgb)
                        except Exception as cam_err:
                            if len(vla_current_episode["frames"]) == 0:
                                print(f"[VLA] Camera {usd_cam_name} error: {cam_err}")
                except Exception as e:
                    if len(vla_current_episode["frames"]) == 0:
                        print(f"[VLA] Camera capture error: {e}")
                
                # Add frame to current episode buffer
                vla_current_episode["frames"].append(frame)
            
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
                    # Save completed episode
                    num_steps = len(vla_current_episode["frames"])
                    if num_steps > 0:
                        if "dataset" in vla_data:
                            # LeRobot native format
                            dataset = vla_data["dataset"]
                            for frame in vla_current_episode["frames"]:
                                dataset.add_frame(frame)
                            dataset.save_episode(task=vla_task_text)
                        else:
                            # Fallback format: save as individual parquet + images
                            import pandas as pd
                            ep_dir = os.path.join(vla_output_dir, "episodes", f"episode_{vla_episode_count}")
                            os.makedirs(ep_dir, exist_ok=True)
                            
                            # Extract tabular data
                            states = np.array([f["observation.state"] for f in vla_current_episode["frames"]])
                            actions = np.array([f["action"] for f in vla_current_episode["frames"]])
                            timestamps = np.arange(num_steps) / fps
                            
                            # Save tabular data as parquet
                            df = pd.DataFrame({
                                "timestamp": timestamps,
                                "task": [vla_task_text] * num_steps,
                            })
                            # Add state columns
                            for i in range(states.shape[1]):
                                df[f"observation.state.{i}"] = states[:, i]
                            # Add action columns
                            for i in range(actions.shape[1]):
                                df[f"action.{i}"] = actions[:, i]
                            
                            df.to_parquet(os.path.join(ep_dir, "data.parquet"))
                            
                            # Save images as video frames (MP4) or individual files
                            for cam_key in ["observation.images.ego", "observation.images.left_wrist", "observation.images.right_wrist"]:
                                cam_name = cam_key.split(".")[-1]
                                cam_dir = os.path.join(ep_dir, cam_name)
                                os.makedirs(cam_dir, exist_ok=True)
                                
                                for idx, frame in enumerate(vla_current_episode["frames"]):
                                    if cam_key in frame and frame[cam_key] is not None:
                                        img = frame[cam_key]
                                        img.save(os.path.join(cam_dir, f"frame_{idx:06d}.png"))
                            
                            # Save initial conditions for replay
                            if "initial_conditions" in vla_current_episode:
                                init_cond = vla_current_episode["initial_conditions"]
                                init_cond_serializable = {
                                    k: v.tolist() for k, v in init_cond.items()
                                }
                                with open(os.path.join(ep_dir, "initial_conditions.json"), "w") as f:
                                    json.dump(init_cond_serializable, f, indent=2)
                        
                        vla_episode_count += 1
                        print(f"[VLA] Episode {vla_episode_count} saved ({num_steps} steps)")
                        
                        # Check if we've captured enough episodes
                        if vla_episode_count >= args_cli.vla_max_episodes:
                            print(f"[VLA] Captured {vla_episode_count} episodes. Stopping...")
                            break
                    
                    # Start new episode with initial conditions
                    robot = unwrapped.scene["robot"]
                    vla_current_episode = {
                        "frames": [],
                        "initial_conditions": {
                            "left_cube_pos": unwrapped.scene["object_left"].data.root_pos_w[0].cpu().numpy().copy(),
                            "right_cube_pos": unwrapped.scene["object_right"].data.root_pos_w[0].cpu().numpy().copy(),
                            "left_cube_quat": unwrapped.scene["object_left"].data.root_quat_w[0].cpu().numpy().copy(),
                            "right_cube_quat": unwrapped.scene["object_right"].data.root_quat_w[0].cpu().numpy().copy(),
                            "robot_qpos": robot.data.joint_pos[0].cpu().numpy().copy(),
                            "robot_qvel": robot.data.joint_vel[0].cpu().numpy().copy(),
                        },
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
            
            # Save any in-progress episode
            num_steps = len(vla_current_episode.get("frames", [])) if vla_current_episode else 0
            if num_steps > 0:
                if "dataset" in vla_data:
                    # LeRobot native format
                    dataset = vla_data["dataset"]
                    for frame in vla_current_episode["frames"]:
                        dataset.add_frame(frame)
                    dataset.save_episode(task=vla_task_text)
                    vla_episode_count += 1
                    print(f"[VLA] Final episode saved ({num_steps} steps)")
                else:
                    # Fallback format
                    import pandas as pd
                    ep_dir = os.path.join(vla_output_dir, "episodes", f"episode_{vla_episode_count}")
                    os.makedirs(ep_dir, exist_ok=True)
                    
                    states = np.array([f["observation.state"] for f in vla_current_episode["frames"]])
                    actions = np.array([f["action"] for f in vla_current_episode["frames"]])
                    timestamps = np.arange(num_steps) / fps
                    
                    df = pd.DataFrame({"timestamp": timestamps, "task": [vla_task_text] * num_steps})
                    for i in range(states.shape[1]):
                        df[f"observation.state.{i}"] = states[:, i]
                    for i in range(actions.shape[1]):
                        df[f"action.{i}"] = actions[:, i]
                    df.to_parquet(os.path.join(ep_dir, "data.parquet"))
                    
                    for cam_key in ["observation.images.ego", "observation.images.left_wrist", "observation.images.right_wrist"]:
                        cam_name = cam_key.split(".")[-1]
                        cam_dir = os.path.join(ep_dir, cam_name)
                        os.makedirs(cam_dir, exist_ok=True)
                        for idx, frame in enumerate(vla_current_episode["frames"]):
                            if cam_key in frame and frame[cam_key] is not None:
                                frame[cam_key].save(os.path.join(cam_dir, f"frame_{idx:06d}.png"))
                    
                    # Save initial conditions for replay
                    if "initial_conditions" in vla_current_episode:
                        init_cond = vla_current_episode["initial_conditions"]
                        init_cond_serializable = {
                            k: v.tolist() for k, v in init_cond.items()
                        }
                        with open(os.path.join(ep_dir, "initial_conditions.json"), "w") as f:
                            json.dump(init_cond_serializable, f, indent=2)
                    
                    vla_episode_count += 1
                    print(f"[VLA] Final episode saved ({num_steps} steps)")
            
            # Finalize dataset
            if "dataset" in vla_data:
                dataset = vla_data["dataset"]
                dataset.finalize()
                print(f"[VLA] LeRobot dataset finalized: {vla_episode_count} episodes")
                print(f"[VLA] Dataset location: {vla_output_dir}")
                print(f"[VLA] To train Pi 0.5, run:")
                print(f"      lerobot-train --dataset.repo_id=local/{os.path.basename(vla_output_dir)} --policy.type=pi05")
            else:
                # Update metadata for fallback format
                if vla_episode_count > 0:
                    vla_data["metadata"]["total_episodes"] = vla_episode_count
                    with open(os.path.join(vla_output_dir, "metadata.json"), "w") as f:
                        json.dump(vla_data["metadata"], f, indent=2)
                    print(f"[VLA] Saved {vla_episode_count} episodes to {vla_output_dir}/")
                    print(f"[VLA] Note: Install LeRobot for native format support")
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
