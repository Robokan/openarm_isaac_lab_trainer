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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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
    
    def on_keyboard_event(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.L:
                reload_requested[0] = True
                return True  # Consume event to prevent UI from handling it
            elif event.input == carb.input.KeyboardInput.R:
                reset_requested[0] = True
                return True  # Consume event to prevent UI from handling it
        return False
    
    # Subscribe to keyboard events (priority subscription to override UI)
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    print("[INFO] Press 'L' to reload checkpoint, 'R' to reset environment, close window to exit")

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
            
            # env stepping (outside inference mode to allow reset)
            obs, _, _, _ = env.step(actions)
            
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
