# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse


from omni.isaac.lab.app import AppLauncher

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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
import copy
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("source","logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    all_episodes_states = []
    max_episodes=500
    episode=0
    total_terminated=0
    total_fail=0
    #simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            
            # Add uniform action noise
            #noise = torch.rand_like(actions) * 3 - 1.5  # Uniform noise in [-1.5,1.5]
            noise = torch.rand_like(actions) * 0.2 - 0.1  # Uniform noise in [-1.5,1.5]
            actions = actions + noise
            
            # Clip actions to ensure they remain within valid bounds
            
            # env stepping
            obs, rew, _, extra = env.step(actions)
            episode += 1

            if episode < max_episodes:
                #print('obs',env.env.obs_buf)
                
                state = env.env.obs_buf['state']
                fail=env.env.obs_buf['fail']#['failure_state']
                rew=rew
                terminated=extra['terminated']
                actions=env.env.action_manager.get_term('joint_pos').processed_actions
                # print('rew',rew)
                #print('terminated',terminated)
                # print(f"State: {state}")
                # print(f"Actions: {actions}")
                step_data = {
                "state": state,
                "fail":fail,
                "action": actions,
                "reward": rew,
                "terminated": terminated
            }
                all_episodes_states.append(copy.deepcopy(step_data))
                print(f"Episode: {episode}")
                # Count the number of True values in the 'terminated' tensor
                true_count = terminated.sum()
                fail_count = fail.sum()
                print(f"Number of True values in 'terminated': {true_count}")
                print(f"Number of True values in 'fail': {fail_count}")
                total_terminated+= true_count
                total_fail += fail_count
            elif episode == max_episodes:
                print('total terminated:',total_terminated)
                print('total fail:',total_fail)
                all_episodes_states_array = np.array(all_episodes_states, dtype=object)
                np.save("episodes_states_real_23dof_new.npy", all_episodes_states_array)
                print("Episodes states saved")
                
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    import random as rand
    random_seed = 43
    rand.seed(random_seed)
    # run the main function
    main()
    # close sim app
    simulation_app.close()
