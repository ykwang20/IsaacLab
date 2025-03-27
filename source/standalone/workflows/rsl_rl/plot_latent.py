import argparse
from omni.isaac.lab.app import AppLauncher
import matplotlib.pyplot as plt


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
parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning with Ray Tune")
parser.add_argument("--load_vae", action="store_true", help="whether to load")
parser.add_argument("--load_path", type=str, default=None, help="Path to load the model from")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
args_cli.headless = True
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



import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from omni.isaac.lab.utils import configclass
from rsl_rl.utils import store_code_state, resolve_nn_activation
from collections import defaultdict
import argparse

#############################################
# Key dimensions
#############################################
KEY_DIMENSIONS = {
    "base_pos": 1,          # z coord position
    "base_lin_vel": 3,      
    "base_ang_vel": 3,      
    "base_lin_vel_w": 3,    
    "base_ang_vel_w": 3,    
    "base_quat": 4,         
    "joint_pos": 23,        
    "joint_vel": 23,        
    "projected_gravity": 3, 
}

#############################################
# TrajectoryDataset for dynamics transitions
#############################################
class TrajectoryDataset(Dataset):
    def __init__(self, data_path: str, obs_keys, pred_targets_keys, seq_len=24):
        self.data = np.load(data_path, allow_pickle=True).tolist()
        self.obs_keys = obs_keys
        self.pred_targets_keys = pred_targets_keys
        self.num_agents = self.data[0]['action'].shape[0]
        self.T = len(self.data)
        self.seq_len = seq_len

        # Process core data fields
        self.observations = self._process_observations()     # (T, num_agents, obs_dim)
        self.actions = self._process_actions()               # (T, num_agents, act_dim)
        self.targets = self._process_prediction_targets()    # (T, num_agents, target_dim)
        self.next_observations = self._process_next_obs()    # (T, num_agents, obs_dim)
        self.terminated = self._process_terminated()         # (T, num_agents)

        # Build valid subsequences for dynamics training
        self.samples = []
        self._build_subsequences()

    def _process_observations(self):
        # shape => (T, num_agents, obs_dim)
        obs_list = []
        for d in self.data:
            state = d["state"]
            feats = [state[k].view(self.num_agents, -1) for k in self.obs_keys]
            obs_list.append(torch.cat(feats, dim=-1))
        return torch.stack(obs_list, dim=0).float()

    def _process_actions(self):
        return torch.stack([d["action"] for d in self.data], dim=0).float()  # (T, num_agents, act_dim)

    def _process_prediction_targets(self):
        t_list = []
        for d in self.data:
            comps = []
            for key in self.pred_targets_keys:
                if key in d:
                    comp = d[key].view(self.num_agents, -1)
                elif "state" in d and key in d["state"]:
                    comp = d["state"][key].view(self.num_agents, -1)
                else:
                    raise KeyError(f"Key '{key}' not found in data.")
                comps.append(comp)
            t_list.append(torch.cat(comps, dim=-1))
        return torch.stack(t_list, dim=0).float()

    def _process_next_obs(self):
        next_obs = torch.zeros_like(self.observations)
        next_obs[:-1] = self.observations[1:]
        next_obs[-1] = self.observations[-1]
        return next_obs

    def _process_terminated(self):
        arr = [d["terminated"] for d in self.data]
        return torch.stack(arr, dim=0).bool()

    def _build_subsequences(self):
        """Build subsequences with terminated states as final step (for GRU training)."""
        for agent_idx in range(self.num_agents):
            t = 0
            while t < self.T:
                subsequence = {
                    'obs': [],
                    'action': [],
                    'next_obs': [],
                    'prediction_targets': [],
                    'terminated': [],
                    'valid_mask': []
                }

                valid_steps = 0
                while valid_steps < self.seq_len and t < self.T:
                    current_terminated = self.terminated[t, agent_idx]

                    subsequence['obs'].append(self.observations[t, agent_idx])
                    subsequence['action'].append(self.actions[t, agent_idx])
                    subsequence['next_obs'].append(self.next_observations[t, agent_idx])
                    subsequence['prediction_targets'].append(self.targets[t, agent_idx])
                    subsequence['terminated'].append(current_terminated)
                    subsequence['valid_mask'].append(True)

                    valid_steps += 1
                    t += 1
                    if current_terminated:
                        break

                if valid_steps == 0:
                    continue

                # Pad
                self.samples.append(subsequence)


    
    def __len__(self):
        # For dynamics, length is # of subsequences
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __repr__(self):
        info = [
            "TrajectoryDataset Summary:",
            f"Total subsequences: {len(self.samples)}",
            f"Observations shape (time x agent): {self.observations.shape}",
            f"Actions shape: {self.actions.shape}",
            f"Targets shape: {self.targets.shape}",
            f"Terminated shape: {self.terminated.shape}"
        ]
        return "\n".join(info)


#############################################
# A simple dataset for i.i.d. VAE states
#############################################
class VAEStatesDataset(Dataset):
    """
    Returns single states i.i.d. from the entire list of observations,
    ignoring subsequences. This way, we meet the VAE's assumption of 
    i.i.d. data for reconstruction.
    """
    def __init__(self, traj_dataset: TrajectoryDataset):
        # All observations shape => (T, num_agents, obs_dim)
        obs = traj_dataset.observations  # a (T, N, obs_dim) tensor
        targets = traj_dataset.targets  # a (T, N, target_dim) tensor
        # Flatten out T * N
        # final shape => (T*N, obs_dim)
        self.all_obs = obs.view(-1, obs.shape[-1])
        self.all_targets = targets.view(-1, targets.shape[-1])

    def __len__(self):
        return self.all_obs.shape[0]
    
    def __repr__(self):
        return (f"VAEStatesDataset: {self.all_obs.shape[0]} samples,"
                f"\nobs_dim={self.all_obs.shape}, "
                f"\ntarget_dim={self.all_targets.shape}")

    def __getitem__(self, idx):
        return {
            'obs': self.all_obs[idx],
            'prediction_targets': self.all_targets[idx]
        }


#############################################
# Network Modules
#############################################
class MLPNetwork(nn.Module):
    def __init__(self, cfg: dict, device: str = 'cuda:0'):
        super().__init__()
        input_dim = cfg["input_dim"]
        hidden_dims = cfg["hidden_dims"]
        output_dim = cfg["output_dim"]
        activation = cfg["activation"]
        activation_at_end = cfg["activation_at_end"]
        name = cfg.get("name", None)

        act_module = resolve_nn_activation(activation)
        layers = []
        current_in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_in_dim, hidden_dim))
            if activation_at_end[i]:
                layers.append(act_module)
            current_in_dim = hidden_dim
        layers.append(nn.Linear(current_in_dim, output_dim))
        if activation_at_end[-1]:
            layers.append(act_module)

        self.net = nn.Sequential(*layers)
        self.net.to(device)
        print(f"{name} MLP: {self.net}")

    def forward(self, x):
        return self.net(x)




class VAEEncoder(nn.Module):
    def __init__(self, cfg: dict, device: str):
        super().__init__()
        self.base_encoder = MLPNetwork(cfg, device=device)
        self.fc_mu = nn.Linear(cfg['output_dim'], cfg['output_dim'], device=device)
        self.fc_logvar = nn.Linear(cfg['output_dim'], cfg['output_dim'], device=device)
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x):
        h = self.base_encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    

class Plotter:
    def __init__(self, encoder:VAEEncoder, dataset:TrajectoryDataset, load_path: str):
        self.encoder = encoder
        self.plot_dir_path = os.path.join(os.path.dirname(load_path), "plots_real")
        os.makedirs(self.plot_dir_path, exist_ok=True)
        self.load_encoder(load_path)       
        self.trajectories = dataset.samples
        

    def load_encoder(self, load_path: str):
        loaded_dict = torch.load(load_path)
        filtered_dict = {k: v for k, v in loaded_dict["model_state_dict"].items() if ( "encoder" in k)}
        print(f"Filtered keys: {filtered_dict.keys()}")
        self.encoder.load_state_dict(filtered_dict,strict=False)
        print(f"Model loaded from {load_path}")

    
    def plot(self, traj_ids=None):
        # Implement your plotting logic here
        if traj_ids is None:
            traj_ids = range(len(self.trajectories))
        
        for traj_id in traj_ids:
            trajectory = self.trajectories[traj_id]
            timesteps = range(len(trajectory['obs']))
            obs =torch.stack(trajectory['obs'])
            print('obs',obs)
            latent,_= self.encoder(obs)
            print('latent',latent.shape)
            latent_norm = torch.linalg.norm(latent, dim=1).cpu().numpy()
            plt.figure(figsize=(12, 6))
    
            plt.plot(timesteps, 
                    latent_norm, 
                    label="Norm of Latent Vector",
                    color="blue",
                    alpha=0.8)
            plt.xlabel("Time Step")
            plt.ylabel("Norm of Latent Vector")
            plt.title(f"Latent Vector Norm for Trajectory {traj_id}")

            #plt.show()
            plt.savefig(f"{self.plot_dir_path}/latent_{traj_id}.png")

            
    def save(self, filename: str):
        # Save the plot to a file
        pass




#############################################
# Example Configs
#############################################
@configclass
class MLPCfg:
    input_dim = 128
    hidden_dims = [128]
    output_dim = 1
    activation = "crelu"
    activation_at_end = [True, False]
    name = "Network"

@configclass
class GRUCfg:
    input_dim = 128
    hidden_dim = 32
    output_dim = 32
    num_layers = 1
    dropout = 0.0
    name = "gru"
    fc_cfg = MLPCfg(
        hidden_dims=[],
        activation_at_end=[False],
        name="gru_out"
    )
    def __post_init__(self):
        self.fc_cfg.input_dim = self.hidden_dim
        self.fc_cfg.output_dim = self.output_dim

@configclass
class LatentModelCfg:
    device = 'cuda:0'
    latent_dim = 16
    obs_dim = 57
    act_dim =23# 37
    pred_targets_dim = 7
    seq_mode = "rnn"
    seq_len = 12
    encoder_cfg = MLPCfg()
    dynamics_cfg = GRUCfg()
    predictor_cfg = MLPCfg()

    def __post_init__(self):
        self.encoder_cfg.input_dim = self.obs_dim
        self.encoder_cfg.output_dim = self.latent_dim
        self.encoder_cfg.name = "encoder"

        self.predictor_cfg.input_dim = self.latent_dim
        self.predictor_cfg.output_dim = self.pred_targets_dim
        self.predictor_cfg.name = "predictor"

        self.dynamics_cfg.input_dim = self.latent_dim + self.act_dim
        self.dynamics_cfg.output_dim = self.latent_dim
        self.dynamics_cfg.name = "dynamics"
        self.dynamics_cfg.__post_init__()

@configclass
class LatentLearnerCfg:
    use_tune = False
    device = 'cuda:0'
    dataset_path = "episodes_states_real_23dof.npy"
    eval_pct = 0.2
    logger = "tensorboard"
    log_dir = "logs/test_separate"  # base log directory
    save_interval = 10
    eval_interval = 2
    seq_len = 100
    batch_size = 64
    learning_rate = 0.001
    grad_norm_clip = 10
    dynamics_weight = 0.01
    prediction_weight = 1
    kl_weight = 1e-3
    dynamics_loss_to_encoder = False

    # For demonstration, we add new fields:
    vae_epoches = 200
    dyn_epoches = 1000
    epoches = 50

    # define keys
    obs_keys = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_quat', 'joint_pos', ]#'joint_vel']
    pred_targets_keys = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_quat', 'joint_pos',]# 'joint_vel']

    act_dim =23# 37
    obs_dim = 57 # place holder. Automatically adjusted with obs keys
    pred_targets_dim = 7 # place holder. Automatically adjusted with pred keys
    latent_dim = 32
    gru_hidden_dim = 32
    model_cfg = LatentModelCfg()

    def __post_init__(self):
        # auto compute dimension
        self.epoches = self.vae_epoches + self.dyn_epoches
        self.obs_dim = sum(KEY_DIMENSIONS[key] for key in self.obs_keys)
        self.pred_targets_dim = sum(KEY_DIMENSIONS[key] for key in self.pred_targets_keys)

        self.model_cfg = LatentModelCfg(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            latent_dim=self.latent_dim,
            pred_targets_dim=self.pred_targets_dim,
            encoder_cfg=MLPCfg(
                hidden_dims=[256, 128],
                activation_at_end=[True, True, False],
                activation="crelu"
            ),
            predictor_cfg=MLPCfg(
                hidden_dims=[256, 128],
                activation_at_end=[True, True, False],
                activation="crelu"
            ),
            dynamics_cfg=GRUCfg(
                hidden_dim=self.gru_hidden_dim
            )
        )
        self.model_cfg.__post_init__()





if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning with Ray Tune")
    # args = parser.parse_args()

    config = LatentLearnerCfg().to_dict()
    encoder_cfg=config["model_cfg"]["encoder_cfg"]
    encoder=VAEEncoder(encoder_cfg, device='cuda:0')

    dataset= TrajectoryDataset(config["dataset_path"],
            pred_targets_keys=config["pred_targets_keys"],
            obs_keys=config["obs_keys"],
            seq_len=config["seq_len"])
    plotter = Plotter(encoder=encoder, dataset=dataset, load_path="/home/legrobot/IsaacLab/logs/test_separate/20250324-002643/model_500.pt")
    traj_ids=range(100)
    plotter.plot(traj_ids=traj_ids)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config["log_dir"] = os.path.join(config["log_dir"], timestamp)
    config["use_tune"] = False
    if args_cli.load_vae:
        if args_cli.load_path is None:
            raise ValueError("Please provide a path to load the model from.")
   