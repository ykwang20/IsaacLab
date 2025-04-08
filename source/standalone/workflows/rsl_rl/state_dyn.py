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
#os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
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
from k_step_evaluate_dyn import compute_batch_k_step_errors

#############################################
# Key dimensions
#############################################
KEY_DIMENSIONS = {
    "base_pos": {'dim':1, 'idx_start':0, 'range': torch.tensor([0.8])},         # z coord position
    "base_lin_vel": {'dim':3, 'idx_start':1, 'range': torch.tensor([1.0]*3)},      
    "base_ang_vel": {'dim':3, 'idx_start':4, 'range': torch.tensor([1.0]*3)},           
    "base_euler": {'dim':3, 'idx_start':7, 'range': torch.tensor([6.28]*3)},             
    "joint_pos": {'dim':23, 'idx_start':10, 'range': torch.tensor([3.0]*23)},            
}


def _process_observations(data, num_agents, obs_keys):
    # shape => (T, num_agents, obs_dim)
    obs_list = []
    for i, d in enumerate(data):
        state = d["state"]
        feats = [state[k].view(num_agents, -1) for k in obs_keys]
        obs_list.append(torch.cat(feats, dim=-1))
    return torch.stack(obs_list, dim=0).float()

def _process_actions(data):
    return torch.stack([d["action"] for d in data], dim=0).float()  # (T, num_agents, act_dim)

def _process_prediction_targets(data, num_agents, pred_targets_keys):
    t_list = []
    for d in data:
        comps = []
        for key in pred_targets_keys:
            if key in d:
                comp = d[key].view(num_agents, -1)
            elif "state" in d and key in d["state"]:
                comp = d["state"][key].view(num_agents, -1)
            else:
                raise KeyError(f"Key '{key}' not found in data.")
            comps.append(comp)
        t_list.append(torch.cat(comps, dim=-1))
    return torch.stack(t_list, dim=0).float()

def _process_next_obs(observations):
    next_obs = torch.zeros_like(observations)
    next_obs[:-1] = observations[1:]
    next_obs[-1] = observations[-1]
    return next_obs

def _process_terminated(data):
    arr_terminated = [d["terminated"] for d in data]
    arr_failure = [d["fail"] for d in data]
    return torch.stack(arr_terminated, dim=0).bool(), torch.stack(arr_failure, dim=0).bool()



#############################################
# TrajectoryDataset for dynamics transitions
#############################################
class StatesDataset(Dataset):
    def __init__(self, data_path: str, obs_keys, pred_targets_keys):
        self.data = np.load(data_path, allow_pickle=True).tolist()
        self.obs_keys = obs_keys
        self.pred_targets_keys = pred_targets_keys
        self.num_agents = self.data[0]['action'].shape[0]
        self.T = len(self.data)

        # Process core data fields
        self.observations = _process_observations(self.data, self.num_agents, self.obs_keys)     # (T, num_agents, obs_dim)
        self.actions = _process_actions(self.data)               # (T, num_agents, act_dim)
        self.targets = _process_prediction_targets(self.data, self.num_agents, self.pred_targets_keys)    # (T, num_agents, target_dim)
        self.next_observations = _process_next_obs(self.observations)    # (T, num_agents, obs_dim)
        self.terminated, self.failed = _process_terminated(self.data)         # (T, num_agents)

        self.all_obs = self.observations.view(-1, self.observations.shape[-1])
        self.all_targets = self.targets.view(-1, self.targets.shape[-1])

    def __len__(self):
        return self.T*self.num_agents
    
    def __repr__(self):
        return (f"StatesDataset: {self.T*self.num_agents} samples,"
                f"\nobs_dim={self.observations.shape}, "
                f"\ntarget_dim={self.targets.shape}"
                f"\nnum_failure={torch.sum(self.failed)}")

    def __getitem__(self, idx):
        t = idx // self.num_agents
        agent = idx % self.num_agents
        return {
            'obs': self.observations[t, agent],
            'prediction_targets': self.targets[t, agent],
            'action': self.actions[t, agent],
            'terminated': self.terminated[t, agent],
            'failed': self.failed[t, agent],
            'next_obs': self.next_observations[t, agent]
        }

class TrajectoryDataset(StatesDataset):
    def __init__(self, data_path: str, obs_keys, pred_targets_keys, seq_len=24):
        super().__init__(data_path, obs_keys, pred_targets_keys)
        self.seq_len = seq_len
        # Build valid subsequences for dynamics training
        self.samples = []
        self._build_subsequences()
        self._validate_dataset()


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
                padded_sample = self._pad_subsequence(subsequence, valid_steps)
                self.samples.append(padded_sample)

    def _pad_subsequence(self, subsequence, valid_steps):
        padded = {
            'obs': torch.zeros(self.seq_len, self.observations.shape[-1]),
            'action': torch.zeros(self.seq_len, self.actions.shape[-1]),
            'next_obs': torch.zeros(self.seq_len, self.observations.shape[-1]),
            'prediction_targets': torch.zeros(self.seq_len, self.targets.shape[-1]),
            'terminated': torch.zeros(self.seq_len, dtype=torch.bool),
            'valid_mask': torch.zeros(self.seq_len, dtype=torch.bool)
        }

        for key in ['obs', 'action', 'next_obs', 'prediction_targets', 'terminated']:
            padded[key][:valid_steps] = torch.stack(subsequence[key])
        padded['valid_mask'][:valid_steps] = True
        return padded

    def _validate_dataset(self):
        for sample in self.samples:
            valid_steps = sample['valid_mask'].sum().item()
            terminated_steps = sample['terminated'].sum().item()
            assert valid_steps > 0, "Empty sequence found"
            if terminated_steps > 0:
                # Termination must be the final step in that sequence
                assert sample['terminated'][valid_steps - 1], "Termination must be final step"
                assert valid_steps <= self.seq_len, "Invalid termination position"

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
            f"State components: {self.data[0]['state'].keys()}"
            f"Actions shape: {self.actions.shape}",
            f"Targets shape: {self.targets.shape}",
            f"Terminated shape: {self.terminated.shape}"
        ]
        return "\n".join(info)


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


class GRUNetwork(nn.Module):
    def __init__(self, cfg: dict, device: str = 'cuda:0'):
        super().__init__()
        num_layers = cfg.get("num_layers", 1)
        hidden_dim = cfg["hidden_dim"]
        name = cfg.get("name", None)

        self.gru = nn.GRU(
            input_size=cfg["input_dim"],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=cfg.get("dropout", 0.0) if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        self.fc = MLPNetwork(cfg["fc_cfg"], device=device)
        self.to(device)
        print(f"{name} GRU Network initialized:\nGRU: {self.gru}\nFC: {self.fc}")

    def forward(self, x):
        hiddens, _ = self.gru(x)
        out = self.fc(hiddens)
        return out


class VAEEncoder(nn.Module):
    def __init__(self, cfg: dict, device: str):
        super().__init__()
        self.base_encoder = MLPNetwork(cfg, device=device)
        self.fc_mu = nn.Linear(cfg['output_dim'], cfg['output_dim'], device=device)
        self.fc_logvar = nn.Linear(cfg['output_dim'], cfg['output_dim'], device=device)

    def forward(self, x):
        h = self.base_encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class LatentDynamicsModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.device = cfg["device"]
        self.encoder_cfg = cfg["encoder_cfg"]
        self.dynamics_cfg = cfg["dynamics_cfg"]
        self.predictor_cfg = cfg.get("predictor_cfg", None)

        self.encoder = VAEEncoder(self.encoder_cfg, device=self.device)
        # self.dynamics = GRUNetwork(self.dynamics_cfg, device=self.device)
        self.dynamics = MLPNetwork(self.dynamics_cfg, device=self.device)
        self.predictor = MLPNetwork(self.predictor_cfg, device=self.device)

        # Optionally freeze GRU from the start if you only want to train VAE first:
        # for param in self.dynamics.parameters():
        #     param.requires_grad = False

    def vae_forward(self, obs):
        """
        A specialized forward that only encodes and decodes (for VAE).
        obs shape: (batch_size, obs_dim)
        """
        mu, logvar = self.encoder(obs)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        # predictor acts like a “decoder” here:
        prediction = self.predictor(z)
        return {
            "mu": mu,
            "logvar": logvar,
            "latent": z,
            "prediction": prediction
        }

    def forward(self, obs, actions):
        """
        Full forward for the joint VAE + dynamics, i.e. to get
        next_latent_pred and reconstruction in the same pass.
        Usually used for the sequence transitions.
        """
        outputs = self.vae_forward(obs)
        dynamics_input = torch.cat([obs, actions], dim=-1)
        next_latent_pred = self.dynamics(dynamics_input)
        outputs["next_latent_pred"] = next_latent_pred
        return outputs

    def latent(self, obs):
        mu, _ = self.encoder(obs)
        return mu

    def next_latent(self, latent, actions):
        dynamics_input = torch.cat([latent, actions], dim=-1)
        next_latent = self.dynamics(dynamics_input)
        return next_latent


def _aggregate_stats(dict_list, keys):
    """
    Returns {key: (mean, std)} for each key in keys.
    If the key is missing in a particular dict, we treat it as np.nan.
    """
    out = {}
    for k in keys:
        vals = []
        for d in dict_list:
            if k in d:
                vals.append(d[k])
            else:
                vals.append(np.nan)
        # compute mean, std for this key
        mean_val = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
        std_val = float(np.nanstd(vals)) if len(vals) > 0 else np.nan
        out[k] = (mean_val, std_val)
    return out


#############################################
# LatentDynamicsLearner
#############################################
class LatentDynamicsLearner:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.seq_len = cfg["seq_len"]
        self.device = cfg["device"]
        self.learning_rate = cfg["learning_rate"]
        self.current_epoch = 0

        # -------------------------------------------
        # 1) Build the full dataset for dynamics
        # -------------------------------------------
        full_dyn_dataset = TrajectoryDataset(
            cfg["dataset_path"],
            pred_targets_keys=cfg["pred_targets_keys"],
            obs_keys=cfg["obs_keys"],
            seq_len=self.seq_len
        )
        print("Trajectory dataset:")
        print(full_dyn_dataset)

        # -------------------------------------------
        # 2) Build the states dataset for VAE
        # -------------------------------------------
        full_vae_dataset = StatesDataset(
            cfg["dataset_path"],
            pred_targets_keys=cfg["pred_targets_keys"],
            obs_keys=cfg["obs_keys"],
        )
        print("VAE States dataset:")
        print(full_vae_dataset)

        # You can split them with the same ratio or different if you want
        train_size_dyn = int((1 - self.cfg["eval_pct"]) * len(full_dyn_dataset))
        eval_size_dyn = len(full_dyn_dataset) - train_size_dyn
        self.train_dyn_dataset, self.eval_dyn_dataset = random_split(
            full_dyn_dataset, [train_size_dyn, eval_size_dyn]
        )
        self.train_dyn_dataloader = DataLoader(self.train_dyn_dataset, batch_size=cfg["batch_size"], shuffle=True)
        self.eval_dyn_dataloader = DataLoader(self.eval_dyn_dataset, batch_size=cfg["batch_size"], shuffle=False)

        train_size_vae = int((1 - self.cfg["eval_pct"]) * len(full_vae_dataset))
        eval_size_vae = len(full_vae_dataset) - train_size_vae
        self.train_vae_dataset, self.eval_vae_dataset = random_split(
            full_vae_dataset, [train_size_vae, eval_size_vae]
        )
        self.train_vae_dataloader = DataLoader(self.train_vae_dataset, batch_size=cfg["batch_size"], shuffle=True)
        self.eval_vae_dataloader = DataLoader(self.eval_vae_dataset, batch_size=cfg["batch_size"], shuffle=False)

        # -------------------------------------------
        # 3) Build the model
        # -------------------------------------------
        self.model_cfg = cfg["model_cfg"]
        self.model = LatentDynamicsModel(self.model_cfg)

        # Single optimizer for everything (VAE + dynamics),
        # or you could do separate if you want separate LR schedules.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Logging
        self.log_dir = cfg["log_dir"]
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()
        self.is_tuning = cfg.get("use_tune", False)
        if not self.is_tuning and self.log_dir is not None:
            # from torch.utils.tensorboard import SummaryWriter
            # self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            if self.logger_type =="wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                from dataclasses import dataclass
                @dataclass
                class Empty:
                    pass
                empty_instance = Empty()
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(env_cfg=empty_instance,runner_cfg=self.cfg,alg_cfg=None, policy_cfg=None)

    #############################################
    # 1) VAE loss
    #############################################
    def vae_loss(self, batch):
        obs = batch['obs']
        outputs = self.model.vae_forward(obs)
        # prediction MSE loss
        prediction_loss = nn.MSELoss(reduction="mean")(outputs['prediction'], batch['prediction_targets'])
        # KL divergence (standard formula)
        kl_loss = -0.5 * torch.mean(1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())

        total_loss = prediction_loss + self.cfg["kl_weight"] * kl_loss
        
        eps = 1e-6
        rel_pred_error = ((outputs["prediction"] - batch["prediction_targets"]).norm(dim=-1) / (batch["prediction_targets"].norm(dim=-1)+eps)).mean()
        results = {
            "total_loss": total_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "kl_loss": kl_loss.item(),
            "relative_prediction_error": rel_pred_error.item()
        }
        return total_loss, results

    #############################################
    # 2) Dynamics loss (essentially your original `loss(...)`)
    #############################################
    def dyn_loss(self, batch):
        obs = batch['obs']
        actions = batch['action']
        next_obs = batch['next_obs']
        terminated = batch['terminated']
        valid_mask = batch["valid_mask"]

        # Full forward
        outputs = self.model(obs, actions)

        # Next-latent target from next_obs
        with torch.no_grad():
            latent_now = self.model.latent(obs)               # (B, seq_len, latent_dim)
            next_latent_target = next_obs  # (B, seq_len, latent_dim)

        if self.cfg["dynamics_loss_to_encoder"]:
            # adapte VAE encoder
            next_latent_pred = outputs["next_latent_pred"]  # shape (B, seq_len, latent_dim)
        else:
            next_latent_pred = self.model.next_latent(obs, actions)

        # We only compute the dynamics loss on non-terminated transitions
        non_terminated_mask = ~terminated
        valid_for_dyn = valid_mask & non_terminated_mask
        valid_dyn_count = valid_for_dyn.sum()

        # dynamics loss for valid sequence fregments
        dynamics_loss = ((nn.MSELoss(reduction='none')(next_latent_pred, next_latent_target).mean(dim=-1) * valid_for_dyn).sum() / valid_dyn_count)
        
        eps = 1e-6
        norm_dyn  = (next_latent_pred - next_latent_target).norm(dim=-1)                   # L2-norm => (B, seq_len)
        denom_dyn = next_latent_target.norm(dim=-1) + eps   # (B, seq_len)
        # multiply by valid_for_dyn so we only sum over valid steps
        rel_dyn_error = (norm_dyn / denom_dyn * valid_for_dyn).sum() / valid_dyn_count

        if self.cfg["dynamics_loss_to_encoder"]:
            vae_total_loss, vae_results = self.vae_loss(batch)
            total_loss = self.cfg["dynamics_weight"] * dynamics_loss + vae_total_loss
        else:
            with torch.no_grad():
                _, vae_results = self.vae_loss(batch)
            total_loss = self.cfg["dynamics_weight"] * dynamics_loss

        # For logging, you might want relative errors as well:
        # e.g. relative_dynamics_error or so. Keep it consistent with your style:
        results = {
            "total_loss": total_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "prediction_loss": vae_results["prediction_loss"],
            "kl_loss": vae_results["kl_loss"],
            "relative_dynamics_error": rel_dyn_error.item()
        }
        return total_loss, results


    def k_step_evaluate(self, batch, stats):
        with torch.no_grad():
            k_step_metric = compute_batch_k_step_errors(
                key_dimensions=KEY_DIMENSIONS,
                k=self.cfg['k_step'],
                learner=self,
                obs_keys=self.cfg['pred_targets_keys'],
                batch=batch,
                quantiles=[0.25, 0.75],
                output="results"
            )
            for key in self.cfg['pred_targets_keys']:
                for i in range(self.cfg["k_step"]+1):
                    for metric in ['abs_norms', 'rel_norms']:
                        stats[f'{key}_{i}step_{metric}_error'] = k_step_metric[key][metric][i]

    
    def evaluate(self, model="vae"):
        self.model.eval()
        if model == "vae":
            dataloader = self.eval_vae_dataloader
            loss_fn = self.vae_loss
        elif model == "dyn":
            dataloader = self.eval_dyn_dataloader
            loss_fn = self.dyn_loss
        else:
            raise ValueError("Invalid model specified for evaluation. Choose 'vae' or 'dyn'.")
        eval_results = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                total_loss, stats = loss_fn(batch)
                if model == "dyn":
                    self.k_step_evaluate(batch, stats)
            
            eval_results.append(stats)
        self.model.train()
        return eval_results
    

    def update(self, batch, model="vae" ):
        self.model.train()
        if model == "vae":
            total_loss, results = self.vae_loss(batch)
        elif model == "dyn":
            total_loss, results = self.dyn_loss(batch)
        else:
            raise ValueError("Invalid model specified for update. Choose 'vae' or 'dyn'.")
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_norm_clip"])
        self.optimizer.step()
        results["grad_norm"] = grad_norm.item()
        if model == "dyn":
            self.k_step_evaluate(batch, results)

        return results
    
    def learn(self, reporter=None):
        """
        Main learn function to train the model.
        """
        while self.current_epoch < self.cfg['epoches']:
            if self.current_epoch < self.cfg['vae_epoches']:
                model = 'vae'
                dataloader = self.train_vae_dataloader
            else:
                model = 'dyn'
                dataloader = self.train_dyn_dataloader
            results = {'train_results':[], 'eval_results': None}
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                result = self.update(batch, model=model)
                results['train_results'].append(result)
            if self.current_epoch % self.cfg["eval_interval"] == 0 or self.current_epoch == self.cfg['vae_epoches'] + self.cfg['dyn_epoches'] - 1:
                results['eval_results'] = self.evaluate(model=model)
            
            processed_results = self.postprocess_results(results, model=model)
            if reporter is not None and self.is_tuning:
                reporter(processed_results)
            
            # if not self.is_tuning and self.log_dir is not None:
            if self.log_dir is not None:
                if self.current_epoch % self.cfg["save_interval"] == 0 or self.current_epoch == self.cfg["epoches"] - 1:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    name = f"model_{len(self.cfg['obs_keys'])}keys_lr{self.learning_rate}_{self.current_epoch}_{timestamp}.pt"
                    self.save(os.path.join(self.log_dir, name))
            
            self.current_epoch += 1

    def postprocess_results(self, results, model):
        required_keys = ["total_loss", "dynamics_loss", "prediction_loss", "kl_loss", "grad_norm", 
                         "relative_dynamics_error", "relative_prediction_error"]
        for key in self.cfg['pred_targets_keys']:
            for i in range(self.cfg["k_step"]+1):
                for metric in ['abs_norms', 'rel_norms']:
                    required_keys += [f'{key}_{i}step_{metric}_error']

        train_list = results["train_results"]  # list of dict
        eval_list = results["eval_results"]    # list of dict or None

        # 1) Aggregate train
        train_agg = _aggregate_stats(train_list, required_keys)

        # 2) Aggregate eval
        if (eval_list is None) or (len(eval_list) == 0):
            eval_agg = {k: (np.nan, np.nan) for k in required_keys}
        else:
            eval_agg = _aggregate_stats(eval_list, required_keys)

        # 3) Build final dictionary
        out_dict = {}
        for k in required_keys:
            train_mean, train_std = train_agg[k]
            eval_mean, eval_std = eval_agg[k]
            out_dict[f"train_{k}_ave"] = train_mean
            out_dict[f"train_{k}_std"] = train_std
            out_dict[f"eval_{k}_ave"] = eval_mean
            out_dict[f"eval_{k}_std"] = eval_std

        # 4) TensorBoard logging if not tuning
        if not self.is_tuning and hasattr(self, 'writer'):
            ep = self.current_epoch
            for k in required_keys:
                self.writer.add_scalar(f"Train/{k}_ave", out_dict[f"train_{k}_ave"], ep)
                self.writer.add_scalar(f"Train/{k}_std", out_dict[f"train_{k}_std"], ep)
                self.writer.add_scalar(f"Eval/{k}_ave",  out_dict[f"eval_{k}_ave"],  ep)
                self.writer.add_scalar(f"Eval/{k}_std",  out_dict[f"eval_{k}_std"],  ep)

            # 5) Print out a short summary to terminal
            # Just print average losses
            print_str = (f"[{model.upper()}] Epoch {self.current_epoch} => "
                        f"train_total_loss={out_dict['train_total_loss_ave']:.4f}, "
                        f"eval_total_loss={out_dict['eval_total_loss_ave']:.4f}")
            print(print_str)

        # Return out_dict for reporter
        return out_dict
    
    def save(self, path: str):
        saved_dict = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.current_epoch,
            "optimizer_state": self.optimizer.state_dict()
        }
        torch.save(saved_dict, path)


    def load(self, path: str, load_dynamics = True, load_optimizer: bool = True):
        loaded_dict = torch.load(path)
        if load_dynamics:
            self.model.load_state_dict(loaded_dict["model_state_dict"])
        else:
            filtered_dict = {k:v for k, v in loaded_dict["model_state_dict"].items() if ("dynamics" not in k)}
            self.model.load_state_dict(filtered_dict, strict=False)
        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state"])


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
    act_dim = 23
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

        self.dynamics_cfg.input_dim = self.obs_dim + self.act_dim
        self.dynamics_cfg.output_dim = self.obs_dim
        self.dynamics_cfg.name = "dynamics"
        self.dynamics_cfg.__post_init__()

@configclass
class LatentLearnerCfg:
    use_tune = False
    device = 'cuda:0'
    dataset_path = "episodes_states_sim_23dof_new_merged_100.npy"
    eval_pct = 0.2
    logger = "wandb"#"tensorboard"
    log_dir = "./logs/VAE"  # base log directory
    save_interval = 50
    eval_interval = 2
    seq_len = 12
    k_step = 10
    batch_size = 64
    learning_rate = 0.001
    grad_norm_clip = 10
    dynamics_weight = 0.01
    prediction_weight = 1
    kl_weight = 1e-3
    dynamics_loss_to_encoder = False

    # For demonstration, we add new fields:
    vae_epoches = 0#250
    dyn_epoches = 600 #600
    epoches = 50

    # define keys
    obs_keys = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_euler', 'joint_pos'] #, 'joint_vel']
    pred_targets_keys = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_euler', 'joint_pos'] #, 'joint_vel']

    act_dim = 23
    obs_dim = 57 # place holder. Automatically adjusted with obs keys
    pred_targets_dim = 7 # place holder. Automatically adjusted with pred keys
    latent_dim = 32
    gru_hidden_dim = 32
    model_cfg = LatentModelCfg()

    wandb_project = "latent-dynamics-vae"

    def __post_init__(self):
        # auto compute dimension
        self.epoches = self.vae_epoches + self.dyn_epoches
        self.obs_dim = sum(KEY_DIMENSIONS[key]['dim'] for key in self.obs_keys)
        self.pred_targets_dim = sum(KEY_DIMENSIONS[key]['dim'] for key in self.pred_targets_keys)

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
            # dynamics_cfg=GRUCfg(
            #     hidden_dim=self.gru_hidden_dim
            # )
            dynamics_cfg=MLPCfg(
                hidden_dims=[256, 128],
                activation_at_end=[True, True, False],
                activation="crelu"
            )
        )
        self.model_cfg.__post_init__()


def tune_model(args, config, model_path=None):
    config['pred_targets_keys'] = config["obs_keys"]
    config_obj = LatentLearnerCfg(**config)
    # Now call .to_dict() to get a dictionary with all __post_init__ changes applied
    updated_config = config_obj.to_dict()
    learner = LatentDynamicsLearner(updated_config)
    if args.load_model:
        assert os.path.exists(model_path)
        learner.load(model_path, load_dynamics=args.load_dynamics, load_optimizer=False)
    learner.learn(reporter=tune.report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning with Ray Tune")
    parser.add_argument("--load_dynamics", action="store_true")
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    base_config = LatentLearnerCfg().to_dict()

    if args.tune:
        from ray import tune
        base_config["dataset_path"] = os.path.abspath(base_config["dataset_path"])
        base_config["log_dir"] = os.path.abspath(base_config["log_dir"])
        root_log_dir = base_config["log_dir"]

        # Define hyperparameter search space.
        # base_config["pred_targets_keys"] = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_euler', 'joint_pos']
        base_config["learning_rate"] = tune.grid_search([1e-3, 1e-4])
        base_config['obs_keys'] = tune.grid_search([
            ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_euler', 'joint_pos'],
            ['base_pos', 'base_lin_vel', 'base_euler', 'joint_pos'],
            ['base_pos', 'base_euler', 'joint_pos']
        ])
        # base_config["batch_size"] = tune.grid_search([128, 256, 512, 1024])

        base_config["use_tune"] = True

        model_path=os.path.abspath(args.model_path) if args.model_path else None
        analysis = tune.run(
            lambda config: tune_model(args, config, model_path=model_path),
            config=base_config,
            storage_path=root_log_dir,
            resources_per_trial={"gpu": 0.3},
            metric="eval_prediction_loss_ave",  # Metric to optimize
            mode="min",  # Mode for optimization (minimize or maximize)
            num_samples=3,
            verbose=1,
            max_concurrent_trials=6
        )
        print("Best config: ", analysis.get_best_config(metric="eval_base_pos_5step_rel_norms_error_ave", mode="min", scope="all"))
        df = analysis.dataframe()
        df.to_csv(f"{root_log_dir}/tune_results.csv", index=False)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_config["log_dir"] = os.path.join(base_config["log_dir"], timestamp)
        base_config["use_tune"] = False
        learner = LatentDynamicsLearner(base_config)
        if args.load_model:
            learner.load(args.model_path, load_dynamics=args.load_dynamics, load_optimizer=False)
        learner.learn()