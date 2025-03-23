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
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from omni.isaac.lab.utils import configclass
from rsl_rl.utils import store_code_state, resolve_nn_activation
from collections import defaultdict
import argparse

#############################################
# Dataset remains the same
#############################################

KEY_DIMENSIONS = {
    "base_pos": 1,          # z coord position
    "base_lin_vel": 3,       # 3D linear velocity
    "base_ang_vel": 3,       # 3D angular velocity
    "base_lin_vel_w": 3,       # 3D linear velocity world
    "base_ang_vel_w": 3,       # 3D angular velocity world
    "base_quat": 4,          # Quaternion (4D)
    "joint_pos": 37,         # Joint positions
    "joint_vel": 37,         # Joint velocities
    "projected_gravity": 3,  # Projected gravity vector (3D)
}

class TrajectoryDataset(Dataset):
    def __init__(self, data_path: str, obs_keys, pred_targets_keys, seq_len=24):
        self.data = np.load(data_path, allow_pickle=True).tolist()
        self.obs_keys = obs_keys
        self.pred_targets_keys = pred_targets_keys
        self.num_agents = self.data[0]['action'].shape[0]
        self.T = len(self.data)
        self.seq_len = seq_len

        # Process core data fields
        self.observations = self._process_observations()
        self.actions = self._process_actions()
        self.targets = self._process_prediction_targets()
        self.next_observations = self._process_next_obs()
        self.terminated = self._process_terminated() # (T, agents)

        # Build valid subsequences
        self.samples = []
        self._build_subsequences()
        self._validate_dataset()

    def _build_subsequences(self):
        """Build subsequences with terminated states as final step"""
        unique_id = 0
        for agent_idx in range(self.num_agents):
            t = 0
            while t < self.T:
                subsequence = {
                    'obs': [],
                    'action': [],
                    'next_obs': [],
                    'prediction_targets': [],
                    'terminated': [],
                    'valid_mask': [],
                    'unique_id': []
                }

                # Collect steps until termination or seq_len
                valid_steps = 0
                while valid_steps < self.seq_len and t < self.T:
                    current_terminated = self.terminated[t, agent_idx]

                    # Always include current step
                    subsequence['obs'].append(self.observations[t, agent_idx])
                    subsequence['action'].append(self.actions[t, agent_idx])
                    subsequence['next_obs'].append(self.next_observations[t, agent_idx])
                    subsequence['prediction_targets'].append(self.targets[t, agent_idx])
                    subsequence['terminated'].append(current_terminated)
                    subsequence['valid_mask'].append(True)
                    
                    valid_steps += 1
                    t += 1
                    unique_id += 1
                    subsequence['unique_id'].append(torch.Tensor([unique_id]).squeeze())

                    # Stop after including terminated step
                    if current_terminated:
                        break

                if valid_steps == 0:
                    continue

                # Pad the subsequence
                padded_sample = self._pad_subsequence(subsequence, valid_steps)
                if valid_steps ==self.seq_len:
                    self.samples.append(padded_sample)

    def _pad_subsequence(self, subsequence, valid_steps):
        """Pad subsequence to seq_len with proper masking"""
        padded = {
            'obs': torch.zeros(self.seq_len, self.observations.shape[-1]),
            'action': torch.zeros(self.seq_len, self.actions.shape[-1]),
            'next_obs': torch.zeros(self.seq_len, self.observations.shape[-1]),
            'prediction_targets': torch.zeros(self.seq_len, self.targets.shape[-1]),
            'terminated': torch.zeros(self.seq_len, dtype=torch.bool),
            'valid_mask': torch.zeros(self.seq_len, dtype=torch.bool),
            'unique_id': torch.zeros(self.seq_len,dtype=torch.long)
        }

        # Copy valid data
        for key in ['obs', 'action', 'next_obs', 'prediction_targets', 'terminated','unique_id']:
            padded[key][:valid_steps] = torch.stack(subsequence[key])

        # Set valid mask
        padded['valid_mask'][:valid_steps] = True
        
        return padded

    def _validate_dataset(self):
        """Ensure terminated states are properly included"""
        for sample in self.samples:
            valid_steps = sample['valid_mask'].sum().item()
            terminated_steps = sample['terminated'].sum().item()
            # print('unique_id:', sample['unique_id'])
            # print('terminated:', sample['terminated'])
            
            assert valid_steps > 0, "Empty sequence found"
            if terminated_steps > 0:
                # Termination should only be in last valid step
                assert sample['terminated'][valid_steps-1], "Termination must be final step"
                assert valid_steps <= self.seq_len, "Invalid termination position"


    # -- The data processing methods are the same as before (just simplified) --
    def _process_observations(self):
        # shape => (T, num_agents, obs_dim)
        obs_list = []
        for d in self.data:
            state = d["state"]
            feats = [state[k].view(self.num_agents, -1) for k in self.obs_keys]
            obs_list.append(torch.cat(feats, dim=-1))
        return torch.stack(obs_list, dim=0).float()

    def _process_actions(self):
        # shape => (T, num_agents, act_dim)
        return torch.stack([d["action"] for d in self.data], dim=0).float()

    def _process_prediction_targets(self):
        # shape => (T, num_agents, target_dim)
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
        # shape => (T, num_agents, obs_dim)
        # shift by 1
        next_obs = torch.zeros_like(self.observations)
        next_obs[:-1] = self.observations[1:]
        next_obs[-1] = self.observations[-1]
        return next_obs

    def _process_terminated(self):
        # shape => (T, num_agents)
        arr = [d["terminated"] for d in self.data]
        return torch.stack(arr, dim=0).bool()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __repr__(self):
        info = [
            "Dataset Summary:",
            f"State key: {self.data[0]['state'].keys()}",
            f"State key shapes: {[(key, self.data[0]['state'][key].shape) for key in self.data[0]['state'].keys()]}"
            f"Total sequences: {len(self)}",
            f"Sequence Length: {self.seq_len}",
            f"Observations shape: {self.observations.shape}",
            f"Actions shape: {self.actions.shape}",
            f"Prediction Targets shape: {self.targets.shape}",
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
        activation = resolve_nn_activation(activation)
        layers = []
        current_in_dim = input_dim
        for layer_index in range(len(hidden_dims)):
            layers.append(nn.Linear(current_in_dim, hidden_dims[layer_index]))
            if activation_at_end[layer_index]:
                layers.append(activation)
            current_in_dim = hidden_dims[layer_index]
        layers.append(nn.Linear(current_in_dim, output_dim))
        if activation_at_end[-1]:
            layers.append(activation)
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
    def __init__(self, cfg: dict, **kwargs):
        super().__init__()
        self.device = cfg["device"]
        self.encoder_cfg = cfg["encoder_cfg"]
        self.dynamics_cfg = cfg["dynamics_cfg"]
        self.predictor_cfg = cfg.get("predictor_cfg", None)
        self.encoder = VAEEncoder(self.encoder_cfg, device=self.device)
        self.dynamics = GRUNetwork(self.dynamics_cfg, device=self.device)
        self.predictor = MLPNetwork(self.predictor_cfg, device=self.device)
        # for param in self.encoder.parameters(): param.requires_grad = False
        # for param in self.predictor.parameters(): param.requires_grad = False

    def forward(self, obs, actions):
        mu, logvar = self.encoder(obs)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        dynamics_input = torch.cat([z, actions], dim=-1)
        next_latent_pred = self.dynamics(dynamics_input)
        prediction = self.predictor(z)
        return {
            "mu": mu,
            "logvar": logvar,
            "latent": z,
            "next_latent_pred": next_latent_pred,
            "prediction": prediction
        }
    
    def latent(self, obs):
        mu, _ = self.encoder(obs)
        return mu
    
    def next_latent(self, latent, actions):
        dynamics_input = torch.cat([latent, actions], dim=-1)
        next_latent = self.dynamics(dynamics_input)
        return next_latent

#############################################
# LatentDynamicsLearner (kept clean from Tune specifics)
#############################################
class LatentDynamicsLearner:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.seq_len = cfg["seq_len"]
        full_dataset = TrajectoryDataset(
            cfg["dataset_path"],
            pred_targets_keys=cfg["pred_targets_keys"],
            obs_keys=cfg["obs_keys"],
            seq_len=self.seq_len
        )
        print(full_dataset)
        train_size = int((1 - self.cfg["eval_pct"]) * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        self.train_dataset, self.eval_dataset = random_split(full_dataset, [train_size, eval_size])
        print('train_dataset:', len(self.train_dataset))
        print('eval_dataset:', len(self.eval_dataset))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=cfg["batch_size"], shuffle=True)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=cfg["batch_size"], shuffle=False)
        self.device = cfg["device"]
        self.learning_rate = cfg["learning_rate"]
        self.current_epoch = 0
        self.model_cfg = cfg["model_cfg"]
        self.model = LatentDynamicsModel(self.model_cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.log_dir = cfg["log_dir"]

    def loss(self, batch):
        obs = batch['obs']
        actions = batch['action']
        next_obs = batch['next_obs']
        terminated = batch['terminated']  # Assuming 'terminated' is part of the batch
        valid_mask = batch["valid_mask"]

        outputs = self.model(obs, actions)
        with torch.no_grad():
            latent = self.model.latent(obs)
            next_latent_target = self.model.latent(next_obs)

        if self.cfg["dynamics_loss_to_encoder"]:
            next_latent_pred = outputs["next_latent_pred"]
        else:
            next_latent_pred = self.model.next_latent(latent, actions)

        # Create a mask for non-terminated steps
        non_terminated_mask = ~terminated
        valid_for_dyn = valid_mask & non_terminated_mask  # shape (B, seq_len)
        # print(f"non_terminated_mask: {non_terminated_mask}")
        # print('non_terminated mask shape:', non_terminated_mask.shape)
        # print(f"valid_mask: {valid_mask}")
        # print('valid mask shape:', valid_mask.shape)
        # print(f"valid_for_dyn: {valid_for_dyn}")
        # print('valid_for_dyn shape:', valid_for_dyn.shape)
        valid_dyn_count = valid_for_dyn.sum()
        valid_loss_count = valid_mask.sum()
        # Apply the mask to the dynamics loss
        # print(next_latent_pred.shape)
        # print("next_latent_pred: ", next_latent_pred[0:5, 0, :])
        # print("next_latent_target: ", next_latent_target[0:5,0,:])
        #dim of latent is (B, seq_len, latent_dim)
        dynamics_loss = ((nn.MSELoss(reduction='none')(next_latent_pred, next_latent_target).mean(dim=-1) * valid_for_dyn).sum() / valid_dyn_count)

        

        prediction_loss = ((nn.MSELoss(reduction='none')(outputs["prediction"], batch["prediction_targets"]).mean(dim=-1) * valid_mask).sum() / valid_loss_count)
        
        
        kl_loss = -0.5 * (((1 + outputs["logvar"] - outputs["mu"]**2 - torch.exp(outputs["logvar"])).mean(dim=-1) * valid_mask).sum() / valid_loss_count)
        
        total_loss = (self.cfg["dynamics_weight"] * dynamics_loss 
                      +  self.cfg["prediction_weight"] * prediction_loss +
                        self.cfg["kl_weight"] * kl_loss
        )
        
        eps = 1e-8

        # -- For dynamics --
        # shape => (B, seq_len, latent_dim)
        diff_dyn  = next_latent_pred - next_latent_target
        norm_dyn  = diff_dyn.norm(dim=-1)                   # L2-norm => (B, seq_len)
        denom_dyn = next_latent_target.norm(dim=-1) + eps   # (B, seq_len)
        # multiply by valid_for_dyn so we only sum over valid steps

        
        rel_dyn_error = (norm_dyn / denom_dyn * valid_for_dyn).sum() / valid_dyn_count
        # eps = 1e-6
        # rel_dyn_error = (torch.sqrt((next_latent_pred - next_latent_target)**2) / (torch.abs(next_latent_target) + eps)).mean(dim=-1)  # (B, seq_len)
        # rel_dyn_error = (rel_dyn_error * valid_for_dyn).sum() / valid_dyn_count 

        # -- For predictions --
        diff_pred  = outputs["prediction"] - batch["prediction_targets"]
        norm_pred  = diff_pred.norm(dim=-1)                  # (B, seq_len)
        denom_pred = batch["prediction_targets"].norm(dim=-1) + eps
        rel_pred_error = (norm_pred / denom_pred * valid_mask).sum() / valid_loss_count
        results = {
            "total_loss": total_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "kl_loss": kl_loss.item(),
            "relative_dynamics_error": rel_dyn_error.item(),
            "relative_prediction_error": rel_pred_error.item()
        }
        return total_loss, results

    def update(self, batch):
        #print("Updating model...")
        total_loss, results = self.loss(batch)
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_norm_clip"])
        self.optimizer.step()
        results["grad_norm"] = grad_norm.item()
        return results

    def evaluate(self):
        self.model.eval()
        eval_results = []
        #print("Evaluating model...")
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                total_loss, results = self.loss(batch)
                eval_results.append(results)
        self.model.train()
        return eval_results

    def learn(self, reporter=None):
        # Determine if we are in tuning mode.
        is_tuning = self.cfg.get("use_tune", False)
        # Only initialize TensorBoard logging if not tuning and log_dir is set.
        if not is_tuning and self.log_dir is not None:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            # (Other logger types can be added here.)

        while self.current_epoch < self.cfg["epoches"]:
            results = []
            for batch in self.train_dataloader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                result = self.update(batch)
                results.append(result)
                
            # Aggregate training metrics
            avg_train_total_loss = np.mean([r["total_loss"] for r in results])
            avg_train_dynamics_loss = np.mean([r["dynamics_loss"] for r in results])
            avg_train_prediction_loss = np.mean([r["prediction_loss"] for r in results])
            avg_train_kl_loss = np.mean([r["kl_loss"] for r in results])
            avg_train_grad_norm = np.mean([r["grad_norm"] for r in results])
            avg_train_dynamcis_error = np.mean([r["relative_dynamics_error"] for r in results])
            avg_train_prediction_error = np.mean([r["relative_prediction_error"] for r in results])
            # Evaluate periodically
            avg_eval_total_loss = np.nan
            avg_eval_dynamics_loss = np.nan
            avg_eval_prediction_loss = np.nan
            avg_eval_kl_loss = np.nan
            avg_eval_dynamics_error = np.nan
            avg_eval_prediction_error = np.nan
            if self.current_epoch % self.cfg["eval_interval"] == 0 or self.current_epoch == self.cfg["epoches"] - 1:
                eval_results = self.evaluate()
                avg_eval_total_loss = np.mean([r["total_loss"] for r in eval_results])
                avg_eval_dynamics_loss = np.mean([r["dynamics_loss"] for r in eval_results])
                avg_eval_prediction_loss = np.mean([r["prediction_loss"] for r in eval_results])
                avg_eval_kl_loss = np.mean([r["kl_loss"] for r in eval_results])
                avg_eval_dynamics_error = np.mean([r["relative_dynamics_error"] for r in eval_results])
                avg_eval_prediction_error = np.mean([r["relative_prediction_error"] for r in eval_results])
            # Build the report dictionary (as in add_scalar)
            report_dict = {
                "train_total_loss": avg_train_total_loss,
                "train_dynamics_loss": avg_train_dynamics_loss,
                "train_prediction_loss": avg_train_prediction_loss,
                "train_kl_loss": avg_train_kl_loss,
                "train_grad_norm": avg_train_grad_norm,
                "train_dynamics_error": avg_train_dynamcis_error,
                "train_prediction_error": avg_train_prediction_error,
                "lr": self.learning_rate,
                "epoch": self.current_epoch,
                "eval_total_loss": avg_eval_total_loss,
                "eval_dynamics_loss": avg_eval_dynamics_loss,
                "eval_prediction_loss": avg_eval_prediction_loss,
                "eval_kl_loss": avg_eval_kl_loss,
                "eval_dynamics_error": avg_eval_dynamics_error,
                "eval_prediction_error": avg_eval_prediction_error
            }
            # When not tuning, log via SummaryWriter
            if not is_tuning and self.log_dir is not None:
                self.writer.add_scalar("Train/total_loss", avg_train_total_loss, self.current_epoch)
                self.writer.add_scalar("Train/dynamics_loss", avg_train_dynamics_loss, self.current_epoch)
                self.writer.add_scalar("Train/prediction_loss", avg_train_prediction_loss, self.current_epoch)
                self.writer.add_scalar("Train/kl_loss", avg_train_kl_loss, self.current_epoch)
                self.writer.add_scalar("Train/grad_norm", avg_train_grad_norm, self.current_epoch)
                self.writer.add_scalar("Train/lr", self.learning_rate, self.current_epoch)
                self.writer.add_scalar("Train/relative_dynamics_error", avg_train_dynamcis_error, self.current_epoch)
                self.writer.add_scalar("Train/relative_prediction_error", avg_train_prediction_error, self.current_epoch)
                if avg_eval_total_loss is not None:
                    self.writer.add_scalar("Eval/total_loss", avg_eval_total_loss, self.current_epoch)
                    self.writer.add_scalar("Eval/dynamics_loss", avg_eval_dynamics_loss, self.current_epoch)
                    self.writer.add_scalar("Eval/prediction_loss", avg_eval_prediction_loss, self.current_epoch)
                    self.writer.add_scalar("Eval/kl_loss", avg_eval_kl_loss, self.current_epoch)
                    self.writer.add_scalar("Eval/relative_dynamics_error", avg_eval_dynamics_error, self.current_epoch)
                    self.writer.add_scalar("Eval/relative_prediction_error", avg_eval_prediction_error, self.current_epoch)
            # Print epoch summary
            if not is_tuning:
                print(f"Epoch {self.current_epoch}: Train total loss = {avg_train_total_loss}, Eval total loss = {avg_eval_total_loss}")
            # If a reporter callback is provided (for Tune), report the metrics
            if is_tuning and reporter is not None:
                reporter(report_dict)
            # Save checkpoints only if not tuning
            if not is_tuning and self.log_dir is not None:
                if self.current_epoch % self.cfg["save_interval"] == 0 or self.current_epoch == self.cfg["epoches"] - 1:
                    self.save(os.path.join(self.log_dir, f"model_{self.current_epoch}.pt"))
            self.current_epoch += 1

    def save(self, path: str):
        saved_dict = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.current_epoch,
            "optimizer_state": self.optimizer.state_dict()
        }
        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = False):
        loaded_dict = torch.load(path)
        print('key of loaded_dict:', loaded_dict['model_state_dict'].keys())
        # self.model.encoder.load_state_dict(loaded_dict["model_state_dict"]["encoder"])
        # self.model.predictor.load_state_dict(loaded_dict["model_state_dict"]["predictor"])
        filtered_dict = {k: v for k, v in loaded_dict["model_state_dict"].items() if (not "dynamics" in k)}
        print(f"Filtered keys: {filtered_dict.keys()}")
        self.model.load_state_dict(filtered_dict,strict=False)
        #self.model.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state"])
        print(f"Loaded model from {path}")
        #self.current_epoch = loaded_dict["epoch"]

#############################################
# Configurations (with new KL weight parameter)
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
    dropout = 0.0  # dropout prob only used if num_layers > 1
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
    act_dim = 37
    pred_targets_dim = 7
    seq_mode = "rnn"
    seq_len = 20
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
    dataset_path = "episodes_states_sim.npy"
    eval_pct = 0.2
    logger = "tensorboard"
    log_dir = "./logs/VAE_GRU_modified"  # base log directory
    save_interval = 10
    eval_interval = 2
    seq_len = 12#24
    epoches = 1000
    batch_size = 64
    learning_rate = 0.001
    grad_norm_clip = 10
    dynamics_weight = 5
    prediction_weight = 1
    kl_weight = 1e-3
    dynamics_loss_to_encoder = False

    # Define observation and prediction target keys
    obs_keys = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_quat', 'joint_pos',] #'joint_vel']
    pred_targets_keys = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_quat', 'joint_pos',] #'joint_vel']

    # Placeholder for dimensions (will be computed in __post_init__)
    obs_dim = None
    pred_targets_dim = None

    # Other configurations
    act_dim = 37
    latent_dim = 32
    gru_hidden_dim =32# 128#400#128#32
    model_cfg = LatentModelCfg()

    def __post_init__(self):
        # Compute obs_dim and pred_targets_dim based on keys and their dimensions
        self.obs_dim = sum(KEY_DIMENSIONS[key] for key in self.obs_keys)
        self.pred_targets_dim = sum(KEY_DIMENSIONS[key] for key in self.pred_targets_keys)

        # Update model_cfg with computed dimensions
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

def tune_model(config):
    config_obj = LatentLearnerCfg(**config)
    # Now call .to_dict() to get a dictionary with all __post_init__ changes applied
    updated_config = config_obj.to_dict()
    learner = LatentDynamicsLearner(updated_config)
    learner.learn(reporter=tune.report)

#############################################
# Main entry point with hyperparameter tuning support via Ray Tune
#############################################
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    
    # args = parser.parse_args()

    if args_cli.tune:
        from ray import tune
        from ray.tune.search import grid_search, BasicVariantGenerator, Repeater

        base_config = LatentLearnerCfg().to_dict()
        base_config["dataset_path"] = os.path.abspath(base_config["dataset_path"])
        base_config["log_dir"] = os.path.abspath(base_config["log_dir"])
        root_log_dir = base_config["log_dir"]
        # Define hyperparameter search space.
        # base_config["batch_size"] = tune.grid_search([64, 256])
        # base_config["learning_rate"] = tune.grid_search([1e-4, 1e-3])

        base_config["kl_weight"] = tune.grid_search([1e-3, 1e-2])
        base_config["dynamics_weight"] = tune.grid_search([0.2, 1.0, 5.0])
        base_config["latent_dim"] = tune.grid_search([16, 32])
        base_config["gru_hidden_dim"] = tune.grid_search([32, 64, 128])
        base_config["dynamics_loss_to_encoder"] = tune.grid_search([False, True])

        base_config["use_tune"] = True

        analysis = tune.run(
            tune_model,
            config=base_config,
            storage_path=root_log_dir,
            resources_per_trial={"gpu": 0.3},
            metric="eval_total_loss",  # Metric to optimize
            mode="min",  # Mode for optimization (minimize or maximize)
            num_samples=2,
            max_concurrent_trials=9
        )
        print("Best config: ", analysis.get_best_config(metric="eval_total_loss", mode="min", scope="all"))
        df = analysis.dataframe()
        df.to_csv(f"{root_log_dir}/tune_results.csv", index=False)
    else:
        config = LatentLearnerCfg().to_dict()
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        config["log_dir"] = os.path.join(config["log_dir"], timestamp)
        config["use_tune"] = False
        learner = LatentDynamicsLearner(config)
        if args_cli.load_vae:
            if args_cli.load_path is None:
                raise ValueError("Please provide a path to load the model from.")
            learner.load(args_cli.load_path)
        learner.learn()