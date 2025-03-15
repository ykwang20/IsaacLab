import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from omni.isaac.lab.utils import configclass
from rsl_rl.utils import store_code_state, resolve_nn_activation
from collections import defaultdict

#############################################
# Dataset remains the same
#############################################
class TrajectoryDataset(Dataset):
    def __init__(self, data_path: str, pred_targets_keys=["reward", "terminated"], seq_len=1):
        self.data = np.load(data_path, allow_pickle=True).tolist()
        self.pred_targets_keys = pred_targets_keys or []
        self.num_agents = self.data[0]['action'].shape[0]
        self.T = len(self.data)  # total timesteps in the dataset
        self.seq_len = seq_len

        # Process each field directly into shape (T, num_agents, feature_dim)
        self._preprocess_dataset(process_fn=[self._makekey_time_to_terminate, self._makekey_termination_in_seq])
        self.observations = self._process_observations()  # shape: (T, num_agents, obs_feature_dim)
        self.actions = self._process_actions()            # shape: (T, num_agents, action_dim)
        self.targets = self._process_prediction_targets()   # shape: (T, num_agents, target_feature_dim)
        self.next_observations = self._process_next_obs()   # shape: (T, num_agents, obs_feature_dim)

    def _preprocess_dataset(self, process_fn:list):
        for fn in process_fn:
            fn()

    def _makekey_time_to_terminate(self):
        time_to_terminate = torch.zeros((self.T, self.num_agents), dtype=torch.float32)
        terminated_flags = torch.stack([d["terminated"] for d in self.data], dim=0)  # shape: (T, num_agents)
        for agent_idx in range(self.num_agents):
            steps_left = 0
            for t in range(self.T - 1, -1, -1):
                if terminated_flags[t, agent_idx]:
                    steps_left = 0
                else:
                    steps_left += 1
                time_to_terminate[t, agent_idx] = steps_left
        time_to_terminate = torch.clamp(time_to_terminate, max=self.seq_len) / self.seq_len
        for t in range(self.T):
            self.data[t]["time_to_terminate"] = time_to_terminate[t].to(torch.float32)

    def _makekey_termination_in_seq(self):
        terminated_flags = torch.stack([d["terminated"] for d in self.data], dim=0)
        T, num_agents = terminated_flags.shape
        future_term = torch.zeros((T, num_agents), dtype=torch.bool)
        for t in range(T):
            end_t = min(t + self.seq_len, T)
            future_term[t] = terminated_flags[t:end_t].any(dim=0)
            self.data[t]["terminate_in_seq"] = future_term[t].to(torch.float32)

    def _process_observations(self):
        obs_list = []
        for d in self.data:
            state = d["state"]
            keys = list(state.keys())
            features = [state[k].view(self.num_agents, -1) for k in keys]
            obs_t = torch.cat(features, dim=-1)
            obs_list.append(obs_t)
        return torch.stack(obs_list, dim=0).cpu().to(torch.float32)

    def _process_actions(self):
        action_list = [d["action"] for d in self.data]
        return torch.stack(action_list, dim=0).cpu().to(torch.float32)

    def _process_prediction_targets(self):
        targets_list = []
        for d in self.data:
            components = []
            for key in self.pred_targets_keys:
                if key in d:
                    comp = d[key].view(self.num_agents, -1)
                elif "state" in d and key in d["state"]:
                    comp = d["state"][key].view(self.num_agents, -1)
                else:
                    raise KeyError(f"Key '{key}' not found in the transition data.")
                components.append(comp)
            targets_t = torch.cat(components, dim=-1)
            targets_list.append(targets_t)
        return torch.stack(targets_list, dim=0).cpu().to(torch.float32)

    def _process_next_obs(self):
        next_obs = torch.zeros_like(self.observations)
        next_obs[:-1] = self.observations[1:]
        next_obs[-1] = self.observations[-1]
        return next_obs

    def __len__(self):
        trajectories_per_agent = self.T - self.seq_len + 1
        return self.num_agents * trajectories_per_agent

    def __getitem__(self, idx):
        trajectories_per_agent = self.T - self.seq_len + 1
        agent = idx // trajectories_per_agent
        start_time = idx % trajectories_per_agent

        sample = {
            'obs': self.observations[start_time:start_time + self.seq_len, agent, :],
            'action': self.actions[start_time:start_time + self.seq_len, agent, :],
            'next_obs': self.next_observations[start_time:start_time + self.seq_len, agent, :],
            'prediction_targets': self.targets[start_time:start_time + self.seq_len, agent, :],
        }
        return sample

    def __repr__(self):
        info = [
            "Dataset Summary:",
            f"State key: {self.data[0]['state'].keys()}",
            f"Total sequences: {len(self)}",
            f"Sequence Length: {self.seq_len}",
            f"Observations shape: {self.observations.shape}",
            f"Actions shape: {self.actions.shape}",
            f"Prediction Targets shape: {self.targets.shape}"
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
            input_size = cfg["input_dim"],
            hidden_size = hidden_dim,
            num_layers = num_layers,
            dropout= cfg.get("dropout", 0.0) if num_layers > 1 else 0,
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
    def __init__(self, cfg:dict, device:str):
        """
        Wraps a base encoder network and outputs both a mean and log variance.
        """
        super().__init__()
        self.base_encoder = MLPNetwork(cfg, device=device)
        # Two separate linear layers to predict mean and logvar from the encoder’s output.
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
        
        # Create a VAE encoder
        self.encoder = VAEEncoder(self.encoder_cfg, device=self.device)
        
        # Dynamics core remains the same
        self.dynamics = GRUNetwork(self.dynamics_cfg, device=self.device)
        
        # Use predictor as the decoder for reconstruction
        self.predictor = MLPNetwork(self.predictor_cfg, device=self.device)

    def forward(self, obs, actions):
        # VAE encoding: get mean and log variance and sample latent vector using reparameterization
        mu, logvar = self.encoder(obs)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        
        # Dynamics prediction from the latent and actions
        dynamics_input = torch.cat([z, actions], dim=-1)
        next_latent_pred = self.dynamics(dynamics_input)
        
        # Reconstruction of the observation from the latent
        predction = self.predictor(z)
        
        return {
            "mu": mu,
            "logvar": logvar,
            "latent": z,
            "next_latent_pred": next_latent_pred,
            "prediction": predction
        }
    
    def latent(self, obs):
        # For evaluation, use the mean of the latent distribution
        mu, _ = self.encoder(obs)
        return mu
    
    def next_latent(self, latent, actions):
        dynamics_input = torch.cat([latent, actions], dim=-1)
        next_latent = self.dynamics(dynamics_input)
        return next_latent

#############################################
# Modified LatentDynamicsLearner including KL loss
#############################################
class LatentDynamicsLearner:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.seq_len = cfg["seq_len"]
        full_dataset = TrajectoryDataset(
            cfg["dataset_path"],
            pred_targets_keys=cfg["pred_targets_keys"],
            seq_len=self.seq_len
        )
        print(full_dataset)
        
        train_size = int((1-self.cfg["eval_pct"]) * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        self.train_dataset, self.eval_dataset = random_split(full_dataset, [train_size, eval_size])
        
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
        
        outputs = self.model(obs, actions)
        
        with torch.no_grad():
            latent = self.model.latent(obs)
            next_latent_target = self.model.latent(next_obs)
        
        # Dynamics loss (latent consistency)
        if self.cfg["dynamics_loss_to_encoder"]:
            next_latent_pred = outputs["next_latent_pred"]
        else:
            next_latent_pred = self.model.next_latent(latent, actions)

        dynamics_loss = torch.mean(
            torch.sum(nn.MSELoss(reduction='none')(next_latent_pred, next_latent_target), dim=1))
        
        # Prediction loss (decoder output vs. target state)
        prediction_loss = torch.mean(
            torch.sum(nn.MSELoss(reduction='none')(outputs["prediction"], batch["prediction_targets"]), dim=1))
        
        # KL divergence loss for the VAE part
        kl_loss = -0.5 * torch.mean(torch.sum(1 + outputs["logvar"] - outputs["mu"]**2 - torch.exp(outputs["logvar"]), dim=1))
        
        total_loss = (
            self.cfg["dynamics_weight"] * dynamics_loss +
            self.cfg["prediction_weight"] * prediction_loss +
            self.cfg["kl_weight"] * kl_loss
        )
        
        results = {
            "total_loss": total_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        
        return total_loss, results
    
    def update(self, batch):
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
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                total_loss, results = self.loss(batch)
                eval_results.append(results)
        self.model.train()
        return eval_results
    
    def learn(self):
        if self.log_dir is not None:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        results = []
        while self.current_epoch < self.cfg["epoches"]:
            for batch in self.train_dataloader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                result = self.update(batch)
                results.append(result)
            
            if self.current_epoch % self.cfg["eval_interval"] == 0 or self.current_epoch == self.cfg["epoches"]-1:
                eval_results = self.evaluate()
            
            if self.log_dir is not None:
                self.log(locals())
                if self.current_epoch % self.cfg["save_interval"] == 0 or self.current_epoch == self.cfg["epoches"]-1:
                    self.save(os.path.join(self.log_dir, f"model_{self.current_epoch}.pt"))
            self.current_epoch += 1
            
    def save(self, path:str):
        saved_dict = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.current_epoch,
            "optimizer_state": self.optimizer.state_dict()
        }
        torch.save(saved_dict, path)

    def load(self, path:str, load_optimizer:bool=True):
        loaded_dict = torch.load(path)
        self.model.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state"])
        self.current_epoch = loaded_dict["epoch"]

    def log(self, locs:dict):
        info_str = ""
        results_list = locs["results"]
        results_dict = defaultdict(list)
        for result_key in results_list[0].keys():
            for result in results_list:
                results_dict[result_key].append(result[result_key])
        
        for result_key in results_dict:
            log_key = result_key
            if "loss" in result_key:
                log_key = f"Loss/{result_key}"
            elif "grad" in result_key:
                log_key = f"Grad/{result_key}"
            self.writer.add_scalar(f"{log_key}_ave", np.mean(results_dict[result_key]), self.current_epoch)
            self.writer.add_scalar(f"{log_key}_std", np.std(results_dict[result_key]), self.current_epoch)
        self.writer.add_scalar("lr", self.learning_rate, self.current_epoch)
        info_str += f"Epoch {self.current_epoch}, Total loss ave: {np.mean(results_dict['total_loss'])}. "
        
        if self.current_epoch % self.cfg["eval_interval"] == 0 or self.current_epoch == self.cfg["epoches"]-1:
            eval_results_list = locs["eval_results"]
            eval_results_dict = defaultdict(list)
            for result_key in eval_results_list[0].keys():
                for result in eval_results_list:
                    eval_results_dict[result_key].append(result[result_key])
            for result_key in eval_results_dict:
                log_key = f"Eval/{result_key}"
                self.writer.add_scalar(f"{log_key}_ave", np.mean(eval_results_dict[result_key]), self.current_epoch)
            info_str += f"Eval loss ave: {np.mean(eval_results_dict['total_loss'])}"
        
        print(info_str)

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
        hidden_dims = [],
        activation_at_end = [False],
        name = "gru_out"
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
    gru_hidden_dim = 16
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
    device = 'cuda:0'

    """dataset"""
    dataset_path = "episodes_states.npy"
    eval_pct = 0.2

    """logging"""
    logger = "tensorboard"
    log_dir = "./logs/VAE_decoupled_loss"
    save_interval = 10
    eval_interval = 2

    """training hyper parameters"""
    seq_len = 24
    epoches = 150
    batch_size = 64
    learning_rate = 0.0001
    grad_norm_clip = 10
    dynamics_weight = 1
    prediction_weight = 1
    kl_weight = 1e-3  # weight for KL divergence term
    dynamics_loss_to_encoder = False # whether propagate dynamics loss through encoder

    """model architecture"""
    # pred_targets_keys contains all state information for reconstruction
    pred_targets_keys = ['base_pos', 'base_lin_vel_w', 'base_ang_vel_w', 'base_lin_vel', 
                         'base_ang_vel', 'projected_gravity', 'base_quat', 'joint_pos']   # full observation
    model_cfg = LatentModelCfg(
        obs_dim=57,
        act_dim=37,
        latent_dim=16,
        pred_targets_dim=57,
        encoder_cfg = MLPCfg(
            hidden_dims = [128],
            activation_at_end = [True, False],
            activation="crelu"
        ),
        predictor_cfg = MLPCfg(
            hidden_dims = [128],
            activation_at_end = [True, False],
            activation="crelu"
        ),
        dynamics_cfg = GRUCfg(
            hidden_dim=64
        )
    )

if __name__ == "__main__":
    # Load trajectory data and configuration
    config = LatentLearnerCfg().to_dict()
    
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config["log_dir"] = os.path.join(config["log_dir"], time_str)
    # Initialize latent dynamics learner with the modified (VAE) model
    latent_learner = LatentDynamicsLearner(config)
    
    # Training loop
    latent_learner.learn()
