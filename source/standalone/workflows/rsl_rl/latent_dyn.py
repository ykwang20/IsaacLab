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

class TrajectoryDataset(Dataset):
    def __init__(self, data_path: str, pred_targets_keys=["reward", "terminated"], seq_len=1):
        """
        Args:
            data_path (str): Path to the .npy file containing the list of transitions.
            pred_targets_keys (list, optional): List of keys in each transition to be used as prediction targets.
                If a key is not at the top level, it will be looked up under 'state'.
            seq_len (int): The desired sequence length (number of contiguous time steps) for each trajectory sample.
        """
        # Load the data (assumes it was saved with np.save using allow_pickle=True)
        self.data = np.load(data_path, allow_pickle=True).tolist()
        self.pred_targets_keys = pred_targets_keys or []
        self.num_agents = self.data[0]['action'].shape[0]
        self.T = len(self.data)  # total timesteps in the dataset
        self.seq_len = seq_len

        # Process each field directly into shape (T, num_agents, feature_dim)
        self.observations = self._process_observations()  # shape: (T, num_agents, obs_feature_dim)
        self.actions = self._process_actions()            # shape: (T, num_agents, action_dim)
        self.targets = self._process_prediction_targets()   # shape: (T, num_agents, target_feature_dim)
        self.next_observations = self._process_next_obs()   # shape: (T, num_agents, obs_feature_dim)

    def _process_observations(self):
        """Process observations into shape (T, num_agents, obs_feature_dim)."""
        obs_list = []
        for d in self.data:
            state = d["state"]
            # Instead of iterating agent-by-agent to build a flat vector,
            # we concatenate each state key along the feature dimension for the entire batch of agents.
            # Each state[key] is assumed to have shape (num_agents, *feature_shape).
            keys = list(state.keys())
            # Flatten the feature dimensions for each key (keeping the agent dimension)
            features = [state[k].view(self.num_agents, -1) for k in keys]
            # Concatenate along the last dimension: result shape (num_agents, total_obs_feature_dim)
            obs_t = torch.cat(features, dim=-1)
            obs_list.append(obs_t)
        # Stack along time dimension: (T, num_agents, total_obs_feature_dim)
        return torch.stack(obs_list, dim=0).cpu().to(torch.float32)

    def _process_actions(self):
        """Process actions into shape (T, num_agents, action_dim)."""
        action_list = [d["action"] for d in self.data]  # each is (num_agents, action_dim)
        return torch.stack(action_list, dim=0).cpu().to(torch.float32)

    def _process_prediction_targets(self):
        """
        For each timestep, build a target tensor of shape (num_agents, target_feature_dim)
        by concatenating the specified keys. For each key, if it's not found at the top level,
        it is retrieved from d['state'].
        """
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
            # Concatenate along feature dimension: shape (num_agents, total_target_feature_dim)
            targets_t = torch.cat(components, dim=-1)
            targets_list.append(targets_t)
        return torch.stack(targets_list, dim=0).cpu().to(torch.float32)

    def _process_next_obs(self):
        """
        Create next observations by shifting observations by one timestep.
        The last timestep is duplicated.
        """
        next_obs = torch.zeros_like(self.observations)
        next_obs[:-1] = self.observations[1:]
        next_obs[-1] = self.observations[-1]
        return next_obs

    def __len__(self):
        # Each agent has (T - seq_len + 1) possible trajectory segments.
        trajectories_per_agent = self.T - self.seq_len + 1
        return self.num_agents * trajectories_per_agent

    def __getitem__(self, idx):
        """
        Convert the flat index to (agent, starting time) pair and return a trajectory segment
        of length seq_len for each field.
        """
        trajectories_per_agent = self.T - self.seq_len + 1
        agent = idx // trajectories_per_agent
        start_time = idx % trajectories_per_agent

        sample = {
            'obs': self.observations[start_time:start_time + self.seq_len, agent, :],
            'action': self.actions[start_time:start_time + self.seq_len, agent, :],
            'next_obs': self.next_observations[start_time:start_time + self.seq_len, agent, :],
            'prediction_targets': self.targets[start_time:start_time + self.seq_len, agent, :]
        }
        return sample

    def __repr__(self):
        info = [
            "Dataset Summary:",
            f"Total sequences: {len(self)}",
            f"Sequence Length: {self.seq_len}",
            f"Observations shape: {self.observations.shape}",
            f"Actions shape: {self.actions.shape}",
            f"Prediction Targets shape: {self.targets.shape}"
        ]
        return "\n".join(info)
    


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
        # Forward pass through GRU
        hiddens, _ = self.gru(x)  # out: batch_size, seq_len, hidden_dim
        # generate output embedding
        out = self.fc(hiddens)    # out: (batch_size, seq_len, output_dim)
        return out
    


class LatentDynamicsModel(nn.Module):
    """Latent dynamics model following IsaacLab's architecture patterns"""
    def __init__(self, 
                 cfg: dict,
                 **kwargs):
        super().__init__()
        
        # Encoder network
        self.device = cfg["device"]
        self.encoder_cfg = cfg["encoder_cfg"]
        self.dynamics_cfg = cfg["dynamics_cfg"]
        self.predictor_cfg = cfg.get("predictor_cfg", None)
    
        self.encoder = MLPNetwork(self.encoder_cfg, device = self.device)
        
        # Dynamics core
        self.dynamics = GRUNetwork(self.dynamics_cfg, device = self.device)

        self.predictor = MLPNetwork(self.predictor_cfg, device = self.device)


    def forward(self, obs, actions):
        # Encode current observation
        latent = self.encoder(obs)  # (batch_size, seq_len, latent_dim)
        
        # Predict next latent state
        dynamics_input = torch.cat([latent, actions], dim=-1) # (batch_size, seq_len, latent_dim+act_dim)
        next_latent_pred = self.dynamics(dynamics_input) # (batch_size, seq_len, latent_dim)
        
        # Predict extra information
        prediction = self.predictor(latent) ## (batch_size, seq_len, pred_dim)
        
        return {
            "latent": latent,
            "next_latent_pred": next_latent_pred,
            "prediction": prediction
        }
    
    def latent(self, obs):
        return self.encoder(obs) # (batch_size, seq_len, latent_dim)


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
        
        # Split dataset into training  and evaluation
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
        
        # Forward pass
        outputs = self.model(obs, actions)
        
        with torch.no_grad():
            next_latent_target = self.model.latent(next_obs)
        
        # Compute loss over the whole sequence
        # dynamics loss, latent consistency
        dynamics_loss = torch.mean(
            torch.sum(
                nn.MSELoss(reduction='none')(outputs["next_latent_pred"], next_latent_target), 
                dim=1  # Sum over seq_len
        ))

        # prediction loss
        pred_loss = torch.mean(
            torch.sum(
                nn.MSELoss(reduction='none')(outputs["prediction"], batch["prediction_targets"]), dim=1
        ))
        
        total_loss = (
            self.cfg["dynamics_weight"] * dynamics_loss +
            self.cfg["prediction_weight"] * pred_loss
        )
        
        results = {
            "total_loss": total_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "predction_loss": pred_loss.item(),
        }

        return total_loss, results
    
    def update(self, batch):
        total_loss, results = self.loss(batch)

        # Optimization step
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
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

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
                result = latent_learner.update(batch)
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
        loaded_dict = torch.load(path, weights_only=False)
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
    log_dir = "./logs/latent_dynamics_gru"
    save_interval = 10
    eval_interval = 2

    """training hyper parameters"""
    seq_len = 24
    epoches = 150
    batch_size = 64
    learning_rate = 0.0001
    grad_norm_clip = 10
    dynamics_weight = 1
    prediction_weight = 5

    """model architecture"""
    pred_targets_keys = ["base_lin_vel", "base_quat"] #, "terminated"] # 8dims
    model_cfg = LatentModelCfg(
        obs_dim=57,
        act_dim=37,
        latent_dim=16,
        pred_targets_dim=7,
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
            hidden_dim=32
        )
    )


# Example usage
if __name__ == "__main__":
    # Load trajectory data
    config = LatentLearnerCfg().to_dict()
    
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config["log_dir"] = os.path.join(config["log_dir"], time_str)
    # Initialize manager with config
    latent_learner = LatentDynamicsLearner(config)
    
    # Training loop
    latent_learner.learn()