import datetime
from tune_architecture import StatesDataset, KEY_DIMENSIONS, MLPNetwork, MLPCfg, VAEEncoder, _aggregate_stats
from omni.isaac.lab.utils import configclass
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
import argparse
import random


class FailureDataset(StatesDataset):
    """
    Make a more balanced dataset for failure classifier through resampling techniques
    """
    def __init__(self, data_path: str, obs_keys, pred_targets_keys, balance_method='oversample', oversample_ratio=9.0, undersample_ratio=0.2):
    #    """
    #     Args:
    #         data_path: Path to the dataset file
    #         obs_keys: Observation keys to use
    #         pred_targets_keys: Target keys to predict
    #         balance_method: Method to balance dataset ('oversample', 'undersample', 'weighted', 'combined')
    #         oversample_ratio: How much to oversample minority class (relative to original size)
    #         undersample_ratio: What portion of majority class to keep (0-1)
    #     """
        # Initialize the parent class first to load and process the data
        super().__init__(data_path, obs_keys, pred_targets_keys)
        
        # Get indices for failed and non-failed samples
        self.failed_indices = torch.nonzero(self.failed.view(-1), as_tuple=False).squeeze().tolist()
        # Handle case when failed_indices is a single number
        if not isinstance(self.failed_indices, list):
            self.failed_indices = [self.failed_indices]
            
        self.normal_indices = torch.nonzero(~self.failed.view(-1).bool(), as_tuple=False).squeeze().tolist()
        # Handle case when normal_indices is a single number
        if not isinstance(self.normal_indices, list):
            self.normal_indices = [self.normal_indices]
        
        # Print original dataset statistics
        self.original_length = super().__len__()
        print(f"Original dataset statistics:")
        print(f"  Total samples: {len(self)}")
        print(f"  Failed samples: {len(self.failed_indices)} ({len(self.failed_indices)/len(self)*100:.2f}%)")
        print(f"  Normal samples: {len(self.normal_indices)} ({len(self.normal_indices)/len(self)*100:.2f}%)")
        
        self.balance_method = balance_method
        self.oversample_ratio = oversample_ratio
        self.undersample_ratio = undersample_ratio
        
        # Store original length
        
        
        # Apply balancing method
        if balance_method == 'oversample':
            self._oversample()
        elif balance_method == 'undersample':
            self._undersample()
        elif balance_method == 'combined':
            self._combined_sampling()
        elif balance_method == 'weighted':
            # For weighted, we don't modify the dataset but will use a sampler
            pass
        else:
            print(f"Warning: Unknown balance method '{balance_method}', using original dataset")
    
    def _oversample(self):
        """
        Oversample the minority class (failed samples) by creating an index mapping
        that includes repeated indices for the minority class
        """
        if len(self.failed_indices) == 0:
            print("Warning: No failed samples found, cannot oversample")
            return
            
        # Calculate how many times to repeat minority samples
        target_failed_samples = int(len(self.failed_indices) * self.oversample_ratio)
        
        # Create index mapping
        self.index_map = self.normal_indices.copy()
        
        # Add repeated failed indices
        if target_failed_samples > 0:
            # Sample with replacement to get the desired number of minority samples
            oversampled_failed = random.choices(self.failed_indices, k=target_failed_samples)
            self.index_map.extend(oversampled_failed)
        
        # Shuffle the indices
        random.shuffle(self.index_map)
        
        print(f"After oversampling:")
        print(f"  Total samples: {len(self.index_map)}")
        print(f"  Failed samples: {target_failed_samples} ({target_failed_samples/len(self.index_map)*100:.2f}%)")
        print(f"  Normal samples: {len(self.normal_indices)} ({len(self.normal_indices)/len(self.index_map)*100:.2f}%)")
    
    def _undersample(self):
        """
        Undersample the majority class (normal samples) by creating an index mapping
        that includes only a subset of the majority class
        """
        if len(self.failed_indices) == 0:
            print("Warning: No failed samples found, keeping original dataset")
            self.index_map = range(self.original_length)
            return
            
        # Calculate how many majority samples to keep
        samples_to_keep = int(len(self.normal_indices) * self.undersample_ratio)
        
        # Randomly select majority samples to keep
        kept_normal_indices = random.sample(self.normal_indices, samples_to_keep)
        
        # Create index mapping with all minority samples and subset of majority
        self.index_map = self.failed_indices + kept_normal_indices
        
        # Shuffle the indices
        random.shuffle(self.index_map)
        
        print(f"After undersampling:")
        print(f"  Total samples: {len(self.index_map)}")
        print(f"  Failed samples: {len(self.failed_indices)} ({len(self.failed_indices)/len(self.index_map)*100:.2f}%)")
        print(f"  Normal samples: {samples_to_keep} ({samples_to_keep/len(self.index_map)*100:.2f}%)")
    
    def _combined_sampling(self):
        """
        Combine undersampling the majority class and oversampling the minority class
        """
        if len(self.failed_indices) == 0:
            print("Warning: No failed samples found, using undersampling only")
            self._undersample()
            return
            
        # Undersample the majority class
        samples_to_keep = int(len(self.normal_indices) * self.undersample_ratio)
        kept_normal_indices = random.sample(self.normal_indices, samples_to_keep)
        
        # Oversample the minority class
        target_failed_samples = int(len(self.failed_indices) * self.oversample_ratio)
        oversampled_failed = random.choices(self.failed_indices, k=target_failed_samples)
        
        # Create index mapping
        self.index_map = kept_normal_indices + oversampled_failed
        
        # Shuffle the indices
        random.shuffle(self.index_map)
        
        print(f"After combined sampling:")
        print(f"  Total samples: {len(self.index_map)}")
        print(f"  Failed samples: {target_failed_samples} ({target_failed_samples/len(self.index_map)*100:.2f}%)")
        print(f"  Normal samples: {samples_to_keep} ({samples_to_keep/len(self.index_map)*100:.2f}%)")
    
    def get_sampler(self):
        """
        Creates a weighted sampler for imbalanced binary classification
        Only used when balance_method='weighted'
        """
        if self.balance_method != 'weighted':
            return None
            
        # Calculate class weights inversely proportional to class frequencies
        num_samples = len(self)
        num_failed = len(self.failed_indices)
        num_normal = len(self.normal_indices)
        
        failed_weight = num_samples / (2.0 * num_failed) if num_failed > 0 else 1.0
        normal_weight = num_samples / (2.0 * num_normal) if num_normal > 0 else 1.0
        
        # Assign weights to each sample
        weights = torch.ones(num_samples)
        for i in self.failed_indices:
            weights[i] = failed_weight
        for i in self.normal_indices:
            weights[i] = normal_weight
            
        # Create and return the sampler
        return WeightedRandomSampler(weights, len(weights))
    
    def __len__(self):
        """Override length to return the length of the index mapping if it exists"""
        if hasattr(self, 'index_map'):
            return len(self.index_map)
        return self.original_length
    
    def __getitem__(self, idx):
        """Map the requested index to the actual data index if resampling is used"""
        if hasattr(self, 'index_map'):
            actual_idx = self.index_map[idx]
            return super().__getitem__(actual_idx)
        return super().__getitem__(idx)


class FailureClassifierLearner:
    def __init__(self, cfg:dict):
        self.cfg = cfg
        self.device = cfg['device']
        self.encoder = VAEEncoder(self.cfg['encoder_cfg'], device=self.device)
        loaded_dict = torch.load(self.cfg['encoder_path'])['model_state_dict']
        encoder_dict = {k[8:]:v for k,v in loaded_dict.items() if ("encoder" in k)}

        self.encoder.load_state_dict(encoder_dict, strict=True)
        self.classifier = MLPNetwork(self.cfg['classifier_cfg'], device=self.device)
        self.learning_rate = cfg["learning_rate"]
        self.current_epoch = 0

        
        # Use FailureDataset instead of StatesDataset for data balancing
        dataset = FailureDataset(
            data_path=self.cfg['dataset_path'],
            obs_keys=self.cfg['obs_keys'],
            pred_targets_keys=self.cfg['pred_targets_keys'],
            balance_method=self.cfg.get('balance_method', 'oversample'),
            oversample_ratio=self.cfg.get('oversample_ratio', 9.0),
            undersample_ratio=self.cfg.get('undersample_ratio', 0.2)
        )
        print("balanced failure dataset:\n", dataset)

        train_size_dyn = int((1 - self.cfg["eval_pct"]) * len(dataset))
        eval_size_dyn = len(dataset) - train_size_dyn

        self.train_dataset, self.eval_dataset = random_split(
            dataset, [train_size_dyn, eval_size_dyn]
        )
        
        # For weighted sampling, use the sampler from the dataset
        if cfg.get('balance_method', '') == 'weighted':
            sampler = dataset.get_sampler()
            if sampler is not None:
                # Apply sample weights only to training data
                train_indices = self.train_dataset.indices
                train_weights = torch.tensor([sampler.weights[i] for i in train_indices])
                train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
                self.train_dataloader = DataLoader(self.train_dataset, batch_size=cfg["batch_size"], sampler=train_sampler)
            else:
                self.train_dataloader = DataLoader(self.train_dataset, batch_size=cfg["batch_size"], shuffle=True)
        else:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=cfg["batch_size"], shuffle=True)
            
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=cfg["batch_size"], shuffle=False)

        # Option to use class weights in loss function
        if cfg.get("use_class_weights", False):
            # Calculate class weights based on inverse of class frequencies
            num_samples = len(dataset)
            num_failed = len(dataset.failed_indices)
            num_normal = num_samples - num_failed
            
            if num_failed > 0:
                pos_weight = num_normal / num_failed
            else:
                pos_weight = 1.0
                
            # Apply the specified weight ratio or use the computed one
            self.pos_weight = torch.tensor([cfg.get("pos_class_weight", pos_weight)]).to(self.device)
        else:
            self.pos_weight = None

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

        # self.load(path='real_classifier_6.pt')
        # scripted=torch.jit.script(self.classifier)
        # scripted.save('real_classifier_6_scripted.pt')


        # Logging
        self.log_dir = cfg["log_dir"]
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()
        self.is_tuning = cfg.get("use_tune", False)
        if not self.is_tuning and self.log_dir is not None:
            if self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                from dataclasses import dataclass, asdict
                @dataclass
                class Empty:
                    pass
                empty_instance = Empty()
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(env_cfg=empty_instance, runner_cfg=self.cfg, alg_cfg=None, policy_cfg=None)



    def loss(self, batch):
        obs = batch["obs"]
        failed = batch['failed'].unsqueeze(-1).float()
        with torch.no_grad():
            latents, _ = self.encoder(obs)
        failed_logits = self.classifier(latents)
        
        # Use weighted BCE loss if pos_weight is provided
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(failed_logits, failed, pos_weight=self.pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(failed_logits, failed)
        
        # Calculate additional performance metrics
        with torch.no_grad():
            predictions = (torch.sigmoid(failed_logits) > 0.5).float()
            accuracy = (predictions == failed).float().mean()
            
            # Calculate precision, recall, and F1 (handling division by zero)
            true_positives = (predictions * failed).sum()
            predicted_positives = predictions.sum()
            actual_positives = failed.sum()
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else torch.tensor(0.0).to(self.device)
            recall = true_positives / actual_positives if actual_positives > 0 else torch.tensor(0.0).to(self.device)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0).to(self.device)
        
        result = {
            "total_loss": loss.item(),
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item()
        }
        return loss, result
    

    def update(self, batch):
        self.classifier.train()
        total_loss, results = self.loss(batch)
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.classifier.parameters(), self.cfg["grad_norm_clip"])
        self.optimizer.step()
        results["grad_norm"] = grad_norm.item()
        return results


    def evaluate(self):
        self.classifier.eval()
        eval_results = []
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                _, stats = self.loss(batch)
            eval_results.append(stats)
        self.classifier.train()
        return eval_results


    def learn(self, reporter=None):
        """
        Main learn function to train the model.
        """
        while self.current_epoch < self.cfg['epoches']:
            results = {'train_results':[], 'eval_results': None}
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                result = self.update(batch)
                results['train_results'].append(result)
            if self.current_epoch % self.cfg["eval_interval"] == 0 or self.current_epoch == self.cfg['epoches'] - 1:
                results['eval_results'] = self.evaluate()
            
            processed_results = self.postprocess_results(results)
            if reporter is not None and self.is_tuning:
                reporter(processed_results)
            
            # if not self.is_tuning and self.log_dir is not None:
            if self.log_dir is not None:
                if self.current_epoch % self.cfg["save_interval"] == 0 or self.current_epoch == self.cfg["epoches"] - 1:
                    # self.save(os.path.join(self.log_dir, f"model_{self.current_epoch}.pt"))
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    name = f"model_{len(self.cfg['obs_keys'])}keys_lr{self.learning_rate}_osr{self.cfg['oversample_ratio']}_{self.current_epoch}_{timestamp}.pt"
                    self.save(os.path.join(self.log_dir, name))

            self.current_epoch += 1

    def postprocess_results(self, results):
        required_keys = ["total_loss", "recall", "precision", "f1", "accuracy", "grad_norm"]

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
            print_str = (f"[classifier] Epoch {self.current_epoch} => "
                        f"train_total_loss={out_dict['train_total_loss_ave']:.4f}, "
                        f"eval_total_loss={out_dict['eval_total_loss_ave']:.4f}")
            print(print_str)

        # Return out_dict for reporter
        return out_dict
    
    def save(self, path: str):
        saved_dict = {
            "model_state_dict": self.classifier.state_dict(),
            "epoch": self.current_epoch,
            "optimizer_state": self.optimizer.state_dict()
        }
        torch.save(saved_dict, path)
        scripted=torch.jit.script(self.classifier)
        scripted.save('real_classifier_9_scripted.pt')

    
    def load(self, path: str, load_dynamics = True, load_optimizer: bool = True):
        loaded_dict = torch.load(path)
        self.classifier.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state"])



@configclass
class ClassifierLearnerCfg:
    device = 'cuda:0'
    use_tune = False
    dataset_path = "real_dataset_100_Apr20_wnoise.npy"
    encoder_path = "./logs/real_all/20250420-200936/model_5keys_lr0.001_300_20250420-203722.pt"
    
    
    eval_pct = 0.25
    logger = "wandb" #"tensorboard"
    wandb_project = "failure_classifier_16dim"
    log_dir = "./logs/real_classifier"  # base log directory
    save_interval = 50
    eval_interval = 2
    batch_size = 64
    learning_rate = 0.0001
    grad_norm_clip = 10
    epoches = 100

    # Balance options for handling imbalanced data (1:9 negative to positive ratio)
    balance_method = 'oversample'  # Options: 'oversample', 'undersample', 'weighted', 'combined'
    oversample_ratio = 9.0 #9.0  # How much to oversample minority class (for 'oversample' and 'combined')
    undersample_ratio = 0.2  # What portion of majority class to keep (for 'undersample' and 'combined')
    use_class_weights = False  # Whether to use weighted loss function
    pos_class_weight = 9.0  # Weight for positive class in BCE loss

    # define keys
    obs_keys = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_euler', 'joint_pos'] 
    pred_targets_keys = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_euler', 'joint_pos']

    act_dim = 23
    obs_dim = 57 # place holder. Automatically adjusted with obs keys
    pred_targets_dim = 7 # place holder. Automatically adjusted with pred keys
    latent_dim = 16
    gru_hidden_dim = 32
    
    classifier_cfg = MLPCfg()
    encoder_cfg = MLPCfg()

    def __post_init__(self):
        # auto compute dimension
        self.obs_dim = sum(KEY_DIMENSIONS[key]['dim'] for key in self.obs_keys)
        self.pred_targets_dim = sum(KEY_DIMENSIONS[key]['dim'] for key in self.pred_targets_keys)
        self.classifier_cfg=MLPCfg(
            input_dim = self.latent_dim,
            output_dim = 1,
            hidden_dims=[256, 128],
            activation_at_end=[True, True, False],
            name = 'classifier'
        )
        self.encoder_cfg=MLPCfg(
            input_dim = self.obs_dim,
            output_dim = self.latent_dim,
            hidden_dims=[256, 128],
            activation_at_end=[True, True, False],
            name = 'encoder'
        )


def tune_model(config):
    config['pred_targets_keys'] = config["obs_keys"]
    config_obj = ClassifierLearnerCfg(**config)
    updated_config = config_obj.to_dict()
    learner = FailureClassifierLearner(updated_config)
    learner.learn(reporter=tune.report)


if __name__ == "__main__":

    base_config = ClassifierLearnerCfg().to_dict()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_config["log_dir"] = os.path.join(base_config["log_dir"], timestamp)
    base_config["use_tune"] = False
    learner = FailureClassifierLearner(base_config)
    learner.learn()