import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List



def compute_batch_k_step_errors(
    key_dimensions,
    k,
    learner,
    obs_keys: List[str],
    batch,
    quantiles: List[float] = [0.25, 0.75],  # Add quantile options
    output = "results", # or "norms"
): # -> tuple | Dict[str, Dict[str, np.ndarray]]:
    learner.model.eval()
    valid_sequences = []
    batch_size, seq_len, _ = batch['obs'].shape

    rel_norms = {key: [] for key in obs_keys}
    obs_keys = obs_keys + ['total']
    abs_norms = {key: [] for key in obs_keys}

    for b in range(batch_size):
        # Find the first invalid index in sequence b
        invalid_positions = torch.where(batch['valid_mask'][b]==False)[0]
        if len(invalid_positions) > 0:
            # T is the *first* time-step that is 'valid_mask == False'
            T = invalid_positions[0].item() - 1
        else:
            # No termination in this sequence; take entire length
            T = seq_len - 2 # T = 10

        # Number of valid timesteps is T+1 
        # (assuming the state at time T is valid, 
        #  and "terminated" means the environment ends *after* T)
        num_valid = T + 1 # if a seq_len = 12 w/o termination, num_valid = 11

        # We only form k-step predictions if we have at least k valid steps
        if num_valid >= k:
            # The set of valid start indices goes from 0 .. (num_valid - k)
            starts = torch.arange(0, num_valid - k + 1, device=learner.device)

            # For each start, record [sequence_index, start_timestep]
            traj_indices = torch.full_like(starts, b)
            valid_sequences.append(torch.stack([traj_indices, starts], dim=1))
    # Concatenate all valid sequences for this batch
    # into a single tensor of shape [num_valid_sequences, 2]    
    valid_sequences = torch.cat(valid_sequences)   # N x 2 (i of batch, j of seq) where N = num_seqs
    # Batch preparation
    obs = batch['obs'][valid_sequences[:, 0], valid_sequences[:, 1]] # N x obs_dim
    actions = torch.stack([
        batch['action'][valid_sequences[:, 0], valid_sequences[:, 1] + i] 
        for i in range(k)
    ], dim=1) # N x k x act_dim
    true_next_obs = torch.stack([obs]+[ 
        batch['next_obs'][valid_sequences[:, 0], valid_sequences[:, 1] + i] 
        for i in range(k)
    ], dim=1) # N x k x obs_dim
    # true_next_obs = torch.stack([obs, true_next_obs], dim=0) # N x k+1 x obs_dim
    # Batched forward passes
    z = learner.model.latent(obs) # N x latent_dim
    pred_curr_obs = learner.model.predictor.forward(z) # N x 1 x obs_dim
    pred_next_obs = [pred_curr_obs]
    current_z = z
    for i in range(k):
        current_z = learner.model.next_latent(current_z, actions[:, i])
        obs_pred = learner.model.predictor.forward(current_z)
        pred_next_obs.append(obs_pred)
    pred_next_obs = torch.stack(pred_next_obs, dim=1)    # N x k+1 x obs_dim
    
    # Compute errors
    for key in obs_keys:
        if key == 'total':
            true_key = true_next_obs[..., :]
            pred_key = pred_next_obs[..., :]
        else:
            dim = key_dimensions[key]['dim']
            idx = key_dimensions[key]['idx_start']
            true_key = true_next_obs[..., idx : idx + dim]
            pred_key = pred_next_obs[..., idx : idx + dim]
        
        # Compute errors
        error = true_key - pred_key
        
        abs_error = error.abs()                                # [N, k, dim]
        abs_norm = abs_error.mean(dim=-1)  # shape [N, k]
        abs_norms[key].append(abs_norm.cpu().numpy())
        if key != 'total':
            rel_error = abs_error / key_dimensions[key]['range'].to(learner.device)  # [N, k, dim]
            rel_norm = rel_error.mean(dim=-1)  # shape [N, k]
            rel_norms[key].append(rel_norm.cpu().numpy())
    
    if output == "norms":
        return abs_norms, rel_norms
    # Aggregate results with quantiles
    results = {}
    for key in obs_keys:
        abs_norm = np.concatenate(abs_norms[key], axis=0)  # shape => [N_total, k]
        results[key] = {
            'abs_norms': np.mean(abs_norm, axis=0),  # [k]
            'abs_std': np.std(abs_norm, axis=0),
            'abs_lower': np.quantile(abs_norm, quantiles[0], axis=0),
            'abs_upper': np.quantile(abs_norm, quantiles[1], axis=0),
        }
        if key != 'total':
            rel_norm = np.concatenate(rel_norms[key], axis=0)  # shape => [N_total, k]
            results[key]['rel_norms'] = np.mean(rel_norm, axis=0)
            results[key]['rel_std'] = np.std(rel_norm, axis=0)
            results[key]['rel_lower'] = np.quantile(rel_norm, quantiles[0], axis=0)
            results[key]['rel_upper'] = np.quantile(rel_norm, quantiles[1], axis=0)

    return results



def compute_k_step_errors(
    k, 
    learner, 
    obs_keys: List[str],
    quantiles: List[float] = [0.25, 0.75]  # Add quantile options
) -> Dict[str, Dict[str, np.ndarray]]:
    model = learner.model
    model.eval()


    rel_norms = {key: [] for key in obs_keys}
    obs_keys = obs_keys + ['total']
    abs_norms = {key: [] for key in obs_keys}

    
    with torch.no_grad():
        for batch in learner.eval_dyn_dataloader:
            batch = {k: v.to(learner.device) for k, v in batch.items()}
            with torch.no_grad():
                abs_norm_batch, rel_norm_batch = compute_batch_k_step_errors(
                    KEY_DIMENSIONS, k, learner, obs_keys, batch, quantiles, output="norms"
                )
            for key in obs_keys:
                for i in range(len(abs_norm_batch[key])):
                    abs_norms[key].append(abs_norm_batch[key][i])
                    if key != 'total':
                        rel_norms[key].append(rel_norm_batch[key][i])
    
    # Aggregate results with quantiles
    results = {}
    for key in obs_keys:
        abs_norm = np.concatenate(abs_norms[key], axis=0)  # shape => [N_total, k]
        results[key] = {
            'abs_norms': np.mean(abs_norm, axis=0),  # [k]
            'abs_std': np.std(abs_norm, axis=0),
            'abs_lower': np.quantile(abs_norm, quantiles[0], axis=0),
            'abs_upper': np.quantile(abs_norm, quantiles[1], axis=0),
        }
        if key != 'total':
            rel_norm = np.concatenate(rel_norms[key], axis=0)  # shape => [N_total, k]
            results[key]['rel_norms'] = np.mean(rel_norm, axis=0)
            results[key]['rel_std'] = np.std(rel_norm, axis=0)
            results[key]['rel_lower'] = np.quantile(rel_norm, quantiles[0], axis=0)
            results[key]['rel_upper'] = np.quantile(rel_norm, quantiles[1], axis=0)

    return results


def plot_k_step_errors(
    results: Dict[str, Dict[str, np.ndarray]], 
    max_k: int,
    error_band_type: str = 'quantile',
    quantiles: List[float] = [0.1, 0.9],
    fig_prefix: str = None,
    ylim_abs: tuple = None,  # Add manual ylim options
    ylim_rel: tuple = None
):
    """Enhanced plotting function with better visibility for error bands."""
    plt.ioff()
    matplotlib.style.use('ggplot')
    
    for key, metrics in results.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'K-step Prediction Errors for {key}', fontsize=16)
        k_range = np.arange(0, max_k+1)
        
        # Prepare data - ensure proper array shapes
        abs_norms = metrics['abs_norms']
        
        # Get error bands based on type
        if error_band_type == 'quantile':
            abs_lower = metrics['abs_lower']
            abs_upper = metrics['abs_upper']
            band_label = f'{quantiles[0]*100:.0f}-{quantiles[1]*100:.0f}%'
        else:
            abs_lower = abs_norms - metrics.get('abs_std', 0)
            abs_upper = abs_norms + metrics.get('abs_std', 0)
            band_label = '±1 std'
        
        # Plot absolute errors with enhanced visibility
        ax1.plot(k_range, abs_norms, 'b-', label='Mean', linewidth=2)
        ax1.fill_between(k_range, abs_lower, abs_upper, 
                        color='blue', alpha=0.15, label=band_label)
        ax1.set_title(f'Absolute Error Norm ({band_label})')
        ax1.set_xlabel('Prediction Steps (k)')
        ax1.set_ylabel('Error Magnitude')
        if ylim_abs:
            ax1.set_ylim(ylim_abs)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Plot relative errors with enhanced visibility
        if key != 'total':
            rel_norms = metrics['rel_norms']
            if error_band_type == 'quantile':
                rel_lower = metrics['rel_lower']
                rel_upper = metrics['rel_upper']
            else:
                rel_lower = rel_norms - metrics.get('rel_std', 0)
                rel_upper = rel_norms + metrics.get('rel_std', 0)
            ax2.plot(k_range, rel_norms, 'r-', label='Mean', linewidth=2)
            ax2.fill_between(k_range, rel_lower, rel_upper, 
                            color='red', alpha=0.15, label=band_label)
            ax2.set_title(f'Relative Error Norm ({band_label})')
            ax2.set_xlabel('Prediction Steps (k)')
            ax2.set_ylabel('Error Magnitude (relative)')
            if ylim_rel:
                ax2.set_ylim(ylim_rel)
            else:
                # Auto-adjust ylim for relative errors to show bands
                y_min = max(0, np.min(rel_lower) * 0.9)
                y_max = np.max(rel_upper) * 1.1
                ax2.set_ylim(y_min, y_max)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        filename = f"{fig_prefix}_{key.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot for {key} to {filename}")



if __name__ == "__main__":
    from sep_vae_dyn import KEY_DIMENSIONS, TrajectoryDataset, LatentDynamicsLearner, LatentDynamicsModel, LatentLearnerCfg
    model_path = 'model_520.pt'
    fig_prefix = 'range_real_dataset_norm'
    dataset_path = 'episodes_states_real_23dof.npy'
    variance_type = 'norm'    # quantile or std
    obs_keys = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_quat', 'joint_pos']
    pred_targets_keys = ['base_pos', 'base_lin_vel', 'base_ang_vel', 'base_quat', 'joint_pos']
    seq_len = 12
    latent_dim = 32
    gru_hidden_dim = 32
    eval_pct = 0.6
    max_k = 10
    quantiles = [0.25, 0.75]
    
    cfg = LatentLearnerCfg(
        eval_pct=eval_pct,
        dataset_path=dataset_path,
        seq_len=seq_len,
        latent_dim=latent_dim,
        gru_hidden_dim=gru_hidden_dim,
        log_dir='./logs'
    )
    
    config = cfg.to_dict()
    learner = LatentDynamicsLearner(config)
    learner.load(model_path)

    # Compute errors with quantiles
    results = compute_k_step_errors(
        k=max_k, 
        learner=learner, 
        obs_keys=pred_targets_keys,
        quantiles=quantiles  # Using 10th and 90th percentiles
    )
    
    # Plot with quantile bands
    plot_k_step_errors(
        results, 
        max_k=max_k,
        error_band_type=variance_type,  # Change to 'quantile' for quantile bands
        quantiles=quantiles,
        fig_prefix = fig_prefix
    )