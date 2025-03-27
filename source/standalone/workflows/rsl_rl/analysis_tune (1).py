import os
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune.analysis import ExperimentAnalysis
import numpy as np

# Define your filtering criteria
def trial_filter(trial, required_params:dict):
    # 1. Filter out errored trials
    if trial.status != "TERMINATED":
        return False
        
    # 2. Filter by config parameters
    config = trial.config
    
    for key, value in required_params.items():
        if config.get(key) != value:
            return False
            
    return True


def get_best_trial(filtered_trials, analysis, metric, mode='min', scope='last'):
    """Flexible best trial selection based on evaluation scope"""
    trial_scores = []
    
    for trial in filtered_trials:
        # Get trial's full result history
        trial_df = analysis.trial_dataframes[trial.trial_id]
        
        if scope == 'last':
            # Use last reported value
            score = trial.last_result.get(metric, np.nan)
        elif scope == 'all':
            # Use best value from all iterations
            if mode == 'min':
                score = trial_df[metric].min()
            else:
                score = trial_df[metric].max()
        else:
            raise ValueError(f"Invalid scope: {scope}. Use 'last' or 'all'")
        
        trial_scores.append((trial, score))
    
    # Filter out trials with missing metrics
    valid_trials = [(t, s) for t, s in trial_scores if not np.isnan(s)]
    if not valid_trials:
        raise ValueError("No trials with valid metric values found")
    
    # Sort by score according to mode
    reverse = (mode == 'max')
    sorted_trials = sorted(valid_trials, key=lambda x: x[1], reverse=reverse)
    
    return sorted_trials[0][0]


def plot_learning_curves(trial_df, fig_name, tuned_params):
    plt.figure(figsize=(20, 12))  # 3 columns, 2 rows
    
    # Helper functions
    def get_smoothed_convergence(data_series, window_frac=0.05, max_iter=500):
        """Calculate convergence using first max_iter iterations"""
        clean_data = data_series[data_series.index <= max_iter].dropna()
        if len(clean_data) == 0:
            return np.nan
        window_size = max(int(len(clean_data) * window_frac), 5)
        return clean_data.iloc[-window_size:].mean()
    
    def get_adaptive_ylim(data_series, convergence_val, n_std=3):
        clean_data = data_series.dropna()
        if len(clean_data) < 10 or np.isnan(convergence_val):
            return [clean_data.min() * 0.9, clean_data.max() * 1.1]
        
        # Use percentiles to ignore extreme outliers
        lower_bound = np.percentile(clean_data, 0)  # 1st percentile
        upper_bound = np.percentile(clean_data, 95)  # 99th percentile
        
        # Calculate IQR-based padding for the main data distribution
        q25, q75 = np.percentile(clean_data, [25, 75])
        iqr = q75 - q25
        iqr_padding = iqr * 1.5
        
        # Create range that captures most data but allows convergence visibility
        data_min = max(lower_bound - iqr_padding, 0)
        data_max = upper_bound + iqr_padding
        
        # Ensure convergence value is visible
        final_min = min(data_min, convergence_val * 0.95)
        final_max = max(data_max, convergence_val * 1.05)
        
        # Add small buffer for visual clarity
        return [final_min * 0.8, final_max * 1.5]
    
    def add_convergence_lines(ax, train_val, eval_val, colors, x_pos):
        """Add convergence markers at iteration 500"""
        ymin, ymax = ax.get_ylim()
        text_offset = (ymax - ymin) * 0.05
        
        # Vertical marker at training end
        ax.axvline(500, color='gray', linestyle=':', alpha=0.5, zorder=0)
        
        # Training convergence
        ax.axhline(train_val, color=colors[0], linestyle='--', alpha=0.7)
        ax.text(x_pos*0.95, train_val + text_offset, f'{train_val:.4f}',
                color=colors[0], ha='right', va='bottom')
        
        # Evaluation convergence
        if not np.isnan(eval_val):
            ax.axhline(eval_val, color=colors[1], linestyle='--', alpha=0.7)
            ax.text(x_pos*0.95, eval_val - text_offset, f'{eval_val:.4f}',
                    color=colors[1], ha='right', va='top')

    # ======================
    # Row 1: Dynamics Metrics
    # ======================
    
    # Subplot 1: Dynamics Loss (500-1500 iterations)
    ax1 = plt.subplot(2, 3, 1)
    dyn_data = trial_df[trial_df["training_iteration"] >= 500]
    train_dyn = dyn_data["train_dynamics_loss_ave"]
    
    # Calculate dynamics convergence using last 5% of dynamics training (500-1500)
    dyn_conv_window_frac = 0.05
    dyn_max_iter = 1500
    conv_train_dyn = get_smoothed_convergence(train_dyn, dyn_conv_window_frac, dyn_max_iter)
    
    ax1.plot(train_dyn.index, train_dyn, color="blue", alpha=0.8, label="Train Dynamics Loss")
    
    # Evaluation curve
    eval_dyn = dyn_data.dropna(subset=["eval_dynamics_loss_ave"])
    conv_eval_dyn = np.nan
    if not eval_dyn.empty:
        ax1.plot(eval_dyn["training_iteration"], eval_dyn["eval_dynamics_loss_ave"], 
                'o--', color="orange", alpha=0.8, label="Eval Dynamics Loss")
        conv_eval_dyn = get_smoothed_convergence(eval_dyn["eval_dynamics_loss_ave"], 
                                                dyn_conv_window_frac, dyn_max_iter)
    
    # Formatting
    ax1.set(xlabel="Training Iteration", ylabel="Loss", xlim=(500, 1500),
           title="Dynamics Loss", ylim=get_adaptive_ylim(train_dyn, conv_train_dyn))
    add_convergence_lines(ax1, conv_train_dyn, conv_eval_dyn, ('blue', 'orange'), x_pos=1500)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Subplot 2: Relative Dynamics Error (500-1500 iterations)
    ax2 = plt.subplot(2, 3, 2)
    train_dyn_err = dyn_data["train_relative_dynamics_error_ave"]
    conv_train_dyn_err = get_smoothed_convergence(train_dyn_err, dyn_conv_window_frac, dyn_max_iter)
    
    ax2.plot(train_dyn_err.index, train_dyn_err, color="navy", alpha=0.8, label="Train Rel. Error")
    
    # Evaluation curve
    eval_dyn_err = dyn_data.dropna(subset=["eval_relative_dynamics_error_ave"])
    conv_eval_dyn_err = np.nan
    if not eval_dyn_err.empty:
        ax2.plot(eval_dyn_err["training_iteration"], 
                eval_dyn_err["eval_relative_dynamics_error_ave"],
                '^--', color="darkorange", alpha=0.8, label="Eval Rel. Error")
        conv_eval_dyn_err = get_smoothed_convergence(eval_dyn_err["eval_relative_dynamics_error_ave"], 
                                                    dyn_conv_window_frac, dyn_max_iter)
    
    # Formatting
    ax2.set(xlabel="Training Iteration", ylabel="Relative Error", xlim=(500, 1500),
           title="Relative Dynamics Error", ylim=get_adaptive_ylim(train_dyn_err, conv_train_dyn_err))
    add_convergence_lines(ax2, conv_train_dyn_err, conv_eval_dyn_err, ('navy', 'darkorange'), x_pos=1500)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ========================
    # Row 2: Prediction Metrics (0-600 iterations)
    # ========================
    
    # Subplot 3: KL Loss (0-600 iterations)
    ax3 = plt.subplot(2, 3, 3)
    kl_data = trial_df[trial_df["training_iteration"] <= 600]
    train_kl = kl_data["train_kl_loss_ave"]
    ax3.plot(train_kl.index, train_kl, color="purple", alpha=0.8, label="Train KL Loss")
    
    # Evaluation curve
    eval_kl = kl_data.dropna(subset=["eval_kl_loss_ave"])
    if not eval_kl.empty:
        ax3.plot(eval_kl["training_iteration"], eval_kl["eval_kl_loss_ave"],
                's--', color="brown", alpha=0.8, label="Eval KL Loss")
    
    # Convergence values (0-500 iterations)
    conv_train_kl = get_smoothed_convergence(train_kl)
    conv_eval_kl = get_smoothed_convergence(eval_kl["eval_kl_loss_ave"]) if not eval_kl.empty else np.nan
    
    ax3.set(xlim=(0, 600), ylim=get_adaptive_ylim(train_kl, conv_train_kl),
           xlabel="Training Iteration", ylabel="Loss", title="KL Loss")
    add_convergence_lines(ax3, conv_train_kl, conv_eval_kl, ('purple', 'brown'), x_pos = 500)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Subplot 4: Prediction Loss (0-600 iterations)
    ax4 = plt.subplot(2, 3, 4)
    pred_data = trial_df[trial_df["training_iteration"] <= 600]
    train_pred = pred_data["train_prediction_loss_ave"]
    ax4.plot(train_pred.index, train_pred, color="green", alpha=0.8, label="Train Prediction Loss")
    
    # Evaluation curve
    eval_pred = pred_data.dropna(subset=["eval_prediction_loss_ave"])
    if not eval_pred.empty:
        ax4.plot(eval_pred["training_iteration"], eval_pred["eval_prediction_loss_ave"],
                'd--', color="red", alpha=0.8, label="Eval Prediction Loss")
    
    # Convergence values (0-500 iterations)
    conv_train_pred = get_smoothed_convergence(train_pred)
    conv_eval_pred = get_smoothed_convergence(eval_pred["eval_prediction_loss_ave"]) if not eval_pred.empty else np.nan
    
    ax4.set(xlim=(0, 600), ylim=get_adaptive_ylim(train_pred, conv_train_pred),
           xlabel="Training Iteration", ylabel="Loss", title="Prediction Loss")
    add_convergence_lines(ax4, conv_train_pred, conv_eval_pred, ('green', 'red'), x_pos = 500)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Subplot 5: Relative Prediction Error (0-600 iterations)
    ax5 = plt.subplot(2, 3, 5)
    train_pred_err = pred_data["train_relative_prediction_error_ave"]
    ax5.plot(train_pred_err.index, train_pred_err, color="darkgreen", alpha=0.8, label="Train Rel. Error")
    
    # Evaluation curve
    eval_pred_err = pred_data.dropna(subset=["eval_relative_prediction_error_ave"])
    if not eval_pred_err.empty:
        ax5.plot(eval_pred_err["training_iteration"], 
                eval_pred_err["eval_relative_prediction_error_ave"],
                'p--', color="darkred", alpha=0.8, label="Eval Rel. Error")
    
    # Convergence values (0-500 iterations)
    conv_train_perr = get_smoothed_convergence(train_pred_err)
    conv_eval_perr = get_smoothed_convergence(eval_pred_err["eval_relative_prediction_error_ave"]) if not eval_pred_err.empty else np.nan
    
    ax5.set(xlim=(0, 600), ylim=get_adaptive_ylim(train_pred_err, conv_train_perr),
           xlabel="Training Iteration", ylabel="Relative Error", title="Relative Prediction Error")
    add_convergence_lines(ax5, conv_train_perr, conv_eval_perr, ('darkgreen', 'darkred'), x_pos=500)
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Subplot 6: Gradient Norm (Adaptive range)
    ax6 = plt.subplot(2, 3, 6)
    grad_norm = trial_df["train_grad_norm_ave"].replace([np.inf, -np.inf], np.nan).dropna()
    
    if not grad_norm.empty:
        # Calculate adaptive y limits using percentiles
        y_min = 0
        y_max = np.percentile(grad_norm, 90) * 1.1  # 90th percentile + 10% padding
        
        ax6.plot(grad_norm.index, grad_norm, color="teal", alpha=0.8, label="Gradient Norm")
        ax6.set_ylim(y_min, y_max)
    else:
        # Fallback if no valid data
        ax6.text(0.5, 0.5, 'No Gradient Data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_ylim(0, 1)
    
    ax6.set(xlim=(0, 1500), 
           xlabel="Training Iteration", 
           ylabel="Norm Value", 
           title="Gradient Norm")
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    if tuned_params:
        param_text = "Tuned Parameters:\n" + "\n".join([f"{k}: {v}" for k, v in tuned_params.items()])
        ax6.text(0.95, 0.05, param_text, 
                transform=ax6.transAxes,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=12,
                fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(f"{root_log_dir}/{fig_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    # Load the experiment
    metric = "eval_relative_dynamics_error_ave"
    mode = "min"
    scope = "all"
    fig_name = "dynamics_best_trial_curve"
    required_params = {
        # "learning_rate": 0.001,
    }
    root_log_dir = "./logs/tune_VAE_dyn_separate"


    analysis = ExperimentAnalysis(
        experiment_checkpoint_path=os.path.abspath(f"{root_log_dir}/tune_model_2025-03-22_20-12-19"),
        default_metric="eval_total_loss",
        default_mode="min"
    )
    
    # Find all trials that match criteria
    filtered_trials = [
        t for t in analysis.trials
        if trial_filter(t, required_params=required_params)
    ]

    if not filtered_trials:
        raise ValueError("No trials matching the criteria found!")

    filtered_df = pd.DataFrame([
        t.last_result for t in filtered_trials
    ])
    filtered_df.to_csv(f"{root_log_dir}/filtered_tune_results.csv", index=False)

    # Find best trial among filtered
    best_trial = get_best_trial(filtered_trials, analysis, 
                            metric=metric,
                            mode=mode, 
                            scope=scope)
    
    tuned_params = {}
    configs = [t.config for t in filtered_trials]
    for key in filtered_trials[0].config.keys():
        values = set(str(c.get(key)) for c in configs)
        if len(values) > 1:  # Parameter had different values across trials
            tuned_params[key] = best_trial.config.get(key)
    
    # Print results
    print("Best trial ID:", best_trial.trial_id)
    print("Best Tuned Config:")
    for key, value in tuned_params.items():
        print(f"  {key}: {value}")

    # Get trial dataframe
    trial_df = analysis.trial_dataframes[best_trial.trial_id]


    plot_learning_curves(trial_df, fig_name, tuned_params = tuned_params)