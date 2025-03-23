import os
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune.analysis import ExperimentAnalysis

# Load the existing experiment
root_log_dir = "./logs/VAE_large_dataset"
analysis = ExperimentAnalysis(experiment_checkpoint_path=os.path.abspath(f"{root_log_dir}/tune_model_2025-03-18_17-54-05"),
                              default_metric="eval_total_loss",
                              default_mode="min")
metric = "eval_prediction_loss"
# Get the best trial based on the metric
best_trial = analysis.get_best_trial(metric=metric, mode="min", scope="all")
print(best_trial)
print("Best config: ", best_trial.config)
df = analysis.dataframe()
df.to_csv(f"{root_log_dir}/tune_results.csv", index=False)


# Get the trial's result dataframe
trial_df = analysis.trial_dataframes[best_trial.trial_id]

def plot_learning_curves(trial_df):
    plt.figure(figsize=(12, 6))
    
    # --- Dynamics Loss Subplot ---
    plt.subplot(1, 3, 1)
    
    # Plot training and evaluation dynamics loss
    plt.plot(trial_df["training_iteration"], 
             trial_df["train_dynamics_loss"], 
             label="Train Dynamics Loss",
             color="blue",
             alpha=0.8)
    
    eval_dynamics = trial_df.dropna(subset=["eval_dynamics_loss"])
    plt.plot(eval_dynamics["training_iteration"], 
             eval_dynamics["eval_dynamics_loss"], 
             'o--',
             label="Eval Dynamics Loss",
             color="orange",
             alpha=0.8)
    
    # Add convergence lines and annotations
    last_train_dyn = trial_df["train_dynamics_loss"].iloc[-1]
    last_eval_dyn = eval_dynamics["eval_dynamics_loss"].iloc[-1]
    max_iter = trial_df["training_iteration"].max()
    
    plt.axhline(last_train_dyn, color='blue', linestyle='--', linewidth=1, alpha=0.7)
    plt.axhline(last_eval_dyn, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    
    text_offset = 3  # Adjust this value to control vertical spacing
    plt.text(max_iter*0.95, last_train_dyn + text_offset, f'{last_train_dyn:.1f}', 
             color='blue', ha='right', va='bottom')
    plt.text(max_iter*0.95, last_eval_dyn - text_offset, f'{last_eval_dyn:.1f}', 
             color='orange', ha='right', va='top')

    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("Dynamics Loss")
    plt.legend()
    plt.ylim([0,100])
    plt.grid(True, alpha=0.3)

    # --- Prediction Loss Subplot ---
    plt.subplot(1, 3, 2)
    
    # Plot training and evaluation prediction loss
    plt.plot(trial_df["training_iteration"], 
             trial_df["train_prediction_loss"], 
             label="Train Prediction Loss",
             color="green",
             alpha=0.8)
    
    eval_prediction = trial_df.dropna(subset=["eval_prediction_loss"])
    plt.plot(eval_prediction["training_iteration"], 
             eval_prediction["eval_prediction_loss"], 
             's--',
             label="Eval Prediction Loss",
             color="red",
             alpha=0.8)
    
    # Add convergence lines and annotations
    last_train_pred = trial_df["train_prediction_loss"].iloc[-1]
    last_eval_pred = eval_prediction["eval_prediction_loss"].iloc[-1]
    
    plt.axhline(last_train_pred, color='green', linestyle='--', linewidth=1, alpha=0.7)
    plt.axhline(last_eval_pred, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.text(max_iter*0.95, last_train_pred + text_offset, f'{last_train_pred:.1f}', 
             color='green', ha='right', va='bottom')
    plt.text(max_iter*0.95, last_eval_pred - text_offset, f'{last_eval_pred:.1f}', 
             color='red', ha='right', va='top')

    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("Prediction Loss")
    plt.legend()
    plt.ylim([0,100])
    plt.grid(True, alpha=0.3)


    # --- KL Loss Subplot ---
    plt.subplot(1, 3, 3)
    # Plot training and evaluation KL loss
    plt.plot(trial_df["training_iteration"], 
             trial_df["train_kl_loss"], 
             label="Train KL Loss",
             color="purple",
             alpha=0.8)
    
    eval_kl = trial_df.dropna(subset=["eval_kl_loss"])
    plt.plot(eval_kl["training_iteration"], 
             eval_kl["eval_kl_loss"], 
             '^--',  # Triangle markers
             label="Eval KL Loss",
             color="brown",
             alpha=0.8)
    
    # Add convergence lines and annotations
    last_train_kl = trial_df["train_kl_loss"].iloc[-1]
    last_eval_kl = eval_kl["eval_kl_loss"].iloc[-1]
    
    plt.axhline(last_train_kl, color='purple', linestyle='--', linewidth=1, alpha=0.7)
    plt.axhline(last_eval_kl, color='brown', linestyle='--', linewidth=1, alpha=0.7)
    
    text_offset = 30
    plt.text(max_iter*0.95, last_train_kl + text_offset, f'{last_train_kl:.1f}', 
             color='purple', ha='right', va='bottom')
    plt.text(max_iter*0.95, last_eval_kl - text_offset, f'{last_eval_kl:.1f}', 
             color='brown', ha='right', va='top')

    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("KL Loss")
    plt.legend()
    plt.ylim([0,1000])  # Adjust this if KL loss has different scale
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{root_log_dir}/best_trial_curve.png", dpi=300, bbox_inches="tight")
    plt.show()


plot_learning_curves(trial_df)