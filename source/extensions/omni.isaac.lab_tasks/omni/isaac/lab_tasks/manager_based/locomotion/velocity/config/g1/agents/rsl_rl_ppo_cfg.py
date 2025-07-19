# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class G1VelBoxPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 29000#19000#3000
    save_interval = 500#50
    experiment_name = "g1_vel_box_23dof"
    empirical_normalization = True#False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # rnd_cfg= {
        #     "weight": 1000,  # initial weight of the RND reward

        #     # note: This is a dictionary with a required key called "mode".
        #     #   Please check the RND module for more information.
        #     "weight_schedule": None,
        #     "reward_normalization": True,  # whether to normalize RND reward
        #     "state_normalization": True,  # whether to normalize RND state observations

        #     # -- Learning parameters
        #     #"learning_rate": 0.001,  # learning rate for RND

        #     # -- Network parameters
        #     # note: if -1, then the network will use dimensions of the observation
        #     "num_outputs": 16,  # number of outputs of RND network
        #     "predictor_hidden_dims": [256,128], # hidden dimensions of predictor network
        #     "target_hidden_dims": [256,128],  # hidden dimensions of target network
        # }
    )
    

@configclass
class G1BoxPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name ="OnPolicyRunnerLip"
    num_steps_per_env = 24
    max_iterations = 19000#3000
    save_interval = 500#50
    experiment_name = "g1_box_23dof"
    empirical_normalization = True#False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOLIP",
        grad_penalty_coef=0,#0.002,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,#0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,#0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # rnd_cfg= {
        #     "weight": 100,  # initial weight of the RND reward

        #     # note: This is a dictionary with a required key called "mode".
        #     #   Please check the RND module for more information.
        #     "weight_schedule": None,
        #     "reward_normalization": True,  # whether to normalize RND reward
        #     "state_normalization": True,  # whether to normalize RND state observations

        #     # -- Learning parameters
        #     #"learning_rate": 0.001,  # learning rate for RND

        #     # -- Network parameters
        #     # note: if -1, then the network will use dimensions of the observation
        #     "num_outputs": 1,  # number of outputs of RND network
        #     "predictor_hidden_dims": [256,128], # hidden dimensions of predictor network
        #     "target_hidden_dims": [256,128],  # hidden dimensions of target network
        # }
         # -- Symmetry Augmentation
        # symmetry_cfg={
        #     "use_data_augmentation": False,#True,  # this adds symmetric trajectories to the batch
        #     "use_mirror_loss": False,  # this adds symmetry loss term to the loss function

        #     # string containing the module and function name to import.
        #     # Example: "legged_gym.envs.locomotion.anymal_c.symmetry:get_symmetric_states"
        #     #
        #     # .. code-block:: python
        #     #
        #     #     @torch.no_grad()
        #     #     def get_symmetric_states(
        #     #        obs: Optional[torch.Tensor] = None, actions: Optional[torch.Tensor] = None, cfg: "BaseEnvCfg" = None, obs_type: str = "policy"
        #     #     ) -> Tuple[torch.Tensor, torch.Tensor]:
        #     #
        #     "data_augmentation_func": "omni.isaac.lab_tasks.utils.symmetry:data_augmentation_func_g1",#None,

        #     # coefficient for symmetry loss term
        #     # if 0, then no symmetry loss is used
        #     "mirror_loss_coeff": 0.0}
        
    )
    logger = "wandb"
    wandb_project = "g1_box-23dof"  # Set the project name for Weights & Biases logging
    

@configclass
class G1TargetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000#3000
    save_interval = 500#50
    experiment_name = "g1_target"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    logger = "wandb"
    wandb_project = "humanoid-target"

@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name ="OnPolicyRunnerLip"
    num_steps_per_env = 24
    max_iterations =8000#3000
    save_interval = 10#50
    experiment_name = "g1_23_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOLIP",
        grad_penalty_coef=0,#0.002,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
