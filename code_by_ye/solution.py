from train_env import get_env, MaskedActionsModel
from agent import Agent
import ray
import time
import random
from collections import defaultdict

from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer

ray.init()


class Trainer:
    def __init__(self, Env, conf_list):
        self.conf_list = conf_list
        self.Env = Env
        self.checkpoint = None
        self.iter = 0
        ModelCatalog.register_custom_model('MaskModel', MaskedActionsModel)

    def train(self, run_time):
        rl_env = get_env(self.Env, self.conf_list)
        # e = rl_env()
        # e.reset()
        # e.step({'M001':1})
        trainer = PPOTrainer(env=rl_env, config={
            # 'train_batch_size':20000,
            'train_batch_size':2000,
            'num_workers':1,
            'num_gpus':1,
            # 'sgd_minibatch_size':2048,
            "sgd_minibatch_size": 128,
            'model':{
                'custom_model': 'MaskModel'
            },
        })
        # DEFAULT_CONFIG = with_common_config({
        #     # Should use a critic as a baseline (otherwise don't use value baseline;
        #     # required for using GAE).
        #     "use_critic": True,
        #     # If true, use the Generalized Advantage Estimator (GAE)
        #     # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        #     "use_gae": True,
        #     # The GAE (lambda) parameter.
        #     "lambda": 1.0,
        #     # Initial coefficient for KL divergence.
        #     "kl_coeff": 0.2,
        #     # Size of batches collected from each worker.
        #     "rollout_fragment_length": 200,
        #     # Number of timesteps collected for each SGD round. This defines the size
        #     # of each SGD epoch.
        #     "train_batch_size": 4000,
        #     # Total SGD batch size across all devices for SGD. This defines the
        #     # minibatch size within each epoch.
        #     "sgd_minibatch_size": 128,
        #     # Whether to shuffle sequences in the batch when training (recommended).
        #     "shuffle_sequences": True,
        #     # Number of SGD iterations in each outer loop (i.e., number of epochs to
        #     # execute per train batch).
        #     "num_sgd_iter": 30,
        #     # Stepsize of SGD.
        #     "lr": 5e-5,
        #     # Learning rate schedule.
        #     "lr_schedule": None,
        #     # Coefficient of the value function loss. IMPORTANT: you must tune this if
        #     # you set vf_share_layers=True inside your model's config.
        #     "vf_loss_coeff": 1.0,
        #     "model": {
        #         # Share layers for value function. If you set this to True, it's
        #         # important to tune vf_loss_coeff.
        #         "vf_share_layers": False,
        #     },
        #     # Coefficient of the entropy regularizer.
        #     "entropy_coeff": 0.0,
        #     # Decay schedule for the entropy regularizer.
        #     "entropy_coeff_schedule": None,
        #     # PPO clip parameter.
        #     "clip_param": 0.3,
        #     # Clip param for the value function. Note that this is sensitive to the
        #     # scale of the rewards. If your expected V is large, increase this.
        #     "vf_clip_param": 10.0,
        #     # If specified, clip the global norm of gradients by this amount.
        #     "grad_clip": None,
        #     # Target value for KL divergence.
        #     "kl_target": 0.01,
        #     # Whether to rollout "complete_episodes" or "truncate_episodes".
        #     "batch_mode": "truncate_episodes",
        #     # Which observation filter to apply to the observation.
        #     "observation_filter": "NoFilter",

        #     # Deprecated keys:
        #     # Share layers for value function. If you set this to True, it's important
        #     # to tune vf_loss_coeff.
        #     # Use config.model.vf_share_layers instead.
        #     "vf_share_layers": DEPRECATED_VALUE,
        # })
        now_time = time.time()
        total_time = 0
        while True:
            last_time = now_time

            result = trainer.train()
            reward = result['episode_reward_mean']
            print(f'Iteration: {self.iter}, reward: {reward}, training iteration: {trainer._iteration}')
            now_time = time.time()
            total_time += now_time - last_time
            trainer.save(f'./work')
            self.checkpoint = f'./work/checkpoint_{trainer._iteration}/checkpoint-{trainer._iteration}'
            self.iter += 1

            if total_time + 2*(now_time-last_time) > run_time:
                break


        return Agent(trainer, rl_env)





