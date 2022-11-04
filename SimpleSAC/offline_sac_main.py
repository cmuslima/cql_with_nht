import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
import torch

import absl.app
import absl.flags

from .sac import SAC
from .replay_buffer import ReplayBuffer, batch_to_torch
from .replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch

from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger
from nht.mujoco_interface import register_NHT_env
from nht.utils import project_d4rl_actions


FLAGS_DEF = define_flags_with_default(
    env='HalfCheetah-v2',
    data_set_name = 'halfcheetah-medium-expert-v2',
    NHT_path = '/Users/cmuslimani/Projects/action_mapping/NHT/training_scripts/trained_maps/NHT/halfcheetah-medium-expert-v2/dataset_percent_0.5/version_0',
    use_NHT = False,
    data_set_percent = 1.0,
    max_traj_length=1000,
    replay_buffer_size=1000000,
    seed=42,
    device='cpu',
    save_model=True,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,
    clip_action= np.sqrt(6/2), #0.999, #np.sqrt(6/2) #.999
    reward_scale=1.0,
    reward_bias=0.0,

    n_epochs=2000,
    bc_epochs=0,
    n_env_steps_per_epoch=1000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    batch_size=256,

    sac=SAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)
    if FLAGS.use_NHT:
        print('using NHT action space')
        register_NHT_env(FLAGS.env, FLAGS.NHT_path) 
        print('finished registering NHT env')
        env = f'NHT_{FLAGS.env}'
        dataset = project_d4rl_actions(FLAGS.NHT_path, FLAGS.data_set_name, FLAGS.data_set_percent)
        print('finished projecting the actions and updated the data set')
    
        
        my_NHT_env = gym.make(env).unwrapped
        eval_sampler = TrajSampler(gym.make(env).unwrapped, FLAGS.max_traj_length)
        print(f'Action space: {my_NHT_env.action_space}')

    else:
        env = FLAGS.env
        print('data_set_name', FLAGS.data_set_name)
        eval_sampler = TrajSampler(gym.make(env).unwrapped, FLAGS.max_traj_length)
        dataset = get_d4rl_dataset(eval_sampler.env)



    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

    #eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)

    policy = TanhGaussianPolicy(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.sac.target_entropy >= 0.0:
        FLAGS.sac.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = SAC(FLAGS.sac, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}
        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = subsample_batch(dataset, FLAGS.batch_size)
                batch = batch_to_torch(batch, FLAGS.device)
                metrics.update(prefix_metrics(sac.train(batch), 'sac'))



        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])

                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')


if __name__ == '__main__':
    absl.app.run(main)
