import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
#from ray import data
import torch
import d4rl

import absl.app
import absl.flags

from SimpleSAC.conservative_sac import ConservativeSAC
from SimpleSAC.replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch
from SimpleSAC.model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from SimpleSAC.sampler import StepSampler, TrajSampler
from SimpleSAC.utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics, remove_unneeded_keys
from SimpleSAC.utils import WandBLogger
from viskit.logging import logger, setup_logger
from NHT.nht.mujoco_interface import register_NHT_env
from NHT.nht.utils import project_d4rl_actions
FLAGS_DEF = define_flags_with_default(
    env='HalfCheetah-v2',
    data_set_name = 'halfcheetah-expert-v2',
    NHT_path = '/home/cmuslima/projects/def-mtaylor3/cmuslima/action_map/action_mapping2/trained_maps/NHT/halfcheetah-expert-v2/2/dataset_percent_0.5_100.0_0.001/version_0',
    use_NHT = True,
    data_set_percent = 1.0,
    max_traj_length=1000,
    seed=42,
    device='cpu',
    save_model=True,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=np.sqrt(6/2), #np.sqrt(6/2) #.999
    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=1000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,
    output_dir = 'halfcheetah-expert-v2-CQL-NHT-1',
    cql=ConservativeSAC.get_default_config(),
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
        dataset = remove_unneeded_keys(dataset)
        print('finished projecting the actions and updated the data set')
        
        my_NHT_env = gym.make(env).unwrapped
        eval_sampler = TrajSampler(gym.make(env).unwrapped, FLAGS.max_traj_length)        
        print('got the eval sampler')        
    else:
        env = FLAGS.env
        eval_sampler = TrajSampler(gym.make(env).unwrapped, FLAGS.max_traj_length)
        dataset = get_d4rl_dataset(eval_sampler.env)

    
    #eval_sampler = TrajSampler(gym.make(env).unwrapped, FLAGS.max_traj_length)


    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)
    print('passed the dataset part')
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

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = subsample_batch(dataset, FLAGS.batch_size)
                batch = batch_to_torch(batch, FLAGS.device)
                metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = 0 #np.mean(
                    #[eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                #)
                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)
