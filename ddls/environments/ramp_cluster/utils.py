import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment

import os
import pickle
import gzip

from collections import defaultdict
import numpy as np


class RLlibRampClusterEnvironmentCallback(DefaultCallbacks):
    '''
    NOTE: RLLib callbacks are hard-coded to store only the min/mean/max values
    for each attribute you track (i.e. if you try to track a list of metrics at each step,
    RLLib will save the min/mean/max of this list).

    Think RLLib does not store per-step/episode/epoch stats, but rather stores the rolling mean/min/max 
    per-step/episode/epoch stats. Seen somewhere online that this is done with a history window 
    of size 100 https://discuss.ray.io/t/custom-metrics-only-mean-value/636/3. Don't think there's 
    a way to change this, and also means plots may look different from what you'd expect...
    '''
    def on_episode_start(self,
                         *,
                         worker: 'RolloutWorker',
                         base_env: BaseEnv,
                         policies: dict,
                         episode: Episode,
                         **kwargs):
        episode.user_data['step_stats'] = defaultdict(list) # store step data in temporary dict

    def on_episode_step(self,
                         *,
                         worker: 'RolloutWorker',
                         base_env: BaseEnv,
                         policies: dict,
                         episode: Episode,
                         **kwargs):
        for env in base_env.get_sub_environments():
            for key, val in env.cluster.step_stats.items():
                episode.user_data['step_stats'][key].append(val)
        
    def on_episode_end(self,
                       *,
                       worker: 'RolloutWorker',
                       base_env: BaseEnv,
                       policies: dict,
                       episode: Episode,
                       **kwargs):
        for key, val in episode.user_data['step_stats'].items():
            episode.custom_metrics[key] = val
        for env in base_env.get_sub_environments():
            for key, val in env.cluster.episode_stats.items():
                try:
                    val = list(val)
                except TypeError:
                    # val is not iterable (e.g. int, float, etc.)
                    val = [val]
                episode.custom_metrics[key] = np.mean(val)
            # episode.custom_metrics['return'] = np.sum(env.cluster.episode_stats['reward'])

def custom_eval_function(algorithm, eval_workers):
    """Taken from RLlib custom evaluation function example: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_eval.py

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.
    Returns:
        metrics: Evaluation metrics dict.
    """

    # # We configured 2 eval workers in the training config.
    # worker_1, worker_2 = eval_workers.remote_workers()

    # # Set different env settings for each worker. Here we use a fixed config,
    # # which also could have been computed in each worker by looking at
    # # env_config.worker_index (printed in SimpleCorridor class above).
    # worker_1.foreach_env.remote(lambda env: env.set_corridor_length(4))
    # worker_2.foreach_env.remote(lambda env: env.set_corridor_length(7))

    # for i in range(5):
        # print("Custom evaluation round", i)
        # # Calling .sample() runs exactly one episode per worker due to how the
        # # eval workers are configured.
        # ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999
    )
    # print(f'episodes: {episodes}') # DEBUG

    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    # print(f'metrics: {metrics}')

    # # You can also put custom values in the metrics dict.
    # metrics["foo"] = 1

    # print(f'returning custom eval func episodes metrics') # DEBUG
    return metrics

def load_ramp_cluster_environment_metrics(base_folder, base_name, ids, agent_to_id=None, default_agent='id'):
    agent_to_episode_stats_dict = defaultdict(list)
    agent_to_episode_completion_stats_dict = defaultdict(list)
    agent_to_episode_blocked_stats_dict = defaultdict(list)

    agent_to_step_stats_dict = defaultdict(list)

    step_metrics = set()

    episode_metrics = RampClusterEnvironment.episode_metrics()
    episode_completion_metrics = RampClusterEnvironment.episode_completion_metrics()
    episode_blocked_metrics = RampClusterEnvironment.episode_blocked_metrics()

    if agent_to_id is not None:
        id_to_agent = {}
        for agent in agent_to_id.keys():
            for _id in agent_to_id[agent]:
                id_to_agent[_id] = agent
    else:
        id_to_agent = None

    for _id in ids:
        if id_to_agent is None:
            agent = default_agent
        else:
            agent = id_to_agent[_id]

        if isinstance(_id, int):
            # use int id to generate string dir to data
            agent_dir = base_folder + f'{base_name}/{base_name}_{_id}/'
        else:
            # string dir to data already provided
            agent_dir = _id

        if os.path.isdir(agent_dir):
            print(f'\nLoading validation data from {agent_dir[:-1]}...')

            completion_stats_found, blocked_stats_found = False, False

            # load episode stats
            with gzip.open(agent_dir+'episode_stats.pkl', 'rb') as f:
                episode_stats = pickle.load(f)
            for metric, result in episode_stats.items():
                # print(metric)
                if metric in episode_metrics:
                    try:
                        agent_to_episode_stats_dict[metric].extend(result)
                    except TypeError:
                        agent_to_episode_stats_dict[metric].append(result)
                elif metric in episode_completion_metrics:
                    completion_stats_found = True
                    try:
                        agent_to_episode_completion_stats_dict[metric].extend(result)
                    except TypeError:
                        agent_to_episode_completion_stats_dict[metric].append(result)
                elif metric in episode_blocked_metrics:
                    blocked_stats_found = True
                    try:
                        agent_to_episode_blocked_stats_dict[metric].extend(result)
                    except TypeError:
                        agent_to_episode_blocked_stats_dict[metric].append(result)
                else:
                    print(f'Unrecognised episode metric {metric}, skipping...')
            agent_to_episode_stats_dict['Agent'].append(agent)
            if completion_stats_found:
                agent_to_episode_completion_stats_dict['Agent'].append(agent)
            if blocked_stats_found:
                agent_to_episode_blocked_stats_dict['Agent'].append(agent)

            # load step stats
            with gzip.open(agent_dir+'step_stats.pkl', 'rb') as f:
                step_stats = pickle.load(f)
            for metric, result in step_stats.items():
                try:
                    agent_to_step_stats_dict[metric].extend(result)
                except TypeError:
                    agent_to_step_stats_dict[metric].append(result)
                step_metrics.add(metric)
            agent_to_step_stats_dict['Agent'].extend([agent for _ in range(len(result))])

            print(f'Checkpoints loaded from {agent_dir[:-1]}.')
        else:
            print(f'\nNo checkpoints/ folder found in {agent_dir[:-1]}')

    return (
        agent_to_episode_stats_dict,
        agent_to_episode_completion_stats_dict,
        agent_to_episode_blocked_stats_dict,
        agent_to_step_stats_dict,
    )
