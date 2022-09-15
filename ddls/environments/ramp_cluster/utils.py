import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
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
