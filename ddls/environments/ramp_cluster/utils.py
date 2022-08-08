import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker

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
