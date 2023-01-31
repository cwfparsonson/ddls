import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment

import os
import pickle
import gzip
import copy
import time
import json
import shutil

from typing import Union

from collections import defaultdict
import numpy as np

import wandb


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

    # # DEBUG TODO TEMP
    # workers = eval_workers.remote_workers()
    # for worker in workers:
        # worker.foreach_env.remote(lambda env: print(f'worker {worker} env cluster jobs_generator max_acceptable_job_completion_time_frac_dist: {env.cluster.jobs_generator.max_acceptable_job_completion_time_frac_dist}'))

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

def load_ramp_cluster_environment_metrics(base_folder, base_name, ids, agent_to_id=None, default_agent='id', hue='Agent'):
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
            agent_to_episode_stats_dict[hue].append(agent)
            if completion_stats_found:
                agent_to_episode_completion_stats_dict[hue].append(agent)
            if blocked_stats_found:
                agent_to_episode_blocked_stats_dict[hue].append(agent)

            # load step stats
            with gzip.open(agent_dir+'step_stats.pkl', 'rb') as f:
                step_stats = pickle.load(f)
            for metric, result in step_stats.items():
                try:
                    agent_to_step_stats_dict[metric].extend(result)
                except TypeError:
                    agent_to_step_stats_dict[metric].append(result)
                step_metrics.add(metric)
            agent_to_step_stats_dict[hue].extend([agent for _ in range(len(result))])

            print(f'Checkpoints loaded from {agent_dir[:-1]}.')
        else:
            print(f'\nNo checkpoints/ folder found in {agent_dir[:-1]}')

    return (
        agent_to_episode_stats_dict,
        agent_to_episode_completion_stats_dict,
        agent_to_episode_blocked_stats_dict,
        agent_to_step_stats_dict,
    )

def load_run_results_dict(run: Union[str, wandb.apis.public.Run], keys_to_ignore=None, verbose=True):
    if keys_to_ignore is None:
        keys_to_ignore = []
        
    def check_if_ignore(key, keys_to_ignore):
        for k in keys_to_ignore:
            if k in key:
                return True
        return False 
        
    if isinstance(run, str):
        # load wandb.apis.public.Run object
        api = wandb.Api()
        run = api.run(run)
    elif isinstance(run, wandb.apis.public.Run):
        # already loaded wandb.apis.public.Run object
        pass
    else:
        raise Exception(f'Unrecognised run type {type(run)}, must be str (path to run) or wandb.apis.public.Run')
    # print(f'api:\n{dir(api)}') # DEBUG
    # print(f'run:\n{dir(run)}') # DEBUG

    results = defaultdict(list)
    recorded_keys, ignored_keys = set(), set()

    # load standard metrics recorded
    history = run.scan_history()
    for log in history:
        for key, val in log.items():
            if not check_if_ignore(key, keys_to_ignore):
                if isinstance(val, dict):
                    # is a reference to an artifact, will load below but can ignore here
                    ignored_keys.add(key)
                else:
                    results[key].append(val)
                    recorded_keys.add(key)
            else:
                ignored_keys.add(key)

    # load any artifacts
    for artifact in run.logged_artifacts():
        path_to_downloaded_artifact = artifact.download()

        # find json file
        attempts = 0
        while 'json' not in path_to_downloaded_artifact:
            path_to_downloaded_artifact += f'/{os.listdir(path_to_downloaded_artifact)[0]}'
            attempts += 1
            if attempts > 50:
                raise Exception(f'Unable to find a .json file in {artifact.download()}. Ensure that this file actually exists with the necessary extension, or if need to add more supported extensions to this block of code.')

        # load json file data into a dict
        with open(path_to_downloaded_artifact) as f:
            artifact_data = json.load(f)

        # create mapping of column header to column values
        formatted_artifact_data = defaultdict(list)
        for row in artifact_data['data']:
            for col_idx in range(len(row)):
                formatted_artifact_data[artifact_data['columns'][col_idx]].append(row[col_idx])

        # record formatted artifact data
        results[artifact.name.split('-')[-1]+'_artifact'] = formatted_artifact_data

    if verbose:
        print(f'\nRecorded keys: {recorded_keys}')
        print(f'Ignored keys: {ignored_keys}')
                                
    return results

def get_results_metric_types(results):
    metric_types = set()
    for key in results.keys():
        metric_types.add(get_key_metric_type(key))
    return metric_types

def get_key_metric_type(key):
    if '_' in key:
        key_end = key.split('_')[-1]
        if key_end in ['mean', 'min', 'max']:
            metric_type = key[:-(len(key_end)+1)]
        else:
            metric_type = key
    else:
        metric_type = key
    return metric_type

def remove_substrings_from_keys(results, substrings_to_remove):
    new_results = {}
    for key in results:
        new_key = copy.copy(key)
        for substring_to_remove in substrings_to_remove:
            if substring_to_remove in key:
                new_key = new_key.replace(substring_to_remove, '')
        new_results[new_key] = results[key]
    return new_results

def load_ramp_cluster_environment_wandb_table_from_wandb_run(agent_to_run: dict,
                                                             keys_to_ignore=None,
                                                             key_substrings_to_remove=None,
                                                             hue='Agent',
                                                             verbose=True):
    if keys_to_ignore is None:
        keys_to_ignore = []

    # gather relevant agent data
    agent_to_results = {agent: load_run_results_dict(run, keys_to_ignore, verbose=verbose) for agent, run in agent_to_run.items()}

    if key_substrings_to_remove is not None:
        agent_to_clean_results = {agent: remove_substrings_from_keys(results, key_substrings_to_remove) for agent, results in agent_to_results.items()}
        # print(f'\nAgent clean results: {agent_to_clean_results}')
    else:
        agent_to_clean_results = agent_to_results

    return agent_to_clean_results


def load_ramp_cluster_environment_metrics_from_wandb_run(agent_to_run: dict, 
                                                         keys_to_ignore=None, 
                                                         key_substrings_to_remove=None, 
                                                         verbose=True):
    '''
    Args:
        agent_to_run: Dict mapping Agent name to run, where run is either a str path to
            a wandb run OR a wandb.apis.public.Run object
    '''
    if keys_to_ignore is None:
        keys_to_ignore = []
        
    # gather relevant agent data
    agent_to_results = {agent: load_run_results_dict(run, keys_to_ignore, verbose=verbose) for agent, run in agent_to_run.items()}
    # print(f'\nAgent results: {agent_to_results}') # DEBUG

    if key_substrings_to_remove is not None:
        agent_to_clean_results = {agent: remove_substrings_from_keys(results, key_substrings_to_remove) for agent, results in agent_to_results.items()}
        # print(f'\nAgent clean results: {agent_to_clean_results}')
    else:
        agent_to_clean_results = agent_to_results

    # get unique metric types with min/max/mean removed so can easily group
    agent_to_metric_types = {agent: get_results_metric_types(results) for agent, results in agent_to_clean_results.items()}
    if verbose:
        print(f'\nUnique metric types: {agent_to_metric_types}')
    
    # organise data so that can get consistent row lengths for pandas dataframe
    agent_to_episode_stats_dict = defaultdict(list)
    agent_to_episode_completion_stats_dict = defaultdict(list)
    agent_to_episode_blocked_stats_dict = defaultdict(list)
    
    agent_to_step_stats_dict = defaultdict(list)
    
    for agent, results in agent_to_clean_results.items():
        episode_stats_found, completion_stats_found, blocked_stats_found = 0, 0, 0
        for metric, result in results.items():
            metric_type = get_key_metric_type(metric)
            if metric_type in RampClusterEnvironment.episode_metrics():
                try:
                    agent_to_episode_stats_dict[metric].extend(result)
                    episode_stats_found = len(result)
                except TypeError:
                    agent_to_episode_stats_dict[metric].append(result)
                    episode_stats_found = 1
            elif metric_type in RampClusterEnvironment.episode_completion_metrics():
                completion_stats_found = True
                try:
                    agent_to_episode_completion_stats_dict[metric].extend(result)
                    completion_stats_found = len(result)
                except TypeError:
                    agent_to_episode_completion_stats_dict[metric].append(result)
                    completion_stats_found = 1
            elif metric_type in RampClusterEnvironment.episode_blocked_metrics():
                blocked_stats_found = True
                try:
                    agent_to_episode_blocked_stats_dict[metric].extend(result)
                    blocked_stats_found = len(result)
                except TypeError:
                    agent_to_episode_blocked_stats_dict[metric].append(result)
                    blocked_stats_found = 1
            else:
                if verbose:
                    print(f'Unrecognised episode metric {metric}, skipping...')
        agent_to_episode_stats_dict[hue].extend([agent for _ in range(episode_stats_found)])
        agent_to_episode_completion_stats_dict[hue].extend([agent for _ in range(completion_stats_found)])
        agent_to_episode_blocked_stats_dict[hue].extend([agent for _ in range(blocked_stats_found)])
            
    return (
        agent_to_episode_stats_dict,
        agent_to_episode_completion_stats_dict,
        agent_to_episode_blocked_stats_dict,
    )

def load_ramp_cluster_environment_metrics_from_wandb_sweep(agent_to_sweep: dict,
                                                           keys_to_ignore=None,
                                                           key_substrings_to_remove=None,
                                                           verbose=False,
                                                           hue='Agent',
                                                           ):
    agent_to_episode_stats_dict, agent_to_episode_completion_stats_dict, agent_to_episode_blocked_stats_dict = defaultdict(list), defaultdict(list), defaultdict(list)

    api = wandb.Api()
    for agent, sweep_path in agent_to_sweep.items():
        sweep = api.sweep(sweep_path)
        
        sweep_params = sweep.config['parameters']
        print(f'\nAgent {agent} sweep {sweep} parameters:')
        for key, val in sweep_params.items():
            print(f'\t{key}:')
            print(f'\t\t{val}')
        num_runs = len(sweep.runs)
        print(f'Loading data from {num_runs} runs...')
        
        sweep_load_start_t = time.time()
        for run_counter, run in enumerate(sweep.runs):
            # get run sweep hparam vals
            run_load_start_t = time.time()
            run_config_dict = json.loads(run.json_config)
                
            # load run data
            agent_to_run = {agent: run}
            _agent_to_episode_stats_dict, _agent_to_episode_completion_stats_dict, _agent_to_episode_blocked_stats_dict = load_ramp_cluster_environment_metrics_from_wandb_run(agent_to_run, keys_to_ignore=keys_to_ignore, key_substrings_to_remove=key_substrings_to_remove, verbose=verbose, hue=hue)
            # print(f'_agent_to_episode_blocked_stats_dict: {_agent_to_episode_blocked_stats_dict}') # DEBUG
            
            # update episode stats dict with values and sweep hparams
            for key, val in _agent_to_episode_stats_dict.items():
                agent_to_episode_stats_dict[key].extend(val)
            for _ in range(len(val)):
                agent_to_episode_stats_dict['config'].append(json.dumps(run_config_dict))
                for hparam in sweep_params:
                    agent_to_episode_stats_dict[hparam].append(run_config_dict[hparam]['value'])
                    
            # update episode completion stats dict with values and sweep hparams
            for key, val in _agent_to_episode_completion_stats_dict.items():
                agent_to_episode_completion_stats_dict[key].extend(val)
            for _ in range(len(val)):
                agent_to_episode_completion_stats_dict['config'].append(json.dumps(run_config_dict))
                for hparam in sweep_params:
                    agent_to_episode_completion_stats_dict[hparam].append(run_config_dict[hparam]['value'])
                    
            # update episode blocked stats dict with values and sweep hparams
            for key, val in _agent_to_episode_blocked_stats_dict.items():
                agent_to_episode_blocked_stats_dict[key].extend(val)
            for _ in range(len(val)):
                agent_to_episode_blocked_stats_dict['config'].append(json.dumps(run_config_dict))
                for hparam in sweep_params:
                    agent_to_episode_blocked_stats_dict[hparam].append(run_config_dict[hparam]['value'])
            
            print(f'Loaded data for run {run_counter+1} of {num_runs} ({run}) in {time.time() - run_load_start_t:.3f}.')
        print(f'Loaded data for agent {agent} sweep {sweep} (num_runs={num_runs}) in {time.time() - sweep_load_start_t:.3f} s.')

    return (
        agent_to_episode_stats_dict,
        agent_to_episode_completion_stats_dict,
        agent_to_episode_blocked_stats_dict,
    )







