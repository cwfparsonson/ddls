from ddls.utils import get_class_from_path, get_module_from_path, get_function_from_path, seed_stochastic_modules_globally, recursively_update_nested_dict

import ray
ray.shutdown()
ray.init()
from ray.rllib.agents import ppo
from ray.tune.registry import register_env

from ray.rllib.models import ModelCatalog

import gym

import collections
from collections import defaultdict
from omegaconf import OmegaConf
import time
import hydra

import pickle
import gzip

import copy

import threading

from typing import Union

import numpy as np

import time



class RLlibEpochLoop:
    def __init__(self,
                 path_to_env_cls: str, # e.g. 'ddls.environments.job_placing.job_placing_all_nodes_environment.JobPlacingAllNodesEnvironment'
                 path_to_rllib_trainer_cls: str, # e.g. 'ray.rllib.agents.ppo.PPOTrainer'
                 rllib_config: dict,
                 path_to_model_cls: str = None,
                 path_to_validator_cls: str = None,
                 validator_rllib_config: dict = None,
                 wandb=None,
                 metric='evaluation/episode_reward_mean', # metric to optimise for when looking for best RLlib checkpoint
                 metric_goal: Union['maximise', 'minimise'] = 'maximise', # whether to maximise or minimise this metric
                 **kwargs):
        self.metric = metric
        self.metric_goal = metric_goal

        rllib_config = OmegaConf.to_container(rllib_config, resolve=False)

        if 'callbacks' in rllib_config:
            if isinstance(rllib_config['callbacks'], str):
                # get callbacks class from string path
                rllib_config['callbacks'] = get_class_from_path(rllib_config['callbacks'])

        if 'custom_eval_function' in rllib_config:
            if isinstance(rllib_config['custom_eval_function'], str):
                # get callable function from string path
                rllib_config['custom_eval_function'] = get_function_from_path(rllib_config['custom_eval_function'])

        if path_to_model_cls is not None:
            # register model with rllib
            ModelCatalog.register_custom_model(rllib_config['model']['custom_model'], get_class_from_path(path_to_model_cls))

        if 'env' in rllib_config:
            # register env with ray
            register_env(rllib_config['env'], lambda env_config: get_class_from_path(path_to_env_cls)(**env_config))

        # merge rllib trainer's default config with specified config
        path_to_agent = '.'.join(path_to_rllib_trainer_cls.split('.')[:-1])
        if 'Apex' in path_to_rllib_trainer_cls:
            # need to load config with specific APEX_DEFAULT_CONFIG attribute
            self.rllib_config = get_module_from_path(path_to_agent).APEX_DEFAULT_CONFIG.copy()
        else:
            # can load default config with DEFAULT_CONFIG attribute
            self.rllib_config = get_module_from_path(path_to_agent).DEFAULT_CONFIG.copy()
        # self.rllib_config = get_module_from_path(path_to_agent).DEFAULT_CONFIG.copy()
        self.rllib_config.update(rllib_config)

        # init rllib trainer
        self.trainer = get_class_from_path(path_to_rllib_trainer_cls)(config=self.rllib_config)
        self.last_agent_checkpoint = None
        self.best_agent_checkpoints = None
        self.best_agent_checkpoint_performances = None

        # # init rllib validator
        # if path_to_validator_cls is None:
            # self.validator = None
        # else:
            # self.validator_rllib_config = self.rllib_config
            # if validator_rllib_config is not None:
                # # update eval config with any overrides
                # self.validator_rllib_config = recursively_update_nested_dict(self.validator_rllib_config, validator_rllib_config)
            # self.validator = get_class_from_path(path_to_validator_cls)(path_to_env_cls=path_to_env_cls,
                                                                        # path_to_rllib_trainer_cls=path_to_rllib_trainer_cls,
                                                                        # rllib_config=self.validator_rllib_config)
        # self.validation_thread = None

        # init trackers
        self.start_time = time.time()
        self.run_time, self.epoch_counter, self.episode_counter, self.actor_step_counter = 0, 0, 0, 0
        self.wandb = wandb
        self.best_agent_wandb_table = self.wandb.Table(columns=['best_agent_checkpoint', 'best_agent_checkpoint_performance'])

    def _log_rllib_key_val(self, key, val, wandb_log, mode: Union['training', 'evaluation']):
        # print(f'mode: {mode}')
        if key == 'info':
            # need to handle rllib learner info differently from other metrics since has multiple nested dicts
            # print(f'Unpacking key-val pairs of log {key} with keys {val.keys()}...')
            if 'default_policy' in val['learner']:
                # ppo etc, unpack nested default_policy dict
                for k, v in val['learner']['default_policy']['learner_stats'].items():
                    # print(f'Unpacked k: {k} | val: {type(v)} {v}')
                    wandb_log[f'{mode}/{k}'] = v
            else:
                # unpack as usual
                for k, v in val.items():
                    if isinstance(v, collections.Mapping):
                        # HACK: Assume no useful extra info stored in dict
                        pass
                    else:
                        # print(f'Unpacked k: {k} | val: {type(v)} {v}')
                        wandb_log[f'{mode}/{k}'] = v

        else:
            if isinstance(val, dict):
                # unpack key-val pairs of dict and log
                # print(f'Unpacking key-val pairs of log {key} with keys {val.keys()}...')
                for k, v in val.items():
                    # print(f'Unpacked k: {k} | val: {type(v)} {v}')
                    try:
                        wandb_log[f'{mode}/{k}'] = np.mean(v)
                    except TypeError:
                        # non-numeric type (e.g. string)
                        wandb_log[f'{mode}/{k}'] = v
            else:
                # print(f'Already unpacked k: {key} | val: {type(val)} {val}')
                try:
                    wandb_log[f'{mode}/{key}'] = np.mean(val)
                except TypeError:
                    # non-numeric type (e.g. string)
                    wandb_log[f'{mode}/{key}'] = val

    def log(self, 
            results, 
            custom_metrics_source: Union [None, 'rllib_results/custom_metrics', 'rllib_results/sampler_results/custom_metrics'] = None, # hack to stop RLlib uneccessarily logging custom_metrics twice https://github.com/ray-project/ray/issues/28265
            ):
        '''Should be called after save_agent_checkpoint() if evaluating so that self.last_agent_checkpoint is updated.'''

        keys_to_ignore = {'config', 'experiment_id', 'trial_id', 'pid', 'hostname', 'node_ip'}
        # HACK: Prevent RLlib logging custom_metrics twice uneccessarily https://github.com/ray-project/ray/issues/28265
        if custom_metrics_source is None:
            pass
        elif custom_metrics_source == 'rllib_results/custom_metrics':
            keys_to_ignore.add('sampler_results')
        elif custom_metrics_source == 'rllib_results/sampler_results/custom_metrics':
            keys_to_ignore.add('custom_metrics')
        else:
            raise Exception(f'Unrecognised custom_metrics_source {custom_metrics_source}')

        if self.wandb is not None:
            # log stats with weights and biases
            # print(f'\n\nSaving training results for epoch with weights and biases...')
            wandb_log = {}
            for log_name, log in results.items():
                if log_name != 'rllib_results':
                    for key, val in log.items():
                        wandb_log[f'{log_name}/{key}'] = val
                else:
                    # print(f'rllib results keys: {results["rllib_results"].keys()}')
                    for key, val in results['rllib_results'].items():
                        if key  == 'evaluation':
                            # update log dict with evaluation data
                            for k, v in val.items():
                                if k not in keys_to_ignore:
                                    # log evaluation metrics
                                    self._log_rllib_key_val(key=k, val=v, wandb_log=wandb_log, mode='evaluation')
                            # add any useful stats to evaluation log
                            wandb_log['evaluation/run_time'] = self.run_time
                            wandb_log['evaluation/epoch_counter'] = self.epoch_counter
                            wandb_log['evaluation/episode_counter'] = self.episode_counter
                            wandb_log['evaluation/actor_step_counter'] = self.actor_step_counter
                            # check if new best checkpoint found
                            if self.best_agent_checkpoints is None:
                                # not yet established a best checkpoint
                                self.best_agent_checkpoints = [copy.copy(self.last_agent_checkpoint)]
                                self.best_agent_checkpoint_performances = [wandb_log[self.metric]]
                            else:
                                if self.metric_goal == 'maximise':
                                    if wandb_log[self.metric] > self.best_agent_checkpoint_performances[-1]:
                                        self.best_agent_checkpoints.append(copy.copy(self.last_agent_checkpoint))
                                        self.best_agent_checkpoint_performances.append(wandb_log[self.metric])
                                    else:
                                        pass
                                elif self.metric_goal == 'minimise':
                                    if wandb_log[self.metric] < self.best_agent_checkpoint_performances[-1]:
                                        self.best_agent_checkpoints.append(copy.copy(self.last_agent_checkpoint))
                                        self.best_agent_checkpoint_performances.append(wandb_log[self.metric])
                                    else:
                                        pass
                                else:
                                    raise Exception(f'Unrecognised metric_goal {self.metric_goal}')
                        else:
                            # update log dict with training data
                            if key not in keys_to_ignore:
                                # log training metrics
                                self._log_rllib_key_val(key=key, val=val, wandb_log=wandb_log, mode='training')
                            # add any useful stats to training log
                            wandb_log['training/run_time'] = self.run_time
                            wandb_log['training/epoch_counter'] = self.epoch_counter
                            wandb_log['training/episode_counter'] = self.episode_counter
                            wandb_log['training/actor_step_counter'] = self.actor_step_counter

                    if self.best_agent_checkpoints is not None:
                        # store historic best evaluations into a table
                        best_agent_wandb_table = self.wandb.Table(
                                                                columns=['best_agent_checkpoint', 'best_agent_checkpoint_performance'],
                                                                data=[[self.best_agent_checkpoints[idx], self.best_agent_checkpoint_performances[idx]] for idx in range(len(self.best_agent_checkpoints))],
                                                             )
                    else:
                        # not yet ran an evaluation
                        best_agent_wandb_table = self.wandb.Table(
                                                                columns=['best_agent_checkpoint', 'best_agent_checkpoint_performance'],
                                                                data=[[None, None]],
                                                             )

            wandb_log['summary/'] = best_agent_wandb_table

            # save log
            self.wandb.log(wandb_log)

    def run(self, *args, **kwargs):
        '''Run one epoch.'''
        # run epoch
        results = {'rllib_results': self.trainer.train()}
        
        # update trackers
        self.run_time = time.time() - self.start_time
        self.epoch_counter += 1
        self.episode_counter = results['rllib_results']['episodes_total']
        self.actor_step_counter = results['rllib_results']['agent_timesteps_total']

        # add any useful additional stats 
        results['trackers'] = {}
        results['trackers']['epoch_counter'] = copy.copy(self.epoch_counter)
        results['trackers']['episode_counter'] = copy.copy(self.episode_counter)
        results['trackers']['actor_step_counter'] = copy.copy(self.actor_step_counter)

        return results

    def save_agent_checkpoint(self, path_to_save):
        self.last_agent_checkpoint = self.trainer.save(path_to_save)

    def validate(self, checkpoint_path, save_results=True, **kwargs):
        if self.validation_thread is not None:
            self.validation_thread.join()
        self.validation_thread = threading.Thread(
                                                    target=self._validate, 
                                                    args=(checkpoint_path, save_results,)
                                                 )
        self.validation_thread.start()

    def _validate(self, checkpoint_path, save_results=True, **kwargs):
        if self.validator is None:
            raise Exception(f'Validator in Epoch Loop is None.')
        else:
            results = self.validator.run(checkpoint_path)

        if save_results:
            # base_path = '/'.join(checkpoint_path.split('/')[:-1])
            base_path = checkpoint_path
            wandb_log = {}

            for log_name, log in results.items():
                log_path = base_path + f'/{log_name}'
                with gzip.open(log_path + '.pkl', 'wb') as f:
                    pickle.dump(log, f)

                if self.wandb is not None:
                    # log valid stats with weights and baises
                    for key, val in log.items():
                        # record average of stat for validation run
                        wandb_log[f'evaluation/{log_name}/{key}'] = np.mean(val)

            if self.wandb is not None:
                self.wandb.log(wandb_log)

        return results
