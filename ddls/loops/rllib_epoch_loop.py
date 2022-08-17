from ddls.utils import get_class_from_path, get_module_from_path, seed_stochastic_modules_globally, recursively_update_nested_dict

import ray
ray.shutdown()
ray.init()
from ray.rllib.agents import ppo
from ray.tune.registry import register_env

from ray.rllib.models import ModelCatalog

import gym

from collections import defaultdict
from omegaconf import OmegaConf
import time
import hydra

import pickle
import gzip

import threading

import numpy as np



class RLlibEpochLoop:
    def __init__(self,
                 path_to_env_cls: str, # e.g. 'ddls.environments.job_placing.job_placing_all_nodes_environment.JobPlacingAllNodesEnvironment'
                 path_to_rllib_trainer_cls: str, # e.g. 'ray.rllib.agents.ppo.PPOTrainer'
                 rllib_config: dict,
                 path_to_model_cls: str = None,
                 path_to_validator_cls: str = None,
                 validator_rllib_config: dict = None,
                 wandb=None,
                 **kwargs):
        rllib_config = OmegaConf.to_container(rllib_config, resolve=False)

        if 'callbacks' in rllib_config:
            if isinstance(rllib_config['callbacks'], str):
                # get callbacks class from string path
                rllib_config['callbacks'] = get_class_from_path(rllib_config['callbacks'])

        if path_to_model_cls is not None:
            # register model with rllib
            ModelCatalog.register_custom_model(rllib_config['model']['custom_model'], get_class_from_path(path_to_model_cls))

        if 'env' in rllib_config:
            # register env with ray
            register_env(rllib_config['env'], lambda env_config: get_class_from_path(path_to_env_cls)(**env_config))

        # merge rllib trainer's default config with specified config
        path_to_agent = '.'.join(path_to_rllib_trainer_cls.split('.')[:-1])
        self.rllib_config = get_module_from_path(path_to_agent).DEFAULT_CONFIG.copy()
        self.rllib_config.update(rllib_config)

        # init rllib trainer
        self.trainer = get_class_from_path(path_to_rllib_trainer_cls)(config=self.rllib_config)
        self.last_agent_checkpoint = None

        # init rllib validator
        if path_to_validator_cls is None:
            self.validator = None
        else:
            self.validator_rllib_config = self.rllib_config
            if validator_rllib_config is not None:
                # update eval config with any overrides
                self.validator_rllib_config = recursively_update_nested_dict(self.validator_rllib_config, validator_rllib_config)
            self.validator = get_class_from_path(path_to_validator_cls)(path_to_env_cls=path_to_env_cls,
                                                                        path_to_rllib_trainer_cls=path_to_rllib_trainer_cls,
                                                                        rllib_config=self.validator_rllib_config)
        self.validation_thread = None

        self.wandb = wandb

    def run(self, *args, **kwargs):
        '''Run one epoch.'''
        results = {'rllib_results': self.trainer.train()}

        if self.wandb is not None:
            # log train epoch stats with weights and biases
            # print(f'\n\nSaving training results for epoch with weights and biases...')
            wandb_log = {}
            for key, val in results['rllib_results'].items():
                if key not in {'config', 'experiment_id', 'trial_id', 'pid', 'hostname', 'node_ip'}:
                    # print(f'key: {key}')
                    # print(f'val: {val}')
                    if key == 'info':
                        # need to handle rllib learner info differently from other metrics since has multiple nested dicts
                        for k, v in val['learner']['default_policy']['learner_stats'].items():
                            # print(f'k: {k} | val: {type(v)} {v}')
                            wandb_log[f'train/{k}'] = v
                    else:
                        if isinstance(val, dict):
                            # unpack key-val pairs of dict and log
                            for k, v in val.items():
                                # print(f'k: {k} | val: {type(v)} {v}')
                                try:
                                    wandb_log[f'train/{k}'] = np.mean(v)
                                except TypeError:
                                    # non-numeric type (e.g. string)
                                    wandb_log[f'train/{k}'] = v
                        else:
                            try:
                                wandb_log[f'train/{key}'] = np.mean(val)
                            except TypeError:
                                # non-numeric type (e.g. string)
                                wandb_log[f'train/{key}'] = val
            self.wandb.log(wandb_log)

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
            base_path = '/'.join(checkpoint_path.split('/')[:-1])
            wandb_log = {}

            for log_name, log in results.items():
                log_path = base_path + f'/{log_name}'
                with gzip.open(log_path + '.pkl', 'wb') as f:
                    pickle.dump(log, f)

                if self.wandb is not None:
                    # log valid stats with weights and baises
                    for key, val in log.items():
                        # record average of stat for validation run
                        wandb_log[f'valid/{log_name}/{key}'] = np.mean(val)

            if self.wandb is not None:
                self.wandb.log(wandb_log)

        return results
