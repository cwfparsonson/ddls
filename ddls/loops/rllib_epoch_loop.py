from ddls.utils import get_class_from_path, get_module_from_path, seed_stochastic_modules_globally, recursively_update_nested_dict

import ray
ray.shutdown()
ray.init()
from ray.rllib.agents import ppo
from ray.tune.registry import register_env

from ray.rllib.models import ModelCatalog

import gym

import collections
from omegaconf import OmegaConf
import time
import hydra



class RLlibEpochLoop:
    def __init__(self,
                 path_to_env_cls: str, # e.g. 'ddls.environments.job_placing.job_placing_all_nodes_environment.JobPlacingAllNodesEnvironment'
                 path_to_rllib_trainer_cls: str, # e.g. 'ray.rllib.agents.ppo.PPOTrainer'
                 rllib_config: dict,
                 path_to_model_cls: str = None,
                 path_to_validator_cls: str = None,
                 validator_rllib_config: dict = None):
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

    def run(self, *args, **kwargs):
        '''Run one epoch.'''
        return {'rllib_results': self.trainer.train()}

    def save_agent_checkpoint(self, path_to_save):
        self.last_agent_checkpoint = self.trainer.save(path_to_save)

    def validate(self, checkpoint_path, *args, **kwargs):
        if self.validator is None:
            results = {}
        else:
            # TODO: Change rllib_config to trainer_rllib_config (in keeping with validator_rllibg_config)

            # TODO: Run in background thread (AND make sure save checkpoint thread has finished before beginning validate thread)
            # TODO: Change eval_loop_config syntax to validator_config ni launcher.py rllib.yaml and this file
            # TODO: Add shufft option for jobs generator -> want to shuffle during training so agent doesn't overfit to specific job order, but keep same order during validation
            # TODO: Save validation results in each checkpoint

            results = self.validator.run(checkpoint_path)

        return results
        






