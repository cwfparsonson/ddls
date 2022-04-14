from ddls.utils import get_class_from_path, get_module_from_path
import ddls.utils

import ray
ray.shutdown()
ray.init()
from ray.rllib.agents import ppo
from ray.tune.registry import register_env

from omegaconf import OmegaConf
import time
import hydra


def recursively_instantiate_classes_in_hydra_config(d):
    for k, v in d.items():
        if isinstance(v, dict):
            recursively_instantiate_classes_in_hydra_config(v)
        else:
            hydra.utils.instantiate(d[k])



class RLlibEpochLoop:
    def __init__(self,
                 path_to_env_cls: str, # e.g. 'ddls.environments.job_placing.job_placing_all_nodes_environment.JobPlacingAllNodesEnvironment'
                 path_to_rllib_trainer_cls: str, # e.g. 'ray.rllib.agents.ppo.PPOTrainer'
                 rllib_config: dict):
        rllib_config = OmegaConf.to_container(rllib_config, resolve=False)

        # register env with ray
        register_env(rllib_config['env'], lambda env_config: get_class_from_path(path_to_env_cls)(**env_config))

        # merge rllib trainer's default config with specified config
        path_to_agent = '.'.join(path_to_rllib_trainer_cls.split('.')[:-1])
        self.rllib_config = get_module_from_path(path_to_agent).DEFAULT_CONFIG.copy()
        self.rllib_config.update(rllib_config)

        # init rllib trainer
        self.trainer = get_class_from_path(path_to_rllib_trainer_cls)(config=self.rllib_config)

    def run(self, *args, **kwargs):
        '''Run one epoch.'''
        return {'rllib_results': self.trainer.train()}
