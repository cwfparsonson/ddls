'''
To set which RLlib checkpoint to test, go into the rllib_config/rllib_loop.yaml file and set
the test_time_checkpoint_path variable to the checkpoint path you want to test.
N.B. This is the .yaml file used for training so that we don't have to have
separate files with different params etc. - you can edit the validator_rllib_config
parameters to change the test-time environment settings etc.
'''

import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='ray')  # noqa

from ddls.utils import seed_stochastic_modules_globally, gen_unique_experiment_folder, get_class_from_path, get_module_from_path, recursively_update_nested_dict
from ddls.ml_models.utils import get_least_used_gpu

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
import os

import ray
ray.shutdown()
ray.init()
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

import time
import pickle
import gzip
import os

# NEED TO IMPORT MODULES WHICH MUST BE SEEDED
import numpy as np
import random
import torch

# to override from command line, do e.g.:
# python <test_rllib_from_config.py --config-path=ramp_job_placement_shaping_configs --config-name=heuristic_config.yaml
@hydra.main(config_path='ramp_job_placement_shaping_configs', config_name='rllib_config.yaml')
def run(cfg: DictConfig):
    if 'cuda_visible_devices' in cfg.experiment:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in cfg.experiment.cuda_visible_devices)
    least_used_gpu = get_least_used_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(least_used_gpu)
    cfg.experiment.cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    print(f'Set CUDA_VISIBLE_DEVICES to least used GPU {least_used_gpu}')

    # merge various configs into rllib config, epoch config, and model config as required
    if 'algo' in cfg:
        cfg.epoch_loop.path_to_rllib_trainer_cls = cfg.algo.path_to_rllib_trainer_cls
        cfg.epoch_loop.rllib_config = {**cfg.epoch_loop.rllib_config, **cfg.algo.algo_config}
    if 'model' in cfg:
        model_dict = OmegaConf.to_container(cfg.model, resolve=False)
        if 'model' in cfg.algo:
            algo_model_dict = OmegaConf.to_container(cfg.algo.model, resolve=False)
            model_dict = recursively_update_nested_dict(model_dict, algo_model_dict, verbose=False)
        cfg.epoch_loop.rllib_config.model = OmegaConf.create(model_dict)
    if 'env_config' in cfg:
        cfg.epoch_loop.rllib_config.env_config = cfg.env_config
    if 'eval_config' in cfg:
        eval_config_dict = OmegaConf.to_container(cfg.eval_config, resolve=False)
        if 'eval_config' in cfg.algo:
            algo_eval_config_dict = OmegaConf.to_container(cfg.algo.eval_config, resolve=False)
            eval_config_dict = recursively_update_nested_dict(eval_config_dict, algo_eval_config_dict, verbose=False) # overwrite with any algorithm-specific eval config settings
        # overwrite rllib config with evaluation config for this test_rllib_from_config script
        rllib_config_dict = OmegaConf.to_container(cfg.epoch_loop.rllib_config, resolve=False)
        rllib_config_dict = recursively_update_nested_dict(rllib_config_dict, eval_config_dict['evaluation_config'], verbose=False) # overwrite with any algorithm-specific eval config settings
        cfg.epoch_loop.rllib_config = OmegaConf.create(rllib_config_dict)

    # seeding
    if 'test_seed' in cfg.experiment:
        np, random, torch = seed_stochastic_modules_globally(default_seed=cfg.experiment.test_seed,
                                         numpy_module=np,
                                         random_module=random,
                                         torch_module=torch,
                                         )
        if 'rllib_config' in cfg.epoch_loop:
            # must seed rllib separately in config
            cfg.epoch_loop.rllib_config.seed = cfg.experiment.test_seed
        # if 'rllib_config' in cfg.epoch_loop:
            # # must seed rllib separately in config
            # cfg.epoch_loop.rllib_config.seed = cfg.experiment.test_seed
            # cfg.epoch_loop.rllib_config.evaluation_config.seed = cfg.experiment.test_seed

    # create dir for saving data
    save_dir = gen_unique_experiment_folder(path_to_save=cfg.experiment.path_to_save, experiment_name=cfg.experiment.name)
    cfg['experiment']['save_dir'] = save_dir

    # init weights and biases
    if 'wandb' in cfg:
        if cfg.wandb is not None:
            import wandb
            hparams = OmegaConf.to_container(cfg)
            wandb.init(config=hparams, **cfg.wandb.init)
        else:
            wandb = None
    else:
        wandb = None

    # print info
    print('\n\n\n')
    print(f'~'*100)
    print(f'Initialised experiment save dir {save_dir}')
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*100)

    # get rllib config
    _rllib_config = OmegaConf.to_container(cfg.epoch_loop.rllib_config, resolve=False)

    if 'callbacks' in _rllib_config:
        if isinstance(_rllib_config['callbacks'], str):
            # get callbacks class from string path
            _rllib_config['callbacks'] = get_class_from_path(_rllib_config['callbacks'])

    # register custom model
    if cfg.epoch_loop.path_to_model_cls is not None:
        # register model with rllib
        ModelCatalog.register_custom_model(_rllib_config['model']['custom_model'], get_class_from_path(cfg.epoch_loop.path_to_model_cls))

    if 'env' in _rllib_config:
        # register env with ray
        register_env(_rllib_config['env'], lambda env_config: get_class_from_path(cfg.epoch_loop.path_to_env_cls)(**env_config))

    # merge rllib trainer's default config with specified config
    path_to_agent = '.'.join(cfg.epoch_loop.path_to_rllib_trainer_cls.split('.')[:-1])
    rllib_config = get_module_from_path(path_to_agent).DEFAULT_CONFIG.copy()
    rllib_config.update(_rllib_config)

    # init rllib eval loop
    eval_loop = get_class_from_path(cfg.epoch_loop.path_to_validator_cls)(path_to_env_cls=cfg.epoch_loop.path_to_env_cls,
                                                                          path_to_rllib_trainer_cls=cfg.epoch_loop.path_to_rllib_trainer_cls,
                                                                          rllib_config=rllib_config,
                                                                          wandb=wandb)
    print(f'Initialised {eval_loop}.')

    start_time = time.time()
    results = eval_loop.run(checkpoint_path=cfg.epoch_loop.test_time_checkpoint_path, verbose=True)
    print(f'Finished validation of {cfg.epoch_loop.test_time_checkpoint_path} in {time.time() - start_time:.3f} s.')
    # print(f'Validation results:\n{results}')

    base_path = '/'.join(save_dir.split('/')[:-1])
    for log_name, log in results.items():
        log_path = base_path + f'/{log_name}'
        with gzip.open(log_path + '.pkl', 'wb') as f:
            pickle.dump(log, f)
        print(f'Saved validation data to {log_path}.pkl')


if __name__ == '__main__':
    run()

