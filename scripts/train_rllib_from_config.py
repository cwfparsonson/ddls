import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='ray')  # noqa

from ddls.utils import seed_stochastic_modules_globally, gen_unique_experiment_folder, get_class_from_path, get_module_from_path, recursively_update_nested_dict
from ddls.ml_models.utils import get_least_used_gpu
from ddls.launchers.launcher import Launcher
from ddls.loops.env_loop import EnvLoop
from ddls.loops.eval_loop import EvalLoop
from ddls.loops.epoch_loop import EpochLoop
from ddls.loggers.logger import Logger
from ddls.checkpointers.checkpointer import Checkpointer

import ray
ray.shutdown()
ray.init()

import numpy as np

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
import os

import subprocess
import pandas as pd
from io import StringIO


# to override from command line, do e.g.:
# python <train_rllib_from_config.py --config-path=ramp_job_placement_shaping_configs --config-name=heuristic_config.yaml
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
            eval_config_dict = recursively_update_nested_dict(eval_config_dict, algo_eval_config_dict, verbose=False)
        # cfg.epoch_loop.rllib_config.eval_config = OmegaConf.create(eval_config_dict)
        cfg.epoch_loop.rllib_config = {**cfg.epoch_loop.rllib_config, **eval_config_dict}

    # seeding
    if 'train_seed' in cfg.experiment:
        import numpy as np
        import random
        import torch
        np, random, torch = seed_stochastic_modules_globally(default_seed=cfg.experiment.train_seed,
                                                             numpy_module=np,
                                                             random_module=random,
                                                             torch_module=torch,
                                                             )
        if 'rllib_config' in cfg.epoch_loop:
            # must seed rllib separately in config
            cfg.epoch_loop.rllib_config.seed = cfg.experiment.train_seed

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

    # save copy of config to the save dir
    OmegaConf.save(config=cfg, f=save_dir+'rllib_config.yaml')

    # print info
    print('\n\n\n')
    print(f'~'*100)
    print(f'Initialised experiment save dir {save_dir}')
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*100)

    # epoch loop for running epochs
    epoch_loop = hydra.utils.instantiate(cfg.epoch_loop, wandb=wandb)
    print(f'Initialised {epoch_loop}.')

    # launcher for running the experiment
    launcher = Launcher(epoch_loop=epoch_loop, **cfg.launcher)
    print(f'Initialised {launcher}.')

    # logger for saving experiment results
    logger = Logger(path_to_save=save_dir, **cfg.logger)
    print(f'Initialised {logger}.')

    # # checkpointer for saving agent checkpoints
    # checkpointer = Checkpointer(path_to_save=save_dir, **cfg.checkpointer)
    checkpointer = Checkpointer(path_to_save=save_dir)
    print(f'Initialised {checkpointer}.')

    # run the experiment
    launcher.run(logger=logger, checkpointer=checkpointer)



if __name__ == '__main__':
    run()

