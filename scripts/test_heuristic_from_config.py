import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='ray')  # noqa

from ddls.utils import seed_stochastic_modules_globally, gen_unique_experiment_folder
from ddls.ml_models.utils import get_least_used_gpu

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
import os

import time
import pickle
import gzip

import cProfile
import pstats

# NEED TO IMPORT MODULES WHICH MUST BE SEEDED
import numpy as np
import random
import torch


# to override from command line, do e.g.:
# python <test_heuristic_from_config.py --config-path=ramp_job_placement_shaping_configs --config-name=heuristic_config.yaml
@hydra.main(config_path='ramp_job_placement_shaping_configs', config_name='heuristic_config.yaml')
def run(cfg: DictConfig):
    # seeding
    if 'seed' in cfg.experiment:
        np, random, torch = seed_stochastic_modules_globally(default_seed=cfg.experiment.seed,
                                         numpy_module=np,
                                         random_module=random,
                                         torch_module=torch,
                                         )

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
    OmegaConf.save(config=cfg, f=save_dir+'heuristic_config.yaml')

    # print info
    print('\n\n\n')
    print(f'~'*100)
    print(f'Initialised experiment save dir {save_dir}')
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*100)

    # eval_loop = hydra.utils.instantiate(actor=actor, env=env)
    eval_loop = hydra.utils.instantiate(cfg.eval_loop, wandb=wandb)
    print(f'Initialised {eval_loop}.')

    start_time = time.time()
    if cfg.experiment.profile_time:
        # 1. Generate a file called <name>.prof
        # 2. Transfer to /home/cwfparsonson/Downloads
        # 3. Run snakeviz /home/cwfparsonson/Downloads/<name>.prof to visualise
        profiler = cProfile.Profile()
        profiler.enable()
    results = eval_loop.run(verbose=True)
    if cfg.experiment.profile_time:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.dump_stats(f'{save_dir}time_profile.prof')
        print(f'Saved time profile to {save_dir}time_profile.prof')
    print(f'Finished validation in {time.time() - start_time:.3f} s.')
    # print(f'Validation results:\n{results}')

    base_path = '/'.join(save_dir.split('/')[:-1])
    for log_name, log in results.items():
        log_path = base_path + f'/{log_name}'
        with gzip.open(log_path + '.pkl', 'wb') as f:
            pickle.dump(log, f)
        print(f'Saved validation data to {log_path}.pkl')

if __name__ == '__main__':
    run()