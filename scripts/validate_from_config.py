import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='ray')  # noqa

from ddls.utils import seed_stochastic_modules_globally, gen_unique_experiment_folder

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil
import os

import time
import pickle
import gzip


@hydra.main(config_path='configs', config_name='validate_config.yaml')
def run(cfg: DictConfig):
    if 'cuda_visible_devices' in cfg.experiment:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(gpu) for gpu in cfg.experiment.cuda_visible_devices)

    # seeding
    if 'seed' in cfg.experiment:
        seed_stochastic_modules_globally(cfg.experiment.seed)

    # create dir for saving data
    save_dir = gen_unique_experiment_folder(path_to_save=cfg.experiment.path_to_save, experiment_name=cfg.experiment.name)

    # save copy of config to the save dir
    OmegaConf.save(config=cfg, f=save_dir+'validate_config.yaml')

    # print info
    print('\n\n\n')
    print(f'~'*80)
    print(f'Initialised experiment save dir {save_dir}')
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*80)

    # eval_loop = hydra.utils.instantiate(actor=actor, env=env)
    eval_loop = hydra.utils.instantiate(cfg.eval_loop)
    print(f'Initialised {eval_loop}.')

    start_time = time.time()
    results = eval_loop.run()
    print(f'Finished validation in {time.time() - start_time:.3f} s.')
    print(f'Validation results:\n{results}')

    base_path = '/'.join(save_dir.split('/')[:-1])
    for log_name, log in results.items():
        log_path = base_path + f'/validation_{log_name}'
        with gzip.open(log_path + '.pkl', 'wb') as f:
            pickle.dump(log, f)
        print(f'Saved validation data to {log_path}.pkl')


if __name__ == '__main__':
    run()

