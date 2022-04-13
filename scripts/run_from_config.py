from ddls.utils import seed_stochastic_modules_globally, gen_unique_experiment_folder
from ddls.launchers.launcher import Launcher
from ddls.loops.env_loop import EnvLoop
from ddls.loops.eval_loop import EvalLoop
from ddls.loops.epoch_loop import EpochLoop
from ddls.loggers.logger import Logger


import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil


@hydra.main(config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
    # create dir for saving data
    save_dir = gen_unique_experiment_folder(path_to_save=cfg.experiment.path_to_save, experiment_name=cfg.experiment.name)

    # save copy of config to the save dir
    OmegaConf.save(config=cfg, f=save_dir+'config.yaml')

    # print info
    print('\n\n\n')
    print(f'~'*80)
    print(f'Initialised experiment save dir {save_dir}')
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*80)
    
    # seeding
    if 'seed' in cfg.experiment:
        seed_stochastic_modules_globally(cfg.experiment.seed)

    # epoch loop for running epochs
    epoch_loop = hydra.utils.instantiate(cfg.epoch_loop)

    # launcher for running the experiment
    launcher = Launcher(epoch_loop=epoch_loop, **cfg.launcher)

    # logger for saving experiment results
    logger = Logger(path_to_save=save_dir, **cfg.logger)

    # TODO: checkpointer for saving agent checkpoints
    checkpointer = None

    # run the experiment
    launcher.run(logger=logger, checkpointer=checkpointer)



if __name__ == '__main__':
    run()

