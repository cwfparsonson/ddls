from ddls.utils import seed_stochastic_modules_globally, gen_unique_experiment_folder
from ddls.launchers.launcher import Launcher
from ddls.loops.env_loop import EnvLoop
from ddls.loops.eval_loop import EvalLoop
from ddls.loops.epoch_loop import EpochLoop


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

    # env
    env = hydra.utils.instantiate(cfg.environment)

    # agent
    agent = hydra.utils.instantiate(cfg.agent)

    # # run agent in env
    # step_counter = 0
    # obs, reward, done, info = env.reset()
    # while not done:
        # action = agent.select_action(obs)
        # prev_obs = obs # save
        # obs, reward, done, info = env.step(action)
        # print(f'\nStep {step_counter}\nObs: {prev_obs}\nAction: {action}\nReward: {reward}\nDone: {done}\nInfo: {info}')
        # step_counter += 1

    # env loop for running episodes
    env_loop = EnvLoop(env, agent)

    # epoch loop for running epochs
    epoch_loop = None

    # launcher
    launcher = Launcher(env_loop=env_loop,
                        epoch_loop=epoch_loop,
                        **cfg.launcher)
                        # **OmegaConf.to_container(cfg.launcher, resolve=False))


    # logger




    # launcher.run
    launcher.run()







if __name__ == '__main__':
    run()

