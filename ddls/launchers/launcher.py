from ddls.loops.env_loop import EnvLoop
from ddls.loops.epoch_loop import EpochLoop
from ddls.loggers.logger import Logger

import numpy as np
import time
from collections import defaultdict








class Checkpointer:
    pass


class Launcher:
    def __init__(self,
                 num_epochs: int = None,
                 num_episodes: int = None,
                 num_actor_steps: int = None,
                 num_eval_episodes: int = None,
                 eval_freq: int = None,
                 env_loop: EnvLoop = None,
                 epoch_loop: EpochLoop = None,
                 epoch_batch_size: int = 1):

        if (epoch_loop is None and env_loop is None):
            raise Exception(f'Must provide env_loop or epoch_loop.')

        self.num_episodes = num_episodes
        self.num_epochs = num_epochs 
        self.num_actor_steps = num_actor_steps
        self.num_eval_episodes = num_eval_episodes
        self.eval_freq = eval_freq

        self.env_loop = env_loop 
        self.epoch_loop = epoch_loop
        self.epoch_batch_size = epoch_batch_size

    def step(self):
        if self.epoch_loop is not None:
            _results = self.epoch_loop.run(self.epoch_batch_size)
            self.epoch_counter += 1
            self.episode_counter += self.epoch_batch_size
            self.actor_step_counter += sum(_results['episode_stats']['num_actor_steps'])
        else:
            _results = self.env_loop.run()
            self.episode_counter += 1
            self.actor_step_counter += _results['episode_stats']['num_actor_steps']
        return _results

    def run(self, 
            logger: Logger = None,
            checkpointer: Checkpointer = None):
        # init trackers
        self.epoch_counter, self.episode_counter, self.actor_step_counter = 0, 0, 0
        results = {}

        # run launcher
        while not self._check_if_should_stop():
            # step the launcher
            results.update(self.step())

            # check if should write log data
            if logger is not None:
                if self._check_if_should_log(logger):
                    # save in-memory results
                    logger.write(results)
                    # reset in-memory results
                    results = {}

    def _check_if_should_log(self, logger):
        should_log = False
        if logger.actor_step_log_freq is not None:
            if self.actor_step_counter % logger.actor_step_log_freq == 0:
                should_log = True
        elif logger.episode_log_freq is not None:
            if self.episode_counter % logger.episode_log_freq == 0:
                should_log = True
        elif logger.epoch_log_freq is not None:
            if self.epoch_counter % logger.epoch_log_freq == 0:
                should_log = True
        else:
            raise Exception('Unrecognised logging condition.')
        return should_log


    def _check_if_should_stop(self):
        stop = False
        if self.num_episodes is None and self.num_epochs is None and self.num_actor_steps is None:
            # never finish
            pass
        else:
            if self.num_epochs is not None:
                if self.epoch_counter >= self.num_epochs:
                    stop = True
            if self.num_episodes is not None:
                if self.episode_counter >= self.num_episodes:
                    stop = True
            if self.num_actor_steps is not None:
                if self.actor_step_counter >= self.num_actor_steps:
                    stop = True
        return stop
























