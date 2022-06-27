from ddls.loops.env_loop import EnvLoop
from ddls.loops.epoch_loop import EpochLoop
from ddls.loggers.logger import Logger
from ddls.checkpointers.checkpointer import Checkpointer

import numpy as np
import time
from collections import defaultdict



class Launcher:
    def __init__(self,
                 num_epochs: int = None,
                 num_episodes: int = None,
                 num_actor_steps: int = None,
                 num_eval_episodes: int = None,
                 eval_freq: int = None,
                 env_loop: EnvLoop = None,
                 epoch_loop: EpochLoop = None,
                 epoch_batch_size: int = 1,
                 verbose: bool = False):

        if (epoch_loop is None and env_loop is None):
            raise Exception(f'Must provide env_loop or epoch_loop.')
        if not isinstance(epoch_batch_size, int):
            raise Exception(f'epoch_batch_size must be int but is {epoch_batch_size}.')

        self.num_episodes = num_episodes
        self.num_epochs = num_epochs 
        self.num_actor_steps = num_actor_steps
        self.num_eval_episodes = num_eval_episodes
        self.eval_freq = eval_freq

        self.env_loop = env_loop 
        self.epoch_loop = epoch_loop
        self.epoch_batch_size = epoch_batch_size

        self.verbose = verbose

    def _step(self):
        if self.epoch_loop is not None:
            _results = self.epoch_loop.run(self.epoch_batch_size)
            self.epoch_counter += 1
            if 'rllib_results' in _results:
                # using rllib epoch loop
                # self.episode_counter += _results['rllib_results']['episodes_this_iter']
                # self.actor_step_counter += _results['rllib_results']['timesteps_this_iter']
                self.episode_counter = _results['rllib_results']['episodes_total']
                self.actor_step_counter = _results['rllib_results']['agent_timesteps_total']
            else:
                # using custom epoch loop
                self.episode_counter += self.epoch_batch_size
                self.actor_step_counter += sum(_results['episode_stats']['num_actor_steps'])
        else:
            _results = self.env_loop.run()
            self.episode_counter += 1
            self.actor_step_counter += _results['episode_stats']['num_actor_steps']
        return _results

    def update_results_log(self, old_results, new_results):
        for log_name, log in new_results.items():
            if log_name not in old_results:
                # initialise log
                # print(f'Initialising old_results {log_name}')
                old_results[log_name] = {}
                for key, val in log.items():
                    # print(f'new_results key: {key} val: {val}')
                    if not isinstance(val, list):
                        val = [val]
                    old_results[log_name][key] = val
                    # print(f'initialised old_results key: {key} val: {old_results[log_name][key]}')
            else:
                # extend log
                # print(f'Extending old_results {log_name}')
                for key, val in log.items():
                    # print(f'new_results key: {key} val: {val}')
                    if not isinstance(val, list):
                        val = [val]
                    if key in old_results[log_name]:
                        # extend log key
                        old_results[log_name][key] += val
                    else:
                        # initialise log key
                        old_results[log_name][key] = val
                    # print(f'extended old_results key: {key} val: {old_results[log_name][key]}')
        return old_results

    def run(self, 
            logger: Logger = None,
            checkpointer: Checkpointer = None):
        if self.verbose:
            print(f'Launching...')

        # init trackers
        self.epoch_counter, self.episode_counter, self.actor_step_counter = 0, 0, 0
        self.start_time = time.time()
        results = {'launcher_stats': {'total_run_time': [0]}}
        if self.epoch_loop is not None:
            # save initial agent checkpoint
            checkpointer.write(self.epoch_loop)

        # run launcher
        while not self._check_if_should_stop():
            # step the launcher
            # results.update(self._step())
            # print(f'\n\n\nresults before update')
            # print(results)
            results = self.update_results_log(results, self._step())
            results['launcher_stats']['total_run_time'].append(time.time() - self.start_time)
            # print(f'\nresults after update')
            # print(results)

            if self.verbose:
                print(f'ELAPSED: Epochs: {self.epoch_counter} | Episodes: {self.episode_counter} | Actor steps: {self.actor_step_counter} | Run time: {results["launcher_stats"]["total_run_time"][-1]:.3f} s')

            # check if should write log data
            if logger is not None:
                if self._check_if_should_log(logger):
                    # save in-memory results
                    logger.write(results)
                    # reset in-memory results
                    results = {'launcher_stats': {'total_run_time': []}}

            # check if should save checkpoint
            if checkpointer is not None and self.epoch_loop is not None:
                if self.epoch_counter % checkpointer.epoch_checkpoint_freq == 0:
                    checkpointer.write(self.epoch_loop)

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
