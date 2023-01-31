from ddls.utils import get_class_from_path, seed_stochastic_modules_globally, recursively_update_nested_dict

from collections import defaultdict
from collections.abc import Iterable
import numpy as np

import copy

from decimal import Decimal

class RLlibEvalLoop:
    def __init__(self,
                 path_to_rllib_trainer_cls: str,
                 path_to_env_cls: str,
                 rllib_config: dict,
                 wandb=None,
                 **kwargs):
        self.rllib_config = rllib_config
        self.actor = get_class_from_path(path_to_rllib_trainer_cls)(config=self.rllib_config)
        self.env = get_class_from_path(path_to_env_cls)(**self.rllib_config['env_config'])
        self.wandb = wandb

    def run(self,
            checkpoint_path: str,
            verbose: bool = False):
        results = {'step_stats': defaultdict(list), 'episode_stats': {}}

        # if 'seed' in self.rllib_config:
            # if self.rllib_config['seed'] is not None:
                # seed_stochastic_modules_globally(self.rllib_config['seed'])

        self.actor.restore(checkpoint_path)

        if verbose:
            print(f'Starting validation...')
        obs, done = self.env.reset(), False
        prev_idx = 0
        step_counter = 1
        while not done:
            start_step_mounted_workers = len(self.env.cluster.mounted_workers)
            job_to_place = list(self.env.cluster.job_queue.jobs.values())[0] # assume event-driven where only ever have one job to queue. Use job_to_place for useful info for heuristics
            action = self.actor.compute_single_action(obs) 
            obs, reward, done, info = self.env.step(action)

            if verbose:
                print(f'Step {step_counter} | Action: {action} | Reward: {reward:.8f} | Jobs arrived: {self.env.cluster.num_jobs_arrived} | Jobs running: {len(self.env.cluster.jobs_running)} | Jobs completed: {len(self.env.cluster.jobs_completed)} | Jobs blocked: {len(self.env.cluster.jobs_blocked)} | Start->end of step mounted workers: {start_step_mounted_workers}->{len(self.env.cluster.mounted_workers)} | Stopwatch: {Decimal(float(self.env.cluster.stopwatch.time())):.3E}')

            results['step_stats']['action'].append(action)
            results['step_stats']['reward'].append(reward)
            for key, val in self.env.cluster.steps_log.items():
                # get vals which have been added to step stats over elapsed cluster steps
                _val = val[int(prev_idx):] 
                try:
                    _val = list(_val)
                except TypeError:
                    # not iterable, put in list
                    _val = [_val]

                # check if need to update steps log idx
                if prev_idx > len(val):
                    # steps log has been reset, reset prev idx
                    prev_idx = 0

                if key != 'step_counter':

                    if key == 'step_start_time':
                        # record time last recorded cluster step started
                        results['step_stats'][key].append(_val[0])

                    elif key == 'step_end_time':
                        # record time most recent cluster step finished
                        results['step_stats'][key].append(_val[-1])

                    elif 'mean' in key:
                        # record average of metric over cluster which have elapsed steps
                        try:
                            if len(_val) > 0:
                                results['step_stats'][key].append(np.mean(_val))
                            else:
                                results['step_stats'][key].append(0)
                        except TypeError:
                            # val is non-numeric, cannot average
                            pass

                    else:
                        # record total of metric over cluster which have elapsed steps
                        try:
                            if len(_val) > 0:
                                results['step_stats'][key].append(np.sum(_val))
                            else:
                                results['step_stats'][key].append(0)
                        except TypeError:
                            # val is non-numeric, cannot average
                            pass

                elif key == 'step_counter':
                    results['step_stats'][key].append(_val[-1])

            prev_idx = copy.deepcopy(len(val))
            step_counter += 1

        results['episode_stats']['return'] = np.sum(results['step_stats']['reward'])
        for key, val in self.env.cluster.episode_stats.items():
            # try:
                # val = list(val)
            # except TypeError:
                # # val is not iterable (e.g. is int or float)
                # val = [val]
            # try:
                # results['episode_stats'][key] = np.mean(val)
            # except TypeError:
                # # val is not numeric (is e.g. a string)
                # results['episode_stats'][key] = val
            results['episode_stats'][key] = np.array(val) # CHANGE
        # print(f'\nstep stats: {results["step_stats"]}')
        # print(f'\nepisode stats: {results["episode_stats"]}\n')


        if self.wandb is not None:
            wandb_log = {}

            # record raw logged metrics in a custom wandb table
            wandb_log[f'valid/completed_jobs_table'] = self._create_raw_logged_metric_wandb_table(results=results, col_headers_to_try=self.env.cluster.episode_completion_metrics())
            wandb_log[f'valid/blocked_jobs_table'] = self._create_raw_logged_metric_wandb_table(results=results, col_headers_to_try=self.env.cluster.episode_blocked_metrics())

            # record mean of logged metrics
            for log_name, log in results.items():
                for key, val in log.items():
                    # record average of stat for validation run
                    # print(f'key: {key} | val: {val}')
                    wandb_log[f'valid/{log_name}/{key}'] = np.mean(val)
            self.wandb.log(wandb_log)

        return results

    def _create_raw_logged_metric_wandb_table(self, results, col_headers_to_try, verbose=False):
        row_idx_to_col_vals = defaultdict(list)
        col_headers = []
        for key in col_headers_to_try:
            if key in results['episode_stats']:
                if isinstance(results['episode_stats'][key], Iterable):
                    if verbose:
                        print(f'{key} vals are iterable, logging raw vals in wandb table...')
                    col_headers.append(key)
                    for row_idx in range(len(results['episode_stats'][key])):
                        row_idx_to_col_vals[row_idx].append(results['episode_stats'][key][row_idx])
                else:
                    if verbose:
                        print(f'{key} vals ({results["episode_stats"][key]}) are not iterable, no need to log raw val in wandb table.')
            else:
                if verbose:
                    print(f'No stats for metric {key} were recorded this episode.')
        if verbose:
            print(f'Table column headers: {col_headers}')
        wandb_table = self.wandb.Table(columns=col_headers)
        for row in row_idx_to_col_vals.values():
            wandb_table.add_data(*row)
        return wandb_table

