'''
Use for validating heuristics, NOT agents.
'''
from ddls.utils import get_class_from_path, seed_stochastic_modules_globally, recursively_update_nested_dict

from collections import defaultdict
import numpy as np

import copy

from decimal import Decimal

class EvalLoop:
    def __init__(self,
                 # seed: int = None,
                 actor,
                 env,
                 wandb=None,
                 **kwargs):
        self.actor = actor
        self.env = env
        self.wandb = wandb
        # self.seed = seed

    def run(self, verbose=False):
        results = {'step_stats': defaultdict(list), 'episode_stats': {}}

        # if 'seed' in self.:
            # if self.seed is not None:
                # seed_stochastic_modules_globally(self.seed)

        if verbose:
            print(f'Starting validation...')
        obs, done = self.env.reset(), False
        prev_idx = 0
        step_counter = 1
        while not done:
            start_step_mounted_workers = len(self.env.cluster.mounted_workers)
            job_to_place = list(self.env.cluster.job_queue.jobs.values())[0] # assume event-driven where only ever have one job to queue. Use job_to_place for useful info for heuristics
            action = self.actor.compute_action(obs, job_to_place=job_to_place) 
            obs, reward, done, info = self.env.step(action)

            if verbose:
                # print(f'Step {step_counter} | Action: {action} | Reward: {reward:.8f} | Jobs arrived: {self.env.cluster.num_jobs_arrived} | Jobs running: {len(self.env.cluster.jobs_running)} | Jobs completed: {len(self.env.cluster.jobs_completed)} | Jobs blocked: {len(self.env.cluster.jobs_blocked)} | Start->end of step mounted workers: {start_step_mounted_workers}->{len(self.env.cluster.mounted_workers)} | Stopwatch: {Decimal(self.env.cluster.stopwatch.time().astype(float)):.3E}')
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

        for key, val in self.env.cluster.episode_stats.items():
            try:
                val = list(val)
            except TypeError:
                # val is not iterable (e.g. is int or float)
                val = [val]
            try:
                results['episode_stats'][key] = np.mean(val)
            except TypeError:
                # val is not numeric (is e.g. a string)
                results['episode_stats'][key] = val
        results['episode_stats']['return'] = np.sum(results['step_stats']['reward'])

        if self.wandb is not None:
            wandb_log = {}
            for log_name, log in results.items():
                for key, val in log.items():
                    # record average of stat for validation run
                    # print(f'key: {key} | val: {val}')
                    wandb_log[f'valid/{log_name}/{key}'] = np.mean(val)
            self.wandb.log(wandb_log)

        # print(f'\nstep stats: {results["step_stats"]}')
        # print(f'\nepisode stats: {results["episode_stats"]}\n')

        return results

