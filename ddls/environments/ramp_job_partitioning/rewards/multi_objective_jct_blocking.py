from ddls.environments.ddls_reward_function import DDLSRewardFunction
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment

import numpy as np
import math
from typing import Union


class MultiObjectiveJCTBlocking(DDLSRewardFunction):
    def __init__(self, 
                 blocking_weight=1, # use to scale the importance of minimising blocking vs. minimising JCT (higher blocking rate -> less blocking, higher JCT)
                 sign: int = -1, 
                 inverse: bool = False, 
                 transform_with_log: bool = False,
                 ):
        self.blocking_weight = blocking_weight
        self.sign = sign
        self.inverse = inverse
        self.transform_with_log = transform_with_log

        # self.verbose = True # DEBUG
        self.verbose = False

    def reset(self, *args, **kwargs):
        pass

    def _get_job_accepted_reward(self, job_idx, env):
        device_type = list(env.cluster.topology.graph.graph['worker_types'])[0]
        if job_idx in env.cluster.jobs_running:
            job = env.cluster.jobs_running[job_idx]
        elif job_idx in env.cluster.jobs_completed:
            job = env.cluster.jobs_completed[job_idx]
        else:
            raise Exception(f'Unable to find job_idx {job_idx} in either cluster running or completed jobs.')

        if self.verbose:
            print(f'Calculating job accepted reward...')
            print(f'job lookahead_job_completion_time: {job.details["lookahead_job_completion_time"]}')
            print(f'job_sequential_completion_time: {job.details["job_sequential_completion_time"][device_type]}')

        return (job.details['lookahead_job_completion_time'] / job.details['job_sequential_completion_time'][device_type])

    def _get_job_blocked_reward(self, job_idx, env):
        device_type = list(env.cluster.topology.graph.graph['worker_types'])[0]
        job = env.cluster.jobs_blocked[job_idx]

        if self.verbose:
            print(f'Calculating job blocked reward...')
            print(f'blocking_weight: {self.blocking_weight}')
            print(f'job_sequential_completion_time: {job.details["job_sequential_completion_time"]}')
            print(f'max_job_sequential_completion_time: {env.cluster.jobs_generator.jobs_params["max_job_sequential_completion_times"]}')
            print(f'min_job_sequential_completion_time: {env.cluster.jobs_generator.jobs_params["min_job_sequential_completion_times"]}')

        return self.blocking_weight * (( (job.details['job_sequential_completion_time'][device_type] - env.cluster.jobs_generator.jobs_params['min_job_sequential_completion_times']) / (env.cluster.jobs_generator.jobs_params['max_job_sequential_completion_times'] - env.cluster.jobs_generator.jobs_params['min_job_sequential_completion_times']) ) + 1)

    def _get_reward(self, env):
        # TODO TEMP: Assume 1 job per step
        job_idx = env.last_job_arrived_job_idx

        if job_idx in env.placed_job_idxs:
            # job was accepted
            reward = self._get_job_accepted_reward(job_idx, env)
        else:
            # job was blocked
            reward = self._get_job_blocked_reward(job_idx, env)

        return reward

    def extract(self, 
                env, # RampJobPlacementShapingEnvironment, 
                done: bool):
        reward = self._get_reward(env)

        if self.verbose:
            print(f'reward before processing: {reward}')

        # do any reward processing
        if self.inverse and reward != 0:
            reward = 1 / reward

        reward *= self.sign

        if self.transform_with_log:
            sign = math.copysign(1, reward)
            reward = sign * math.log(1 + abs(reward), 10)

        if self.verbose:
            print(f'reward after processing: {reward}')

        return reward
