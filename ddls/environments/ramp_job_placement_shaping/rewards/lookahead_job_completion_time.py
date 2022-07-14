from ddls.environments.ddls_reward_function import DDLSRewardFunction
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment

import numpy as np
import math
from typing import Union


class LookaheadJobCompletionTime(DDLSRewardFunction):
    def __init__(self, 
                 # fail_reward: Union[int, float] = -1, 
                 # sign: int = 1, 
                 # inverse: bool = True, 
                 # transform_with_log: bool = False,
                 # fail_reward: Union[int, float] = 1000, 
                 fail_reward: Union[int, float, 'job_sequential_completion_time'] = 'job_sequential_completion_time', # reward to issue agent for a job being blocked
                 fail_reward_factor: int = 2, # factor by which to multiply fail reward by so e.g. do not punish agent for placing all ops on one server by same amount as for blocking job
                 sign: int = -1, 
                 inverse: bool = False, 
                 transform_with_log: bool = True,
                 ):
        self.fail_reward = fail_reward 
        self.fail_reward_factor = fail_reward_factor
        self.sign = sign
        self.inverse = inverse
        self.transform_with_log = transform_with_log

    def reset(self, *args, **kwargs):
        self.job_idxs_processed = set()

    def extract(self, 
                env, # RampJobPlacementShapingEnvironment, 
                done: bool):
        # TODO TEMP: Assume 1 job per step
        job_idx = env.last_job_arrived_job_idx

        # print(f'job_idx: {job_idx}')
        # print(f'jobs placed: {env.placed_job_idxs}')
        # print(f'jobs running: {env.cluster.jobs_running}')
        # print(f'jobs completed: {env.cluster.jobs_completed}')
        # print(f'jobs blocked: {env.cluster.jobs_blocked}')

        if job_idx in env.placed_job_idxs:
            # job was placed, set reward as job completion time
            if job_idx in env.cluster.jobs_running:
                reward = env.cluster.jobs_running[job_idx].details['lookahead_job_completion_time']
            elif job_idx in env.cluster.jobs_completed:
                reward = env.cluster.jobs_completed[job_idx].details['lookahead_job_completion_time']
            else:
                raise Exception(f'Unable to find job_idx {job_idx} in either cluster running or completed jobs.')
        else:
            # job was not successfully placed, set reward as fail reward
            if isinstance(self.fail_reward, int) or isinstance(self.fail_reward, float):
                reward = copy.deepcopy(self.fail_reward) * self.fail_reward_factor
            elif isinstance(self.fail_reward, str):
                if self.fail_reward == 'job_sequential_completion_time':
                    # TODO TEMP: Currently assuming one device in whole cluster, should update to handle multiple different device types?
                    device_type = list(env.cluster.topology.graph.graph['worker_types'])[0]
                    reward = env.cluster.jobs_blocked[job_idx].details['job_sequential_completion_time'][device_type] * self.fail_reward_factor
                else:
                    raise Exception(f'Unrecognised fail_reward {self.fail_reward}')

        # do any reward processing
        if self.inverse and reward != 0:
            reward = 1 / reward

        reward *= self.sign

        if self.transform_with_log:
            sign = math.copysign(1, reward)
            reward = sign * math.log(1 + abs(reward), 10)

        return reward
