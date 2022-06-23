from ddls.environments.ddls_reward_function import DDLSRewardFunction
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment

import numpy as np
import math
from typing import Union


class LookaheadJobCompletionTime(DDLSRewardFunction):
    def __init__(self, 
                 fail_reward: Union[int, float] = -1, 
                 sign: int = 1, 
                 inverse: bool = True, 
                 transform_with_log: bool = False):
        self.fail_reward = fail_reward 
        self.sign = sign
        self.inverse = inverse
        self.transform_with_log = transform_with_log

    def reset(self, *args, **kwargs):
        self.job_idxs_processed = set()

    def extract(self, 
                env, # RampJobPlacementShapingEnvironment, 
                done: bool):
        reward = self.fail_reward
        for job_idx in env.placed_job_idxs:
            if job_idx in env.cluster.jobs_running:
                reward = env.cluster.jobs_running[job_idx].details['lookahead_job_completion_time']
            elif job_idx in env.cluster.jobs_completed:
                reward = env.cluster.jobs_completed[job_idx].details['lookahead_job_completion_time']
            else:
                raise Exception(f'Unable to find job_idx {job_idx} in either cluster running or completed jobs.')

        if self.inverse and reward != 0:
            reward = 1 / reward

        reward *= self.sign

        if self.transform_with_log:
            sign = math.copysign(1, reward)
            reward = sign * math.log(1 + abs(reward), 10)

        return reward
