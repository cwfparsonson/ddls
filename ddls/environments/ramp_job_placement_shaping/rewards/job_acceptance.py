from ddls.environments.ddls_reward_function import DDLSRewardFunction
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment

import numpy as np
import math
from typing import Union


class JobAcceptance(DDLSRewardFunction):
    def __init__(self, 
                 fail_reward: Union[int, float] = -1,
                 success_reward: Union[int, float] = 1,
                 ):
        self.fail_reward = fail_reward
        self.success_reward = success_reward

    def reset(self, *args, **kwargs):
        self.job_idxs_processed = set()

    def extract(self, 
                env, # RampJobPlacementShapingEnvironment, 
                done: bool):
        # TODO TEMP: Assume 1 job per step
        job_idx = env.last_job_arrived_job_idx

        if job_idx in env.placed_job_idxs:
            # job was placed
            reward = self.success_reward
        else:
            # job was not successfully placed
            reward = self.fail_reward

        return reward
