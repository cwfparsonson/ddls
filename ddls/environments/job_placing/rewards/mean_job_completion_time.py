from ddls.environments.ddls_reward_function import DDLSRewardFunction
from ddls.environments.cluster.cluster_environment import ClusterEnvironment
from ddls.utils import transform_with_log

import numpy as np
import math


class MeanJobCompletionTime(DDLSRewardFunction):
    def __init__(self, 
                 sign: int = -1,
                 transform_with_log: bool = True):
        self.sign = sign
        self.transform_with_log = transform_with_log

    def reset(self, cluster: ClusterEnvironment):
        pass

    def extract(self, cluster: ClusterEnvironment, done: bool):
        num_jobs_completed = cluster.step_stats['num_jobs_completed']
        if num_jobs_completed != 0:
            reward = np.mean(cluster.sim_log['job_completion_time'][-num_jobs_completed:])
        else:
            reward = 0

        if self.transform_with_log:
            reward = transform_with_log(reward)

        return self.sign * reward

