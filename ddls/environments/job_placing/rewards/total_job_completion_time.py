from ddls.environments.ddls_reward_function import DDLSRewardFunction
from ddls.environments.cluster.cluster_environment import ClusterEnvironment

import numpy as np


class TotalJobCompletionTime(DDLSRewardFunction):
    def __init__(self, sign: int = -1):
        self.sign = sign

    def reset(self, cluster: ClusterEnvironment):
        pass

    def extract(self, cluster: ClusterEnvironment, done: bool):
        num_jobs_completed = cluster.step_stats['num_jobs_completed']
        if num_jobs_completed != 0:
            reward = np.sum(cluster.sim_log['job_completion_time'][-num_jobs_completed:])
        else:
            reward = 0
        return self.sign * reward

