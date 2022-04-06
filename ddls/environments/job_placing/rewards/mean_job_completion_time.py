from ddls.environments.ddls_reward_function import DDLSRewardFunction

import numpy as np


class MeanJobCompletionTime(DDLSRewardFunction):
    def __init__(self, sign: int = -1):
        self.sign = sign

    def reset(self):
        self.started = False

    def extract(self, cluster, done):
        if not self.started:
            self.started = True
            reward = 0
        else:
            num_jobs_completed = cluster.step_stats['num_jobs_completed']
            if num_jobs_completed != 0:
                reward = np.mean(cluster.sim_log['job_completion_time'][-num_jobs_completed:])
            else:
                reward = 0
        return self.sign * reward

