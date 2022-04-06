from ddls.environments.ddls_reward_function import DDLSRewardFunction

class WorkerComputeUtilisation(DDLSRewardFunction):
    def reset(self):
        self.started = False

    def extract(self, cluster, done):

        if not self.started:
            self.started = True
            reward = 0
        else:
            reward = cluster.step_stats['mean_worker_compute_utilisation']

        return reward

        
