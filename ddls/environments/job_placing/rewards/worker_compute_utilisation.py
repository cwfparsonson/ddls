from ddls.environments.ddls_reward import DDLSReward

class WorkerComputeUtilisation(DDLSReward):
    def reset(self):
        self.started = False

    def extract(self, cluster, done):
        if not self.started:
            self.started = True
            return 0
        else:
            return cluster.step_stats['mean_worker_compute_utilisation']

        
