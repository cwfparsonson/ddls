from ddls.environments.ddls_reward_function import DDLSRewardFunction
from ddls.environments.cluster.cluster_environment import ClusterEnvironment

class WorkerComputeUtilisation(DDLSRewardFunction):
    def reset(self, cluster: ClusterEnvironment):
        pass

    def extract(self, cluster: ClusterEnvironment, done: bool):
        return cluster.step_stats['mean_worker_compute_utilisation']

        
