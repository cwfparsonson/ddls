from ddls.environments.cluster.cluster_environment import ClusterEnvironment

class ClusterObservation:
    def __init__(self):
        pass

    def before_reset(self, 
                     cluster: ClusterEnvironment):
        pass

    def extract(self, 
                cluster: ClusterEnvironment,
                done: bool):
        return


