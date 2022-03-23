from ddls.clusters.cluster import Cluster

class ClusterObservation:
    def __init__(self):
        pass

    def before_reset(self, 
                     cluster: Cluster):
        pass

    def extract(self, 
                cluster: Cluster,
                done: bool):
        return


