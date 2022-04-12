from ddls.environments.cluster.cluster_environment import ClusterEnvironment

from abc import ABC, abstractmethod

class DDLSRewardFunction(ABC):
    @abstractmethod
    def reset(self, cluster: ClusterEnvironment):
        pass

    @abstractmethod
    def extract(self, cluster: ClusterEnvironment, done: bool):
        pass
