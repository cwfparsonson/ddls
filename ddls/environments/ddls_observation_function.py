from ddls.environments.cluster.cluster_environment import ClusterEnvironment

from abc import ABC, abstractmethod

class DDLSObservationFunction(ABC):
    @abstractmethod
    def reset(self, cluster: ClusterEnvironment):
        pass

    @abstractmethod
    def extract(self, cluster: ClusterEnvironment):
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass

    @abstractmethod
    def get_action_set_and_action_mask(self, cluster: ClusterEnvironment):
        pass
