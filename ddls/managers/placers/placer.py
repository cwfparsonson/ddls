from ddls.demands.jobs.job import Job
from ddls.environments.cluster.cluster_environment import ClusterEnvironment

from abc import ABC, abstractmethod

class Placer(ABC):
    @abstractmethod
    def __init__(self, 
                 parallelisation: str = 'data_parallelisation'):
        self.parallelisation = parallelisation
    
    @abstractmethod
    def get(self, 
            cluster: ClusterEnvironment):
        '''Place jobs in the cluster.'''
        pass
