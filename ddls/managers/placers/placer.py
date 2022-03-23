from ddls.demands.jobs.job import Job
from ddls.clusters.cluster import Cluster

from abc import ABC, abstractmethod

class Placer(ABC):
    @abstractmethod
    def __init__(self, 
                 parallelisation: str = 'data_parallelisation'):
        self.parallelisation = parallelisation
    
    @abstractmethod
    def get_placement(self, 
                      jobs: list[Job], 
                      cluster: Cluster):
        '''Place jobs in the cluster.'''
        pass
