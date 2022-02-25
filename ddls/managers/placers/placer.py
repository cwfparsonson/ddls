from abc import ABC, abstractmethod

class Placer(ABC):
    @abstractmethod
    def __init__(self, 
                 parallelisation: str = 'data_parallelisation'):
        self.parallelisation = parallelisation
    
    @abstractmethod
    def place_job(self, job, cluster):
        '''Place a job in the cluster.'''
        pass
