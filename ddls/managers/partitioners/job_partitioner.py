
from abc import ABC, abstractmethod

class JobPartitioner(ABC):
    @abstractmethod
    def partition(self, job):
        '''Partitions operations in a job computation graph into sub-operations.'''
        return