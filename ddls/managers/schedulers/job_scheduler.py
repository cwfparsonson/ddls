from abc import ABC, abstractmethod

class JobScheduler(ABC):
    @abstractmethod
    def schedule(self, operations):
        '''Set order in which a set of operations should be scheduled.'''
        return
