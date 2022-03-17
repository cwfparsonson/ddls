
from abc import ABC, abstractmethod

class JobPrioritiser(ABC):
    @abstractmethod
    def prioritise(self, jobs):
        '''Set order in which a set of jobs should be prioritised.'''
        return
