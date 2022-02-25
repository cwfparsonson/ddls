from abc import ABC, abstractmethod

class Scheduler(ABC):
    @abstractmethod
    def prioritise_jobs(self, jobs):
        '''Set order in which a set of jobs should be prioritised.'''
        return
