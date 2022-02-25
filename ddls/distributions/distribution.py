from abc import ABC, abstractmethod

class Distribution(ABC):

    @abstractmethod
    def sample(self,
               num_samples: int):
        '''Generates data by sampling num_samples from the distribution.'''
        return
