
from abc import ABC, abstractmethod

class JobCommunicator(ABC):
    @abstractmethod
    def communicate(self, architectures):
        '''Communicates information amongst devices in an architecture to synchronise workers.'''
        return