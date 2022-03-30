from abc import ABC, abstractmethod

class DDLSObservation(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def extract(self):
        pass
