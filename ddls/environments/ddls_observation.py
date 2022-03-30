from abc import ABC, abstractmethod

class DDLSObservation(ABC):
    @abstractmethod
    def before_reset(self):
        pass

    @abstractmethod
    def extract(self):
        pass
