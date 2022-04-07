from abc import ABC, abstractmethod

class DDLSObservationFunction(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def extract(self):
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass


