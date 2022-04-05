from abc import ABC, abstractmethod

class DDLSReward(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def extract(self):
        pass
