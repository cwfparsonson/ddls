from abc import ABC, abstractmethod

class DDLSReward(ABC):
    @abstractmethod
    def before_reset(self):
        pass

    @abstractmethod
    def extract(self):
        pass
