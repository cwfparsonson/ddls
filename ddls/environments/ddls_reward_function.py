from abc import ABC, abstractmethod

class DDLSRewardFunction(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def extract(self):
        pass
