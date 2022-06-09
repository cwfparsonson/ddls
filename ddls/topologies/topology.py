from abc import ABC, abstractmethod
from typing import Optional


class Topology(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _build_topology(self, *args, **kwargs):
        pass
    
