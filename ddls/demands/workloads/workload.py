from abc import ABC, abstractmethod

class Workload(ABC):
    @abstractmethod
    def get_workload_size(self):
        '''Return total memory size occupied by the workload when mounted onto a device.'''
        return
    
    @abstractmethod
    def get_run_time(self, device):
        '''Return total run time of the workload when mounted on the mounted device.'''
        return
    
    def step(self, device, time):
        '''Run the workload for some amount of time on its mounted device.'''
        return
