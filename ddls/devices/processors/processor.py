from abc import ABC, abstractmethod

class Processor(ABC):
    @abstractmethod
    def mount(self, workload):
        '''Mount a workload onto the device.'''
        return

    @abstractmethod
    def unmount(self, workload):
        '''Unmount a workload from the device.'''
        return
        
    @abstractmethod
    def step(self, time):
        '''Spend some amount of time running the workloads on the device.'''
        return
