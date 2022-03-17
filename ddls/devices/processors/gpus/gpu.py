from ddls.devices.processors.processor import Processor
import math


class GPU(Processor):
    def __init__(self,
                 device_id: int = None,
                 device_type: str = 'GPU',
                 memory_capacity: int = int(40e9),
                 memory_bandwidth: int = int(1.555e9),
                 num_streaming_multiprocessors: int = 8,
                 num_tensor_cores_per_streaming_multiprocessor: int = 8,
                 base_clock_frequency: int = int(1095e6)):
        '''
        Refer to https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/ for more details.
        '''
        if device_id is None:
            self.device_id = id(self)
        else:
            self.device_id = device_id
        self.device_type = device_type
        
        self.memory_capacity = memory_capacity
        self.memory_occupied = 0
        self.memory_bandwidth = memory_bandwidth
        
        self.num_streaming_multiprocessors = num_streaming_multiprocessors
        self.num_tensor_cores_per_streaming_multiprocessor = num_tensor_cores_per_streaming_multiprocessor
        self.num_tensor_cores = num_streaming_multiprocessors * num_tensor_cores_per_streaming_multiprocessor
        
        self.base_clock_frequency = base_clock_frequency
        
        self.mounted_workloads = {}
        
    def __str__(self):
        return f'{self.device_type}_{self.device_id}'
    
    def mount(self, workload):
        if self.memory_occupied + workload.get_workload_size() > self.memory_capacity:
            raise Exception(f'Trying to allocate {workload.get_workload_size()} of memory but have only {self.memory_capacity - self.memory_occupied} available.')
        self.memory_occupied += workload.get_workload_size()
        self.mounted_workloads[workload.workload_id] = workload
        
    def unmount(self, workload):
        self.memory_occupied -= workload.get_workload_size()
        del self.mounted_workloads[workload.workload_id]
        
    def step(self, time):
        for workload in self.mounted_workloads.values():
            workload.step(self, time)
