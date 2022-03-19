from ddls.devices.processors.processor import Processor

class A100(Processor):
    def __init__(self,
                 device_id: int = None):
        if device_id is None:
            self.device_id = id(self)
        else:
            self.device_id = device_id
            
        self.device_type = 'A100'

        self.memory_capacity = int(40e9)
        self.memory_occupied = 0
        self.memory_bandwidth = int(1.555e9)

        self.num_streaming_multiprocessors = 8
        self.num_tensor_cores_per_streaming_multiprocessor = 8
        self.num_tensor_cores = self.num_streaming_multiprocessors * self.num_tensor_cores_per_streaming_multiprocessor

        self.base_clock_frequency = int(1095e6)
        
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