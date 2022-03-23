from ddls.devices.processors.processor import Processor

from collections import defaultdict


class A100(Processor):
    def __init__(self,
                 processor_id: int = None):
        if processor_id is None:
            self.processor_id = id(self)
        else:
            self.processor_id = processor_id
            
        self.device_type = 'A100'

        self.memory_capacity = int(40e9)
        self.memory_occupied = 0
        self.memory_bandwidth = int(1.555e9)

        self.num_streaming_multiprocessors = 8
        self.num_tensor_cores_per_streaming_multiprocessor = 8
        self.num_tensor_cores = self.num_streaming_multiprocessors * self.num_tensor_cores_per_streaming_multiprocessor

        self.base_clock_frequency = int(1095e6)
        
        self.mounted_job_to_ops = defaultdict(set)
        
    def __str__(self):
        return f'{self.device_type}_{self.processor_id}'
    
    def mount(self, job, op_id):
        if self.memory_occupied + job.computation_graph.nodes[op_id]['memory_cost'] > self.memory_capacity:
            raise Exception(f'Trying to allocate {job.nodes[op_id]["memory_cost"]} of memory for job {job} op {op_id} but have only {self.memory_capacity - self.memory_occupied} available on processor {self.processor_id}.')
        self.mounted_job_to_ops[job.job_id].add(op_id)
        
    def unmount(self, job, op_id):
        self.memory_occupied -= job.computation_graph.nodes[op_id]['memory_cost']
        self.mounted_job_to_ops[job.job_id][op_id].remove()
        
    def step(self, time):
        # for workload in self.mounted_workloads.values():
            # workload.step(self, time)
        pass
