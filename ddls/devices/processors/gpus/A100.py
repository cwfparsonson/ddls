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
        # self.memory_bandwidth = int(1.555e9)

        # self.num_streaming_multiprocessors = 8
        # self.num_tensor_cores_per_streaming_multiprocessor = 8
        # self.num_tensor_cores = self.num_streaming_multiprocessors * self.num_tensor_cores_per_streaming_multiprocessor

        # self.base_clock_frequency = int(1095e6)

        self.reset()
        
        
    def __str__(self):
        return f'{self.device_type}_{self.processor_id}'

    def reset(self):
        self.memory_occupied = 0
        self.mounted_job_idx_to_ops = defaultdict(set)
        self.mounted_job_op_to_priority = dict() # job op schedule of mounted ops

    def mount(self, job, op_id):
        '''Returns job with initialised remaining_run_time for each op.'''
        if self.device_type not in job.computation_graph.nodes[op_id]['compute_cost']:
            raise Exception(f'Tried to mount op on device type {self.device_type} but only profile op compute cost for {job.computation_graph.nodes[op_id]["compute_cost"]}')
        if self.memory_occupied + job.computation_graph.nodes[op_id]['memory_cost'] > self.memory_capacity:
            raise Exception(f'Trying to allocate {job.nodes[op_id]["memory_cost"]} of memory for job {job} op {op_id} but have only {self.memory_capacity - self.memory_occupied} available on processor {self.processor_id}.')
        self.mounted_job_idx_to_ops[job.details['job_idx']].add(op_id)
        self.memory_occupied += job.computation_graph.nodes[op_id]['memory_cost']
        
    def unmount(self, job, op_id):
        self.memory_occupied -= job.computation_graph.nodes[op_id]['memory_cost']
        self.mounted_job_idx_to_ops[job.details['job_idx']].remove(op_id)
        del self.mounted_job_op_to_priority[f'{job.details["job_idx"]}_{job.job_id}_{op_id}']
        if len(self.mounted_job_idx_to_ops[job.details['job_idx']]) == 0:
            del self.mounted_job_idx_to_ops[job.details['job_idx']]
