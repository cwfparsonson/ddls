from ddls.demands.workloads.workload import Workload
from ddls.demands.jobs.job import Job
from ddls.devices.processors.gpu import GPU

from typing import Union
import math


class DataParallelWorkload(Workload):
    def __init__(self,
                 workload_id: Union[int, str],
                 job: Job,
                 local_batch_size: int,
                 details: dict = {}):
        self.workload_id = workload_id
        self.job = job
        self.local_batch_size = local_batch_size
        self.details = details
        
        self.time_ran_for = 0
        self.started, self.completed = False, False
        
    def get_workload_size(self):
        return self.job.get_model_size() + (self.local_batch_size * self.job.sample_size)
    
    def get_run_time(self, device):
        '''
        Returns the time taken to do a forward and backward pass of the local batch
        through the model to obtain a set of gradients.
        '''
        return self.get_time_per_model_pass(device) * 2
        
    def get_time_per_model_pass(self, device):
        return self.job.num_layers * self.get_time_per_layer_pass(device)
        
    def get_time_per_layer_pass(self, device):
        '''Return time per pass through each layer.
        
        GPU: Each tensor core can do a 4x4 matrix operation in one clock cycle. If we assume the model
        has num_dims_per_layer, and that in each layer a num_dims_per_layer x num_dims_per_layer
        matrix multiplication must be performed, then we can calculate the number of cycles needed
        at each layer and therefore the total number of cycles needed per layer.
        Using as rough guide: https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/
        
        Args:
            device: Device onto which the workload is mounted.
        '''
        if self.workload_id not in device.mounted_workloads:
            raise Exception(f'Must first mount workload onto device memory.')
            
        if type(device) is GPU:
            # get number of tensor core ops needed for layer matrix operation on the local batch of the workload
            if self.job.num_dims_per_layer % 4 != 0:
                raise Exception(f'Layer dimensions must be divisible by a 4x4 op tensor core.')
            # need to do a batch_size x input_dims x output_dims calculation using 4x4 tensor core(s)
            num_tensor_core_ops = (self.local_batch_size / 4) * (self.job.num_dims_per_layer / 4) * (self.job.num_dims_per_layer / 4)
            batches = math.ceil(num_tensor_core_ops / device.num_tensor_cores)

            # get time to read data from global GPU memory
            time_per_read = self.get_workload_size() / device.memory_bandwidth

            return (batches / device.base_clock_frequency) + (batches * time_per_read)
        
        else:
            raise Exception(f'Unrecognised device type {type(device)}')
            
    def step(self, device, time):
        if not self.started:
            self.started = True # started running workload
        self.time_ran_for += time
        if self.time_ran_for >= self.get_run_time(device):
            self.completed = True
    
    def __str__(self):
        descr = f'Workload ID: {self.workload_id}'
        descr += f' | Local batch size: {self.local_batch_size}'
        descr += f' | Total workload memory size: {self.get_workload_size():.3e}'
        descr += f' | Parent job ID: {self.job.job_id}'
        return descr
