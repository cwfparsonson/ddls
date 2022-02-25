from ddls.demands.jobs.job import Job

from typing import Union
import copy


class JobQueue:
    def __init__(self, 
                 queue_capacity: int):
        self.jobs = []
        self.queue_occupancy = 0
        self.queue_capacity = queue_capacity
        
    def __len__(self):
        return len(self.jobs)
        
    def add(self, jobs: Union[list[Job, ...], Job]):
        if type(jobs) is not list:
            jobs = [jobs]
        if self.can_fit(jobs):
            self.jobs.extend(jobs)
        else:
            raise Exception(f'Cannot fit all jobs, only have {self.queue_capacity - self.queue_occupancy} of space remaining.')
    
    def can_fit(self, jobs):
        if type(jobs) is not list:
            jobs = [jobs]
        new_occupancy = copy.deepcopy(self.queue_occupancy)
        for job in jobs:
            new_occupancy += (job.get_model_size() + job.get_dataset_size())
        if new_occupancy <= self.queue_capacity:
            return True
        else:
            return False
    
    def remove(self, jobs: Union[list[Job, ...], Job]):
        if type(jobs) is not list:
            jobs = [jobs]
        for job in jobs:
            self.jobs.remove(job)
            self.queue_occupancy -= (job.get_model_size() + job.get_dataset_size())
