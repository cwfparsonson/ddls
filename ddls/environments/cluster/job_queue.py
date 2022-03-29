from ddls.demands.jobs.job import Job

from collections import OrderedDict
from typing import Union
import copy


class JobQueue:
    def __init__(self, 
                 queue_capacity: int):
        self.jobs = OrderedDict()
        self.queue_capacity = queue_capacity
        
    def __len__(self):
        return len(self.jobs)
        
    def add(self, jobs: Union[list[Job], Job]):
        if type(jobs) is not list:
            jobs = [jobs]
        if self.can_fit(jobs):
            for job in jobs:
                self.jobs[job.job_id] = job
        else:
            raise Exception(f'Cannot fit all jobs, only have {self.queue_capacity - len(self)} of space remaining.')
    
    def can_fit(self, jobs):
        if type(jobs) is not list:
            jobs = [jobs]
        if len(self) + len(jobs) <= self.queue_capacity:
            return True
        else:
            return False
    
    def remove(self, jobs: Union[list[Job], Job]):
        if type(jobs) is not list:
            jobs = [jobs]
        for job in jobs:
            del self.jobs[job.job_id]
