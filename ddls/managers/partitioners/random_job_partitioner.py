
from ddls.managers.partitioners.job_partitioner import JobPartitioner

class RandomJobPartitioner(JobPartitioner):
    def __init__(self):
        pass
    
    def partition(self, jobs):
        raise Exception('Not implemented.')