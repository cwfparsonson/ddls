from ddls.managers.schedulers.job_scheduler import JobScheduler 

import numpy as np

class SRPTJobScheduler(JobScheduler):
    def __init__(self):
        pass
    
    def schedule(self, operations):
        raise Exception('Not implemented.')