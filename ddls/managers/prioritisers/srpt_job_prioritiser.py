
from ddls.managers.prioritisers.job_prioritiser import JobPrioritiser

import numpy as np

class SRPTJobScheduler(JobPrioritiser):
    def __init__(self):
        pass
    
    def prioritise(self, jobs):
        sizes = np.array([job.get_model_size() + job.get_dataset_size() for job in jobs])
        return np.array(jobs)[np.argsort(sizes)] 