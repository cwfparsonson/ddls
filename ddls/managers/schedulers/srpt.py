from ddls.managers.schedulers.scheduler import Scheduler

import numpy as np

class SRPT(Scheduler):
    def __init__(self):
        pass
    
    def prioritise_jobs(self, jobs):
        sizes = np.array([job.get_model_size() + job.get_dataset_size() for job in jobs])
        return np.array(jobs)[np.argsort(sizes)] 
