
from ddls.managers.communicators.job_communicator import JobCommunicator 

class SRPTJobScheduler(JobCommunicator):
    def __init__(self):
        pass
    
    def communicate(self, jobs):
        raise Exception('Not implemented.')