from ddls.demands.workloads.workload import Workload

class ModelParallelWorkload(Workload):
    def __init__(self,
                 workload_id: int,
                 job: Job):
        self.workload_id = workload_id
        self.job = job
