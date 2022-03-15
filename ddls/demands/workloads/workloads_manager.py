class DataParallelWorkloadsManager:
    def __init__(self,
                 job,
                 node_to_workloads):
        '''
        Args:
            node_to_workloads (dict): Maps cluster node to workload(s) allocated.
        '''
        self.job = job
        self.node_to_workloads = node_to_workloads

        self.workloads_remaining = copy.deepcopy(node_to_workloads)
        self.workloads_completed = {}

    def register_completed_workload(self, workload):
        self.workloads_completed[workload.workload_id] = workload
        del self.workloads_remaining[workload.workload_id]

