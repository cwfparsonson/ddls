from ddls.environments.ramp_cluster.actions.op_partition import OpPartition
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.utils import update_dep_run_times

from collections import defaultdict

class OpPlacement:
    def __init__(self,
                 action: dict,
                 op_partition: OpPartition,
                 cluster: RampClusterEnvironment):
        '''
        Args:
            action: Mapping of job_id -> operation_id -> worker_id.
            op_partition: OpPartition object whose partitioned jobs will have their
                dependency run times updated given the placement and network parameters.
        '''
        self.action = action

        self.job_ids, self.worker_ids, self.worker_to_ops, self.job_id_to_worker_ids = set(), set(), defaultdict(list), defaultdict(set)
        for job_id in action.keys():
            self.job_ids.add(job_id)
            for op_id in self.action[job_id].keys():
                self.worker_ids.add(self.action[job_id][op_id])
                self.worker_to_ops[self.action[job_id][op_id]].append({'op_id': op_id, 'job_id': job_id})
                self.job_id_to_worker_ids[job_id].add(self.action[job_id][op_id])

        # set partitioned jobs' dependency run times given their op placements, dependency sizes, and the network's processor and channel link parameters
        # update_dep_run_times(cluster=cluster, op_partition=op_partition, op_placement=self, verbose=True) # DEBUG
        update_dep_run_times(cluster=cluster, op_partition=op_partition, op_placement=self, verbose=False)

    def __str__(self):
        descr = ''
        for job_id in self.action.keys():
            descr += f'\nJob ID: {job_id}'
            descr += f'\n\tWorkers ({len(self.job_id_to_worker_ids[job_id])}): {self.job_id_to_worker_ids[job_id]})'
            for op_id in self.action[job_id].keys():
                worker_id = self.action[job_id][op_id]
                descr += f'\n\tOp ID {op_id} -> Worker ID {worker_id}'
        return descr


