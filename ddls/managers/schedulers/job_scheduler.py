from ddls.demands.jobs.job import Job
from ddls.clusters.cluster import Cluster

from abc import ABC, abstractmethod

class JobScheduler(ABC):
    @abstractmethod
    def get_schedule(self, 
                     new_placements: dict,
                     cluster: Cluster):
        '''Set order in which a set of operations should be scheduled.

        Args:
            new_placements: Any new placement(s) about to be chosen this step by
                the job placer.

        Returns nested dict mapping worker_id -> job_id -> op_id -> priority, where priority
        is an integer giving the priority of each op on each worker. Operations 
        should be scheduled in order of priority, with the op with the highest
        priority value being scheduled first etc.
        '''
        pass
        
    def get_worker_to_ops(self, 
                          placement: dict):
        '''Gather which ops have been placed on each worker.'''
        worker_to_ops = defaultdict(list)
        for job_id in placement.keys():
            for op_id in placement[job_id].keys():
                worker_to_ops[placement[job_id][op_id]].append({'op_id': op_id, 'job_id': job_id})
        return worker_to_ops
