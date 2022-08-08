from ddls.demands.jobs.job import Job
from ddls.environments.cluster.cluster_environment import ClusterEnvironment

from abc import ABC, abstractmethod
from collections import defaultdict

class JobScheduler(ABC):
    @abstractmethod
    def get(self, 
            job_placement,
            cluster: ClusterEnvironment):
        '''Set order in which a set of operations should be scheduled.

        Returns nested dict mapping worker_id -> job_id -> op_id -> priority, where priority
        is an integer giving the priority of each op on each worker. Operations 
        should be scheduled in order of priority, with the op with the highest
        priority value being scheduled first etc.
        '''
        pass
        
