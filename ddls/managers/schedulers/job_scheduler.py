from ddls.demands.jobs.job import Job

from abc import ABC, abstractmethod

class JobScheduler(ABC):
    @abstractmethod
    def get_schedule(self, 
            jobs: list[Job],
                     placement: dict):
        '''Set order in which a set of operations should be scheduled.

        Args:
            jobs: Jobs which have been placed.
            placement: Nested dict of placement decisions mapping 
                job_id -> operation_id -> worker_id

        Returns nested dict mapping worker_id -> job_id -> op_id -> priority, where priority
        is an integer giving the priority of each op on each worker. Operations 
        should be scheduled in order of priority, with the op with the highest
        priority value being scheduled first etc.
        '''
        pass
        
