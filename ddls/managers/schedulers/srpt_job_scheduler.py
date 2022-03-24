from ddls.managers.schedulers.job_scheduler import JobScheduler 
from ddls.demands.jobs.job import Job

import numpy as np
from collections import defaultdict

class SRPTJobScheduler(JobScheduler):
    def __init__(self):
        pass

    def _get_worker_to_ops(self, 
                           placement: dict):
        '''Gather which ops have been placed on each worker.

        
        '''
        worker_to_ops = defaultdict(list)
        for job_id in placement.keys():
            for op_id in placement[job_id].keys():
                worker_to_ops[placement[job_id][op_id]].append({'op_id': op_id, 'job_id': job_id})
        return worker_to_ops

    def get_schedule(self, 
                     jobs: list[Job],
                     placement: dict,
                     worker_to_type: dict):
        '''
        Args:
            worker_to_type: Maps worker id to its processor/device type so that
                can query profiled job computation graph to get op run times.
        '''
        job_id_to_job = {job.job_id: job for job in jobs}
        worker_to_ops = self._get_worker_to_ops(placement)

        # if remaining run time not initialised for op, initialise
        for job in job_id_to_job.values():
            test_node = list(job.computation_graph.nodes())[0] 
            if job.computation_graph.nodes[test_node]['remaining_run_time'] is None:
                # initialise so can use SRPT
                for op_id in job.computation_graph.nodes:
                    worker_id = placement[job.job_id][op_id]
                    job.reset_op_remaining_run_time(op_id, device_type=worker_to_type[worker_id])
        
        worker_to_job_to_op_to_priority = defaultdict(lambda: defaultdict(dict))
        for worker_id, ops in worker_to_ops.items():
            # get cost of each op
            job_op_to_cost = {f'{op["job_id"]}_{op["op_id"]}': job_id_to_job[op['job_id']].computation_graph.nodes[op['op_id']]['remaining_run_time'] for op in ops}
            # sort ops in descending order of cost
            sorted_job_op_to_cost = sorted(job_op_to_cost, key=job_op_to_cost.get, reverse=False)
            # highest cost ops have lowest priority
            for priority, job_op in enumerate(list(sorted_job_op_to_cost)):
                job_id, op_id = job_op.split('_') 
                worker_to_job_to_op_to_priority[worker_id][int(job_id)][int(op_id)] = priority

        return worker_to_job_to_op_to_priority 




