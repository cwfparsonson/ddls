from ddls.managers.schedulers.job_scheduler import JobScheduler
from ddls.demands.jobs.job import Job
from ddls.clusters.cluster import Cluster

import numpy as np
import random
from collections import defaultdict
import copy


class FIFOJobScheduler(JobScheduler):
    def __init__(self):
        super().__init__()

    def get_schedule(self, 
                     new_placements: dict,
                     cluster: Cluster):
        # initialise job op schedule for each worker
        worker_to_job_to_op_to_priority = defaultdict(lambda: defaultdict(dict))

        if len(new_placements) == 0:
            # no new placements made, job op schedule is unchanged
            return worker_to_job_to_op_to_priority
        else:
            # new job(s) will be mounted on cluster, need to get new job op schedule
            pass

        # combine current cluster placement status with new placement decisions so can schedule ops
        placement = copy.deepcopy(cluster.placement)
        for job_id in new_placements.keys():
            placement[job_id] = new_placements[job_id]

        # gather the placed jobs for which an op schedule is needed
        jobs = [job for job in cluster.job_queue.jobs.values() if job_id in new_placements]
        for job in cluster.jobs_running.values():
            jobs.append(job)

        # initialise useful mappings
        job_id_to_job = {job.job_id: job for job in jobs}
        worker_to_type = cluster.topology.graph.graph['worker_to_type'] # maps worker id to its device type so that can query profiled job computation time -> get op run times
        worker_to_ops = self.get_worker_to_ops(placement)

        # schedule ops on each worker
        for worker_id, ops in worker_to_ops.items():
            # get cost of each op
            job_op_to_cost = {f'{op["job_id"]}_{op["op_id"]}': job_id_to_job[op['job_id']].details['time_arrived'] for op in ops}
            # sort ops in descending order of cost
            sorted_job_op_to_cost = sorted(job_op_to_cost, key=job_op_to_cost.get, reverse=True)
            # highest cost ops have lowest priority
            for priority, job_op in enumerate(list(sorted_job_op_to_cost)):
                job_id, op_id = [int(i) for i in job_op.split('_')]
                worker_to_job_to_op_to_priority[worker_id][job_id][op_id] = priority

        return worker_to_job_to_op_to_priority 
















