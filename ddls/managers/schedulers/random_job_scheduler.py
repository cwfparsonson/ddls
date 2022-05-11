from ddls.managers.schedulers.job_scheduler import JobScheduler
from ddls.demands.jobs.job import Job
from ddls.environments.cluster.cluster_environment import ClusterEnvironment

import numpy as np
import random
from collections import defaultdict
import copy


class RandomJobScheduler(JobScheduler):
    def __init__(self):
        pass

    def get_schedule(self, 
                     new_placements: dict,
                     cluster: ClusterEnvironment):
        # initialise job op schedule for each worker
        worker_to_job_to_op_to_priority = defaultdict(lambda: defaultdict(dict))

        if len(new_placements) == 0:
            # no new placements made, job op schedule is unchanged
            return worker_to_job_to_op_to_priority
        else:
            # new job(s) will be mounted on cluster, need to get new job op schedule
            pass

        # combine current cluster placement status with new placement decisions so can schedule ops
        placement = copy.deepcopy(cluster.job_op_placement)
        for job_id in new_placements.keys():
            placement[job_id] = new_placements[job_id]

        # gather the placed jobs for which an op schedule is needed
        jobs = [job for job in cluster.job_queue.jobs.values() if job_id in new_placements]
        for job in cluster.jobs_running.values():
            jobs.append(job)

        # initialise useful mappings
        job_id_to_job = {job.job_id: job for job in jobs}
        worker_to_type = cluster.topology.graph.graph['worker_to_type'] # maps worker id to its device type so that can query profiled job computation time -> get op run times
        worker_to_ops = super().get_worker_to_ops(placement)

        # schedule ops on each worker
        for worker_id, ops in worker_to_ops.items():
            # randomly shuffle order of ops
            shuffled_ops = list(ops)
            random.shuffle(shuffled_ops)

            # assign priorities based on random shuffle order
            for priority, op in enumerate(shuffled_ops):
                job_id, op_id = op['job_id'], op['op_id']
                worker_to_job_to_op_to_priority[worker_id][job_id][op_id] = priority

        return worker_to_job_to_op_to_priority 
















