from ddls.managers.schedulers.job_scheduler import JobScheduler
from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.job_schedule import JobSchedule

import numpy as np
from collections import defaultdict
import copy

class SRPTOpScheduler:

    def get(self, 
            job_placement,
            cluster: RampClusterEnvironment):
        # get new placements made by job placer
        new_placements = job_placement.placement

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

        # if remaining run time not initialised for op, initialise so can calc op costs for scheduling
        for job in job_id_to_job.values():
            test_node = list(job.computation_graph.nodes())[0] 
            if job.computation_graph.nodes[test_node]['remaining_run_time'] is None:
                # initialise so can use SRPT
                for op_id in job.computation_graph.nodes:
                    worker_id = placement[job.job_id][op_id]
                    job.reset_op_remaining_run_time(op_id, device_type=worker_to_type[worker_id])
        
        # schedule ops on each worker
        for worker_id, ops in job_placement.worker_to_ops.items():
            # get cost of each op
            job_op_to_cost = {f'{op["job_id"]}_{op["op_id"]}': job_id_to_job[op['job_id']].computation_graph.nodes[op['op_id']]['remaining_run_time'] for op in ops}
            # sort ops in descending order of cost
            sorted_job_op_to_cost = sorted(job_op_to_cost, key=job_op_to_cost.get, reverse=True)
            # highest cost ops have lowest priority
            for priority, job_op in enumerate(list(sorted_job_op_to_cost)):
                job_id, op_id = [int(i) for i in job_op.split('_')]
                worker_to_job_to_op_to_priority[worker_id][job_id][op_id] = priority

        return JobSchedule(worker_to_job_to_op_to_priority)




