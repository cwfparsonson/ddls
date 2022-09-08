# from ddls.utils import gen_job_dep_str, load_job_dep_str
from ddls.managers.schedulers.job_scheduler import JobScheduler
from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.op_schedule import OpSchedule
from ddls.environments.ramp_cluster.actions.op_partition import OpPartition
from ddls.environments.ramp_cluster.actions.op_placement import OpPlacement

import numpy as np
from collections import defaultdict
import copy
import json

class SRPTOpScheduler:

    def get(self, 
            op_partition: OpPartition,
            op_placement: OpPlacement,
            cluster: RampClusterEnvironment):
        # get new placements made by job placer
        new_placements = op_placement.action

        # gather the placed jobs for which an op schedule is needed
        jobs = [job for job_id, job in op_partition.partitioned_jobs.items() if job_id in new_placements]
        for job in cluster.jobs_running.values():
            jobs.append(job)

        # print(op_partition)
        # print(op_placement)

        # initialise job op schedule for each worker
        worker_to_job_to_op_to_priority = defaultdict(lambda: defaultdict(dict))

        if len(new_placements) == 0:
            # no new placements made, job op schedule is unchanged
            return OpSchedule(worker_to_job_to_op_to_priority)
        else:
            # new job(s) will be mounted on cluster, need to get new job op schedule
            pass

        # combine current cluster placement status with new placement decisions so can schedule ops
        # placement = copy.deepcopy(cluster.job_op_placement)
        # for job_id in new_placements.keys():
            # placement[job_id] = new_placements[job_id]
        placement = new_placements
        placement.update(cluster.job_op_placement)

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
        for worker_id, ops in op_placement.worker_to_ops.items():
            # get cost of each op
            job_op_to_cost = {json.dumps(op["job_id"]) + '_' + json.dumps(op["op_id"]): job_id_to_job[op['job_id']].computation_graph.nodes[op['op_id']]['remaining_run_time'] for op in ops}
            # job_op_to_cost = {}
            # for op in ops:
                # try:
                    # job_op_to_cost[json.dumps(op["job_id"]) + '_' + json.dumps(op["op_id"])] = job_id_to_job[op['job_id']].computation_graph.nodes[op['op_id']]['remaining_run_time']
                # except KeyError:
                    # # TODO TEMP DEBUG
                    # print(f'key error when getting job_op_to_cost for op {op}')
                    # print(f'worker_id: {worker_id}')
                    # print(f'ops placed on worker id: {ops}')
                    # print(f'job_id_to_job[op["job_id"]]: {job_id_to_job[op["job_id"]]}')
                    # print(f'job_id_to_job: {job_id_to_job}')
                    # print(f'job computation graph nodes: {job_id_to_job[op["job_id"]].computation_graph.nodes}')
                    # print(f'op_placement:\n{op_placement}')
                    # raise Exception()

            # sort ops in descending order of cost
            sorted_job_op_to_cost = sorted(job_op_to_cost, key=job_op_to_cost.get, reverse=True)
            # highest cost ops have lowest priority
            for priority, job_op in enumerate(list(sorted_job_op_to_cost)):
                # job_id, op_id = [int(i) for i in job_op.split('_')]
                job_id, op_id = [json.loads(i) for i in job_op.split('_')]
                worker_to_job_to_op_to_priority[worker_id][job_id][op_id] = priority

        return OpSchedule(worker_to_job_to_op_to_priority)




