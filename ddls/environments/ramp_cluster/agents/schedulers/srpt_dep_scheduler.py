from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.dep_schedule import DepSchedule
from ddls.environments.ramp_cluster.actions.dep_placement import DepPlacement
from ddls.environments.ramp_cluster.actions.op_partition import OpPartition

import numpy as np
from collections import defaultdict
import copy
import json

class SRPTDepScheduler:

    def get(self, 
            op_partition: OpPartition,
            dep_placement: DepPlacement, # flow dep placements
            cluster: RampClusterEnvironment):
        new_placements = dep_placement.action

        # initialise flow dep schedule for each channel mapping channel_id -> job_id -> dep_id -> priority
        channel_to_job_to_dep_to_priority = defaultdict(lambda: defaultdict(dict))

        if len(new_placements) == 0:
            # no new placements made, flow dep schedule is unchanged
            return DepSchedule(channel_to_job_to_dep_to_priority)
        else:
            # new dep(s) will be mounted on cluster, need to get new job dep schedule
            pass

        # combine current cluster placement status with new placement decisions so can schedule deps 
        placement = copy.deepcopy(cluster.job_dep_placement)
        for job_id in new_placements.keys():
            placement[job_id] = new_placements[job_id]

        # gather the placed jobs for which flow schedule is needed
        # jobs = [job for job in cluster.job_queue.jobs.values() if job_id in new_placements]
        jobs = [job for job_id, job in op_partition.partitioned_jobs.items() if job_id in new_placements]

        # initialise useful mappings
        job_id_to_job = {job.job_id: job for job in jobs}

        channel_to_deps = defaultdict(list)
        for job_id, job in job_id_to_job.items():
            for dep_id in job.computation_graph.edges:
                channel_id = dep_placement.job_to_dep_to_channel[job_id][dep_id]
                if len(channel_id) > 0:
                    # dep is a flow -> has a placement
                    channel_to_deps[channel_id].append(f'{json.dumps(job_id)}_{json.dumps(dep_id)}')
                else:
                    # dep not a flow -> has no placement
                    pass

        # if remaining run time not initialised for dep, initialise so can calc dep costs for scheduling
        for job in job_id_to_job.values():
            edges = [edge for edge in job.computation_graph.edges]
            u, v, k = edges[0] 
            if job.computation_graph[u][v][k]['remaining_run_time'] is None:
                # initialise so can use SRPT
                for dep_id in job.computation_graph.edges:
                    job.reset_dep_remaining_run_time(dep_id)

        # assign a cost to each flow dependency in the network
        jobdep_to_cost = {}
        for jobdep in dep_placement.jobdeps:
            job_id, dep_id = jobdep.split('_')
            job_id = json.loads(job_id)
            dep_id = tuple(json.loads(dep_id))
            
            job = job_id_to_job[job_id]
            u, v, k = dep_id

            jobdep_to_cost[jobdep] = job.computation_graph[u][v][k]['remaining_run_time']

        # sort deps in descending order of cost
        sorted_jobdep_to_cost = sorted(jobdep_to_cost, key=jobdep_to_cost.get, reverse=True)

        # highest cost deps have lowest priority
        for priority, jobdep in enumerate(list(sorted_jobdep_to_cost)):
            job_id, dep_id = jobdep.split('_')
            job_id = json.loads(job_id)
            dep_id = tuple(json.loads(dep_id))
            for channel_id in dep_placement.jobdep_to_channels[jobdep]:
                channel_to_job_to_dep_to_priority[channel_id][job_id][dep_id] = priority

        return DepSchedule(channel_to_job_to_dep_to_priority)








