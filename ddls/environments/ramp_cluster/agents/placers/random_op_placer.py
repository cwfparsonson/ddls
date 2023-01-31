from ddls.managers.placers.placer import Placer
from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.op_placement import OpPlacement
from ddls.environments.ramp_cluster.actions.op_partition import OpPartition

import numpy as np
import copy
from collections import defaultdict
import random


class RandomOpPlacer(Placer):
    def __init__(self):
        pass

    def get(self, 
            op_partition: OpPartition,
            cluster: RampClusterEnvironment):
        '''
        Places operations in a job onto available worker(s) in a cluster, where the clusters
        nodes are servers which contain >=1 worker(s) which may or may not have sufficient 
        memory available for a given operation. 
        
        Returns a mapping of job_id -> operation_id -> worker_id. If no valid placement for the operation
        could be found, the job will not be included in the placement mapping.
        '''
        # gather jobs which are requesting to be placed
        jobs = op_partition.partitioned_jobs.values()

        # check how much memory is available on each worker
        worker_to_available_memory = self._get_workers_available_memory(cluster, sort=True)
        
        # make placements
        workers_used = set() # track workers used across jobs
        job_to_operation_to_worker = dict()
        for job in jobs:
            _workers_used = set() # track workers used for this job
            job_to_operation_to_worker[job.job_id] = dict()
            for op in job.computation_graph.nodes:
                # find which worker placements would be valid for this operation
                # valid_placements = [worker for worker in worker_to_available_memory if worker_to_available_memory[worker] >= job.computation_graph.nodes[op]['memory_cost']]
                valid_placements = self._get_valid_placements(job, op, cluster, worker_to_available_memory, workers_used)
                if len(valid_placements) == 0:
                    # no valid placement for operation, cannot place this job
                    del job_to_operation_to_worker[job.job_id]
                    break
                else:
                    worker = random.choice(valid_placements)                
                    _workers_used.add(worker)
                    worker_to_available_memory[worker] -= job.computation_graph.nodes[op]['memory_cost']
                    job_to_operation_to_worker[job.job_id][op] = worker

            # update workers used across jobs
            for worker in _workers_used:
                workers_used.add(worker)
                
        return OpPlacement(job_to_operation_to_worker, op_partition=op_partition, cluster=cluster)

    def _get_valid_placements(self, job, op, cluster, worker_to_available_memory, workers_used):
        # cannot exceed worker available memory
        valid_mem_placements = set([worker_id for worker_id in worker_to_available_memory if worker_to_available_memory[worker_id] >= job.computation_graph.nodes[op]['memory_cost']])

        # Ramp Rule 1: No worker can have ops from more than one job.
        valid_r1_placements = set()
        for worker_id in valid_mem_placements:
            node_id = cluster.topology.graph.graph['worker_to_node'][worker_id]
            worker = cluster.topology.graph.nodes[node_id]['workers'][worker_id]
            if job.details['job_idx'] not in worker.mounted_job_idx_to_ops:
                if len(list(worker.mounted_job_idx_to_ops.keys())) > 0 or worker_id in workers_used:
                    # already have another job mounted on this worker
                    pass
                else:
                    valid_r1_placements.add(worker_id)

        return list(set.intersection(valid_mem_placements, valid_r1_placements))

    def _get_workers_available_memory(self, 
                                      cluster: RampClusterEnvironment, 
                                      sort: bool = True):
        '''
        Maps worker ids to available memory. 

        Args:
            sort: If true, returned dict is in order of memory available,
                with the worker with the most memory available first, etc.
        '''
        worker_to_available_memory = dict()
        for worker_id, node_id in cluster.topology.graph.graph['worker_to_node'].items():
            node_id = cluster.topology.graph.graph['worker_to_node'][worker_id]
            worker = cluster.topology.graph.nodes[node_id]['workers'][worker_id]
            worker_to_available_memory[worker_id] = worker.memory_capacity - worker.memory_occupied
        if sort:
            worker_to_available_memory = dict(sorted(worker_to_available_memory.items(), key=lambda x:x[1], reverse=True))
        return worker_to_available_memory