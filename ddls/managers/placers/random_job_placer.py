from ddls.managers.placers.placer import Placer
from ddls.demands.jobs.job import Job
from ddls.clusters.cluster import Cluster
#from ddls.demands.workloads.data_parallel_workload import DataParallelWorkload
#from ddls.demands.workloads.workloads_manager import DataParallelWorkloadsManager

import numpy as np
import copy
from collections import defaultdict
import random










class RandomJobPlacer(Placer):
    def __init__(self):
        pass

    def place(self, 
              job: Job, 
              cluster: Cluster):
        '''
        Places operations in a job onto available worker(s) in a cluster, where the clusters
        nodes are servers which contain >=1 worker(s) which may or may not have sufficient 
        memory available for a given operation. 
        
        Returns a mapping of operation_id -> worker_id. If no valid placement for the operation
        could be found, worker_id will be None.
        '''
        # check how much memory is available on each worker
        worker_to_available_memory = self._get_workers_available_memory(cluster, sort=True)
        
        # make placements
        operation_to_worker = dict()
        for op in job.computation_graph.nodes:
            # find which worker placements would be valid for this operation
            valid_placements = [worker for worker in worker_to_available_memory if worker_to_available_memory[worker] >= cluster.topology.graph.nodes[op]['memory_cost']]
            if len(valid_placements) == 0:
                operation_to_worker = None
            else:
                worker = random.choice(valid_placements)                
                worker_to_available_memory[worker] -= cluster.topology.graph.nodes[op]['memory_cost']
                operation_to_worker[op] = worker
                
        return operation_to_worker
                
    def _get_workers_available_memory(self, 
                                      cluster: Cluster, 
                                      sort: bool = True):
        '''
        Maps worker ids to available memory. Returned dict is in order of memory available,
        with the worker with the most memory available first, etc.
        '''
        worker_to_available_memory = dict()
        for worker_id, node_id in cluster.topology.graph.graph['worker_to_node'].items():
            worker = cluster.topology.graph.graph['worker_to_node'][worker_id]
            worker_to_available_memory[worker_id] = worker.memory_capacity - worker.memory_occupied
        if sort:
            worker_to_available_memory = dict(sorted(worker_to_available_memory.items(), key=lambda x:x[1], reverse=True))
        return worker_to_available_memory
        
            
        
                

        












# class OldRandomJobPlacer(Placer):
#     def __init__(self,
#                  parallelisation: str = 'data_parallelisation'):
#         self.parallelisation = parallelisation
#         
#     def place(self, job, cluster):
#         '''
#         Places operations of a job onto workers in a cluster, where the cluster is made up
#         of nodes/servers which each contain a worker.
#         '''
#         # consider all cluster nodes as potential nodes/servers
#         num_workers = len(cluster.topology.topology.nodes)
#         
#         # create workloads from job
#         local_batch_size = int(job.batch_size / num_workers)
#         if self.parallelisation == 'data_parallelisation':
#             workloads = [DataParallelWorkload(workload_id=i, job=job, local_batch_size=local_batch_size) for i in range(num_workers)]
#         else:
#             raise Exception(f'Unrecognised parallelisation {self.parallelisation}')
#             
#         # map workloads to cluster nodes
#         nodes = np.array(copy.deepcopy(cluster.topology.topology.nodes))
#         node_to_workloads = defaultdict(lambda: [])
#         for workload in workloads:
#             np.random.shuffle(nodes)
#             for counter, node in enumerate(nodes):
#                 device = cluster.topology.topology.nodes[node]['device']
#                 if device.memory_occupied + workload.get_workload_size() <= device.memory_capacity:
#                     node_to_workloads[node].append(workload)
#                 else:
#                     if counter == len(nodes) - 1:
#                         # cannot place workload on any node in cluster
#                         return None
# 
#         workloads_manager = DataParallelWorkloadsManager(job, node_to_workloads)
#                     
#         return workloads_manager
