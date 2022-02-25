from ddls.managers.placers.placer import Placer
from ddls.demands.workloads.data_parallel_workload import DataParallelWorkload

import numpy as np
import copy
from collections import defaultdict


class RandomPlacer(Placer):
    def __init__(self,
                 parallelisation: str = 'data_parallelisation'):
        self.parallelisation = parallelisation
        
    def place_job(self, job, cluster):
        '''
        Divides job into parallelisation workloads and attempts to map workloads to cluster nodes.
        If cannot fit all workloads onto cluster, returns None.
        '''
        # consider all cluster nodes as potential workers
        num_workers = len(cluster.topology.topology.nodes)
        
        # create workloads from job
        local_batch_size = int(job.batch_size / num_workers)
        if self.parallelisation == 'data_parallelisation':
            workloads = [DataParallelWorkload(workload_id=i, job=job, local_batch_size=local_batch_size) for i in range(num_workers)]
            
        # map workloads to cluster nodes
        nodes = np.array(copy.deepcopy(cluster.topology.topology.nodes))
        node_to_workloads = defaultdict(lambda: [])
        for workload in workloads:
            np.random.shuffle(nodes)
            for counter, node in enumerate(nodes):
                device = cluster.topology.topology.nodes[node]['device']
                if device.memory_occupied + workload.get_workload_size() <= device.memory_capacity:
                    node_to_workloads[node].append(workload)
                else:
                    if counter == len(nodes) - 1:
                        # cannot place workload on any node in cluster
                        return None
                    
        return node_to_workloads
