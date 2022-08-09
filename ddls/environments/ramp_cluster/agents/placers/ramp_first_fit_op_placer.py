from ddls.managers.placers.placer import Placer
from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.op_placement import OpPlacement
from ddls.environments.ramp_cluster.actions.op_partition import OpPartition
from ddls.environments.ramp_cluster.actions.job_placement_shape import JobPlacementShape
from ddls.utils import get_forward_graph
from ddls.environments.ramp_cluster.agents.placers.utils import get_allocation_preamble, get_parents_and_children, topo_sort, find_meta_block, ff_meta_block, get_meta_block, check_block, dummy_ramp, parent_collective_placement, regular_collective_placement, find_sub_block, ff_block, get_factor_pairs, get_block, get_block_shapes, allocate

import numpy as np
import copy
from collections import defaultdict, deque
import random
import math







# class RampRandomOpPlacer(Placer):
class RampFirstFitOpPlacer(Placer):
    def __init__(self):
        pass

    def get(self, 
            op_partition: OpPartition,
            job_placement_shape: JobPlacementShape,
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

        # get shape of ramp topology
        ramp_shape = (cluster.topology.num_communication_groups, cluster.topology.num_racks_per_communication_group, cluster.topology.num_servers_per_rack)
        ramp_topology = dummy_ramp(ramp_shape, cluster)

        # place job ops
        job_to_operation_to_worker = defaultdict(lambda: defaultdict(lambda: None))
        # for partitioned_job in jobs:
        for key in job_placement_shape.action.keys():
            partitioned_job = op_partition.partitioned_jobs[key]
            job_id = partitioned_job.job_id

            # get original job
            original_job = cluster.job_queue.jobs[job_id]

            # collapse mirrored graph into only forward pass nodes
            forward_graph = get_forward_graph(original_job.computation_graph)
            
            # get partitioning decisions made
            mp_split_ids = op_partition.job_id_to_mp_split_forward_op_ids[job_id]
            mp_splits = op_partition.job_id_to_mp_splits[job_id]

            # specify shape of meta-block to be used for this job
            # meta_shape = tuple([random.randint(1, dim) for dim in ramp_shape])
            meta_shape = job_placement_shape.action[job_id]
            
            # get useful info
            sequence, splits, op_server_info, parents, children = get_allocation_preamble(forward_graph, mp_split_ids, mp_splits)

            # get a meta-block of a particular shape which the heuristic allocator will try to pack the job fully into
            meta_block_info = find_meta_block(ramp_topology, ramp_shape, meta_shape)

            # # DEBUG
            # print(f'\nramp_topology: {ramp_topology}')
            # print(f'ramp_shape: {ramp_shape}')
            # print(f'forward_graph: {forward_graph}')
            # print(f'sequence: {sequence}')
            # print(f'splits: {splits}')
            # print(f'meta_block_info: {meta_block_info}')
            # print(f'parents: {parents}')
            # print(f'op_server_info: {op_server_info}')

            if meta_block_info:
                # valid meta block successfully found, try to allocate the job
                allocated = allocate(ramp_topology,ramp_shape,forward_graph,sequence,splits,meta_block_info,parents,op_server_info)
                # print(f'allocated: {allocated}') # DEBUG
                if allocated:
                    # update the topology and op-server info for use in the next job allocation
                    ramp_topology, op_server_info = allocated

                # update job placement dict
                for n in ramp_topology.keys():
                    c, r, s = n
                    node_id = f'{c}-{r}-{s}'
                    # HACK: assume 1 worker per server
                    worker_id = list(cluster.topology.graph.nodes[node_id]['workers'].keys())[0]
                    for op_id in ramp_topology[n]['ops']:
                        # ensure op_id is string for consistency
                        job_to_operation_to_worker[job_id][str(op_id)] = worker_id

            else:
                # unable to find valid meta block
                pass

        return OpPlacement(job_to_operation_to_worker, op_partition=op_partition, cluster=cluster)
