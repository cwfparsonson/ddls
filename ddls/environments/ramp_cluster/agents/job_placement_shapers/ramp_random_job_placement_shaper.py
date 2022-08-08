from ddls.environments.ramp_cluster.actions.op_partition import OpPartition
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.job_placement_shape import JobPlacementShape
from ddls.environments.ramp_cluster.agents.placers.utils import dummy_ramp

import random
import random

class RampRandomJobPlacementShaper:
    def __init__(self):
        pass

    def get(self,
            op_partition: OpPartition,
            cluster: RampClusterEnvironment):
        '''
        Randomly generates a meta block shape for each job.

        Returns a mapping of job_id -> meta_block_shape, where meta_block_shape
        is a tuple of (num_communication_groups, num_racks, num_nodes_per_rack) specifying
        amount of cluster resources to use for each job.
        '''
        # get shape of ramp topology
        ramp_shape = (cluster.topology.num_communication_groups, cluster.topology.num_racks_per_communication_group, cluster.topology.num_servers_per_rack)
        ramp_topology = dummy_ramp(ramp_shape, cluster)

        # generate meta block shapes for each job requesting to be placed
        job_to_meta_block_shape = {}
        for job_id in op_partition.partitioned_jobs.keys():
            # meta block shape is (num_communication_groups, num_racks, num_nodes)
            # total number of servers to use for job is given by num_servers = num_racks * num_nodes
            # given a partitioned job where ops have been partitioned up to max_partition_degree times, need
            # to ensure that num_servers >= max_partition_degree to ensure partitioned ops can be spread across servers.
            if ramp_shape[1] * ramp_shape[2] < op_partition.job_id_to_max_partition_degree[job_id]:
                raise Exception(f'ERROR: Ramp shape is {ramp_shape} -> have num_racks x num_racks_per_server = {ramp_shape[1]} x {ramp_shape[2]} = {int(ramp_shape[1] * ramp_shape[2])} servers, but op partition for job_id {job_id} has max op partition degree {op_partition.job_id_to_max_partition_degree[job_id]}. Either increase the number of servers in your RAMP cluster, or decrease the maximum partition degree of your op partitioning agent.')
            count = 1
            while True:
                job_to_meta_block_shape[job_id] = tuple([int(math.ceil(random.randint(1, dim) / 2) * 2) for dim in ramp_shape])
                if job_to_meta_block_shape[job_id][1] * job_to_meta_block_shape[job_id][2] >= op_partition.job_id_to_max_partition_degree[job_id]:
                    break
                else:
                    if count > 10000:
                        raise Exception(f'Unable to find valid meta block shape after {count} attempts. Lower max partition degree or increase number of servers in cluster to help find valid meta blocks.')
                count += 1

        return JobPlacementShape(job_to_meta_block_shape)
