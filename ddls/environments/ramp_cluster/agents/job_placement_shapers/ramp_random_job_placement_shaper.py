from ddls.environments.ramp_cluster.actions.op_partition import OpPartition
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.job_placement_shape import JobPlacementShape
# from ddls.environments.ramp_cluster.agents.placers.utils import dummy_ramp, find_meta_block
from ddls.environments.ramp_cluster.agents.placers.utils import get_partitioned_job_valid_meta_block_shapes

import random
import numpy as np

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
        # generate meta block shapes for each job requesting to be placed
        job_to_meta_block_shape = {}
        for job_id in op_partition.partitioned_jobs.keys():
            # find which meta block shape(s) valid for this job
            meta_block_shapes, mask = get_partitioned_job_valid_meta_block_shapes(cluster, op_partition.job_id_to_max_partition_degree[job_id])

            # choose a meta block shape for this job
            valid_meta_block_shapes = meta_block_shapes[mask]
            if len(valid_meta_block_shapes) > 0:
                job_to_meta_block_shape[job_id] = random.choice(list(valid_meta_block_shapes))
            else:
                # no valid meta block shapes available for this job
                pass

        return JobPlacementShape(job_to_meta_block_shape)
