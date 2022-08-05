from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.op_partition import OpPartition
from ddls.utils import get_forward_graph

from collections import defaultdict
import random


class RandomOpPartitioner:
    def __init__(self, 
                # max_partitions_per_op: int = 2,
                **kwargs):
        # if max_partitions_per_op < 1:
            # raise Exception(f'max_partitions_per_op must be >= 1 but is {max_partitions_per_op}')
        # self.max_partitions_per_op = max_partitions_per_op
        pass

    def get(self, 
            cluster: RampClusterEnvironment,
            max_partitions_per_op: int = 2,
            **kwargs
            ):
        if max_partitions_per_op < 1:
            raise Exception(f'max_partitions_per_op must be >= 1 but is {max_partitions_per_op}')
        if max_partitions_per_op > 1 and max_partitions_per_op % 2 != 0:
            raise Exception(f'max_partitions_per_op must be an even number but is {max_partitions_per_op}')

        # gather jobs which are requesting to be placed
        jobs = cluster.job_queue.jobs.values()

        # make partition decisions
        job_id_to_op_id_to_num_partitions = defaultdict(lambda: defaultdict(lambda: 1))
        job_id_to_mp_split_forward_op_ids, job_id_to_mp_splits = defaultdict(list), defaultdict(list)
        for job in jobs:
            job_id = job.job_id

            # collapse mirrored graph into only forward pass nodes
            forward_graph = get_forward_graph(job.computation_graph)

            for forward_op_id in forward_graph.nodes:
                # choose a number of times to partition this op
                num_partitions = random.randint(1, max_partitions_per_op)

                # partition this forward op
                job_id_to_op_id_to_num_partitions[job_id][forward_op_id] = num_partitions

                # apply same partitioning to the backward op
                backward_op_id = job.computation_graph.nodes[forward_op_id]['backward_node_id']
                job_id_to_op_id_to_num_partitions[job_id][backward_op_id] = num_partitions

                # update trackers
                if num_partitions > 1:
                    job_id_to_mp_split_forward_op_ids[job_id].append(forward_op_id)
                    job_id_to_mp_splits[job_id].append(num_partitions)

        return OpPartition(job_id_to_op_id_to_num_partitions, cluster=cluster)
