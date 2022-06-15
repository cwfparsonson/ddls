from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.agents.partitioners.utils import data_split_node, model_split_node

from collections import defaultdict

class OpPartition:
    def __init__(self,
                 action: dict,
                 cluster: RampClusterEnvironment):
        '''
        Args:
            action: Mapping of job_id -> operation_id -> num_partitions (e.g. If op 1
            of job 1 was going to be partitioned into 2 ops, would have action[1][1] = 2)
        '''
        self.action = action

        # collect useful stats
        self.job_id_to_mp_split_forward_op_ids, self.job_id_to_mp_splits = defaultdict(list), defaultdict(list)
        for job_id in action:
            for op_id in action[job_id]:
                num_partitions = action[job_id][op_id]
                # update trackers
                if num_partitions > 1:
                    self.job_id_to_mp_split_forward_op_ids[job_id].append(op_id)
                    self.job_id_to_mp_splits[job_id].append(num_partitions)

        # create dict mapping job_id -> partitioned_job object
        self.job_ids, self.partitioned_jobs = set(), {}
        for job_id, job in cluster.job_queue.jobs.items():
            self.job_ids.add(job_id)
            computation_graph = job.computation_graph

            # apply data parallelism partitioning
            partitioned_computation_graph = data_split_node(computation_graph, dp_splits=0)

            # apply model parallelism partitioning
            partitioned_computation_graph = model_split_node(partitioned_computation_graph, mp_split_ids=self.job_id_to_mp_split_forward_op_ids[job_id], mp_splits=self.job_id_to_mp_splits[job_id])

            # record partitioned job
            self.partitioned_jobs[job_id] = Job(computation_graph=partitioned_computation_graph,
                                                num_training_steps=job.num_training_steps,
                                                job_id=job_id,
                                                job_type=job.job_type,
                                                details=job.details)

    def __len__(self):
        return len(list(self.action.keys()))

    def __str__(self):
        descr = ''
        for job_id in self.action.keys():
            descr += f'\nJob ID: {job_id}'
            for op_id in self.action[job_id].keys():
                num_partitions = self.action[job_id][op_id]
                if num_partitions > 1:
                    descr += f'\n\tOp ID {op_id} -> Partition into {num_partitions} sub-ops'
                else:
                    descr += f'\n\tOp ID {op_id} -> No partitioning applied'
        return descr


