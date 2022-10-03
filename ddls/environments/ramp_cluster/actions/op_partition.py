from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.agents.partitioners.utils import data_split_node, model_split_node

from collections import defaultdict
import copy

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
        self.job_id_to_forward_op_id_to_mp_splits = defaultdict(lambda: defaultdict(lambda: None))
        self.job_id_to_max_partition_degree = defaultdict(lambda: 1)
        for job_id in action:
            for op_id in action[job_id]:
                num_partitions = action[job_id][op_id]
                if num_partitions != 1 and num_partitions % 2 != 0:
                    raise Exception(f'Invalid num_partitions={num_partitions} for job_id {job_id} op_id {op_id}; RAMP placer expects partitions (number of splits per op) to be even numbers.')
                # update trackers
                if num_partitions > 1:
                    self.job_id_to_mp_split_forward_op_ids[job_id].append(op_id)
                    self.job_id_to_mp_splits[job_id].append(num_partitions)
                    self.job_id_to_forward_op_id_to_mp_splits[job_id][op_id] = num_partitions
                    if num_partitions > self.job_id_to_max_partition_degree[job_id]:
                        self.job_id_to_max_partition_degree[job_id] = num_partitions
        # print(f'OpPartition job_id_to_max_partition_degree: {self.job_id_to_max_partition_degree}') # TODO TEMP DEBUG

        # create dict mapping job_id -> partitioned_job object and job_id -> original_job object
        self.job_ids, self.partitioned_jobs, self.original_jobs = set(), {}, {}
        self.job_id_to_partitioned_computation_graph = {}
        for job_id in action: # NEW
            job = cluster.job_queue.jobs[job_id]
            self.job_ids.add(job_id)
            self.original_jobs[job_id] = job

            # get init job details from cluster to avoid having to repeat computations if already computed for same model and max num partitions in the past
            model, max_num_partitions = job.details['model'], self.job_id_to_max_partition_degree[job_id]
            partitioned_computation_graph = cluster.job_model_to_max_num_partitions_to_init_details[model][max_num_partitions]['partitioned_computation_graph']
            job_total_operation_memory_cost = cluster.job_model_to_max_num_partitions_to_init_details[model][max_num_partitions]['job_total_operation_memory_cost']
            job_total_dependency_size = cluster.job_model_to_max_num_partitions_to_init_details[model][max_num_partitions]['job_total_dependency_size']
            init_job_immutable_details = cluster.job_model_to_max_num_partitions_to_init_details[model][max_num_partitions]['init_job_immutable_details']

            if partitioned_computation_graph is None:
                # not yet partitioned computation graph for this model and max partition degree, need to get partitioned computation graph
                computation_graph = job.computation_graph

                # apply data parallelism partitioning
                partitioned_computation_graph = data_split_node(computation_graph, dp_splits=0)

                # apply model parallelism partitioning
                partitioned_computation_graph = model_split_node(partitioned_computation_graph, mp_split_ids=self.job_id_to_mp_split_forward_op_ids[job_id], mp_splits=self.job_id_to_mp_splits[job_id])

                self.job_id_to_partitioned_computation_graph[job_id] = partitioned_computation_graph
            else:
                self.job_id_to_partitioned_computation_graph[job_id] = partitioned_computation_graph

            # record partitioned job
            details = copy.deepcopy(job.details)
            details['max_partitions_per_op'] = self.job_id_to_max_partition_degree[job_id]
            self.partitioned_jobs[job_id] = Job(computation_graph=partitioned_computation_graph,
                                                num_training_steps=job.num_training_steps,
                                                max_acceptable_job_completion_time_frac=job.max_acceptable_job_completion_time_frac,
                                                job_id=copy.deepcopy(job_id),
                                                original_job=job,
                                                details=details,
                                                job_total_operation_memory_cost=job_total_operation_memory_cost,
                                                job_total_dependency_size=job_total_dependency_size,
                                                init_job_immutable_details=init_job_immutable_details,
                                                )

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


