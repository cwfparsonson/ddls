from ddls.environments.ddls_observation import DDLSObservation
from ddls.environments.cluster.cluster_environment import ClusterEnvironment
from ddls.demands.jobs.job import Job
from ddls.utils import flatten_list

import networkx as nx
import numpy as np

class JobPlacingAllNodesObservation(DDLSObservation):
    def reset(self):
        pass

    def extract(self, 
                cluster: ClusterEnvironment,
                done: bool):
        '''
        Return features of each node in computation graph whose nodes (ops) need
        to be scheduled.

        Each node should be encoded with the following features:

            Op features:
                compute_cost (float): Compute cost of op normalised by compute cost of
                    node with highest compute cost. Do for each worker device type.
                is_highest_compute_cost (binary): If the op has the highest compute cost
                    of all nodes. Do for each worker.
                memory_cost (float): Memory cost of op normalised by memory cost of
                    node with highest memory cost.
                is_highest_memory_cost (binary): If the op has the highest memory cost
                    of all nodes.
                parents_compute_cost_sum (float): Sum of parent(s) compute cost normalised
                    by compute cost of node with highest compute cost. Do for each 
                    worker device type.
                parents_memory_cost_sum (float): Sum of parent(s) memory cost normalised
                    by memory cost of node with highest memory cost.
                children_compute_cost_sum (float): Sum of child(s) compute cost normalised
                    by compute cost of node with highest compute cost. Do for each
                    worker device type.
                children_memory_cost_sum (float): Sum of child(s) memory cost normalised
                    by memory cost of node with highest memory cost.
                node_depth (float): Distance of node from source normalised by the
                    distance of node furthest from source.

            Job features:
                num_training_steps_remaining (float): Number of training steps remaining
                    for job normalised by the initial number of training steps.

            Network worker features:
                num_ready_ops: For each worker, number of ready ops normalised
                    by number of mounted ops.
                num_mounted_ops: For each worker, number of mounted ops normalised
                    by total number of mounted ops in cluster.

            Network global features:
                num_active_workers (float): Number of active workers normalised
                    by the total number of workers in the cluster.
        '''
        # get job to place
        # TEMPORARY: Just assume placing 1st job in queue
        # TODO: Implement where get given job and do per-job encoding?
        job = list(cluster.job_queue.jobs.values())[0] # assume event-driven where only ever have one job to queue

        encoded_obs = []

        # only need to compute job and network features once
        job_features = self._get_job_features(job)
        network_worker_features = self._get_network_worker_features(job, cluster)
        network_global_features = self._get_network_global_features(job, cluster)

        # compute op features for each node and create node features
        for node in job.computation_graph.nodes:
            node_features = flatten_list([self._get_op_features(node, job, cluster), job_features, network_worker_features, network_global_features])
            encoded_obs.append(np.array(node_features, dtype=object))

        return np.array(encoded_obs, dtype=object)

    def _get_job_features(self, job):
        job_features = []
        num_training_steps_remaining = (job.num_training_steps - job.training_step_counter) / job.num_training_steps
        job_features.append(np.array(num_training_steps_remaining, dtype=object))
        return job_features

    def _get_network_worker_features(self, job, cluster):
        network_worker_features = []
        num_ready_ops, num_mounted_ops = [], []
        for worker_id, server_id in cluster.topology.graph.graph['worker_to_node'].items():
            worker = cluster.topology.graph.nodes[server_id]['workers'][worker_id]
            ready_op_counter = 0
            mounted_op_counter = 0
            for job_idx, op_ids in worker.mounted_job_idx_to_ops.items():
                job = cluster.running_jobs[job_idx]
                for op_id in op_ids:
                    mounted_op_counter += 1
                    if op_id in job.computation_graph.graph['ops_ready']:
                        ready_op_counter += 1
            try:
                num_ready_ops.append(ready_op_counter / mounted_op_counter)
            except ZeroDivisionError:
                num_ready_ops.append(0)
            try:
                num_mounted_ops.append(mounted_op_counter / cluster.num_mounted_ops)
            except ZeroDivisionError:
                num_mounted_ops.append(0)
        network_worker_features.append(np.array(num_ready_ops, dtype=object))
        network_worker_features.append(np.array(num_mounted_ops, dtype=object))
        return network_worker_features

    def _get_network_global_features(self, job, cluster):
        network_global_features = []
        num_active_workers = cluster.num_active_workers / len(list(cluster.topology.graph.graph['worker_to_node'].keys()))
        network_global_features.append(np.array(num_active_workers, dtype=object))
        return network_global_features

    def _get_op_features(self, op, job, cluster):
        op_features = []

        # op compute cost
        compute_cost = []
        for device_type in cluster.topology.graph.graph['worker_types']:
            op_compute_cost = job.computation_graph.nodes[op]['compute_cost'][device_type]
            try:
                compute_cost.append(op_compute_cost / job.details['max_compute_cost'][device_type])
            except ZeroDivisionError:
                compute_cost.append(0)
        is_highest_compute_cost = op == job.details['max_compute_node']
        op_features.append(np.array(compute_cost, dtype=object))
        op_features.append(np.array(is_highest_compute_cost, dtype=object))

        # op memory cost
        try:
            memory_cost = job.computation_graph.nodes[op]['memory_cost'] / job.details['max_memory_cost']
        except ZeroDivisionError:
            memory_cost = 0
        is_highest_memory_cost = op == job.details['max_memory_node']
        op_features.append(np.array(memory_cost, dtype=object))
        op_features.append(np.array(is_highest_memory_cost, dtype=object))

        # parents compute and memory cost sum
        parents_memory_cost_sum = 0
        if len(list(job.computation_graph.predecessors(op))) == 0:
            # op has no parents
            parents_compute_cost_sum = [0]
        else:
            parents_compute_cost_sum = []
            for parent in job.computation_graph.predecessors(op):
                # parent compute cost
                parent_compute_cost_sum = 0
                for device_type in cluster.topology.graph.graph['worker_types']:
                    parent_compute_cost = job.computation_graph.nodes[parent]['compute_cost'][device_type]
                    try:
                        parent_compute_cost_sum += (parent_compute_cost / job.details['max_compute_cost'][device_type])
                    except ZeroDivisionError:
                        pass
                parents_compute_cost_sum.append(parent_compute_cost_sum)
                # parent memory cost
                try:
                    parents_memory_cost_sum += (job.computation_graph.nodes[parent]['memory_cost'] / job.details['max_memory_cost'])
                except ZeroDivisionError:
                    pass
        op_features.append(np.array(parents_compute_cost_sum, dtype=object))
        op_features.append(np.array(parents_memory_cost_sum, dtype=object))

        # children compute and memory cost sum
        children_memory_cost_sum = 0
        if len(list(job.computation_graph.successors(op))) == 0:
            # op has no children
            children_compute_cost_sum = [0]
        else:
            children_compute_cost_sum = []
            for child in job.computation_graph.successors(op):
                # child compute cost
                child_compute_cost_sum = 0
                for device_type in cluster.topology.graph.graph['worker_types']:
                    child_compute_cost = job.computation_graph.nodes[child]['compute_cost'][device_type]
                    try:
                        child_compute_cost_sum += (child_compute_cost / job.details['max_compute_cost'][device_type])
                    except ZeroDivisionError:
                        pass
                children_compute_cost_sum.append(child_compute_cost_sum)
                # child memory cost
                try:
                    children_memory_cost_sum += (job.computation_graph.nodes[child]['memory_cost'] / job.details['max_memory_cost'])
                except ZeroDivisionError:
                    pass
        op_features.append(np.array(children_compute_cost_sum, dtype=object))
        op_features.append(np.array(children_memory_cost_sum, dtype=object))

        # op depth
        node_depth = len(nx.shortest_path(job.computation_graph, source=0, target=op)) / job.details['max_depth']
        op_features.append(np.array(node_depth, dtype=object))

        return op_features





                















