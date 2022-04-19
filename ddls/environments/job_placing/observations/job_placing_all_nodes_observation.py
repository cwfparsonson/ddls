from ddls.environments.ddls_observation_function import DDLSObservationFunction
from ddls.environments.ddls_observation import DDLSObservation
from ddls.environments.cluster.cluster_environment import ClusterEnvironment
from ddls.demands.jobs.job import Job
from ddls.utils import flatten_list, flatten_numpy_array

import gym
import torch
import networkx as nx
import numpy as np
import copy

class JobPlacingAllNodesObservation(DDLSObservationFunction):
    def __init__(self,
                 pad_obs_kwargs: dict = None):
        '''
        Args:
            pad_obs_kwargs: If not None will look at jobs_config, get max number of nodes and
                edges across all jobs, and pad each obs to ensure dimensionality of
                each obs is consistent even for observations with varying graph sizes.
                pad_obs_kwargs must be dict of {'max_nodes': <int>, 'max_edges': <int>}
                UPDATE: Only needs to be {'max_nodes': <int>}, will then calc max edges
                by assuming fully connected graph of max_nodes.
        '''
        self.pad_obs_kwargs = pad_obs_kwargs

        # init obs space
        self._observation_space = None

        # init any hard-coded feature min and max values
        self.node_features_low, self.node_features_high = 0, 1
        self.edge_features_low, self.edge_features_high = 0, 1
        self.graph_features_low, self.graph_features_high = 0, 1

    def reset(self, 
              cluster: ClusterEnvironment,
              flatten: bool = True):
        # get job which is going to be encoded in obs
        job = self._get_job_to_encode(cluster)
    
        # encode the initial obs
        obs = self._encode_obs(job, cluster, flatten=flatten)

        if self.pad_obs_kwargs is not None:
            max_nodes =  self.pad_obs_kwargs['max_nodes']
            max_edges = int(max_nodes*(max_nodes-1)/2) #number of edges in a fully connected graph
        else:
            max_nodes, max_edges = 0, 0

        # use the encoded obs to initialise the observation space
        self.observation_space = gym.spaces.Dict({
                'node_features': gym.spaces.Box(low=self.node_features_low, high=self.node_features_high, shape=obs['node_features'].shape),
                'edge_features': gym.spaces.Box(low=self.edge_features_low, high=self.edge_features_high, shape=obs['edge_features'].shape),
                'graph_features': gym.spaces.Box(low=self.graph_features_low, high=self.graph_features_high, shape=obs['graph_features'].shape),
                'edges_src': gym.spaces.Box(low=0, high=max(obs['edges_src'])+1, shape=obs['edges_src'].shape),
                'edges_dst': gym.spaces.Box(low=0, high=max(obs['edges_dst'])+1, shape=obs['edges_dst'].shape),
                'node_split': gym.spaces.Box(low=0, high=max_nodes, shape=(1,)),
                'edge_split': gym.spaces.Box(low=0, high=max_edges, shape=(1,))
            })


    def extract(self, 
                cluster: ClusterEnvironment,
                done: bool,
                flatten: bool = True):
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
                # parents_compute_cost_sum (float): Sum of parent(s) compute cost normalised
                    # by compute cost of node with highest compute cost. Do for each 
                    # worker device type.
                # parents_memory_cost_sum (float): Sum of parent(s) memory cost normalised
                    # by memory cost of node with highest memory cost.
                # children_compute_cost_sum (float): Sum of child(s) compute cost normalised
                    # by compute cost of node with highest compute cost. Do for each
                    # worker device type.
                # children_memory_cost_sum (float): Sum of child(s) memory cost normalised
                    # by memory cost of node with highest memory cost.
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

        
        Args:
            flatten: If True, will flatten feature dims. I.e. if have 100 nodes where 
                each node has 10 features but dimensions of each feature are different,
                then will flatten per-node features to get a (100, <total_feat_dims>)
                node feature observation. If False, will leave the node observation
                as (100, <num_node_features>, <ambiguous_node_feature_dims>), where
                ambiguous_node_feature_dims changes for each node.

        '''
        return self._encode_obs(self._get_job_to_encode(cluster), cluster, flatten=flatten)



    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space: gym.spaces.Dict):
        self._observation_space = observation_space

    def _get_job_to_encode(self, cluster):
        # TEMPORARY: Just assume placing 1st job in queue
        # TODO: Implement where get given job and do per-job encoding?
        return list(cluster.job_queue.jobs.values())[0] # assume event-driven where only ever have one job to queue

    def _pad_obs(self, obs):
        padded_obs = copy.deepcopy(obs)

        edges_src = torch.Tensor(obs['edges_src'])
        edges_dst = torch.Tensor(obs['edges_dst'])
        node_features = torch.Tensor(obs['node_features'])
        edge_features = torch.Tensor(obs['edge_features'])

        max_nodes = self.pad_obs_kwargs['max_nodes']
        max_edges = int(max_nodes*(max_nodes-1)/2) #number of edges in a fully connected graph

        src_padding = torch.zeros((max_edges-len(edges_src),))
        dst_padding = torch.zeros((max_edges-len(edges_dst),))

        edges_src = torch.cat((edges_src,src_padding),dim=0)
        edges_dst = torch.cat((edges_dst,dst_padding),dim=0)

        edge_feature_padding = torch.zeros(
            max_edges-edge_features.shape[0],
            edge_features.shape[1]
        )
        edge_features = torch.cat((edge_features,edge_feature_padding),dim=0)

        node_feature_padding = torch.zeros(
            max_nodes-node_features.shape[0],
            node_features.shape[1]
        )

        node_features = torch.cat((node_features,node_feature_padding),dim=0)

        padded_obs['node_features'] = node_features.numpy().astype(np.float32)
        padded_obs['edge_features'] = edge_features.numpy().astype(np.float32)
        padded_obs['edges_src'] = edges_src.numpy().astype(np.float32)
        padded_obs['edges_dst'] = edges_dst.numpy().astype(np.float32)
        padded_obs['node_split'] = np.array([len(obs['node_features'])], dtype=np.float32)
        padded_obs['edge_split'] = np.array([len(obs['edge_features'])], dtype=np.float32)

        return padded_obs

    def _encode_obs(self, 
                    job: Job, 
                    cluster: ClusterEnvironment, 
                    flatten: bool = True):
        edges_src, edges_dst = self._extract_edges_src_dst(job)
        obs =   {
                    'node_features': np.array(self._extract_node_features(job, cluster), dtype=np.float32),
                    'edge_features': np.array(self._extract_edge_features(job, cluster), dtype=np.float32),
                    'graph_features': np.array(self._extract_graph_features(job, cluster), dtype=np.float32),
                    'edges_src': np.array(edges_src, dtype=np.float32),
                    'edges_dst': np.array(edges_dst, dtype=np.float32),
                    'node_split': None,
                    'edge_split': None
                 }
        if self.pad_obs_kwargs is not None:
            obs = self._pad_obs(obs)
        return obs

    def _extract_node_features(self, job, cluster):
        return [self._get_op_features(node, job, cluster) for node in job.computation_graph.nodes]

    def _extract_edge_features(self, job, cluster):
        # TODO: Encode edge features with useful info
        return [np.array([1]) for _ in job.computation_graph.edges]

    def _extract_graph_features(self, job, cluster):
        return flatten_list([self._get_job_features(job), 
                             self._get_network_worker_features(job, cluster), 
                             self._get_network_graph_features(job, cluster)
                            ])

    def _extract_edges_src_dst(self, job):
        srcs, dsts = [], []
        for edge in job.computation_graph.edges:
            srcs.append(edge[0])
            dsts.append(edge[1])
        return srcs, dsts

    def _get_job_features(self, job, flatten=True):
        job_features = []
        num_training_steps_remaining = (job.num_training_steps - job.training_step_counter) / job.num_training_steps
        job_features.append(np.array(num_training_steps_remaining, dtype=object))
        if flatten:
            job_features = flatten_numpy_array(job_features)
        return job_features

    def _get_network_worker_features(self, job, cluster, flatten=True):
        network_worker_features = []
        num_ready_ops, num_mounted_ops = [], []
        for worker_id, server_id in cluster.topology.graph.graph['worker_to_node'].items():
            worker = cluster.topology.graph.nodes[server_id]['workers'][worker_id]
            ready_op_counter = 0
            mounted_op_counter = 0
            for job_idx, op_ids in worker.mounted_job_idx_to_ops.items():
                job = cluster.jobs_running[job_idx]
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
        if flatten:
            network_worker_features = flatten_numpy_array(network_worker_features)
        return network_worker_features

    def _get_network_graph_features(self, job, cluster, flatten=True):
        network_graph_features = []
        num_active_workers = cluster.num_active_workers / len(list(cluster.topology.graph.graph['worker_to_node'].keys()))
        network_graph_features.append(np.array(num_active_workers, dtype=object))
        if flatten:
            network_graph_features = flatten_numpy_array(network_graph_features)
        return network_graph_features

    def _get_op_features(self, op, job, cluster, flatten=True):
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

        # TODO: Fix below parent and child feats (currently get changing feature dimensions for each node so have bug)
        # # parents compute and memory cost sum
        # parents_memory_cost_sum = 0
        # if len(list(job.computation_graph.predecessors(op))) == 0:
            # # op has no parents
            # parents_compute_cost_sum = [0]
        # else:
            # parents_compute_cost_sum = []
            # for parent in job.computation_graph.predecessors(op):
                # # parent compute cost
                # parent_compute_cost_sum = 0
                # for device_type in cluster.topology.graph.graph['worker_types']:
                    # parent_compute_cost = job.computation_graph.nodes[parent]['compute_cost'][device_type]
                    # try:
                        # parent_compute_cost_sum += (parent_compute_cost / job.details['max_compute_cost'][device_type])
                    # except ZeroDivisionError:
                        # pass
                # parents_compute_cost_sum.append(parent_compute_cost_sum)
                # # parent memory cost
                # try:
                    # parents_memory_cost_sum += (job.computation_graph.nodes[parent]['memory_cost'] / job.details['max_memory_cost'])
                # except ZeroDivisionError:
                    # pass
        # op_features.append(np.array(parents_compute_cost_sum, dtype=object))
        # op_features.append(np.array(parents_memory_cost_sum, dtype=object))

        # # children compute and memory cost sum
        # children_memory_cost_sum = 0
        # if len(list(job.computation_graph.successors(op))) == 0:
            # # op has no children
            # children_compute_cost_sum = [0]
        # else:
            # children_compute_cost_sum = []
            # for child in job.computation_graph.successors(op):
                # # child compute cost
                # child_compute_cost_sum = 0
                # for device_type in cluster.topology.graph.graph['worker_types']:
                    # child_compute_cost = job.computation_graph.nodes[child]['compute_cost'][device_type]
                    # try:
                        # child_compute_cost_sum += (child_compute_cost / job.details['max_compute_cost'][device_type])
                    # except ZeroDivisionError:
                        # pass
                # children_compute_cost_sum.append(child_compute_cost_sum)
                # # child memory cost
                # try:
                    # children_memory_cost_sum += (job.computation_graph.nodes[child]['memory_cost'] / job.details['max_memory_cost'])
                # except ZeroDivisionError:
                    # pass
        # op_features.append(np.array(children_compute_cost_sum, dtype=object))
        # op_features.append(np.array(children_memory_cost_sum, dtype=object))

        # op depth
        node_depth = len(nx.shortest_path(job.computation_graph, source=0, target=op)) / job.details['max_depth']
        op_features.append(np.array(node_depth, dtype=object))

        if flatten:
            op_features = flatten_numpy_array(op_features)

        return op_features





                















