import ddls
from ddls.environments.ddls_observation_function import DDLSObservationFunction
from ddls.environments.ddls_observation import DDLSObservation
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.demands.jobs.job import Job
from ddls.utils import flatten_list, flatten_numpy_array
from ddls.environments.ramp_cluster.agents.placers.utils import find_meta_block, dummy_ramp, get_factor_pairs, get_block_shapes, get_block

import gym
import torch
import networkx as nx
import numpy as np
import copy

class RampJobPartitioningObservation(DDLSObservationFunction):
    def __init__(self,
                 max_partitions_per_op: int,
                 pad_obs_kwargs: dict = None,
                 machine_epsilon: float = 1e-7,
                 ):
        '''
        Args:
            pad_obs_kwargs: If not None will look at jobs_config, get max number of nodes and
                edges across all jobs, and pad each obs to ensure dimensionality of
                each obs is consistent even for observations with varying graph sizes.
                pad_obs_kwargs must be dict of {'max_nodes': <int>, 'max_edges': <int>}
                UPDATE: Only needs to be {'max_nodes': <int>}, will then calc max edges
                by assuming fully connected graph of max_nodes.
            machine_epsilon: Add to obs feat values to stop getting negative feats due
                to python floating point arithmetic.
        '''
        self.max_partitions_per_op = max_partitions_per_op
        self.pad_obs_kwargs = pad_obs_kwargs
        self.machine_epsilon = machine_epsilon

        # init obs space
        self._observation_space = None
        # self.observation_space = gym.spaces.Dict({})

        # init any hard-coded feature min and max values
        self.node_features_low, self.node_features_high = 0, 1
        self.edge_features_low, self.edge_features_high = 0, 1
        self.graph_features_low, self.graph_features_high = 0, 1
        self.edges_src_low, self.edges_src_high = 0, self.pad_obs_kwargs['max_nodes'] - 1
        self.edges_dst_low, self.edges_dst_high = 0, self.pad_obs_kwargs['max_nodes'] - 1

    def reset(self, 
              env, # RampJobPlacementShapingEnvironment
              flatten: bool = True):
        if self.pad_obs_kwargs is not None:
            self.max_nodes =  self.pad_obs_kwargs['max_nodes']
            self.max_edges = int(self.max_nodes*(self.max_nodes-1)/2) #number of edges in a fully connected graph
        else:
            self.max_nodes, self.max_edges = 0, 0

        # get job which is going to be encoded in obs
        job = self._get_job_to_encode(env)
    
        # encode the initial obs
        obs = self._encode_obs(job, env, flatten=flatten)

        # use the encoded obs to initialise the observation space
        self.observation_space = gym.spaces.Dict({
                'action_set': gym.spaces.Box(low=min(obs['action_set']), high=max(obs['action_set']), shape=obs['action_set'].shape, dtype=obs['action_set'].dtype),
                'action_mask': gym.spaces.Box(low=0, high=1, shape=obs['action_mask'].shape, dtype=obs['action_mask'].dtype),
                'node_features': gym.spaces.Box(low=self.node_features_low, high=self.node_features_high, shape=obs['node_features'].shape, dtype=obs['node_features'].dtype),
                'edge_features': gym.spaces.Box(low=self.edge_features_low, high=self.edge_features_high, shape=obs['edge_features'].shape, dtype=obs['edge_features'].dtype),
                'graph_features': gym.spaces.Box(low=self.graph_features_low, high=self.graph_features_high, shape=obs['graph_features'].shape, dtype=obs['graph_features'].dtype),
                # 'edges_src': gym.spaces.Box(low=0, high=max(obs['edges_src'])+1, shape=obs['edges_src'].shape, dtype=obs['edges_src'].dtype),
                # 'edges_dst': gym.spaces.Box(low=0, high=max(obs['edges_dst'])+1, shape=obs['edges_dst'].shape, dtype=obs['edges_dst'].dtype),
                'edges_src': gym.spaces.Box(low=self.edges_src_low, high=self.edges_src_high, shape=obs['edges_src'].shape, dtype=obs['edges_src'].dtype),
                'edges_dst': gym.spaces.Box(low=self.edges_src_low, high=self.edges_src_high, shape=obs['edges_dst'].shape, dtype=obs['edges_dst'].dtype),
                'node_split': gym.spaces.Box(low=0, high=self.max_nodes, shape=(1,), dtype=obs['node_split'].dtype),
                'edge_split': gym.spaces.Box(low=0, high=self.max_edges, shape=(1,), dtype=obs['edge_split'].dtype)
            })

        # print(f'\n\n\n--------------------------------------------------')
        # print(f'\nobs_space:\n{self.observation_space}')
    
    def get_action_set_and_action_mask(self, env):
        # action_set, action_mask = [0], [True] # action = 0 (do not place job) is always valid
        # for action in range(1, env.cluster.topology.graph.graph['num_workers']+1):
            # action_set.append(action)
            # is_valid = False
            
            # # if partitoning op (i.e. num partitions >1), then number of times op is partitioned must be even
            # if (action > 1 and action % 2 == 0) or (action == 1):
                # # cannot partition an op more times than max_partitions_per_op
                # if action <= env.max_partitions_per_op:
                    # # cannot partition an op more times than there are available workers
                    # if action <= env.cluster.topology.graph.graph['num_workers'] - len(env.cluster.mounted_workers):
                        # is_valid = True

            # # # DEBUG
            # # print(f'action: {action} | max_partitions_per_op: {env.max_partitions_per_op} | num available workers: {env.cluster.topology.graph.graph["num_workers"] - len(env.cluster.mounted_workers)} -> is_valid: {is_valid}')

            # action_mask.append(is_valid)

        ramp_shape = (env.cluster.topology.num_communication_groups, env.cluster.topology.num_racks_per_communication_group, env.cluster.topology.num_servers_per_rack)
        action_set, action_mask = [0], [True] # action = 0 (do not place job) is always valid
        # for action in range(1, env.cluster.topology.graph.graph['num_workers']+1):
        for action in range(1, env.max_partitions_per_op+1):
            action_set.append(action)
            is_valid = False

            # if partitoning op (i.e. num partitions >1), then number of times op is partitioned must be even
            if (action > 1 and action % 2 == 0) or (action == 1):
                # cannot partition an op more times than max_partitions_per_op
                if action <= env.max_partitions_per_op:
                    # cannot partition an op more times than there are available workers
                    if action <= env.cluster.topology.graph.graph['num_workers'] - len(env.cluster.mounted_workers):
                        if action == 1:
                            # run job sequentially on one worker
                            is_valid = True
                        else:
                            # check if number of partitions has a valid shape which meets the ramp rule requirements
                            pairs = get_factor_pairs(action)
                            block_shapes = get_block_shapes(pairs, ramp_shape)
                            b = []
                            for shape in block_shapes:
                                block = get_block(shape[0], shape[1], shape[2], ramp_shape)
                                b.extend(block)
                            if len(b) > 0:
                                is_valid = True

            # # DEBUG
            # print(f'action: {action} | max_partitions_per_op: {env.max_partitions_per_op} | num available workers: {env.cluster.topology.graph.graph["num_workers"] - len(env.cluster.mounted_workers)} -> is_valid: {is_valid}')

            action_mask.append(is_valid)

        return action_set, action_mask

    def extract(self, 
                env, # RampJobPlacementShapingEnvironment,
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
        return self._encode_obs(self._get_job_to_encode(env), env, flatten=flatten)

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space: gym.spaces.Dict):
        self._observation_space = observation_space

    def _get_job_to_encode(self, env):
        # TODO: Implement where get given job and do per-job encoding?
        # return list(env.op_partition.partitioned_jobs.values())[0] # assume event-driven where only ever have one job to queue
        return list(env.cluster.job_queue.jobs.values())[0] # assume event-driven where only ever have one job to queue

    def _pad_obs(self, obs):
        padded_obs = copy.deepcopy(obs)

        edges_src = torch.Tensor(obs['edges_src'])
        edges_dst = torch.Tensor(obs['edges_dst'])
        node_features = torch.Tensor(obs['node_features'])
        edge_features = torch.Tensor(obs['edge_features'])

        self.max_nodes = self.pad_obs_kwargs['max_nodes']
        self.max_edges = int(self.max_nodes*(self.max_nodes-1)/2) #number of edges in a fully connected graph

        src_padding = torch.zeros((self.max_edges-len(edges_src),))
        dst_padding = torch.zeros((self.max_edges-len(edges_dst),))

        edges_src = torch.cat((edges_src,src_padding),dim=0)
        edges_dst = torch.cat((edges_dst,dst_padding),dim=0)

        edge_feature_padding = torch.zeros(
            self.max_edges-edge_features.shape[0],
            edge_features.shape[1]
        )
        edge_features = torch.cat((edge_features,edge_feature_padding),dim=0)

        node_feature_padding = torch.zeros(
            self.max_nodes-node_features.shape[0],
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
                    env, # RampJobPlacementShapingEnvironment, 
                    flatten: bool = True):
        # # DEBUG
        # print(f'\nEncoding obs for job {job}')
        # print(f'self.max_nodes: {self.max_nodes} | self.max_edges: {self.max_edges}')
        # print(f'job nodes: {len(list(job.computation_graph.nodes()))} | job edges: {len(list(job.computation_graph.edges()))}')

        # check that can encode job
        if len(list(job.computation_graph.nodes())) > self.max_edges:
            raise Exception(f'ERROR: Trying to encode job {job} with {len(list(job.computation_graph.nodes()))} nodes but max nodes set to {self.max_nodes}. Increase max nodes or use smaller computation graphs.')
        if len(list(job.computation_graph.edges())) > self.max_edges:
            raise Exception(f'ERROR: Trying to encode job {job} with {len(list(job.computation_graph.edges()))} nodes but max edges set to {self.max_nodes}. Increase max edges or use smaller computation graphs.')

        edges_src, edges_dst = self._extract_edges_src_dst(job)
        action_set, action_mask = self.get_action_set_and_action_mask(env)
        obs =   {
                    'action_set': np.array(action_set, dtype=np.int16),
                    'action_mask': np.array(action_mask, dtype=np.int16),
                    'node_features': np.array(self._extract_node_features(job, env.cluster), dtype=np.float32),
                    'edge_features': np.array(self._extract_edge_features(job, env.cluster), dtype=np.float32),
                    'graph_features': np.array(self._extract_graph_features(job, env.cluster), dtype=np.float32),
                    'edges_src': np.array(edges_src, dtype=np.float32),
                    'edges_dst': np.array(edges_dst, dtype=np.float32),
                    # 'node_split': None,
                    # 'edge_split': None
                    'node_split': np.array(np.nan), # init as NaN and can replace if using obs padding
                    'edge_split': np.array(np.nan) # init as NaN and can replace if using obs padding
                 }

        # add action mask to graph features
        obs['graph_features'] = np.concatenate((obs['graph_features'], action_mask))

        # pad obs if required
        if self.pad_obs_kwargs is not None:
            obs = self._pad_obs(obs)

        # # DEBUG
        # print(f'encoded obs after padding: {obs}')
        # for key, val in obs.items():
            # try:
                # print(f'{key} -> shape {np.array(val).shape} | dtype {type(val[0][0])} | min {np.min(val)} | max {np.max(val)}')
            # except:
                # print(f'{key} -> shape {np.array(val).shape} | dtype {type(val[0])} | min {np.min(val)} | max {np.max(val)}')
        # print(f'observation space of env:')
        # if env.observation_space is not None:
            # for key, val in env.observation_space.items():
                # print(f'{key} -> {val}')
        # else:
            # print(env.observation_space)

        # check for any invalid values
        for key, val in obs.items():
            if key not in set(['node_split', 'edge_split']):
                if not np.isfinite(val).any():
                    raise Exception(f'{key} in observation contains NaN or inf value(s).')

        # # DEBUG
        # print(f'\nobs:\n{obs}')
        # for key, val in obs.items():
            # print(f'{key} | shape = {val.shape} | dtype = {val.dtype} | min: {np.min(val)} | max: {np.max(val)}')

        return obs

    def _extract_node_features(self, job, cluster):
        node_features = [self._get_op_features(node, job, cluster) for node in job.computation_graph.nodes]

        if np.min(node_features) < self.node_features_low:
            raise Exception(f'node_features_low set to {self.node_features_low} but min feature val is {np.min(node_features)} at index {np.argmin(node_features)}')
        if np.max(node_features) > self.node_features_high:
            raise Exception(f'node_features_high set to {self.node_features_high} but max feature val is {np.max(node_features)} at index {np.argmax(node_features)}')

        return node_features

    def _extract_edge_features(self, job, cluster):
        # edge_features = [np.array([1]) for _ in job.computation_graph.edges]

        edge_features = [self._get_dep_features(edge, job, cluster) for edge in job.computation_graph.edges]

        if np.min(edge_features) < self.edge_features_low:
            raise Exception(f'edge_features_low set to {self.edge_features_low} but min feature val is {np.min(edge_features)} at index {np.argmin(edge_features)}')
        if np.max(edge_features) > self.edge_features_high:
            raise Exception(f'edge_features_high set to {self.edge_features_high} but max feature val is {np.max(edge_features)} at index {np.argmax(edge_features)}')

        return edge_features

    def _extract_graph_features(self, job, cluster):
        graph_features = flatten_list([self._get_job_features(job, cluster), 
                             # self._get_network_worker_features(job, cluster), 
                             self._get_network_graph_features(job, cluster),
                             # self._get_action_mask_features(job, cluster)
                            ])
        # print(f'graph_features: {graph_features}') # DEBUG
        if np.min(graph_features) < self.graph_features_low:
            raise Exception(f'graph_features_low set to {self.graph_features_low} but min feature val is {np.min(graph_features)} at index {np.argmin(graph_features)}')
        if np.max(graph_features) > self.graph_features_high:
            raise Exception(f'graph_features_high set to {self.graph_features_high} but max feature val is {np.max(graph_features)} at index {np.argmax(graph_features)}')

        return graph_features

    def _extract_edges_src_dst(self, job):
        # need to ensure node ids are ints
        node_to_node_int = {}
        for node_int, node in enumerate(job.computation_graph.nodes):
            node_to_node_int[node] = node_int

        # collect int src-dst pairs
        srcs, dsts = [], []
        for edge in job.computation_graph.edges:
            srcs.append(node_to_node_int[edge[0]])
            dsts.append(node_to_node_int[edge[1]])

        return srcs, dsts

    def _get_job_features(self, job, cluster, flatten=True):
        job_features = []

        # num_training_steps_remaining = (job.num_training_steps - job.training_step_counter) / job.num_training_steps
        # job_features.append(np.array(num_training_steps_remaining, dtype=object))

        if cluster.jobs_generator.jobs_params['max_job_total_num_ops'] - cluster.jobs_generator.jobs_params['min_job_total_num_ops'] != 0:
            num_ops = (len(job.computation_graph.nodes()) - cluster.jobs_generator.jobs_params['min_job_total_num_ops']) / (cluster.jobs_generator.jobs_params['max_job_total_num_ops'] - cluster.jobs_generator.jobs_params['min_job_total_num_ops'])
        else:
            num_ops = 1
        job_features.append(np.array(num_ops, dtype=object))

        if cluster.jobs_generator.jobs_params['max_job_total_num_deps'] - cluster.jobs_generator.jobs_params['min_job_total_num_deps'] != 0:
            # # DEBUG
            # print(f'job: {job}')
            # print(f'job details: {job.details}')
            # print(f'graph.graph: {job.computation_graph.graph}')
            # print(f'job edges: {len(list(job.computation_graph.edges()))}')
            # print(f'min job total num deps: {cluster.jobs_generator.jobs_params["min_job_total_num_deps"]}')
            # print(f'max job total num deps: {cluster.jobs_generator.jobs_params["max_job_total_num_deps"]}')
            num_deps = (len(job.computation_graph.edges()) - cluster.jobs_generator.jobs_params['min_job_total_num_deps']) / (cluster.jobs_generator.jobs_params['max_job_total_num_deps'] - cluster.jobs_generator.jobs_params['min_job_total_num_deps'])
        else:
            num_deps = 1
        job_features.append(np.array(num_deps, dtype=object))

        if cluster.jobs_generator.jobs_params['max_job_sequential_completion_times'] - cluster.jobs_generator.jobs_params['min_job_sequential_completion_times'] != 0:
            job_sequential_completion_time = (job.details['job_sequential_completion_time'][list(cluster.topology.graph.graph['worker_types'])[0]] - cluster.jobs_generator.jobs_params['min_job_sequential_completion_times']) / (cluster.jobs_generator.jobs_params['max_job_sequential_completion_times'] - cluster.jobs_generator.jobs_params['min_job_sequential_completion_times'])
        else:
            job_sequential_completion_time = 1
        job_features.append(np.array(job_sequential_completion_time, dtype=object))

        if cluster.jobs_generator.jobs_params['max_max_acceptable_job_completion_times'] - cluster.jobs_generator.jobs_params['min_max_acceptable_job_completion_times'] != 0:
            max_acceptable_job_completion_time = (job.details['max_acceptable_job_completion_time'][list(cluster.topology.graph.graph['worker_types'])[0]] - cluster.jobs_generator.jobs_params['min_max_acceptable_job_completion_times']) / (cluster.jobs_generator.jobs_params['max_max_acceptable_job_completion_times'] - cluster.jobs_generator.jobs_params['min_max_acceptable_job_completion_times'])
        else:
            max_acceptable_job_completion_time = 1
        job_features.append(np.array(max_acceptable_job_completion_time, dtype=object))

        if cluster.jobs_generator.jobs_params['max_max_acceptable_job_completion_time_fracs'] - cluster.jobs_generator.jobs_params['min_max_acceptable_job_completion_time_fracs'] != 0:
            max_acceptable_job_completion_time_frac = (job.max_acceptable_job_completion_time_frac - cluster.jobs_generator.jobs_params['min_max_acceptable_job_completion_time_fracs']) / (cluster.jobs_generator.jobs_params['max_max_acceptable_job_completion_time_fracs'] - cluster.jobs_generator.jobs_params['min_max_acceptable_job_completion_time_fracs'])
        else:
            max_acceptable_job_completion_time_frac = 1
        job_features.append(np.array(max_acceptable_job_completion_time_frac, dtype=object)) # effectively tells agent priority of job relative to all other jobs which can come

        job_features.append(np.array(job.max_acceptable_job_completion_time_frac, dtype=object)) # give agent the raw fraction so explicitly tell it roughly how many times job must be partitioned to meet the maximum acceptable job completion time requirement of the job

        if cluster.jobs_generator.jobs_params['max_job_total_op_memory_costs'] - cluster.jobs_generator.jobs_params['min_job_total_op_memory_costs'] != 0:
            job_total_op_memory_cost = (job.details['job_total_op_memory_cost'] - cluster.jobs_generator.jobs_params['min_job_total_op_memory_costs']) / (cluster.jobs_generator.jobs_params['max_job_total_op_memory_costs'] - cluster.jobs_generator.jobs_params['min_job_total_op_memory_costs'])
        else:
            job_total_op_memory_cost = 1
        job_features.append(np.array(job_total_op_memory_cost, dtype=object))

        if cluster.jobs_generator.jobs_params['max_job_total_dep_sizes'] - cluster.jobs_generator.jobs_params['min_job_total_dep_sizes'] != 0:
            job_total_dep_size = (job.details['job_total_dep_size'] - cluster.jobs_generator.jobs_params['min_job_total_dep_sizes']) / (cluster.jobs_generator.jobs_params['max_job_total_dep_sizes'] - cluster.jobs_generator.jobs_params['min_job_total_dep_sizes'])
        else:
            job_total_dep_size = 1
        job_features.append(np.array(job_total_dep_size, dtype=object))

        if cluster.jobs_generator.jobs_params['max_job_num_training_steps'] - cluster.jobs_generator.jobs_params['min_job_num_training_steps'] != 0:
            job_num_training_steps = (job.num_training_steps - cluster.jobs_generator.jobs_params['min_job_num_training_steps']) / (cluster.jobs_generator.jobs_params['max_job_num_training_steps'] - cluster.jobs_generator.jobs_params['min_job_num_training_steps'])
        else:
            job_num_training_steps = 1
        job_features.append(np.array(job_num_training_steps, dtype=object))

        op_compute_costs, op_memory_costs = [], []
        for op in job.computation_graph.nodes():
            for device_type in cluster.topology.graph.graph['worker_types']:
                op_compute_costs.append(job.computation_graph.nodes[op]['compute_cost'][device_type] / job.details['max_compute_cost'][device_type])
            op_memory_costs.append(job.computation_graph.nodes[op]['memory_cost'] / job.details['max_memory_cost'])
        job_features.append(np.array(np.mean(op_compute_costs), dtype=object))
        job_features.append(np.array(np.median(op_compute_costs), dtype=object))
        job_features.append(np.array(np.mean(op_memory_costs), dtype=object))
        job_features.append(np.array(np.median(op_memory_costs), dtype=object))

        dep_sizes = []
        for dep in job.computation_graph.edges:
            u, v, k = dep
            dep_sizes.append(job.computation_graph[u][v][k]['size'])
        job_features.append(np.array(np.mean(dep_sizes) / job.details['max_dep_size'], dtype=object))
        job_features.append(np.array(np.median(dep_sizes) / job.details['max_dep_size'], dtype=object))

        if flatten:
            job_features = flatten_numpy_array(job_features)

        # job_features = np.array(job_features, dtype=object) + self.machine_epsilon
        for idx, el in enumerate(job_features):
            if el < self.graph_features_low:
                job_features[idx] += self.machine_epsilon

        return job_features.tolist()

    # def _get_network_worker_features(self, job, cluster, flatten=True):
        # network_worker_features = []
        # num_ready_ops, num_mounted_ops = [], []
        # for worker_id, server_id in cluster.topology.graph.graph['worker_to_node'].items():
            # worker = cluster.topology.graph.nodes[server_id]['workers'][worker_id]
            # ready_op_counter = 0
            # mounted_op_counter = 0
            # for job_idx, op_ids in worker.mounted_job_idx_to_ops.items():
                # job = cluster.jobs_running[job_idx]
                # for op_id in op_ids:
                    # mounted_op_counter += 1
                    # if op_id in job.computation_graph.graph['ops_ready']:
                        # ready_op_counter += 1
            # try:
                # num_ready_ops.append(ready_op_counter / mounted_op_counter)
            # except ZeroDivisionError:
                # num_ready_ops.append(0)
            # try:
                # num_mounted_ops.append(mounted_op_counter / cluster.num_mounted_ops)
            # except ZeroDivisionError:
                # num_mounted_ops.append(0)
        # network_worker_features.append(np.array(num_ready_ops, dtype=object))
        # network_worker_features.append(np.array(num_mounted_ops, dtype=object))
        # if flatten:
            # network_worker_features = flatten_numpy_array(network_worker_features)
        # return network_worker_features

    def _get_network_graph_features(self, job, cluster, flatten=True):
        network_graph_features = []

        # num_active_workers = cluster.num_active_workers / len(list(cluster.topology.graph.graph['worker_to_node'].keys()))
        # network_graph_features.append(np.array(num_active_workers, dtype=object))

        num_mounted_workers = len(cluster.mounted_workers) / len(list(cluster.topology.graph.graph['worker_to_node'].keys()))
        network_graph_features.append(np.array(num_mounted_workers, dtype=object))

        # num_mounted_channels = len(cluster.mounted_channels) / (cluster.topology.num_channels * len(list(cluster.topology.graph.edges())))
        # network_graph_features.append(np.array(num_mounted_channels, dtype=object))

        num_jobs_running = len(list(cluster.jobs_running.keys())) / len(list(cluster.topology.graph.graph['worker_to_node'].keys()))
        network_graph_features.append(np.array(num_jobs_running, dtype=object))

        if flatten:
            network_graph_features = flatten_numpy_array(network_graph_features)

        # network_graph_features = np.array(network_graph_features, dtype=object) + self.machine_epsilon
        for idx, el in enumerate(network_graph_features):
            if el < self.graph_features_low:
                network_graph_features[idx] += self.machine_epsilon

        return network_graph_features.tolist()

    def _get_action_mask_features(self, job, cluster):
        pass

    def _get_dep_features(self, dep, job, cluster, flatten=True):
        dep_features = []
        u, v, k = dep

        # dep size
        dep_features.append(np.array((job.computation_graph[u][v][k]['size'] / job.details['max_dep_size']), dtype=object))
        is_highest_dep_size = dep == job.details['max_dep_size_dep']
        dep_features.append(np.array(is_highest_dep_size, dtype=object))

        if flatten:
            dep_features = flatten_numpy_array(dep_features)

        # dep_features = np.array(dep_features, dtype=object) + self.machine_epsilon
        for idx, el in enumerate(dep_features):
            if el < self.edge_features_low:
                dep_features[idx] += self.machine_epsilon

        return dep_features.tolist()

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
        # if np.max(op_features) > 1:
            # raise Exception(f'found > 1 feat')

        # op memory cost
        try:
            memory_cost = job.computation_graph.nodes[op]['memory_cost'] / job.details['max_memory_cost']
        except ZeroDivisionError:
            memory_cost = 0
        is_highest_memory_cost = op == job.details['max_memory_node']
        op_features.append(np.array(memory_cost, dtype=object))
        op_features.append(np.array(is_highest_memory_cost, dtype=object))
        # if np.max(op_features) > 1:
            # print(f'op: {op} | memory cost: {job.computation_graph.nodes[op]["memory_cost"]} | job max memory_cost: {job.details["max_memory_cost"]} | job max memory node: {job.details["max_memory_node"]}')
            # raise Exception(f'found > 1 feat')

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
        try:
            node_depth = len(nx.shortest_path(job.computation_graph, source=job.computation_graph.graph['source_nodes'][0], target=op)) / job.details['max_depth']
        except nx.NetworkXNoPath:
            # sibling nodes have no directional path to oneanother
            node_depth = 0
        op_features.append(np.array(node_depth, dtype=object))
        # if np.max(op_features) > 1:
            # raise Exception(f'found > 1 feat')

        if flatten:
            op_features = flatten_numpy_array(op_features)

        # op_features = np.array(op_features, dtype=object) + self.machine_epsilon
        for idx, el in enumerate(op_features):
            if el < self.node_features_low:
                op_features[idx] += self.machine_epsilon

        return op_features.tolist()
