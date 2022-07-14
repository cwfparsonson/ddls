from ddls.plotting.plotting import plot_computation_graph

import networkx as nx
import copy
from typing import Union
from collections import defaultdict
import json


class Job:
    def __init__(self,
                 computation_graph: nx.MultiDiGraph,
                 num_training_steps: int,
                 job_id: int = None,
                 details: dict = None):
        '''
        A ddls deep learning job consists of a computation_graph which contains the
        operations and dependencies of one forward and backward pass of a DNN model.
        One execution of the computation graph corresponds to one training step of 
        the model. Therefore, the task is to execute the computation graph some
        num_training_steps times.
        
        Before the job can begin to be executed, you **must** call Job.reset()

        Args:
            computation_graph: Computation graph of made up of operations,
                control, and data dependencies describing the processes
                which need to be executed to perform one epoch/iteration/update/training step
                of the model (i.e. a forward and backward pass).
            num_training_steps: Number of epochs/iterations/updates/training step to perform on the model
                (i.e. number of times to run the computation graph).
        '''
        self.computation_graph = computation_graph
        self.num_training_steps = num_training_steps
        self.training_step_counter = 0

        if job_id is None:
            self.job_id = id(self)
        else:
            self.job_id = job_id 

        if details is None:
            self.details = {}
        else:
            self.details = details

        self.reset_job(self.details)


    def _init_job_details(self, details: dict = None):
        '''Initialises some additional useful details about the job.'''
        if details is None:
            details = {}

        details['max_compute_node'], details['max_compute_cost'], details['max_memory_node'], details['max_memory_cost'], details['max_depth_node'], details['max_depth'] = self.get_max_node_details()
        details['max_dep_size_dep'], details['max_dep_size'] = self.get_max_edge_details()
        details['job_sequential_completion_time'] = self.get_job_sequential_completion_time()
        details['job_total_op_memory_cost'] = self.get_job_total_memory_cost()
        details['job_total_dep_size'] = self.get_job_total_dep_size()

        # do not override values which are already provided
        if 'communication_overhead_time' not in details:
            details['communication_overhead_time'] = 0
        if 'computation_overhead_time' not in details:
            details['computation_overhead_time'] = 0
        if 'mounted_workers' not in details:
            details['mounted_workers'] = set()
        if 'mounted_channels' not in details:
            details['mounted_channels'] = set()

        # details.update(self.details)

        return details

    def get_job_sequential_completion_time(self):
        '''
        Assuming all job ops are loaded onto one device, computes the total 
        compute cost of sequentially computing these ops.
        '''
        job_sequential_completion_time = defaultdict(lambda: 0)
        for op in self.computation_graph.nodes:
            for device_type, compute_cost in self.computation_graph.nodes[op]['compute_cost'].items():
                job_sequential_completion_time[device_type] += compute_cost
        for device_type in job_sequential_completion_time.keys():
            job_sequential_completion_time[device_type] *= self.num_training_steps
        return job_sequential_completion_time

    def get_job_total_memory_cost(self):
        job_total_memory_cost = 0
        for op in self.computation_graph.nodes:
            job_total_memory_cost += self.computation_graph.nodes[op]['memory_cost']
        return job_total_memory_cost

    def get_job_total_dep_size(self):
        job_total_dep_size = 0
        for dep in self.computation_graph.edges:
            u, v, k = dep
            job_total_dep_size += self.computation_graph[u][v][k]['size']
        return job_total_dep_size 

    def get_max_node_details(self):
        '''
        Goes through each op in computation graph and finds info about nodes with
        maximum stats for various metrics.

        For compute cost info, will return max_compute_node and max_compute_node
        which are dicts mapping device types to the corresponding info.

        For memory cost info, is device-agnostic so no need for dict mapping.

        For depth info, gets maximum depth of any node from source.
        '''
        # print(f'\n>>> Getting max node details <<<')
        max_compute_node, max_compute_cost = defaultdict(lambda: 0), defaultdict(lambda: 0)
        max_memory_node, max_memory_cost = 0, 0
        max_depth_node, max_depth = 0, 0
        for node in self.computation_graph.nodes:
            # print(f'node {node} -> mem cost {self.computation_graph.nodes[node]["memory_cost"]}')
            # check if update max cost info
            for device_type, compute_cost in self.computation_graph.nodes[node]['compute_cost'].items():
                if self.computation_graph.nodes[node]['compute_cost'][device_type] > max_compute_cost[device_type]:
                    max_compute_node[device_type] = copy.deepcopy(node)
                    max_compute_cost[device_type] = copy.deepcopy(compute_cost)
            # check if update max memory info
            if self.computation_graph.nodes[node]['memory_cost'] > max_memory_cost:
                max_memory_node = copy.deepcopy(node)
                max_memory_cost = copy.deepcopy(self.computation_graph.nodes[max_memory_node]['memory_cost'])
            # check if update max depth info
            try:
                node_depth = len(nx.shortest_path(self.computation_graph, source=self.computation_graph.graph['source_nodes'][0], target=node))
            except nx.NetworkXNoPath:
                # sibling nodes have no directional path to oneanother
                node_depth = 0
            if node_depth > max_depth:
                max_depth_node = copy.deepcopy(node)
                max_depth = copy.deepcopy(node_depth)
        # print(f'max_compute_node: {max_compute_node} | max_compute_cost: {max_compute_cost} | max_memory_node: {max_memory_node} | max_memory_cost: {max_memory_cost}')
        return max_compute_node, max_compute_cost, max_memory_node, max_memory_cost, max_depth_node, max_depth

    def get_max_edge_details(self):
        max_dep_size, max_dep_size_dep = 0, None
        for edge in self.computation_graph.edges:
            u, v, k = edge
            if self.computation_graph[u][v][k]['size'] > max_dep_size:
                max_dep_size_dep = copy.deepcopy(edge)
                max_dep_size = copy.deepcopy(self.computation_graph[u][v][k]['size'])
        return max_dep_size_dep, max_dep_size

    def reset_job(self, details, computation_graph=None):
        '''Resets whole job. If computation_graph is not None, will update job's computation graph.'''
        if computation_graph is not None:
            self.computation_graph = computation_graph

        self.job_total_operation_memory_cost = self._init_job_total_operation_memory_cost()
        self.job_total_dependency_size = self._init_job_total_dependecy_size()
        
        self.reset_job_training_step()
        # self.details = self._init_job_details(details)
        self.details.update(self._init_job_details(details))
        
    def reset_job_training_step(self):
        '''Resets the job ready for a training step to be executed.'''
        # initialse additional node-, edge-, and graph-level info self._init_node_info()
        self._init_node_info()
        self._init_edge_info()
        self._init_graph_info()

    def register_job_arrived(self, 
                             time_arrived: Union[int, float], 
                             job_idx: int):
        '''When job arrives in simulation, call this method to register the details.

        Args:
            time_arrived: Time job arrived in simulation.
            job_idx: Index assigned to job ID which is unique to the job across
                the whole of the simulation.
        '''
        self.details['time_arrived'] = copy.deepcopy(time_arrived)
        self.details['time_started'] = None
        self.details['time_completed'] = None
        self.details['job_idx'] = copy.deepcopy(job_idx)

    def register_job_running(self,
                             time_started: Union[int, float]):
        '''When start running job on cluster, call this method to register the details.

        Args:
            time_started: Time job started running on cluster.
        '''
        self.details['time_started'] = copy.deepcopy(time_started)

    def register_job_completed(self,
                               time_completed: Union[int, float]):
        '''
        When completed the job by executing the computation_graph some
        num_training_steps number of times, call this method to register the details.

        Args:
            time_completed: Time job was completed.
        '''
        self.details['time_completed'] = copy.deepcopy(time_completed)
        
    def _init_node_info(self):
        for node in self.computation_graph.nodes:
            self.computation_graph.nodes[node]['job_id'] = self.job_id
            self.computation_graph.nodes[node]['parent_deps_completed'] = set()
            if 'mounted_device_type' not in self.computation_graph.nodes[node]:
                # not yet mounted this node
                self.computation_graph.nodes[node]['mounted_device_type'] = None
                self.computation_graph.nodes[node]['remaining_run_time'] = None
            else:
                # have already mounted this node and resetting ready for another training step, do not change mounted device type, just reset remaining run time
                self.reset_op_remaining_run_time(node, device_type=self.computation_graph.nodes[node]['mounted_device_type'])
        
    def _init_edge_info(self):
        for edge in self.computation_graph.edges:
            u, v, k = edge

            self.computation_graph[u][v][k]['job_id'] = self.job_id

            # can only know dep run time after placement (since depends on network, collectives, etc.), so init as None and must update later
            self.set_dep_init_run_time(edge, None)

            if 'remaining_run_time' not in self.computation_graph[u][v][k]:
                # not yet mounted this dep
                self.computation_graph[u][v][k]['remaining_run_time'] = None
            else:
                # have already mounted this dep and resetting ready for another training step
                self.reset_dep_remaining_run_time(edge)

    def set_dep_init_run_time(self, dep_id, run_time):
        u, v, k = dep_id
        self.computation_graph[u][v][k]['init_run_time'] = run_time
        self.computation_graph[u][v][k]['remaining_run_time'] = run_time

    def _init_graph_info(self):
        # if 0 not in self.computation_graph.nodes:
            # raise Exception(f'Source node of computation graph must have node ID 0.')
        # else:
            # if len(list(self.computation_graph.predecessors(0))) != 0:
                # raise Exception(f'Source node of computation graph must have node ID 0.')

        # get source/root node of DAG
        self.computation_graph.graph['source_nodes'] = []
        for node in self.computation_graph.nodes:
            if self.computation_graph.in_degree(node) == 0:
                self.computation_graph.graph['source_nodes'].append(node)

        # init source node as being ready to run
        # self.computation_graph.graph['ops_ready'] = {list(nx.topological_sort(self.computation_graph))[0]}
        self.computation_graph.graph['ops_ready'] = {source_node for source_node in self.computation_graph.graph['source_nodes']}
        self.computation_graph.graph['ops_completed'] = set()
        self.computation_graph.graph['deps_ready'] = set()
        self.computation_graph.graph['deps_completed'] = set()

    def check_if_op_ready(self, op_id):
        return len(self.get_op_parents(op_id)) == self.computation_graph.nodes[op_id]['parent_deps_completed']

    def register_ready_op(self, op_id):
        self.computation_graph.graph['ops_ready'].add(op_id)

    def register_completed_op(self, op_id):
        self.computation_graph.graph['ops_completed'].add(op_id)
        self.computation_graph.graph['ops_ready'].remove(op_id)

        for child_dep in self.computation_graph.out_edges(op_id):
            # parent op completed, child dependency is ready to be executed
            self.register_ready_dep(child_dep)

        if self.is_training_step_complete():
            self.training_step_counter += 1

    def register_ready_dep(self, dep_id):
        if len(dep_id) == 2:
            dep_id = (dep_id[0], dep_id[1], 0) # multi graph requires u, v, k edge
        self.computation_graph.graph['deps_ready'].add(dep_id)

    def get_op_parents(self, op_id):
        '''
        Op A is only a parent of op B if A has an edge going to B and B does not have an edge going to A.

        If op B is both a child and a parent of op A (i.e. have edges A -> B and B -> A), then edges
        A -> B and B -> A are treated as children of ops A and B rather than parents, and will
        be executed after each have been completed. This is because if A -> B and B -> A are treated
        as parents of B and A respectively, then there's an infinite loop where A and B can never
        start since their parent dependencies can never start.
        '''
        parent_ids = set()
        for parent_id in self.computation_graph.predecessors(op_id):
            if parent_id not in self.computation_graph.successors(op_id):
                parent_ids.add(parent_id)

        return parent_ids

    def register_completed_dep(self, dep_id):
        if dep_id not in self.computation_graph.graph['deps_completed']:
            self.computation_graph.graph['deps_completed'].add(dep_id)
            self.computation_graph.graph['deps_ready'].remove(dep_id)
            child = dep_id[1]
            self.computation_graph.nodes[child]['parent_deps_completed'].add(dep_id)
            if len(self.computation_graph.nodes[child]['parent_deps_completed']) == len(self.get_op_parents(child)):
                # all parent dependencies completed, child op is ready to be executed
                self.register_ready_op(child)
        else:
            # already registered dep as completed
            pass

    def reset_op_remaining_run_time(self, op_id, device_type):
        '''Given that an op has just been mounted on a device, reset the remaining run time for each op.'''
        if device_type is not None:
            self.computation_graph.nodes[op_id]['remaining_run_time'] = copy.deepcopy(self.computation_graph.nodes[op_id]['compute_cost'][device_type])
        else:
            self.computation_graph.nodes[op_id]['remaining_run_time'] = None
        self.computation_graph.nodes[op_id]['mounted_device_type'] = device_type

    def reset_dep_remaining_run_time(self, dep_id):
        u, v, k = dep_id
        self.computation_graph[u][v][k]['remaining_run_time'] = copy.deepcopy(self.computation_graph[u][v][k]['init_run_time'])

    def is_job_complete(self):
        return self.training_step_counter == self.num_training_steps

    def is_training_step_complete(self):
        # return len(self.computation_graph.graph['ops_completed']) == len(self.computation_graph.nodes)
        return len(self.computation_graph.graph['ops_completed']) == len(self.computation_graph.nodes) and len(self.computation_graph.graph['deps_completed']) == len(self.computation_graph.edges)

    def tick_op(self, op_id, tick):
        op = self.computation_graph.nodes[op_id]
        op['remaining_run_time'] -= min(tick, op['remaining_run_time'])
        if op['remaining_run_time'] == 0:
            self.register_completed_op(op_id)

    def tick_dep(self, dep_id, tick):
        u, v, k = dep_id
        self.computation_graph[u][v][k]['remaining_run_time'] -= min(tick, self.computation_graph[u][v][k]['remaining_run_time'])
        if self.computation_graph[u][v][k]['remaining_run_time'] == 0:
            self.register_completed_dep(dep_id)
           
    def _init_job_total_operation_memory_cost(self):
        job_operation_memory_cost = 0
        for node in self.computation_graph.nodes:
           job_operation_memory_cost += self.computation_graph.nodes[node]['memory_cost']
        return job_operation_memory_cost 

    def _init_job_total_dependecy_size(self):
        job_dependency_size = 0
        for edge in self.computation_graph.edges:
            job_dependency_size += self.computation_graph[edge[0]][edge[1]][edge[2]]['size']
        return job_dependency_size

    def __str__(self):
        descr = f'Job ID: {self.job_id}'
        descr += f' | # nodes: {len(self.computation_graph.nodes)}'
        descr += f' | # edges: {len(self.computation_graph.edges)}'
        descr += f' | # training steps: {self.num_training_steps}'
        descr += f' | Total op mem cost: {self.job_total_operation_memory_cost}'
        descr += f' | Total dep size: {self.job_total_dependency_size}'
        return descr
    
    def render(self, scaling_factor=3, title='computation_graph', show_fig=True, verbose=False):
        return plot_computation_graph(self.computation_graph, 
                                      scaling_factor=scaling_factor, 
                                      title=title, 
                                      show_fig=show_fig, 
                                      verbose=verbose)
