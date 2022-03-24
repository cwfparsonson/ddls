import networkx as nx
import copy


class Job:
    def __init__(self,
                 computation_graph: nx.MultiDiGraph,
                 num_training_steps: int,
                 job_id: int = None,
                 job_type: str = None,
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

        if job_id is None:
            self.job_id = id(self)
        else:
            self.job_id = job_id 

        if details is None:
            self.details = {}
        else:
            self.details = details

        self.job_type = job_type

        # initialise graph-level info
        self.job_total_operation_memory_cost = self._init_job_total_operation_memory_cost()
        self.job_total_dependency_size = self._init_job_total_dependecy_size()
        
        # initialse additional node- and edge-level info
        self._init_node_info()
        self._init_edge_info()
        self._init_graph_info()
        
    def reset(self):
        '''Resets the job ready for a training step to be executed.'''
        # reset record of completed and uncompleted operations and dependencies
        self.completed_ops, self.uncompleted_ops = {}, {node for node in self.computation_graph.nodes}
        self.completed_deps, self.uncompleted_deps = {}, {edge for edge in self.computation_graph.edges}
        
    def register_completed_op(self, op):
        self.completed_operations.add(op)
        self.uncompleted_ops.remove(op)
        
    def register_completed_dep(self, dep):
        self.completed_deps.add(dep)
        self.uncompleted_deps.remove(dep)
        
    def _init_node_info(self):
        for node in self.computation_graph.nodes:
            self.computation_graph.nodes[node]['job_id'] = self.job_id
            # self.computation_graph.nodes[node]['remaining_run_time'] = self.computation_graph.nodes[node]['compute_cost']
            self.computation_graph.nodes[node]['remaining_run_time'] = None
            self.computation_graph.nodes[node]['parent_dependencies_satisfied'] = set()
        
    def _init_edge_info(self):
        for edge in self.computation_graph.edges:
            self.computation_graph[edge[0]][edge[1]][edge[2]]['job_id'] = self.job_id

    def _init_graph_info(self):
        self.computation_graph.graph['ops_ready'] = {list(nx.topological_sort(self.computation_graph))[0]}
        self.computation_graph.graph['ops_completed'] = set()

    def check_if_op_ready(self, op_id):
        return len(self.computation_graph.in_edges(op_id)) == self.computation_graph.nodes[op_id]['parent_dependencies_satisfied']

    def register_ready_op(self, op_id):
        self.computation_graph.graph['ops_ready'].add(op_id)

    def register_completed_op(self, op_id):
        self.computation_graph.graph['ops_completed'].add(op_id)
        self.computation_graph.graph['ops_ready'].remove(op_id)
        if not self.is_placed:
            # job must have been placed by now to have had an op completed
            self.is_placed = True

    def reset_op_remaining_run_time(self, op_id, device_type):
        '''Given that an op has just been mounted on a device, reset the remaining run time for each op.'''
        self.computation_graph.nodes[op_id]['remaining_run_time'] = copy.deepcopy(self.computation_graph.nodes[op_id]['compute_cost'][device_type])

    def is_completed(self):
        return len(self.computation_graph.graph['ops_completed']) == len(self.computation_graph.nodes)

    def register_satisfied_dependency(self, edge):
        child = edge[1]
        print(f'Registering dependency {edge}')
        self.computation_graph.nodes[child]['parent_dependencies_satisfied'].add(child)
        if len(self.computation_graph[child]['parent_dependencies_satisfied']) == len(self.computation_graph.in_edges(child)):
            # all parent dependencies satisfied, child op is ready to be executed
            self.register_ready_op(child)

    def tick_op(self, op_id, tick):
        op = self.computation_graph.nodes[op_id]
        op['remaining_run_time'] -= min(tick, op['remaining_run_time'])
        if op['remaining_run_time'] == 0:
            self.register_completed_op(op_id)
           
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
    
    
    





    




#class OldJob:
#    def __init__(self,
#                 num_layers: int,
#                 num_dims_per_layer: int,
#                 weight_size: int,
#                 num_weights: int,
#                 batch_size: int,
#                 sample_size: int, # memory per data set sample
#                 num_samples: int, # number of samples in data set
#                 num_epochs: int,
#                 job_id: int = None,
#                 details: dict = {},
#                 job_type: str = 'DNN'):
#        
#        if job_id is None:
#            self.job_id = id(self)
#        else:
#            self.job_id = job_id 
#
#        self.weight_size = weight_size
#        self.num_weights = num_weights
#        self.num_layers = num_layers
#        self.num_dims_per_layer = num_dims_per_layer
#        self.batch_size = batch_size
#        self.sample_size = sample_size
#        self.num_samples = num_samples
#        self.num_epochs = num_epochs
#        self.details = details
#        self.job_type = job_type
#    
#    def __str__(self):
#        descr = f'Job ID: {self.job_id}'
#        descr += f' | Job type: {self.job_type}'
#        descr += f' | Per-weight memory: {self.weight_size}'
#        descr += f' | # weights: {self.num_weights:.3e}'
#        descr += f' | Model memory: {self.get_model_size():.3e}'
#        descr += f' | # layers: {self.num_layers}'
#        descr += f' | Per-layer # dims: {self.num_dims_per_layer}'
#        descr += f' | Batch size: {self.batch_size}'
#        descr += f' | Per-sample memory: {self.sample_size}'
#        descr += f' | # samples: {self.num_samples:.3e}'
#        descr += f' | Data set size: {self.get_dataset_size():.3e}'
#        descr += f' | # epochs: {self.num_epochs}'
#        return descr
#    
#    def __eq__(self, other):
#        return self.job_id == other.job_id
#    
#    def get_model_size(self):
#        return self.weight_size * self.num_weights
#
#    def get_dataset_size(self):
#        return self.sample_size * self.num_samples

