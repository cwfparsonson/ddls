import numpy as np
import random
import copy
import glob
from collections import defaultdict
import time
import json
import networkx as nx
import random


def seed_stochastic_modules_globally(default_seed=0, 
                                     numpy_seed=None, 
                                     random_seed=None):
    '''Seeds any stochastic modules so get reproducible results.'''
    if numpy_seed is None:
        numpy_seed = default_seed
    if random_seed is None:
        random_seed = default_seed
    
    np.random.seed(numpy_seed)
    random.seed(random_seed)

class Sampler:
    def __init__(self, 
                 pool: list,
                 sampling_mode: str):
        '''
        Args:
            sampling_mode ('replace', 'remove', 'remove_and_repeat')
        '''
        self.original_pool = pool
        self.sample_pool = copy.deepcopy(self.original_pool)
        self.sampling_mode = sampling_mode
    
    def sample(self):
        idx = np.random.randint(low=0, high=len(self.sample_pool))
        datum = self.sample_pool[idx]
        
        if self.sampling_mode == 'replace':
            pass
        elif self.sampling_mode == 'remove':
            self.sample_pool.pop(idx)
        elif self.sampling_mode == 'remove_and_repeat':
            self.sample_pool.pop(idx)
            if len(self.sample_pool) == 0:
                self.sample_pool = copy.deepcopy(self.original_pool)
            
        return datum

    def __str__(self):
        descr = f'Original pool: {self.original_pool} | Current pool: {self.sample_pool}'
        descr += f' | Sampling mode: {self.sampling_mode}'
        return descr

    def __len__(self):
        return len(self.sample_pool)





def pbtxt_nodes_from_pbtxt_file(file_path, verbose=False):
    '''
    Load a tensorflow computation graph from a .pbtxt file.

    Assumes structure of .pbtxt is the same as the CostGraphDef .pbtxt files
    open-sourced by DeepMind's REGAL paper https://openreview.net/attachment?id=rkxDoJBYPB&name=original_pdf
    (see https://www.tensorflow.org/jvm/api_docs/java/org/tensorflow/proto/framework/CostGraphDef).

    In the pbtxt files, each node has:
        input_info: Data dependency edges. preceding_node gives parent node of edge.
            preceding_port can be ignored since is always set to 1.
        output_info: Data dependency edges. size gives size (in Bytes) of data
            to be sent to the child node, alias_input_port can be ignored since
            always set to -1.
        control_input: Control dependency edges. Values give parent node of edge.
            Can have multiple entries per node.
        compute_cost: Gives estiamted time (in ms) to run the operation.
    '''
    graph, start_time = [], time.time_ns()
    with open(file_path, 'r') as file:
        node_info = None
        for line in file:
            line = line.replace(' ', '').replace('\n', '')
            if line == 'node{':
                # new node entry, save previous node's info and reset
                if node_info is not None:
                    graph.append(copy.deepcopy(node_info))
                node_info = defaultdict(list)  
            elif line == '}':
                # ended previouse element entry
                pass
            elif 'id' in line:
                node_info['id'] = int(line.split(":", 1)[1].strip())
            elif 'name' in line:
                if '_SOURCE' in line:
                    node_info['id'] = 0
            elif 'input_info' in line:
                pass
            elif 'preceding_node' in line:
                node_info['input_info'].append(int(line.split(":", 1)[1].strip()))
            elif 'preceding_port' in line:
                pass
            elif 'output_info' in line:
                pass
            elif 'size' in line:
                node_info['output_info'].append(int(line.split(":", 1)[1].strip()))
            elif 'alias_input_port' in line:
                pass
            elif 'control_input' in line:
                node_info['control_input'].append(int(line.split(":", 1)[1].strip()))
            elif 'compute_cost' in line:
                node_info['compute_cost'] = int(line.split(":", 1)[1].strip())
            else:
                raise Exception(f'Unrecognised line {line}')
    if verbose:
        print(f'Parsed file {file_path} in {(time.time_ns() - start_time)*1e-9:.3f} s')
                
    return graph


def pbtxt_graph_from_pbtxt_nodes(nodes, verbose=False):
    '''
    Converts a list of nodes read from a .pbtxt file into a networkx pbtxt graph.

    Hack: The .pbtxt open-sourced by DeepMind do not say which child each data
    dependency is connected to, therefore if have e.g. 1 parent node connected
    to 2 child nodes via 2 data dependencies, then cannot know which data
    dependency is applied to which child node. Therefore, in below
    implementation, if there are multiple possible data dependency sizes, we
    simply randomly sample a size for the dependency from amongst the possible
    sizes so that we retain the original distribution of sizes open-sourced by
    DeepMind.
    '''
    graph = nx.MultiDiGraph()
    
    for node in nodes:
        # add operation
        graph.add_node(node['id'], compute_time=0, output_info=[])
        for attr in node:
            graph.nodes[node['id']][attr] = node[attr]
        
        # add preceding data dependencies
        for parent in node['input_info']:
            # hack: randomly sampling a size if have multiple possibilities (see docstring)
            graph.add_edge(parent, node['id'], size=random.choice(graph.nodes[parent]['output_info']))
                           
        # add preceding control dependencies
        for parent in node['control_input']:
            graph.add_edge(parent, node['id'], size=0)
            
    if verbose:
        print(f'Num nodes: {len(graph.nodes)}')
        print(f'Num edges: {len(graph.edges)}')
        
    return graph


def ddls_graph_from_pbtxt_graph(pbtxt_graph: nx.MultiDiGraph, 
                                processor_type_profiled: str = 'A100', 
                                verbose: bool = False):
    '''
    Returns a directed multi-graph (i.e. can have multiple edges between
    a given parent and child node). Since is a multi-graph, each edge is
    a 3-tuple (u, v, k) where u is the parent node, v is the child node,
    and k is the index of the multi-edge used to distringuish between
    the multi-edges of a pair of nodes.
    
    Node attributes:
        compute_cost: {processor_type_profiled: Operation compute time (ms)}
        memory_cost: Operation memory size (B)
        
    Edge attributes:
        size: Size of tensor being transferred by dependency (B)
        
    Args:
        processor_type_profiled: Processor device type profiled to get the compute 
            cost of the operation(s) in the computation graph. 
    '''
    ddls_graph = nx.MultiDiGraph()
    
    if verbose:
        print('\n\n~~~ Adding Nodes ~~~')
    for node in pbtxt_graph.nodes:
        node_attrs = pbtxt_graph.nodes[node]
        if verbose:
            print(f'\npbtxt node {node} attrs:')
            print(node_attrs)
        
        ddls_graph.add_node(node,
                            compute_cost={processor_type_profiled: node_attrs['compute_cost'] if 'compute_cost' in node_attrs else 0},
                            memory_cost=node_attrs['memory_cost'] if 'memory_cost' in node_attrs else 0)
            
        if verbose:
            print(f'ddls node {node} attrs:')
            print(ddls_graph.nodes[node])
            
    if verbose:
        print('\n\n~~~ Adding Edges ~~~')
    for edge in pbtxt_graph.edges:
        edge_attrs = pbtxt_graph[edge[0]][edge[1]][edge[2]]
        if verbose:
            print(f'\npbtxt edge {edge} attrs:')
            print(edge_attrs)
            
        ddls_graph.add_edge(u_for_edge=edge[0],
                            v_for_edge=edge[1],
                            key=edge[2],
                            size=edge_attrs['size'] if 'size' in edge_attrs else 0)
        
        if verbose:
            print(f'ddls edge {edge} attrs:')
            print(ddls_graph[edge[0]][edge[1]][edge[2]])
            
    if verbose:
        print(f'Num nodes: {len(ddls_graph.nodes)}')
        print(f'Num edges: {len(ddls_graph.edges)}')
    
    return ddls_graph

def ddls_graph_from_pbtxt_file(file_path: str, 
                               processor_type_profiled: str,
                               verbose: bool = False):
    pbtxt_nodes = pbtxt_nodes_from_pbtxt_file(file_path, verbose=verbose)
    pbtxt_computation_graph = pbtxt_graph_from_pbtxt_nodes(pbtxt_nodes)
    return ddls_graph_from_pbtxt_graph(pbtxt_computation_graph, 
                                       processor_type_profiled=processor_type_profiled, 
                                       verbose=verbose)




class Stopwatch:
    def __init__(self):
        self.reset()

    def reset(self):
        self._time = 0

    def tick(self, tick=1):
        self._time += tick

    def time(self):
        return self._time

























