import numpy as np
import random
import copy
import glob
import collections
from collections import defaultdict
import time
import json
import networkx as nx
import random
import pathlib
import math
import importlib
import torch
import dgl
from omegaconf import OmegaConf
from functools import reduce


def seed_stochastic_modules_globally(numpy_module,
                                     random_module,
                                     torch_module,
                                     # dgl_module,
                                     default_seed=0,
                                     numpy_seed=None, 
                                     random_seed=None,
                                     torch_seed=None,
                                     dgl_seed=None):
    '''Seeds any stochastic modules so get reproducible results.'''
    if numpy_seed is None:
        numpy_seed = default_seed
    if random_seed is None:
        random_seed = default_seed

    if numpy_seed is None:
        numpy_seed = default_seed
    if random_seed is None:
        random_seed = default_seed
    if torch_seed is None:
        torch_seed = default_seed
    if dgl_seed is None:
        dgl_seed = default_seed

    numpy_module.random.seed(numpy_seed)

    random_module.seed(random_seed)

    torch_module.manual_seed(torch_seed)
    torch_module.cuda.manual_seed(torch_seed)
    torch_module.cuda.manual_seed_all(torch_seed)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True

    # dgl.seed(dgl_seed)

# def seed_stochastic_modules_globally(default_seed=0, 
                                     # numpy_seed=None, 
                                     # random_seed=None,
                                     # torch_seed=None,
                                     # dgl_seed=None):
    # '''Seeds any stochastic modules so get reproducible results.'''
    # if numpy_seed is None:
        # numpy_seed = default_seed
    # if random_seed is None:
        # random_seed = default_seed

    # if numpy_seed is None:
        # numpy_seed = default_seed
    # if random_seed is None:
        # random_seed = default_seed
    # if torch_seed is None:
        # torch_seed = default_seed
    # if dgl_seed is None:
        # dgl_seed = default_seed

    # np.random.seed(numpy_seed)

    # random.seed(random_seed)

    # torch.manual_seed(torch_seed)
    # torch.cuda.manual_seed(torch_seed)
    # torch.cuda.manual_seed_all(torch_seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # # dgl.seed(dgl_seed)


class Sampler:
    def __init__(self, 
                 pool: list,
                 sampling_mode: str,
                 shuffle: bool = False, # whether or not to shuffle sampling pool when reset
                 automatically_change_ids: bool = True, # if True, when call reset(), will assume pool is pool of jobs and that should go through and change job IDs of each job so that do not duplicate job IDs when sample from reset pool
                 ):
        '''
        Args:
            sampling_mode ('replace', 'remove', 'remove_and_repeat')
        '''
        self.original_pool = pool
        self.sample_pool = copy.deepcopy(self.original_pool)
        self.sampling_mode = sampling_mode
        self.shuffle = shuffle
        self.automatically_change_ids = automatically_change_ids
        self.reset_counter = 0
        self.reset()
    
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
                self.reset()
        else:
            raise Exception(f'Unrecognised sampling_mode {self.sampling_mode}')
            
        return datum

    def __str__(self):
        descr = f'Original pool length: {len(self.original_pool)} | Current pool length: {len(self.sample_pool)}'
        descr += f' | Sampling mode: {self.sampling_mode}'
        return descr

    def __len__(self):
        return len(self.sample_pool)

    def reset(self):
        self.sample_pool = copy.deepcopy(self.original_pool)
        if self.automatically_change_ids:
            # assume pool is pool of jobs, and go through and adjust job IDs so do not duplicate on reset
            base_id = len(self.original_pool) * self.reset_counter
            for idx, job in enumerate(self.sample_pool):
                job.job_id = int(base_id + job.job_id)
                self.sample_pool[idx] = job
        if self.shuffle:
            random.shuffle(self.sample_pool)
        self.reset_counter += 1





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

def pipedream_graph_from_txt_file(file_path, shift_node_ids_by=0, verbose=False):
    '''
    Args:
        shift_node_ids_by (int): How much to shift all node IDs by. Use to e.g. 
            ensure node ids start at your desired root ID.
    '''
    graph = nx.DiGraph()

    f = list(open(file_path,'r'))

    nodes = []
    edges = []

    for line in f:

        line = line.split(' -- ')
        for idx, el in enumerate(line):
            line[idx] = el.split('\t')[-1]

        #if this line represents a node
        if len(line) > 2:

            node_features = {}

            #get node id
            node_id = str(int(line[0][4:]) + shift_node_ids_by)

            #get op id
            op_id = line[1].split('(')[0]

            node_features['type'] = op_id

            #get op details
            op_details = str(line[1].split(op_id)[1][1:-1]).split(', ')

            #get compute time and memory details
            comp_and_memory = line[2].split(', ')
            comp_memory_feats = ['forward','backward','activation','parameter']
            for i in range(len(comp_memory_feats)):
                # OLD
                # feat_val = comp_and_memory[i].split('=')[1].replace('\n', '')

                # NEW (works with pipedream translation computation graphs as well)
                feat_val = json.loads(comp_and_memory[i].split('=')[1].replace('\n', '').replace(';', ','))
                if isinstance(feat_val, list):
                    # HACK: Some of pipedream activation values are given as list, assume sum of this list is total activation size value for this operation
                    feat_val = np.sum(feat_val)

                node_features[comp_memory_feats[i]] = float(feat_val)

            nodes.append((node_id,node_features))
        else:

            src = int(line[0][4:]) + shift_node_ids_by
            dst = int(line[1][4:]) + shift_node_ids_by

            edges.append((str(src),str(dst))) #assume only 1 data channel for now

    # get initial graph
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph

def mirror_graph(graph):
    forward_graph = graph.copy()
    n = len(forward_graph.nodes())

    backward_nodes = []
    # for i in range(1,len(forward_graph.nodes())+1):
    for i in forward_graph.nodes():
        backward_node_id = str(2*n-(int(i)-1))

        # set extra attrs for backward node
        forward_graph.nodes[str(i)]['forward_node_id'] = i
        forward_graph.nodes[str(i)]['backward_node_id'] = None
        backward_nodes.append((backward_node_id,forward_graph.nodes[str(i)]))

        # update extra attrs for forward node
        forward_graph.nodes[str(i)]['forward_node_id'] = None
        forward_graph.nodes[str(i)]['backward_node_id'] = backward_node_id

    backward_edges = list(forward_graph.edges())
    for i in range(len(backward_edges)):
        backward_edges[i] = (str(2*n-(int(backward_edges[i][1])-1)),str(2*n-(int(backward_edges[i][0])-1)))

    backward_graph = nx.DiGraph()
    # for node, node_attrs in zip(nodes, nodes_attrs):
        # backward_graph.add_node(node, **node_attrs)
    backward_graph.add_nodes_from(backward_nodes)
    backward_graph.add_edges_from(backward_edges)

    return forward_graph, backward_graph

def combine_graphs(forward, backward):
    forward_nodes = list(forward.nodes())
    backward_nodes = list(backward.nodes())

    for i in list(forward.nodes()):
        forward.nodes[str(i)]['compute'] = forward.nodes[str(i)]['forward']
        forward.nodes[str(i)]['pass_type'] = 'forward_pass'
        del forward.nodes[str(i)]['forward']
        del forward.nodes[str(i)]['backward']

    for i in list(backward.nodes()):
        backward.nodes[str(i)]['compute'] = backward.nodes[str(i)]['backward']
        backward.nodes[str(i)]['pass_type'] = 'backward_pass'
        backward.nodes[str(i)]['forward_node_id'] = backward.nodes[str(i)]['forward_node_id']
        del backward.nodes[str(i)]['forward']
        del backward.nodes[str(i)]['backward']

    join_0, join_1 = max([int(nd) for nd in forward_nodes]),min([int(nd) for nd in backward_nodes])
    joined = nx.union(forward,backward)

    joined.add_edge(str(join_0),str(join_1))

    for edge in joined.edges():
        edge = tuple(edge)
        joined.edges[edge[0],edge[1]]['communication'] = joined.nodes[edge[0]]['activation']

    return joined

def ddls_graph_from_pipedream_graph(pipedream_graph,
                                    processor_type_profiled: str = 'A100',
                                    verbose: bool = False):
    if verbose:
        print('\n\n~~~ Original Pipedream Graph Nodes ~~~')
        for node in pipedream_graph.nodes:
            node_attrs = pipedream_graph.nodes[node]
            print(f'\npipedream node {node} attrs:')
            print(node_attrs)
    
    # get mirrored forward and backward computation graph
    forward, backward = mirror_graph(pipedream_graph)
    fb_pipedream_graph = combine_graphs(forward=forward, backward=backward)

    # init ddls graph
    ddls_graph = nx.MultiDiGraph()

    if verbose:
        print('\n\n~~~ Adding Nodes from F-B Pipedream Graph to DDLS Graph ~~~')
    for node in fb_pipedream_graph.nodes:
        node_attrs = fb_pipedream_graph.nodes[node]
        if verbose:
            print(f'\nF-B pipedream node {node} attrs:')
            print(node_attrs)
        
        node = int(node)
        ddls_graph.add_node(node,
                            compute_cost={processor_type_profiled: node_attrs['compute'] if 'compute' in node_attrs else 0},
                            memory_cost=node_attrs['activation'] + node_attrs['parameter'],
                            pass_type=node_attrs['pass_type'],
                            forward_node_id=node_attrs['forward_node_id'] if 'forward_node_id' in node_attrs else None,
                            backward_node_id=node_attrs['backward_node_id'] if 'backward_node_id' in node_attrs else None)
            
        if verbose:
            print(f'DDLS node {node} attrs:')
            print(ddls_graph.nodes[node])

    if verbose:
        print('\n\n~~~ Adding Edges from F-B Pipedream Graph to DDLS Graph ~~~')
    for edge in fb_pipedream_graph.edges:
        u, v = edge
        k = 0 # multi-graph key index hardcoded as 0
        edge_attrs = fb_pipedream_graph[u][v]
        if verbose:
            print(f'\nF-B pipedream edge {edge} attrs:')
            print(edge_attrs)
            
        u, v, k = int(u), int(v), int(k)
        ddls_graph.add_edge(u_for_edge=u,
                            v_for_edge=v,
                            key=k,
                            size=edge_attrs['communication'] if 'communication' in edge_attrs else 0)
        
        if verbose:
            print(f'DDLS edge {edge} attrs:')
            print(ddls_graph[u][v][k])
            
    if verbose:
        print(f'\nNum nodes: {len(ddls_graph.nodes)}')
        print(f'Num edges: {len(ddls_graph.edges)}')
    
    return ddls_graph
    


def ddls_graph_from_pipedream_txt_file(file_path: str,
                                       processor_type_profiled: str,
                                       verbose: bool = False):
    '''Assumes .txt file follows the convention of the pipedream .txt graph profiles.'''
    pipedream_computation_graph = pipedream_graph_from_txt_file(file_path, verbose=verbose)
    graph = ddls_graph_from_pipedream_graph(pipedream_computation_graph, 
                                           processor_type_profiled=processor_type_profiled, 
                                           verbose=verbose)
    graph.graph['file_path'] = file_path
    # graph.graph['graph_name'] = file_path.split('/')[-1].split('.')[0]
    return graph

def get_forward_graph(computation_graph):
    '''Removes nodes and edges from a graph which originally contains both the forward and backward pass.'''
    forward_graph = copy.deepcopy(computation_graph)
    for node in computation_graph.nodes():
        if computation_graph.nodes[node]['pass_type'] == 'backward_pass':
            forward_graph.remove_node(node)
    return forward_graph 

class Stopwatch:
    def __init__(self):
        self.reset()

    def reset(self):
        self._time = 0

    def tick(self, tick=1):
        self._time += tick

    def time(self):
        return self._time

def flatten_list(t):
    '''Flattens a list of lists t.'''
    return [item for sublist in t for item in sublist]

def flatten_numpy_array(a, dtype=object):
    return np.hstack(np.array(a, dtype=dtype).flatten())

def get_module_from_path(path):
    '''
    Path must be the path to the module **without** the .py extension.

    E.g. ddls.module_name
    '''
    return importlib.import_module(path)
     
def get_class_from_path(path):
    '''
    Path must be the path to the class **without** the .py extension.

    E.g. ddls.module_name.ModuleClass
    '''
    ClassName = path.split('.')[-1]
    path_to_class = '.'.join(path.split('.')[:-1])
    module = __import__(path_to_class, fromlist=[ClassName])
    return getattr(module, ClassName)

def get_function_from_path(path):
    module_path = '.'.join(path.split('.')[:-1])
    module = get_module_from_path(module_path)
    func = path.split('.')[-1]
    return getattr(module, func)

def gen_unique_experiment_folder(path_to_save, experiment_name):
    # init highest level folder
    path = path_to_save + '/' + experiment_name + '/'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    # init folder for this experiment
    path_items = glob.glob(path+'*')
    ids = sorted([int(el.split('_')[-1]) for el in path_items])
    if len(ids) > 0:
        _id = ids[-1] + 1
    else:
        _id = 0
    foldername = f'{experiment_name}_{_id}/'
    pathlib.Path(path+foldername).mkdir(parents=True, exist_ok=False)

    return path + foldername

def transform_with_log(val):
    return math.copysign(1, val) * math.log(1 + abs(val), 10)

def gen_channel_id(src, dst, channel_number):
    '''
    src and dst are the two server nodes between which the channel exists on a link, 
    channel_number is the global channel number of the channel.
    '''
    return f'src_{src}_dst_{dst}_channel_{channel_number}'

def init_nested_hash():
    return defaultdict(init_nested_hash)

def gen_job_dep_str(job_idx, job_id, dep_id):
    return json.dumps(job_idx) + '_' + json.dumps(job_id) + '_' + json.dumps(dep_id)

def load_job_dep_str(job_dep, conv_lists_to_tuples=True):
    job_idx, job_id, dep_id = [json.loads(i) for i in job_dep.split('_')]
    if isinstance(dep_id, list) and conv_lists_to_tuples:
        # is an edge dependency, convert to hashable type tuple as in networkx (json has no concept of tuples so was mistakenly json'd as list rather than tuple)
        dep_id = tuple(dep_id)
    return job_idx, job_id, dep_id

def recursively_instantiate_classes_in_hydra_config(d):
    for k, v in d.items():
        if isinstance(v, dict):
            recursively_instantiate_classes_in_hydra_config(v)
        else:
            hydra.utils.instantiate(d[k])

def recursively_update_nested_dict(orig_dict, overrides, verbose=False):
    if verbose:
        print(f'\nRecursively updating orig_dict {orig_dict} with overrides {overrides}')
    for key, val in overrides.items():
        if verbose:
            print(f'key: {key} | val: {type(val)} {val}')
        if key not in orig_dict:
            if verbose:
                print(f'key not in orig_dict, adding...')
            orig_dict[key] = val
        else:
            if isinstance(val, collections.Mapping):
                if verbose:
                    print(f'val is a Mapping, re-running recursion...')
                orig_dict[key] = recursively_update_nested_dict(orig_dict[key], val)
            else:
                if verbose:
                    print(f'val is not a Mapping, updating...')
                orig_dict[key] = val
    if verbose:
        print(f'Recursively updated orig_dict: {orig_dict}')
    return orig_dict



def get_nested_dict_keys_val(dictionary, *keys):
    return reduce(lambda d, key: d.get(key) if d else None, keys, dictionary)


def map_agent_id_to_hparams(base_folder, base_name, ids, hparams, verbose=True):
    # map IDs to parameters you want to look at
    id_to_hparams = defaultdict(lambda: defaultdict(lambda: None))
    for _id in ids:
        # load config
        path_to_config = glob.glob(f'{base_folder}/{base_name}/{base_name}_{_id}/*config.yaml')
        if len(path_to_config) != 1:
            raise Exception(f'Unable to locate a single *config.yaml file in {base_folder}/{base_name}/{base_name}_{_id}/, found {path_to_config}')
        else:
            path_to_config = path_to_config[0]
            if verbose:
                print(f'\nLoaded config for ID {_id} from {path_to_config}')
        config = OmegaConf.load(path_to_config)
        
        # get hparam value(s) for this id
        for hparam in hparams:
            keys = hparam.split('.')
            val = get_nested_dict_keys_val(config, *keys)
            if verbose:
                print(f'ID {_id} has hparam {hparam} value {val}')
            id_to_hparams[_id][hparam] = val

    return id_to_hparams


