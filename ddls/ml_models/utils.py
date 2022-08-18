import networkx as nx
import random as rnd
import numpy as np
import dgl
import torch
import os
import subprocess
import pandas as pd
from io import StringIO

def stack_features(g,**kwargs):

    #################NODES#################

    node_stack = None

    if 'node_attrs' in kwargs.keys():
        node_feats = kwargs['node_attrs']
    else:
        node_feats = list(g.ndata.keys())

    for node_feat in node_feats:
        
        feat_shape = list(g.ndata[node_feat].shape)

        if len(feat_shape) > 2:
            new_dim = np.prod(feat_shape[1:])
        else:
            new_dim = 1

        g.ndata[node_feat] = torch.reshape(g.ndata[node_feat],(feat_shape[0],new_dim))

        if node_stack is None:
            node_stack = g.ndata[node_feat]
        else:
            node_stack = torch.cat((node_stack,g.ndata[node_feat]),dim=1)

    for node_feat in list(node_feats):
        del g.ndata[node_feat]

    g.ndata['z'] = node_stack.type(torch.FloatTensor)

    #################EDGES#################

    edge_stack = None

    if 'edge_attrs' in kwargs.keys():
        edge_feats = kwargs['edge_attrs']
    else:
        edge_feats = list(g.edata.keys())

    for edge_feat in edge_feats:
        
        feat_shape = list(g.edata[edge_feat].shape)

        if len(feat_shape) > 2:
            new_dim = np.prod(feat_shape[1:])
        else:
            new_dim = 1

        g.edata[edge_feat] = torch.reshape(g.edata[edge_feat],(feat_shape[0],new_dim))

        if edge_stack is None:
            edge_stack = g.edata[edge_feat]
        else:
            edge_stack = torch.cat((edge_stack,g.edata[edge_feat]),dim=1)

    for edge_feat in list(edge_feats):
        del g.edata[edge_feat]

    g.edata['z'] = edge_stack.type(torch.FloatTensor)

    return g
    

def pad_graph(num_nodes,node_split,edge_split,edges_src,edges_dst,node_features,edge_features):
        '''
        NOTE: the args to this function should be the standard observation variables
        before padding (e.g. node_features = DGLGraph.ndata['z'] for some DGLGraph
        that has been extracted from the simulation network with feature flattening
        already applied). num_nodes is just the upper limit of the number of nodes
        (which should have been supplied in the config).

        This function just adds zero padding onto the node and edge data that is passed
        into the rllib gnn-model, so that different sized graphs have the same sized
        observation and are compatible.
        '''

        max_nodes = num_nodes
        max_edges = int(num_nodes*(num_nodes-1)/2) #number of edges in a fully connected graph

        src_padding = torch.zeros((max_edges-len(edges_src),))
        dst_padding = torch.zeros((max_edges-len(edges_dst),))

        edges_src = torch.cat((edges_src,src_padding),dim=0).numpy().astype(np.float32)
        edges_dst = torch.cat((edges_dst,dst_padding),dim=0).numpy().astype(np.float32)

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

        return edges_src, edges_dst, node_features, edge_features, node_split, edge_split

def get_least_used_gpu():
    '''Returns the GPU index on the current server with the most available memory.'''
    # get devices visible to cuda
    cuda_visible_devices = [int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]

    # get string output of nvidia-smi memory query
    gpu_stats = subprocess.run(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"], stdout=subprocess.PIPE).stdout.decode('utf-8')

    # process query into StringIO object
    gpu_stats_2 = u''.join(gpu_stats)
    gpu_stats_3 = StringIO(gpu_stats_2)
    gpu_stats_3.seek(0)

    # read into dataframe
    gpu_df = pd.read_csv(gpu_stats_3,
                         names=['memory.used', 'memory.free'],
                         skiprows=1)

    # filter any devices not in cuda visible devices
    gpu_df = gpu_df[gpu_df.index.isin(cuda_visible_devices)]

    # get GPU with most free memory
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' MiB'))
    idx = int(gpu_df['memory.free'].astype(float).idxmax())

    return idx
