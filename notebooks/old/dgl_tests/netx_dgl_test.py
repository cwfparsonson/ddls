import networkx as nx
import random as rnd
import numpy as np
import dgl
import torch
from sage import MeanPool, GNN

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
    

if __name__ == '__main__':
    g_tmp = nx.complete_graph(10)
    g = nx.DiGraph()
    g.add_nodes_from(g_tmp.nodes())
    g.add_edges_from(g_tmp.edges())

    

    for node in g.nodes():
        g.nodes[node]['feat_0'] = np.random.rand(1)
        g.nodes[node]['feat_1'] = np.random.rand(2,5)

    for edge in g.edges():
        g.edges[edge]['feat_0'] = np.random.rand(1)
        g.edges[edge]['feat_1'] = np.random.rand(3,6)

    g = dgl.from_networkx(g,node_attrs=['feat_0','feat_1'],edge_attrs=['feat_0','feat_1'])
    
    g = stack_features(g)

    print(f"EDGES: {g.edges()[0].numpy().shape}")
    # print(g)

    # model_config = {
    #     'in_features_node':11,
    #     'in_features_edge':19,
    #     'out_features_msg':8,
    #     'out_features_hidden':16,
    #     'out_features':4,
    #     'num_layers':1
    # }

    # gnn = GNN(in_features_node=11,
    #         in_features_edge=19,
    #         out_features_msg=8,
    #         out_features_hidden=16,
    #         out_features=4,
    #         num_layers=1)
    # print(gnn(g))