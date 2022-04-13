from typing import Sequence
import gym
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict 
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as FC

from sage import GNN
import dgl
import numpy as np
import torch.nn as nn
from gym.spaces import Box

torch, nn = try_import_torch()

'''
General:
    - this implements an RLlib model that uses a GNN for demonstration
    - the model expects an obs with the following facets:
        -- 'node_features'
        -- 'edge_features'
        -- 'edges_src'
        -- 'edges_dst'
        -- 'graph_features'
    - this works with a simple network-based Env which should form
    an acceptable framework for how any Env built for DDL should interface
    (i.e. what do the returns of reset and step look like)

TODO (or consider):
    - understand the value function feature of rllib torch models better
        -- consider if better way or if fine as is (seems fine)
    - how to handle intra-batch differences in graph size
        -- Dict obs is batched (i.e. each feature of the Dict obs is stacked)
        -- this is done per-batch (e.g. feature with dim=10 and batch of 32 gives a
        [32,10] shape obs from input_dict)
        -- not sure how this would work if the graph (and therefore obs sizes) change
        with each reset
        -- GNN part is fine, but more about how to aggregate intra-graph node embeddings
        and also whether the rllib/gym observation batching supports this
        -- currently this demo env is just expecting the same size topology each time
        -- might be better to run through each graph separately and append logits
'''

class GNNPolicy(TorchModelV2, nn.Module):

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name
    ):

        nn.Module.__init__(self)
        super(GNNPolicy, self).__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name
        )
        
        self.config = model_config['custom_model_config']
        
        self.gnn = GNN(self.config)

        self.graph_layer = nn.Linear(self.config['in_features_graph'],self.config['out_features_graph'])

        self.logit_layer = FC(
            Box(-1,1,shape=(self.config['out_features_graph']+self.config['out_features'],)),
            action_space,
            2,
            model_config,
            name + "_logits"
        )

        

        self.initialising = True
        self.initialised = False

    def forward(self, input_dict, state, seq_lens):

        src_nodes = input_dict['obs']['edges_src']
        dst_nodes = input_dict['obs']['edges_dst']
        node_features = input_dict['obs']['node_features']
        edge_features = input_dict['obs']['edge_features']

        og_node_feature_shape = node_features.shape

        '''
        Note on initialisation section:

        rllib initialises by sending dummy sample observations using the lower
        limit of each observation facet.

        This means that for the dummy obs, the src and dst tensors are just zeros.

        This implies a 0-node graph when constructing from these features.  

        As such, in this case (i.e. during initialisation) we create an 
        all to all edge set and populate them with an arbitrary edge feature
        (the feature of the first edge in the dummy obs).

        This is only done when the model sees that the src nodes are all zero
        since this implies an initialisation phase (there are typically 3 for
        rllib).

        Otherwise it just uses the normal edge features etc. 
        '''
        self.initialising = (torch.sum(src_nodes) == 0)
        if self.initialising:
            n_nodes = node_features.shape[1]

            src_nodes_tmp = []
            dst_nodes_tmp = []
            for i in range(n_nodes-1):
                for j in range(i+1,n_nodes):
                    src_nodes_tmp.append(i)
                    dst_nodes_tmp.append(j)

            src_nodes_tmp = [src_nodes_tmp]*src_nodes.shape[0]
            dst_nodes_tmp = [dst_nodes_tmp]*dst_nodes.shape[0]

            src_nodes = torch.Tensor(src_nodes_tmp)
            dst_nodes = torch.Tensor(dst_nodes_tmp)

            edge_feature_shape = list(edge_features.shape)
            edge_features = edge_features[0][0].tolist()
            edge_features = torch.Tensor([edge_features]*edge_feature_shape[0]*edge_feature_shape[1])
        else:
            #reshape edge features if they're batched to make a single set (so can apply to a batched DGL graph)
            edge_features = torch.reshape(edge_features,(edge_features.shape[0]*edge_features.shape[1],edge_features.shape[2]))

        graphs = []

        for i in range(node_features.shape[0]):

            graph = dgl.graph((np.array(src_nodes[i]).astype(int),np.array(dst_nodes[i]).astype(int)))
            graphs.append(graph)

        graph = dgl.batch(graphs)

        node_features = torch.reshape(node_features,(node_features.shape[0]*node_features.shape[1],node_features.shape[2]))

        graph.ndata['z'] = torch.Tensor(node_features)
        graph.edata['z'] = torch.Tensor(edge_features)

        emb_nodes = self.gnn(graph)
        emb_nodes = torch.reshape(emb_nodes,
                                    (int(emb_nodes.shape[0]/og_node_feature_shape[1]),
                                    og_node_feature_shape[1],
                                    emb_nodes.shape[-1])
        )
        emb_nodes = torch.mean(emb_nodes,1)
        emb_graph = self.graph_layer(input_dict['obs']['graph_features'])
        
        #ignoring graph embedding in final embedding for now
        final_emb = torch.cat((emb_nodes,emb_graph),dim=1)

        logits, _ = self.logit_layer({
            'obs':final_emb
        })

        return logits, state

    def value_function(self):
        return self.logit_layer.value_function()