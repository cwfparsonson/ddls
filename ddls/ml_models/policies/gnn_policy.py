from ddls.ml_models.models.gnn import GNN
from ddls.ml_models.utils import get_torch_module_from_str

from typing import Sequence, Union
import gym
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict 
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as FC

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
        name,
    ):
        '''
        Uses a custom GNN to parameterise an RLlib policy and output logits. 

        model_config:
            action_space_type ('continuous', 'discrete'): Whether the action space
                is continuous or discrete. If continuous, policy will output
                2*action_space.n logits, where outputting the mean and variance of a Gaussian
                distribution over each dimension of the action space. If discrete,
                policy will output action_space.n logits; one logit for each possible
                discrete action. Can then mask appropriately.
        '''
        # num_outputs = action_space.n # HACK
        nn.Module.__init__(self)
        super(GNNPolicy, self).__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name
        )

        
        self.config = model_config['custom_model_config']
        
        # init message passing GNN which will generate per-node embeddings using node and edge features
        self.gnn_module = GNN(self.config)

        # init a graph module which will generate a graph-level embedding using graph features
        if self.config['module_depth'] < 1:
            raise Exception(f'Require config[\"module_depth\"] >= 1.')
        self.activation_layer = get_torch_module_from_str(self.config['aggregator_activation'])
        # add input layer
        graph_module = [
                torch.nn.LayerNorm(self.config['in_features_graph'] + action_space.n),
                torch.nn.Linear(self.config['in_features_graph'] + action_space.n, self.config['out_features_graph']), 
                ]
        # add any extra layers
        for _ in range(self.config['module_depth']-1):
            graph_module.extend([
                    torch.nn.Linear(self.config['out_features_graph'], self.config['out_features_graph']),
                    self.activation_layer(),
                ])
        self.graph_module = torch.nn.Sequential(*graph_module)

        # # init a logits module which will generate logits (i.e. outputs for each possible action in a gym environment) using the concatenated node- and graph-level embeddings
        # if self.config['action_space_type'] == 'continuous':
            # num_logits = 2 * action_space.n
        # elif self.config['action_space_type'] == 'discrete':
            # num_logits = action_space.n
        # else:
            # raise Exception(f'Unrecognised model_config action_space_type {self.config["action_space_type"]}.')
        self.logit_module = FC(
            Box(-1,1,shape=(self.config['out_features_graph']+self.config['out_features_node'],)),
            action_space,
            # num_logits,
            num_outputs, # NEW HACK
            model_config,
            name + "_logits"
        )
        # print(f'action_space: {action_space.n} | num_logits: {num_logits} | logit_module: {self.logit_module}') # DEBUG
        # print(f'obs_space: {obs_space}')
        # print(f'action_space: {action_space}')
        # print(f'num_outputs: {num_outputs}')


        self.initialising = True
        self.initialised = False

    def forward(self, input_dict, state, seq_lens):

        src_nodes = input_dict['obs']['edges_src']
        dst_nodes = input_dict['obs']['edges_dst']
        node_features = input_dict['obs']['node_features']
        edge_features = input_dict['obs']['edge_features']
        node_splits = input_dict['obs']['node_split']
        edge_splits = input_dict['obs']['edge_split']

        og_node_feature_shape = node_features.shape

        # get device of tensors
        device = node_features.device

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
        '''
        If initialising (regardless of padding or not etc) create a fully connected graph
        to be used only for Rllib initialisation. This requires making each src/dst node 
        not have a value of 0 otherwise there are multiple node and edge features but 
        only a single node. This is not done for other scenarios, only the dummy 
        '''
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

            # src_nodes = torch.LongTensor(src_nodes_tmp).to(device)
            # dst_nodes = torch.LongTensor(dst_nodes_tmp).to(device)
            src_nodes = torch.LongTensor(src_nodes_tmp)
            dst_nodes = torch.LongTensor(dst_nodes_tmp)

            edge_feature_shape = list(edge_features.shape)
            edge_features = edge_features[0][0].tolist()
            # edge_features = torch.tensor([edge_features]*edge_feature_shape[0]*edge_feature_shape[1], device=device)
            edge_features = torch.tensor([edge_features]*edge_feature_shape[0]*edge_feature_shape[1])


            #if initialising, then just batch all fake graphs and do one big pass through
            #this is possible becuase no care has to be taken about taking the mean of the
            #node embeddings etc since they all have the same (maximum) size
            graphs = []
            for i in range(node_features.shape[0]):
                graph = dgl.graph((src_nodes[i],dst_nodes[i])).to(device)
                # graph = dgl.graph((src_nodes[i],dst_nodes[i]))
                graphs.append(graph)

            node_features = torch.reshape(node_features,(node_features.shape[0]*node_features.shape[1],node_features.shape[2]))

            graph = dgl.batch(graphs).to(device)
            # graph = dgl.batch(graphs)
            graph.ndata['z'] = node_features.to(device)
            graph.edata['z'] = edge_features.to(device)

            emb_nodes = self.gnn_module(graph)


            emb_nodes = torch.reshape(emb_nodes,
                                        (int(emb_nodes.shape[0]/og_node_feature_shape[1]),
                                        og_node_feature_shape[1],
                                        emb_nodes.shape[-1])
            )
            emb_nodes = torch.mean(emb_nodes,1)
        
        else:
            emb_nodes = []

            # print('node feat shape: {}'.format(n s))
            for batch_id in range(node_features.shape[0]):

                #chop off the padding nodes
                node_ft = node_features[batch_id]
                node_split = int(node_splits[batch_id])
                node_ft_reduced, node_pads = torch.split(node_ft,[node_split,node_ft.shape[0]-node_split],dim=0)

                #chop off the padding edges
                edge_ft = edge_features[batch_id]
                edge_split = int(edge_splits[batch_id])
                edge_ft_reduced, edge_pads = torch.split(edge_ft,[edge_split,edge_ft.shape[0]-edge_split],dim=0)

                src = src_nodes[batch_id]
                src, src_pads = torch.split(src,[edge_split,edge_ft.shape[0]-edge_split],dim=0)

                dst = dst_nodes[batch_id]
                dst, dst_pads = torch.split(dst,[edge_split,edge_ft.shape[0]-edge_split],dim=0)

                #construct a graph and get its embeddings
                graph = dgl.graph((src.cpu().numpy(),dst.cpu().numpy())).to(device)
                graph.ndata['z'] = node_ft_reduced
                graph.edata['z'] = edge_ft_reduced

                #take the element-wise mean of the embeddings in that graph
                embs = self.gnn_module(graph)
                emb_nodes.append(torch.mean(embs,0))

            emb_nodes = torch.stack(emb_nodes)

        #concatenate graph-averaged node embeddings and graph feature embeddings
        emb_graph = self.graph_module(input_dict['obs']['graph_features'])
        final_emb = torch.cat((emb_nodes,emb_graph),dim=1)

        #calculate logits/output from this final representation
        logits, _ = self.logit_module({
            'obs': final_emb
        })

        # TEMP DEBUG TODO: Commented below while debugging DQN
        # if self.config['action_space_type'] == 'discrete':
        if self.config['apply_action_mask']:
            # apply action masking; use inf action mask where invalid actioins have the smallest possible float value (so that will be 0 when RLLib applies softmax over logits and therefore will never be sampled) and 0 otherwise (so logit values will be unchanged)
            inf_mask = torch.maximum(
                                     torch.log(input_dict['obs']['action_mask']).to(device), 
                                     torch.tensor(torch.finfo(torch.float32).min).to(device)
                                    ).to(device)
            logits += inf_mask

        return logits, state

    def value_function(self):
        return self.logit_module.value_function()

    def get_q_values(self, *args, **kwargs):
        raise Exception(f'CALLING get_q_values IN GNN POLICY')

    def get_q_value_distributions(self, *args, **kwargs):
        raise Exception(f'CALLING get_q_value_distributions IN GNN POLICY')

