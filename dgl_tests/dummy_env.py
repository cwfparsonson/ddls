import numpy as np
import gym
from gym.spaces import Discrete, Box, Dict
import networkx as nx
import dgl
from netx_dgl_test import stack_features
import random
import torch

class DummyEnv(gym.Env):

    def __init__(self, env_config):

        super(DummyEnv, self).__init__()

        self.observation_space = Dict({
            "obs_0":Box(-1,1,shape=(5,)),
            "obs_1":Discrete(10)
        })

        self.action_space = Discrete(8)


        self.dummy_obs = {
            "obs_0":self.observation_space['obs_0'].sample(),
            "obs_1":self.observation_space['obs_1'].sample()
        }
    
    def reset(self):
        
        return self.dummy_obs

    def step(self,action):
        
        return self.dummy_obs, 1, False, {}

class DummyNetworkEnv(gym.Env):

    def __init__(self, env_config):

        super(DummyNetworkEnv, self).__init__()
        
        #init graph and convert to DGL with single features
        self.config = env_config
        self.reset()

    def reset(self):
        

        # # g_tmp = nx.complete_graph(self.config['num_nodes'])
        # num_nodes = random.randint(1,self.config['num_nodes'])
        # print('NUMBER OF NODES: {}'.format(num_nodes))
        # g_tmp = nx.complete_graph(num_nodes)
        # g = nx.DiGraph()
        # g.add_nodes_from(g_tmp.nodes())
        # g.add_edges_from(g_tmp.edges())

        # for node in g.nodes():
        #     g.nodes[node]['feat_0'] = np.random.rand(self.config['node_features_shape']['feat_0'])
        #     g.nodes[node]['feat_1'] = np.random.rand(
        #         self.config['node_features_shape']['feat_1'][0],
        #         self.config['node_features_shape']['feat_1'][1]
        #     )

        # for edge in g.edges():
        #     g.edges[edge]['feat_0'] = np.random.rand(self.config['edge_features_shape']['feat_0'])
        #     g.edges[edge]['feat_1'] = np.random.rand(
        #         self.config['edge_features_shape']['feat_1'][0],
        #         self.config['edge_features_shape']['feat_1'][1]
        #     )

        # g = dgl.from_networkx(
        #     g,
        #     node_attrs=list(self.config['node_features_shape'].keys()),
        #     edge_attrs=list(self.config['edge_features_shape'].keys())
        # )

        # self.g = stack_features(g)

        # edges_src = self.g.edges()[0].numpy().astype(np.float32)
        # edges_dst = self.g.edges()[1].numpy().astype(np.float32)

        # create action and observation space based on these features
        # self.action_space = Discrete(len(self.g.ndata))

        g = self._get_unpadded_graph()
        edges_src, edges_dst, node_features, edge_features, node_split, edge_split = self._pad_graph(g)

        self.action_space = Box(0,1,shape=(1,))
        self.observation_space = Dict({
            'node_features':Box(0,1,shape=np.array(node_features).shape),
            'edge_features':Box(0,1,shape=np.array(edge_features).shape),
            'graph_features':Box(0,1,shape=(self.config['graph_features'],)),
            'edges_src':Box(-2,self.config['num_nodes']+1,shape=edges_src.shape),
            'edges_dst':Box(-2,self.config['num_nodes']+1,shape=edges_dst.shape),
            'node_split':Box(0,self.config['num_nodes']+1,shape=(1,)),
            'edge_split':Box(0,self.config['num_nodes']*(self.config['num_nodes']+1)/2,shape=(1,))
        })

        self.dummy_obs = {
            'node_features':np.array(node_features),
            'edge_features':np.array(edge_features),
            'graph_features':np.random.rand(self.config['graph_features']).astype(np.float32),
            'edges_src':edges_src,
            'edges_dst':edges_dst,
            'node_split':np.array([node_split]).astype(np.float32),
            'edge_split':np.array([edge_split]).astype(np.float32)
        }

        #if loading a new graph with different size, need to re-define the obs spaces (right?)
        # print('reset obs: {}'.format(self.dummy_obs))
        return self.dummy_obs

    def _get_unpadded_graph(self):

        # g_tmp = nx.complete_graph(self.config['num_nodes']-5)
        num_nodes = random.randint(2,self.config['num_nodes'])
        # print('NUMBER OF NODES: {}'.format(num_nodes))
        g_tmp = nx.complete_graph(num_nodes)
        g = nx.DiGraph()
        g.add_nodes_from(g_tmp.nodes())
        g.add_edges_from(g_tmp.edges())

        for node in g.nodes():
            g.nodes[node]['feat_0'] = np.random.rand(self.config['node_features_shape']['feat_0'])
            g.nodes[node]['feat_1'] = np.random.rand(
                self.config['node_features_shape']['feat_1'][0],
                self.config['node_features_shape']['feat_1'][1]
            )

        for edge in g.edges():
            g.edges[edge]['feat_0'] = np.random.rand(self.config['edge_features_shape']['feat_0'])
            g.edges[edge]['feat_1'] = np.random.rand(
                self.config['edge_features_shape']['feat_1'][0],
                self.config['edge_features_shape']['feat_1'][1]
            )

        g = dgl.from_networkx(
            g,
            node_attrs=list(self.config['node_features_shape'].keys()),
            edge_attrs=list(self.config['edge_features_shape'].keys())
        )

        g = stack_features(g)

        return g

    def _pad_graph(self,g):

        node_split = len(g.nodes())
        edge_split = len(g.edges()[0])
        edges_src = g.edges()[0]
        edges_dst = g.edges()[1]
        node_features = g.ndata['z']
        edge_features = g.edata['z']

        max_nodes = self.config['num_nodes']
        max_edges = int(self.config['num_nodes']*(self.config['num_nodes']-1)/2) #number of edges in a fully connected graph

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





    def step(self,action):

        return self.dummy_obs, -1, True, {}

        # self.observation_space = Dict({
        #     'node_features':np.array(self.g.ndata),
        #     'edge_features':np.array(self.g.edata)
        # })

if __name__ == '__main__':

    # self.config = {
    #             'obs_0':{
    #                 'upper':-1,
    #                 'lower':1,
    #                 'dim':5
    #             },
    #             'obs_1':{
    #                 'dim':10
    #             }                        
    # }

    # env = DummyEnv(self.config)

    config = {
        'num_nodes':10,
        'node_features_shape':{
            'feat_0':1,
            'feat_1':(3,6)
        },
        'edge_features_shape':{
            'feat_0':1,
            'feat_1':(5,2)
        },
        'graph_features':4
    }

    env = DummyNetworkEnv(config)
