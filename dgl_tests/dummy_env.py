import numpy as np
import gym
from gym.spaces import Discrete, Box, Dict
import networkx as nx
import dgl
from netx_dgl_test import stack_features

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

        g_tmp = nx.complete_graph(env_config['num_nodes'])
        g = nx.DiGraph()
        g.add_nodes_from(g_tmp.nodes())
        g.add_edges_from(g_tmp.edges())

        for node in g.nodes():
            g.nodes[node]['feat_0'] = np.random.rand(env_config['node_features_shape']['feat_0'])
            g.nodes[node]['feat_1'] = np.random.rand(
                env_config['node_features_shape']['feat_1'][0],
                env_config['node_features_shape']['feat_1'][1]
            )

        for edge in g.edges():
            g.edges[edge]['feat_0'] = np.random.rand(env_config['edge_features_shape']['feat_0'])
            g.edges[edge]['feat_1'] = np.random.rand(
                env_config['edge_features_shape']['feat_1'][0],
                env_config['edge_features_shape']['feat_1'][1]
            )

        g = dgl.from_networkx(
            g,
            node_attrs=list(env_config['node_features_shape'].keys()),
            edge_attrs=list(env_config['edge_features_shape'].keys())
        )

        self.g = stack_features(g)

        edges_src = self.g.edges()[0].numpy().astype(np.float32)
        edges_dst = self.g.edges()[1].numpy().astype(np.float32)

        #create action and observation space based on these features
        self.action_space = Discrete(len(self.g.ndata))
        self.observation_space = Dict({
            'node_features':Box(0,1,shape=np.array(self.g.ndata['z']).shape),
            'edge_features':Box(0,1,shape=np.array(self.g.edata['z']).shape),
            'edges_src':Box(0,max(edges_src)+1,shape=edges_src.shape),
            'edges_dst':Box(0,max(edges_dst)+1,shape=edges_dst.shape)
        })

        self.dummy_obs = {
            'node_features':np.array(self.g.ndata['z']),
            'edge_features':np.array(self.g.edata['z']),
            'edges_src':edges_src,
            'edges_dst':edges_dst
        }

        # self.dummy_obs = np.ones((3,))
        # self.observation_space = Box(-1,1,shape=(3,))

    def reset(self):

        return self.dummy_obs

    def step(self,action):

        return self.dummy_obs, 1, False, {}

        # self.observation_space = Dict({
        #     'node_features':np.array(self.g.ndata),
        #     'edge_features':np.array(self.g.edata)
        # })

if __name__ == '__main__':

    # env_config = {
    #             'obs_0':{
    #                 'upper':-1,
    #                 'lower':1,
    #                 'dim':5
    #             },
    #             'obs_1':{
    #                 'dim':10
    #             }                        
    # }

    # env = DummyEnv(env_config)

    env_config = {
        'num_nodes':10,
        'node_features_shape':{
            'feat_0':1,
            'feat_1':(3,6)
        },
        'edge_features_shape':{
            'feat_0':1,
            'feat_1':(5,2)
        }
    }

    env = DummyNetworkEnv(env_config)
