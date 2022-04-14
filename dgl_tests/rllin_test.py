import argparse
import os

import ray
from ray import tune

from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env

from ray.rllib.utils.framework import try_import_tf, try_import_torch

from dummy_env import DummyEnv, DummyNetworkEnv
from rllib_model_test import GNNPolicy 

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

env_config = {
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

model_config = {
        'in_features_node':19,
        'in_features_edge':11,
        'out_features_msg':8,
        'out_features_hidden':16,
        'out_features':4,
        'in_features_graph':4,
        'out_features_graph':4,
        'num_layers':3,
        'aggregator_type':'mean'
    }

if __name__ == '__main__':

    ray.shutdown()
    ray.init()

    # register_env('dummy_env', lambda config: DummyEnv(config))
    register_env('dummy_network_env', lambda config: DummyNetworkEnv(config))
    ModelCatalog.register_custom_model('dummy_model', GNNPolicy)

    config = {
        'env':'dummy_network_env',
        'env_config':env_config,
        'model':{
            'fcnet_hiddens':[8],
            'fcnet_activation':'relu',
            'custom_model':'dummy_model',
            'custom_model_config':model_config
        },
        'framework':'torch',
        'num_workers':1,
    }



    stop = {
        'training_iteration':2
    }

    results = tune.run(
                        'PPO',
                        config=config,
                        stop=stop,
                        verbose=2
                    )