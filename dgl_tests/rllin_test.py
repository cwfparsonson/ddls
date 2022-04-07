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
        }
    }

if __name__ == '__main__':

    ray.shutdown()
    ray.init()

    # register_env('dummy_env', lambda config: DummyEnv(config))
    register_env('dummy_env', lambda config: DummyNetworkEnv(config))

    config = {
        'env':'dummy_env',
        'env_config':env_config,
        'model':{
            'fcnet_hiddens':[8],
            'fcnet_activation':'relu'
        },
        'framework':'torch',
        'num_workers':1
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