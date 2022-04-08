from typing import Sequence
import gym
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict 

from sage import GNN
import dgl

torch, nn = try_import_torch()

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

        self.initialising = True

    def forward(self, input_dict, state, seq_lens):
        
        #do something here to configure a dummy graph to allow initialisation passthrough
        #this is required since the dummy pass on init for rllib zeros all obs vectors, so there
        #it no real graph information being conveyed here (can just be a 2 node graph)
        self.initialising = False

        #implement the regular from-obs graph construction (probably will have some overlap with)
        #the previous method

        #do normal passthrough stuff with gnn
        
        pass