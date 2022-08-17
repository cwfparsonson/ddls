from ddls.ml_models.models.mean_pool import MeanPool

import torch

class GNN(torch.nn.Module):
    '''
    Simple GNN model. Stacks an aggregation and message passing module (e.g. MeanPool)
    num_rounds times and passes input graph throuugh stack to generate per-node embeddings, 

    Given input args, this model works effectively like a regular Sequential model
    (i.e. there is an input dimension, hidden output dimension and final output dimension).

    This model expects an input of a DGL graph where nodes and edges have a single feature
    denoted as 'z'. This is something that should be handled by preprocessing.

    Arguments (__init__):
        in_features_node (int): dimension of the node features seen by this layer
        in_features_edge (int): dimension of the edge features seen by this layer
        out_features_msg (int): dimension of the message that will be sent between nodes
        out_features_hidden (int): dimension of the embedding of the hidden layer(s)
        out_features_reduce (int): dimension of the embedding of the final layer
        num_rounds (int): number of message passing layers (including first and last)
        aggregator_type (str): reference to a particular type of message-passing/aggregation layer

    Arguments (forward):
        graph (DGL DiGraph): a DGL DiGraph, where nodes and edges have only a single feature vector 'z' 
    '''

    def __init__(self,
        config
    ):
        torch.nn.Module.__init__(self)
        super(GNN, self).__init__()

        self.config = config
        self.init_nn_modules(self.config)

    def init_nn_modules(self, config):
        if config['num_rounds'] < 2:
            raise Exception(f'num_rounds must be >= 2.')

        # init GNN aggregator used to aggregate nodes' and their neighbours' representations into per-node representations
        if config['aggregator_type'] == 'mean':
            agg = MeanPool
        else:
            raise Exception(f'Unrecognised aggregator type {config["aggregator_type"]}')

        # init list of message passing gnn layers
        self.layers = []

        # add first layer
        self.layers.append(agg(in_features_node=config['in_features_node'],
                                in_features_edge=config['in_features_edge'],
                                out_features_msg=config['out_features_msg'],
                                out_features_reduce=config['out_features_hidden'],
                                aggregator_activation=config['aggregator_activation'],
                                module_depth=config['module_depth'],
                                ))

        #add hidden layers
        for _ in range(config['num_rounds']-2):
            self.layers.append(agg(in_features_node=config['out_features_hidden'],
                                    in_features_edge=config['in_features_edge'],
                                    out_features_msg=config['out_features_msg'],
                                    out_features_reduce=config['out_features_hidden'],
                                    aggregator_activation=config['aggregator_activation'],
                                    module_depth=config['module_depth'],
                                    ))

        #add output layer
        self.layers.append(agg(in_features_node=config['out_features_hidden'],
                                in_features_edge=config['in_features_edge'],
                                out_features_msg=config['out_features_msg'],
                                out_features_reduce=config['out_features_node'],
                                aggregator_activation=config['aggregator_activation'],
                                module_depth=config['module_depth'],
                                ))
        
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self,graph):
        for layer in self.layers:
            #generate node embeddings    
            output = layer(graph)

            #set node features as embeddings
            graph.ndata['z'] = output

        return graph.ndata['z']
