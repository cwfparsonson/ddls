import dgl
import torch.nn as nn
import torch


class MeanPool(nn.Module):
    '''
    mean-pooling layer for use in GNN. 

    Message-passing method:
        - node features are passed through a node-specific nn.Linear layer (self.node_layer)
            ==> intermediate node representation
        - edge features are passed through a edge-specific nn.Linear layer (self.edge_layer)
            ==> intermediate edge representation
        - messages are sent as the intermediate node representation concatenated with the
            intermediate edge representation along which the message is being passed
        
    Aggregation method:
        - each node creates an intermediate node and edge representation of itself
            (in this case the intermediate edge representation is just zeros)
        - these intermediate representations are concatenated (as they were for messages)
            so that each node has an effective 'message' containing information about
            itself
        - this 'message' is concatenated to the rest of the messages received by their 
            neighbour
        - this total set of messages is passed through a nn.Linear layer (self.reduce_layer)
            and is then mean-reduced to a single vector (which can/should be set as the new
            node feature for the next message passing phase)

    Arguments:
        in_features_node (int): dimension of the node features seen by this layer
        in_features_edge (int): dimension of the edge features seen by this layer
        out_features_msg (int): dimension of the message that will be sent between nodes
        out_features_reduce (int): dimension of the output embedding for each node
    '''

    def __init__(self,
                 in_features_node,
                 in_features_edge,
                 out_features_msg,
                 out_features_reduce):

        super(MeanPool, self).__init__()


        self.node_layer = nn.Linear(in_features_node,int(out_features_msg/2))
        self.edge_layer = nn.Linear(in_features_edge,int(out_features_msg/2))
        self.out_features_msg = out_features_msg
        self.reduce_layer = nn.Linear(out_features_msg,out_features_reduce)
        self.activation = nn.ReLU()

    
    def forward(self,graph):

        graph.update_all(
            message_func=self.mp_func,
            reduce_func=self.reduce_func
        )

        return graph.ndata['z']
    
    def mp_func(self,edges):

        #generate intermediate representations of node and edge features
        # print(edges.src['z'])
        # print(edges.data['z'])
        nodes = self.node_layer(edges.src['z'])
        edges = self.edge_layer(edges.data['z'])

        #concatenate intermediate node and edge features to create the message
        msg = torch.cat((nodes,edges),-1)

        return {'m':msg}

    def reduce_func(self,nodes):

        #generate intermediate representations of the receiving node's node and edge features
        local_node = self.node_layer(nodes.data['z'])

        #create padded edge feature for each self-node in the batch
        local_edge = torch.zeros((len(local_node),int(self.out_features_msg/2)))
        local_state = torch.cat((local_node,local_edge),-1)

        #reshape local representation so it matches dimensions of received messages (dimension extension)
        local_shape = local_state.shape
        local_state = torch.reshape(local_state,(local_shape[0],1,local_shape[1]))

        #combine local and received messages to create a single set of messages
        all_states = torch.cat((local_state,nodes.mailbox['m']),1)

        #embed all messages with a nn.Linear layer (self.reduce_layer)
        embedded = self.reduce_layer(all_states)

        #take element-wise mean of all embedded messages and return
        aggregated = torch.mean(embedded,dim=1)

        return {'z':aggregated}


class GNN(nn.Module):
    '''
    Simple GNN model.

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
        num_layers (int): number of message passing layers (including first and last)
        aggregator_type (str): reference to a particular type of message-passing/aggregation layer

    Arguments (forward):
        graph (DGL DiGraph): a DGL DiGraph, where nodes and edges have only a single feature vector 'z' 
    '''

    def __init__(self,
        config
    ):
    
        nn.Module.__init__(self)
        super(GNN, self).__init__()

        if config['aggregator_type'] == 'mean':
            agg = MeanPool

        self.layers = []
        #add first layer
        self.layers.append(agg(in_features_node=config['in_features_node'],
                                in_features_edge=config['in_features_edge'],
                                out_features_msg=config['out_features_msg'],
                                out_features_reduce=config['out_features_hidden']))

        #add hidden layers
        for _ in range(config['num_layers']-2):
            self.layers.append(agg(in_features_node=config['out_features_hidden'],
                                    in_features_edge=config['in_features_edge'],
                                    out_features_msg=config['out_features_msg'],
                                    out_features_reduce=config['out_features_hidden']))

        #add output layer
        self.layers.append(agg(in_features_node=config['out_features_hidden'],
                                in_features_edge=config['in_features_edge'],
                                out_features_msg=config['out_features_msg'],
                                out_features_reduce=config['out_features']))
        
        self.layers = nn.ModuleList(self.layers)

    def forward(self,graph):
        for layer in self.layers:
            #generate node embeddings    
            output = layer(graph)

            #set node features as embeddings
            graph.ndata['z'] = output

        return graph.ndata['z']
