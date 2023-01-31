from ddls.ml_models.utils import get_torch_module_from_str

import torch

class MeanPool(torch.nn.Module):
    '''
    mean-pooling layer for use in GNN. Does message passing and aggregation to generate per-node embeddings.

    Message-passing method:
        - node features are passed through a node-specific torch.nn.Linear layer (self.node_layer)
            ==> intermediate node representation
        - edge features are passed through a edge-specific torch.nn.Linear layer (self.edge_layer)
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
        - this total set of messages is passed through a torch.nn.Linear layer (self.reduce_layer)
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
                 out_features_reduce,
                 aggregator_activation='leaky_relu',
                 module_depth=1,
                 **kwargs):

        super(MeanPool, self).__init__()

        if module_depth < 1:
            raise Exception(f'Require module_depth >= 1.')

        self.activation_layer = get_torch_module_from_str(aggregator_activation)

        # NODE MODULE
        # add input layer
        node_module = [
                torch.nn.LayerNorm(in_features_node), 
                torch.nn.Linear(in_features_node, int(out_features_msg / 2)),
                self.activation_layer(),
                ]
        # add any extra layers
        for _ in range(module_depth-1):
            node_module.extend([
                    torch.nn.Linear(int(out_features_msg / 2), int(out_features_msg / 2)),
                    self.activation_layer(),
                ])
        self.node_module = torch.nn.Sequential(*node_module)

        # EDGE MODULE
        # add input layer
        edge_module = [
                torch.nn.LayerNorm(in_features_edge), 
                torch.nn.Linear(in_features_edge, int(out_features_msg / 2)),
                self.activation_layer(),
                ]
        # add any extra layers
        for _ in range(module_depth-1):
            edge_module.extend([
                    torch.nn.Linear(int(out_features_msg / 2), int(out_features_msg / 2)),
                    self.activation_layer(),
                ])
        self.edge_module = torch.nn.Sequential(*edge_module)

        # REDUCE MODULE
        # add input layer
        reduce_module = [
                torch.nn.LayerNorm(out_features_msg), 
                torch.nn.Linear(out_features_msg, out_features_reduce),
                self.activation_layer(),
                ]
        # add any extra layers
        for _ in range(module_depth-1):
            reduce_module.extend([
                    torch.nn.Linear(out_features_reduce, out_features_reduce),
                    self.activation_layer(),
                ])
        self.reduce_module = torch.nn.Sequential(*reduce_module)

        self.out_features_msg = out_features_msg
        
    def to(self, device):
        '''Mount torch model(s) onto device.'''
        self.node_module = self.node_module.to(device)
        self.edge_module = self.edge_module.to(device)
        self.reduce_module = self.reduce_module.to(device)
        # self.activation_layer = self.activation_layer.to(device)
    
    def forward(self,graph):
        self.to(graph.device)

        graph.update_all(
            message_func=self.mp_func,
            reduce_func=self.reduce_func
        )

        return graph.ndata['z']
    
    def mp_func(self,edges):
        #generate intermediate representations of node and edge features
        nodes = self.node_module(edges.src['z'])
        edges = self.edge_module(edges.data['z'])

        #concatenate intermediate node and edge features to create the message
        msg = torch.cat((nodes,edges),-1)

        return {'m':msg}

    def reduce_func(self,nodes):
        device = nodes.data['z'].device 

        #generate intermediate representations of the receiving node's node and edge features
        local_node = self.node_module(nodes.data['z'])

        #create padded edge feature for each self-node in the batch
        local_edge = torch.zeros((len(local_node),int(self.out_features_msg/2)), device=device)
        local_state = torch.cat((local_node,local_edge),-1)

        #reshape local representation so it matches dimensions of received messages (dimension extension)
        local_shape = local_state.shape
        local_state = torch.reshape(local_state,(local_shape[0],1,local_shape[1]))

        #combine local and received messages to create a single set of messages
        all_states = torch.cat((local_state,nodes.mailbox['m']),1)

        #embed all messages with a torch.nn.Linear layer (self.reduce_layer)
        embedded = self.reduce_module(all_states)

        #take element-wise mean of all embedded messages and return
        aggregated = torch.mean(embedded,dim=1)

        return {'z':aggregated}
