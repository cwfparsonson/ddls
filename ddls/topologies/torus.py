from ddls.topologies.topology import Topology

import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
    

class Torus(Topology):
    def __init__(self,
                 x_dims: int,
                 y_dims: int = 1,
                 z_dims: int = 1,
                 num_channels: int = 1,
                 channel_capacity: int = int(1.25e9)):
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.z_dims = z_dims
        self.num_channels = num_channels
        self.channel_capacity = channel_capacity
        
        self.topology = nx.Graph()
        self._build_topology()
            
    def _build_topology(self):
        # init nodes
        x_dim_to_nodes, y_dim_to_nodes, z_dim_to_nodes = defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: [])
        for x in range(1, self.x_dims+1):
            for y in range(1, self.y_dims+1):
                for z in range(1, self.z_dims+1):
                    node = f'{x}-{y}-{z}'
                    x_dim_to_nodes[x].append(node)
                    y_dim_to_nodes[y].append(node)
                    z_dim_to_nodes[z].append(node)
                        
        # for each y_dim (row), connect along x-axis
        for y_dim in range(1, self.y_dims+1):
            self._connect_nodes_in_dim(y_dim, y_dim_to_nodes)
            
        if self.y_dims > 1:
            # for each x_dim (col), connect along y-axis
            for x_dim in range(1, self.x_dims+1):
                self._connect_nodes_in_dim(x_dim, x_dim_to_nodes)
                
        if self.z_dims > 1:
            # for each z_dim, connect along x-axis
            for z_dim in range(1, self.z_dims+1):
                self._connect_nodes_in_dim(z_dim, z_dim_to_nodes)
                
        self._init_link_channels()
        
        # initialise node devices as being empty
        for node in self.topology.nodes:
            self.topology.nodes[node]['device'] = None
            
    def _init_link_channels(self):
        '''Initialise link channels where each direction has 50% of total channel capacity.'''
        channel_names = [f'channel_{channel}' for channel in range(self.num_channels)]
        for link in self.topology.edges:
            self.topology.edges[link[0], link[1]][f'{link[0]}_to_{link[1]}'] = {channel: self.channel_capacity/2 for channel in channel_names}
            self.topology.edges[link[1], link[0]][f'{link[1]}_to_{link[0]}'] = {channel: self.channel_capacity/2 for channel in channel_names}
        self.topology.graph['channel_names'] = channel_names
        self.topology.graph['channel_capacity'] = self.channel_capacity
                         
    def _connect_nodes_in_dim(self, dim, dim_to_nodes):
        for idx in range(len(dim_to_nodes[dim][:-1])):
            self.topology.add_edge(dim_to_nodes[dim][idx], dim_to_nodes[dim][idx+1])
        self.topology.add_edge(dim_to_nodes[dim][-1], dim_to_nodes[dim][0])
                
    def render(self, 
               label_node_names=False, 
               label_node_devices=False,
               node_size=20,
               figsize=(10,10)):
        fig = plt.figure(figsize=figsize)
        
        pos = nx.spring_layout(self.topology)
        
        node_labels = {}
        for node in self.topology.nodes:
            node_label = ''
            if label_node_names:
                node_label += node
            if label_node_devices:
                if self.topology.nodes[node]['device'] is None:
                    node_label += 'None'
                else:
                    node_label += str(self.topology.nodes[node]['device'])
            node_labels[node] = node_label
        
        nx.draw_networkx_nodes(self.topology,
                               pos,
                               label=node_labels,
                               node_size=node_size)
        nx.draw_networkx_edges(self.topology,
                               pos)
        
        nx.draw_networkx_labels(self.topology, pos, labels=node_labels)
        
        plt.show()
