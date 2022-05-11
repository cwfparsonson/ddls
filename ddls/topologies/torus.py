from ddls.topologies.topology import Topology
from ddls.devices.channels.channel import Channel

import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
    

class Torus(Topology):
    def __init__(self,
                 x_dims: int,
                 y_dims: int = 1,
                 z_dims: int = 1,
                 num_channels: int = 1,
                 channel_bandwidth: int = int(1.25e9)):
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.z_dims = z_dims
        self.num_channels = num_channels
        self.channel_bandwidth = channel_bandwidth
        
        self.graph = nx.Graph()
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
                
        # initialise node workers as being empty
        for node in self.graph.nodes:
            self.graph.nodes[node]['workers'] = dict()

        # initialise link channels
        for link in self.graph.edges:
            u, v = link
            for channel_num in range(self.num_channels):
                # initialise separate link channel object for each direction
                channel_one = Channel(u, v, channel_num, channel_bandwidth=self.channel_bandwidth)
                self.graph[u][v]['channels'][channel_one.channel_id] = channel_one
                channel_two = Channel(v, u, channel_num, channel_bandwidth=self.channel_bandwidth)
                self.graph[v][u]['channels'][channel_two.channel_id] = channel_two
            
    def _connect_nodes_in_dim(self, dim, dim_to_nodes):
        for idx in range(len(dim_to_nodes[dim][:-1])):
            u, v = dim_to_nodes[dim][idx], dim_to_nodes[dim][idx+1]
            self.graph.add_edge(u, v, channels={u: {v: {}}, v: {u: {}}})
        u, v = dim_to_nodes[dim][-1], dim_to_nodes[dim][0]
        self.graph.add_edge(u, v, channels={u: {v: {}}, v: {u: {}}})
                
    def render(self, 
               label_node_names=False, 
               node_size=20,
               figsize=(10,10)):
        fig = plt.figure(figsize=figsize)
        
        pos = nx.spring_layout(self.graph)
        
        node_labels = {}
        for node in self.graph.nodes:
            node_label = ''
            if label_node_names:
                node_label += node
            node_labels[node] = node_label
        
        nx.draw_networkx_nodes(self.graph,
                               pos,
                               label=node_labels,
                               node_size=node_size)
        nx.draw_networkx_edges(self.graph,
                               pos)
        
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels)
        
        plt.show()
