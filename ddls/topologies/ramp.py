from ddls.topologies.topology import Topology
from ddls.devices.channels.channel import Channel

import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
from typing import Union
    

class Ramp(Topology):
    def __init__(self,
                 num_communication_groups: int = 4,
                 num_racks_per_communication_group: int = 2,
                 num_servers_per_rack: int = 4,
                 num_channels: int = 1,
                 channel_bandwidth: int = int(1.25e9),
                 switch_reconfiguration_latency: Union[int, float] = 1.25e-6,
                 worker_io_latency: Union[int, float] = 100e-9):
        self.num_communication_groups = num_communication_groups
        self.num_racks_per_communication_group = num_racks_per_communication_group
        self.num_servers_per_rack = num_servers_per_rack
        self.num_channels = num_channels
        self.channel_bandwidth = channel_bandwidth
        self.switch_reconfiguration_latency = switch_reconfiguration_latency
        self.worker_io_latency = worker_io_latency
        
        self.graph = nx.Graph()
        self._build_topology()

    def _build_topology(self):
        # init nodes
        for comm_group_id in range(self.num_communication_groups):
            for rack_id in range(self.num_racks_per_communication_group):
                for server_id in range(self.num_servers_per_rack):
                    node = f'{comm_group_id}-{rack_id}-{server_id}'
                    self.graph.add_node(node, workers=dict())

        # init edges
        for u in self.graph.nodes:
            for v in self.graph.nodes:
                if u != v and not self.graph.has_edge(v, u) and not self.graph.has_edge(u, v):
                    self.graph.add_edge(u, v, channels={u: {v: {}}, v: {u: {}}})

        # init edge channels
        self.channel_id_to_channel = {}
        for link in self.graph.edges:
            u, v = link
            for channel_num in range(self.num_channels):
                # initialise separate link channel object for each direction
                channel_one = Channel(u, v, channel_num, channel_bandwidth=self.channel_bandwidth)
                self.graph[u][v]['channels'][channel_one.channel_id] = channel_one
                self.channel_id_to_channel[channel_one.channel_id] = channel_one

                channel_two = Channel(v, u, channel_num, channel_bandwidth=self.channel_bandwidth)
                self.graph[v][u]['channels'][channel_two.channel_id] = channel_two
                self.channel_id_to_channel[channel_two.channel_id] = channel_two
                
    def render(self, 
               label_node_names=False, 
               node_size=20,
               figsize=(10,10)):
        fig = plt.figure(figsize=figsize)
        
        pos = nx.circular_layout(self.graph)
        
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
