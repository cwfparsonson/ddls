import networkx as nx
import copy


def data_split_node(graph, dp_splits: int = 0):

        '''goal: 
        - create n_splits copies of a given graph with no repeated node ids
        - do this before node splits, so that the same set of nodes splits can be applied to all of these graphs at once
        '''

        og_nodes = [int(node) for node in list(graph.nodes())]
        og_edges = [(int(edge[0]),int(edge[1])) for edge in graph.edges()] 
        highest_og_node = max(og_nodes)

        all_highest_nodes = []

        #currently assuming that data splitting doesn't require model size to change in any way
        new_features = [graph.nodes[node] for node in og_nodes]

        new_graph = nx.MultiDiGraph()

        for i in range(dp_splits+1):
            id_shift = i*highest_og_node
            new_nodes = [(str(og_nodes[j]+id_shift),new_features[j]) for j in range(len(og_nodes))]
            new_edges = [(str(edge[0]+id_shift),str(edge[1]+id_shift),0) for edge in og_edges]

            all_highest_nodes.append(str(highest_og_node+id_shift))

            new_graph.add_nodes_from(new_nodes)
            new_graph.add_edges_from(new_edges)

        edge_features = {}
        for edge in new_graph.edges:
            u, v, k = edge
            edge_features[edge] = {'size': new_graph.nodes[u]['memory_cost']}

        nx.set_edge_attributes(new_graph,edge_features)
        
        return new_graph

def model_split_node(_graph, mp_split_ids, mp_splits, dp_splits: int = 0):
    graph = copy.deepcopy(_graph)

    og_nodes = [int(node) for node in list(graph.nodes())]
    og_edges = [(int(edge[0]),int(edge[1])) for edge in graph.edges()]

    highest_og_node = max(og_nodes)

    in_edge_features = {}
    out_edge_features = {}
    
    #do for each op that should be split
    for i in range(len(mp_split_ids)):
        if str(mp_split_ids[i]) in graph.nodes:
            if graph.nodes[str(mp_split_ids[i])]['pass_type'] == 'forward_pass': # only consider forward pass then apply to backward pass simulatneously
                n_splits = mp_splits[i]
                #do across each data-parallel split graph
                for j in range(dp_splits+1):
                    node_ids = [
                                str(int(mp_split_ids[i])+(j*int(highest_og_node))), # forward op
                                str(int(highest_og_node) - (int(mp_split_ids[i])-1)+(j*int(highest_og_node))) # backward op
                               ]
                                
                    for k in range(len(node_ids)):
                        
                        node_id = node_ids[k]
                        
                        in_edges = [edge[0] for edge in graph.in_edges(node_id)]
                        out_edges = [edge[1] for edge in graph.out_edges(node_id)]

                        new_feature = {
                                       'compute_cost': {processor: compute_cost/n_splits for processor, compute_cost in graph.nodes[node_id]['compute_cost'].items()},
                                       'memory_cost': graph.nodes[node_id]['memory_cost']/n_splits,
                                       'pass_type': graph.nodes[node_id]['pass_type'],
                                      }

                        nodes = [('{}{}'.format(node_id,chr(97+k)),new_feature) for k in range(n_splits)]

                        new_edges = []
                        for node in nodes:
                            new_edges += [(edge,node[0],0) for edge in in_edges]
                            new_edges += [(node[0],edge,0) for edge in out_edges]

                            #account for edge features for incoming edges to split nodes
                            for edge in in_edges:
                                in_edge_features[(edge,node[0],0)] = {'size': graph.nodes[edge]['memory_cost']/len(nodes)}

                            #account for edge features for incoming edges to split nodes
                            for edge in out_edges:
                                out_edge_features[(node[0],edge,0)] = {'size': graph.nodes[edge]['memory_cost']/len(nodes)}

                        #extra step if on the back-prop for collective between all sub-ops to sync weights
                        if k == 1:
                            for l in range(len(nodes)):
                                for m in range(len(nodes)):
                                    if l == m:
                                        continue
                                    else:
                                        new_edges += [(nodes[l][0],nodes[m][0],0)]
                                        in_edge_features[(nodes[l][0],nodes[m][0],0)] = {'size': nodes[l][1]['memory_cost']}

                        graph.remove_node(node_id)
                        graph.add_nodes_from(nodes)
                        graph.add_edges_from(new_edges)
            
    nx.set_edge_attributes(graph,in_edge_features)
    nx.set_edge_attributes(graph,out_edge_features)

    return graph
