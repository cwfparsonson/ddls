from ddls.environments.ddls_observation import DDLSObservation

class JobPlacingAllNodesObservation(DDLSObservation):
    def reset(self):
        pass

    def extract(self, cluster, done):
        '''
        Return features of each node in computation graph.

        Each node should have the following features:
            valid_actions (one-hot vector): Which machines the op can be placed on.
            compute_cost (float): Compute cost of op normalised by compute cost of
                node with highest compute cost.
            memory_cost (float): Memory cost of op normalised by memory cost of
                node with highest memory cost.
            is_highest_compute_cost (binary): If the op has the highest compute cost
                of all nodes.
            is_highest_memory_cost (binary): If the op has the highest memory cost
                of all nodes.
            parents_compute_cost_sum (float): Sum of parent(s) compute cost normalised
                by compute cost of node with highest compute cost.
            children_compute_cost_sum (float): Sum of child(s) compute cost normalised
                by compute cost of node with highest compute cost.
            parents_memory_cost_sum (float): Sum of parent(s) memory cost normalised
                by memory cost of node with highest memory cost.
            children_memory_cost_sum (float): Sum of child(s) memory cost normalised
                by memory cost of node with highest memory cost.
            
        '''
        pass
