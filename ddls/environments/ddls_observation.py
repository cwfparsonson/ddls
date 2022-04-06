import numpy as np

class DDLSObservation:
    def __init__(self,
                 node_features: np.array = None,
                 edge_features: np.array = None,
                 global_features: np.array = None,
                 action_set: np.array = None,
                 action_mask: np.array = None):
        self._action_set = action_set
        self._action_mask = action_mask
        self._node_features = node_features
        self._edge_features = edge_features
        self._global_features = global_features

    def __str__(self):
        node_info = 'Node feats:'
        if self.node_features is not None:
            node_info += f' # nodes: {self.node_features.shape[0]}'
            node_info += f' | # feats per node: {self.node_features.shape[1]}'
            node_info += f' | # flattened feats per node: {np.hstack(self.node_features[0]).shape[0]}'
        else:
            node_info += ' None'

        edge_info = ' || Edge feats:'
        if self.edge_features is not None:
            edge_info += f' # edges: {self.edge_features.shape[0]}'
            edge_info += f' | # feats per edge: {self.edge_features.shape[1]}'
            edge_info += f' | # flattened feats per edge: {np.hstack(self.edge_features[0]).shape[0]}'
        else:
            edge_info += ' None'

        global_info = ' || Global feats:'
        if self.global_features is not None:
            global_info += f' # global features: {self.global_features.shape[0]}'
            global_info += f' | # flattened global feats: {np.hstack(self.global_features).shape[0]}'
        else:
            global_info += ' None'

        action_info = ' || Action info:'
        if self.action_set is not None:
            action_info += f' action space: {self.action_set.shape}'
            if self.action_mask is not None:
                action_info  += f' | # valid candidate actions: {len(self.action_set[self.action_mask])}'

        return node_info + edge_info + global_info + action_info

    @property
    def node_features(self):
        return self._node_features

    @node_features.setter
    def node_features(self, node_features: np.array):
        self._node_features = node_features

    @property
    def edge_features(self):
        return self._edge_features

    @edge_features.setter
    def edge_features(self, edge_features: np.array):
        self._edge_features = edge_features

    @property
    def action_set(self):
        return self._action_set

    @action_set.setter
    def action_set(self, action_set: np.array):
        self._action_set = action_set

    @property
    def action_mask(self):
        return self._action_mask

    @action_mask.setter
    def action_mask(self, action_mask: np.array):
        self._action_mask = action_mask

    @property
    def global_features(self):
        return self._global_features

    @global_features.setter
    def global_features(self, global_features: np.array):
        self._global_features = global_features








