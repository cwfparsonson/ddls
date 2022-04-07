import numpy as np

class RandomJobPlacingAgent:
    def __init__(self, name: str = 'random'):
        self.name = name
    
    def select_action(self, obs):
        return np.random.choice(obs.action_set[obs.action_mask])
