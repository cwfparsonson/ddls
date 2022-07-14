class LastFit:
    def __init__(self, name: str = 'random'):
        self.name = name

    def compute_action(self, obs, *args, **kwargs):
        return obs['action_set'][obs['action_mask'].astype(bool)][-1]
