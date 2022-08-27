class NoParallelism:
    def __init__(self, name: str = 'no_parallelism', **kwargs):
        self.name = name

    def compute_action(self, obs, *args, **kwargs):
        valid_actions = obs['action_set'][obs['action_mask'].astype(bool)]
        if len(valid_actions) > 1:
            # do not partition job (i.e. max num op partitions = 1)
            action = 1
        else:
            # only action = 0 (do not place job) is available
            action = 0
        return action
