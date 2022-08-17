class MinParallelism:
    def __init__(self, name: str = 'min_parallelism'):
        self.name = name

    def compute_action(self, obs, *args, **kwargs):
        valid_actions = obs['action_set'][obs['action_mask'].astype(bool)]
        if len(valid_actions) > 2:
            # min number of times can partition each op is 2x
            action = 2
        elif len(valid_actions) == 2:
            # can either no allocate job or run job sequentially (no parallelism)
            action = 1
        else:
            # only action = 0 (do not place job) is available
            action = 0
        return action
