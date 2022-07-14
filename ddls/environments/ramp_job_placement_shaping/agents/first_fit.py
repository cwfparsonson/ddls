class FirstFit:
    def __init__(self, name: str = 'random'):
        self.name = name

    def compute_action(self, obs, *args, **kwargs):
        valid_actions = obs['action_set'][obs['action_mask'].astype(bool)]
        if len(valid_actions) > 1:
            # mask out action = 0 (not placing job) since other valid actions are available
            action = valid_actions[1:][0]
        else:
            # only action = 0 (do not place job) is available
            action = valid_actions[0]
        return action
