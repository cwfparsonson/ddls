class LastFit:
    def __init__(self, name: str = 'random'):
        self.name = name

    def compute_action(self, obs, *args, **kwargs):
        # print(f'action set: {obs["action_set"]} | action mask: {obs["action_mask"]}')
        valid_actions = obs['action_set'][obs['action_mask'].astype(bool)]
        # print(f'valid_actions: {valid_actions}')
        if len(valid_actions) > 1:
            # mask out action = 0 (not placing job) since other valid actions are available
            action = valid_actions[1:][-1]
        else:
            # only action = 0 (do not place job) is available
            action = valid_actions[0]
        return action
