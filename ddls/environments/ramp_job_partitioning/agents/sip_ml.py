class SiPML:
    def __init__(self, max_partitions_per_op: int = 2, name: str = 'sip_ml', **kwargs):
        '''Will always partition all jobs upto a statically fixed max_partitions_per_op number of times.'''
        self.max_partitions_per_op = max_partitions_per_op
        self.name = name

    def compute_action(self, obs, *args, **kwargs):
        valid_actions = obs['action_set'][obs['action_mask'].astype(bool)]
        if len(valid_actions) > 1:
            # partition up to max_partitions_per_op (env will partition less if min quantum is different)
            max_allowed_partitions_per_op = valid_actions[-1]
            action = min(self.max_partitions_per_op, max_allowed_partitions_per_op)
        else:
            # only action = 0 (do not place job) is available
            action = valid_actions[0]
        return action
