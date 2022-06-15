

class Action:
    def __init__(self,
                 op_partition = None,
                 op_placement = None,
                 op_schedule = None,
                 dep_placement = None,
                 dep_schedule = None,
                 verbose = False):
        # gather actions which were made
        self.actions = {}
        if op_partition is not None:
            self.actions['op_partition'] = op_partition
        if op_placement is not None:
            self.actions['op_placement'] = op_placement
        if op_schedule is not None:
            self.actions['op_schedule'] = op_schedule
        if dep_placement is not None:
            self.actions['dep_placement'] = dep_placement
        if dep_schedule is not None:
            self.actions['dep_schedule'] = dep_schedule

        # find which job ids were successfully handled by ALL actions
        self.job_ids = set(set.intersection(*[action.job_ids for action in self.actions.values()]))
        print(f'action job ids: {self.job_ids}')

        # for each action, filter any job ids which were not handled by all actions
        for key, action in self.actions.items():
            self._filter_action(key, action)

    def _filter_action(self, key, action):
        '''Removes any job ids from action which were not also handled by all other actions.'''
        if key in {'op_partition', 'op_placement', 'dep_placement'}:
            action_job_ids = list(action.action.keys())
            for action_job_id in action_job_ids:
                if action_job_id not in self.job_ids:
                    del action.action[action_job_id]
        elif key in {'op_schedule', 'dep_schedule'}:
            for device_id in action.action.keys():
                action_job_ids = list(action.action[device_id].keys())
                for action_job_id in action_job_ids:
                    if action_job_id not in self.job_ids:
                        del action.action[device_id][action_job_id]
        else:
            raise Exception(f'Unrecognised action key {key}')

    def __str__(self):
        descr = ''
        for key, action in self.actions.items():
            descr += f'\n{key}:'
            descr += f'{action}'
        return descr
        

