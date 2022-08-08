import numpy as np

class DepSchedule:
    def __init__(self,
                 action: dict):
        '''
        Args:
            action: Mapping of channel_id -> job_id -> dep_id -> priority.
        '''
        self.action = action 

        self.job_ids = set()
        for channel_id in self.action.keys():
            job_id = list(self.action[channel_id].keys())[0]
            self.job_ids.add(job_id)

    def __str__(self):
        descr = ''
        for channel_id in self.action.keys():
            descr += f'\nChannel ID: {channel_id}'
            jobs, deps, priorities = [], [], []
            for job_id in self.action[channel_id].keys():
                for dep_id, priority in self.action[channel_id][job_id].items():
                    jobs.append(job_id)
                    deps.append(dep_id)
                    priorities.append(priority)
            sorted_idxs = np.argsort(priorities)
            jobs, deps, priorities = np.array(jobs)[sorted_idxs], np.array(deps)[sorted_idxs], np.array(priorities)[sorted_idxs]
            for job, dep, priority in zip(jobs, deps, priorities):
                descr += f'\n\tJob ID {job} Dep ID {dep} -> Priority {priority}'
        return descr
