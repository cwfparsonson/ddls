import numpy as np

class DepSchedule:
    def __init__(self,
                 schedule: dict):
        '''
        Args:
            schedule: Mapping of channel_id -> job_id -> dep_id -> priority.
        '''
        self.schedule = schedule 

    def __str__(self):
        descr = ''
        for channel_id in self.schedule.keys():
            descr += f'\n\tChannel ID: {channel_id}'
            jobs, deps, priorities = [], [], []
            for job_id in self.schedule[channel_id].keys():
                for dep_id, priority in self.schedule[channel_id][job_id].items():
                    jobs.append(job_id)
                    deps.append(dep_id)
                    priorities.append(priority)
            sorted_idxs = np.argsort(priorities)
            jobs, deps, priorities = np.array(jobs)[sorted_idxs], np.array(deps)[sorted_idxs], np.array(priorities)[sorted_idxs]
            for job, dep, priority in zip(jobs, deps, priorities):
                descr += f'\n\t\tJob ID {job} Dep ID {dep} -> Priority {priority}'
        return descr
