import numpy as np

class JobSchedule:
    def __init__(self,
                 schedule: dict):
        '''
        Args:
            schedule: Mapping of worker_id -> job_id -> op_id -> priority.
        '''
        self.schedule = schedule 

    def __str__(self):
        descr = ''
        for worker_id in self.schedule.keys():
            descr += f'\n\tWorker ID: {worker_id}'
            jobs, ops, priorities = [], [], []
            for job_id in self.schedule[worker_id].keys():
                for op_id, priority in self.schedule[worker_id][job_id].items():
                    jobs.append(job_id)
                    ops.append(op_id)
                    priorities.append(priority)
            sorted_idxs = np.argsort(priorities)
            jobs, ops, priorities = np.array(jobs)[sorted_idxs], np.array(ops)[sorted_idxs], np.array(priorities)[sorted_idxs]
            for job, op, priority in zip(jobs, ops, priorities):
                descr += f'\n\t\tJob ID {job} Op ID {op} -> Priority {priority}'
        return descr



