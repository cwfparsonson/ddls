import numpy as np

class OpSchedule:
    def __init__(self,
                 action: dict):
        '''
        Args:
            action: Mapping of worker_id -> job_id -> op_id -> priority.
        '''
        self.action = action 

        self.job_ids = set()
        for worker_id in self.action.keys():
            job_id = list(self.action[worker_id].keys())[0]
            self.job_ids.add(job_id)

    def __str__(self):
        descr = ''
        for worker_id in self.action.keys():
            descr += f'\n\tWorker ID: {worker_id}'
            jobs, ops, priorities = [], [], []
            for job_id in self.action[worker_id].keys():
                for op_id, priority in self.action[worker_id][job_id].items():
                    jobs.append(job_id)
                    ops.append(op_id)
                    priorities.append(priority)
            sorted_idxs = np.argsort(priorities)
            jobs, ops, priorities = np.array(jobs)[sorted_idxs], np.array(ops)[sorted_idxs], np.array(priorities)[sorted_idxs]
            for job, op, priority in zip(jobs, ops, priorities):
                descr += f'\n\t\tJob ID {job} Op ID {op} -> Priority {priority}'
        return descr



