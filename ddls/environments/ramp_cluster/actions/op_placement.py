from collections import defaultdict

class OpPlacement:
    def __init__(self,
                 action: dict):
        '''
        Args:
            action: Mapping of job_id -> operation_id -> worker_id.
        '''
        self.action = action

        self.job_ids, self.worker_ids, self.worker_to_ops = set(), set(), defaultdict(list)
        for job_id in action.keys():
            self.job_ids.add(job_id)
            for op_id in self.action[job_id].keys():
                self.worker_ids.add(self.action[job_id][op_id])
                self.worker_to_ops[self.action[job_id][op_id]].append({'op_id': op_id, 'job_id': job_id})

    def __str__(self):
        descr = ''
        for job_id in self.action.keys():
            descr += f'\nJob ID: {job_id}'
            for op_id in self.action[job_id].keys():
                worker_id = self.action[job_id][op_id]
                descr += f'\n\tOp ID {op_id} -> Worker ID {worker_id}'
        return descr


