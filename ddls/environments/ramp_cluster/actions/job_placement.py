from collections import defaultdict

class JobPlacement:
    def __init__(self,
                 placement: dict):
        '''
        Args:
            placement: Mapping of job_id -> operation_id -> worker_id.
        '''
        self.placement = placement

        self.job_ids, self.worker_ids, self.worker_to_ops = set(), set(), defaultdict(list)
        for job_id in placement.keys():
            self.job_ids.add(job_id)
            for op_id in self.placement[job_id].keys():
                self.worker_ids.add(self.placement[job_id][op_id])
                self.worker_to_ops[self.placement[job_id][op_id]].append({'op_id': op_id, 'job_id': job_id})

    def __str__(self):
        descr = ''
        for job_id in self.placement.keys():
            descr += f'\n\tJob ID: {job_id}'
            for op_id in self.placement[job_id].keys():
                worker_id = self.placement[job_id][op_id]
                descr += f'\n\t\tOp ID {op_id} -> Worker ID {worker_id}'
        return descr



