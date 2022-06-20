
class JobPlacementShape:
    def __init__(self,
                 action: dict):
        '''
        Args:
            action: Mapping of job_id -> job_placement_shape
        '''
        self.action = action

        self.job_ids = set()
        for job_id in action.keys():
            self.job_ids.add(job_id)

    def __str__(self):
        descr = ''
        for job_id, meta_shape in self.action.items():
            descr += f'\nJob ID: {job_id}\n\tMeta block job placement shape: {meta_shape}'
        return descr


