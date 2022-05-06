from collections import defaultdict

class Channel:
    def __init__(self,
                 channel_id: int = None,
                 channel_bandwidth: int = int(1.25e9)):
        if channel_id is None:
            self.channel_id = id(self)
        else:
            self.channel_id = channel_id
        self.channel_bandwidth = channel_bandwidth
        self.reset()

    def __str__(self):
        return f'Channel_{self.channel_id}'
        
    def reset(self):
        self.mounted_job_idx_to_deps = defaultdict(set)
        self.mounted_job_dep_to_priority = dict() # job flow schedule for mounted flows

    def mount(self, job, dep):
        self.mounted_job_idx_to_deps[job.details['job_idx']].add(str(dep))

    def unmount(self, job, dep):
        self.mounted_job_idx_to_deps[job.details['job_idx']].remove(str(dep))
        del self.mounted_job_dep_to_priority[f'{job.details["job_idx"]}_{job.job_id}_{dep}']
        if len(self.mounted_job_idx_to_deps[job.details['job_idx']]) == 0:
            del self.mounted_job_idx_to_deps[job.details['job_idx']]
