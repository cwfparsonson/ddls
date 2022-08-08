from ddls.utils import gen_channel_id

from collections import defaultdict
from typing import Union
import json

class Channel:
    def __init__(self,
                 src: Union[int, str],
                 dst: Union[int, str],
                 channel_number: int,
                 channel_bandwidth: int = int(1.25e9)):
        self.src = src
        self.dst = dst
        if channel_number is None:
            self.channel_number = id(self)
        else:
            self.channel_number = channel_number 
        self.channel_id = gen_channel_id(self.src, self.dst, self.channel_number)
        self.channel_bandwidth = channel_bandwidth
        self.reset()

    def __str__(self):
        return f'Channel_{self.channel_id}'
        
    def reset(self):
        self.mounted_job_idx_to_deps = defaultdict(set)
        self.mounted_job_dep_to_priority = dict() # job flow schedule for mounted flows

    def mount(self, job, dep):
        self.mounted_job_idx_to_deps[job.details['job_idx']].add(dep)

    def unmount(self, job, dep):
        self.mounted_job_idx_to_deps[job.details['job_idx']].remove(dep)
        del self.mounted_job_dep_to_priority[self._gen_job_dep_str(job.details['job_idx'], job.job_id, dep)]
        if len(self.mounted_job_idx_to_deps[job.details['job_idx']]) == 0:
            del self.mounted_job_idx_to_deps[job.details['job_idx']]

    def _gen_job_dep_str(self, job_idx, job_id, dep_id):
        return f'{json.dumps(job_idx)}_{json.dumps(job_id)}_{json.dumps(dep_id)}'
