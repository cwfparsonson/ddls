from ddls.utils import gen_channel_id

from collections import defaultdict
from typing import Union

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
        # self.mounted_job_idx_to_deps = {self.src: {self.dst: defaultdict(set)}, self.dst: {self.src: defaultdict(set)}}
        # self.mounted_job_dep_to_priority = {src: {dst: dict()}, dst: {src: dict()}} # job flow schedule for mounted flows
        self.mounted_job_idx_to_deps = defaultdict(set)
        self.mounted_job_dep_to_priority = dict() # job flow schedule for mounted flows

    def mount(self, job, dep):
        self.mounted_job_idx_to_deps[job.details['job_idx']].add(str(dep))

    def unmount(self, job, dep):
        self.mounted_job_idx_to_deps[job.details['job_idx']].remove(str(dep))
        del self.mounted_job_dep_to_priority[f'{job.details["job_idx"]}_{job.job_id}_{dep}']
        if len(self.mounted_job_idx_to_deps[job.details['job_idx']]) == 0:
            del self.mounted_job_idx_to_deps[job.details['job_idx']]
