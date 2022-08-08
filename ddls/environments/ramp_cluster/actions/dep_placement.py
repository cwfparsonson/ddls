from collections import defaultdict
from ddls.utils import init_nested_hash, gen_channel_id

import json

class DepPlacement:
    def __init__(self,
                 action: dict):
        '''
        Args:
            # action: Mapping of job_id -> dependency_id -> path -> channel_num.
            action: Mapping of job_id -> dependency_id -> channel_ids
        '''
        self.action = action

        self.job_ids = set()
        self.channel_ids = set()
        self.jobdeps = set() # <job_id>_<dep_id> str
        self.channel_to_job_to_deps = defaultdict(lambda: defaultdict(set)) # maps channel_id -> job_id -> dep_ids
        self.job_to_dep_to_channel = init_nested_hash() # maps job_id -> dep_id -> channel_id
        self.channel_to_jobdeps, self.jobdep_to_channels = defaultdict(set), defaultdict(set)
        # self.src_to_dst_to_channel_to_deps = init_nested_hash()
        for job_id in action.keys():
            self.job_ids.add(job_id)
            for dep_id in self.action[job_id].keys():
                for channel_id in self.action[job_id][dep_id]:
                    self.channel_ids.add(channel_id)
                    self.channel_to_job_to_deps[channel_id][job_id].add(dep_id)
                    self.job_to_dep_to_channel[job_id][dep_id] = channel_id

                    jobdep = f'{json.dumps(job_id)}_{json.dumps(dep_id)}'
                    self.jobdeps.add(jobdep)
                    self.channel_to_jobdeps[channel_id].add(jobdep)
                    self.jobdep_to_channels[jobdep].add(channel_id) 


                # path = self.action[job_id][dep_id]
                # if len(path) > 0:
                    # # dependency became a flow and therefore required action
                    # path, channel_num = list(self.action[job_id][dep_id].items())[0]
                    # path = json.loads(path)
                    # for idx in range(len(path) - 1):
                        # src, dst = (path[idx], path[idx+1])
                        # channel_id = gen_channel_id(src, dst, channel_num)
                        # self.channel_ids.add(channel_id)
                        # self.channel_to_job_to_deps[channel_id][job_id].add(dep_id)
                        # self.job_to_dep_to_channel[job_id][dep_id] = channel_id

                        # jobdep = f'{json.dumps(job_id)}_{json.dumps(dep_id)}'
                        # self.jobdeps.add(jobdep)
                        # self.channel_to_jobdeps[channel_id].add(jobdep)
                        # self.jobdep_to_channels[jobdep].add(channel_id) 
                # else:
                    # # dependency either had size == 0 or src == dst -> not a flow
                    # pass

    def __str__(self):
        # descr = ''
        # for job_id in self.action.keys():
            # descr += f'\n\tJob ID: {job_id}'
            # for dep_id in self.action[job_id].keys():
                # path = self.action[job_id][dep_id]
                # if len(path) > 0:
                    # # dep was placed
                    # path, channel = list(self.action[job_id][dep_id].items())[0]
                    # descr += f'\n\t\tDep ID {dep_id} -> Path {path} channel {channel}'
                # else:
                    # # dep had either size == 0 or src == dst -> not a network flow -> no placement 
                    # descr += f'\n\t\tDep ID {dep_id} -> N/A'
        # return descr

        descr = ''
        for job_id in self.action.keys():
            descr += f'\nJob ID: {job_id}'
            for dep_id in self.action[job_id].keys():
                channel_ids = self.action[job_id][dep_id]
                descr += f'\n\tDep ID {dep_id} -> Channel IDs {channel_ids}'
        return descr


