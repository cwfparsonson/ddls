from ddls.utils import Sampler
from ddls.distributions.distribution import Distribution
from ddls.utils import ddls_graph_from_pbtxt_file, ddls_graph_from_pipedream_txt_file
from ddls.demands.jobs.job import Job

import glob
from typing import Union
from collections import defaultdict
import numpy as np
import copy

class JobsGenerator:
    def __init__(self, 
                 path_to_files: str, 
                 job_interarrival_time_dist: Distribution,
                 # job_interarrival_time_dist: Union[Distribution, str], # either a Distribution object or a path leading to the distbution path
                 max_files: int = None, 
                 job_sampling_mode: Union['replace', 'remove', 'remove_and_repeat'] = 'remove_and_repeat',
                 shuffle_files: bool = False, # whether or not to shuffle loaded file order when re-load files
                 num_training_steps: int = 1,
                 ):
        self.shuffle_files = shuffle_files

        # get file paths
        _file_paths = glob.glob(path_to_files + '/*')

        # only use valid file types for loading graphs
        valid_types, file_paths = set(['pbtxt', 'txt']), []
        for f in _file_paths:
            _type = f.split('.')[-1]
            if _type in valid_types:
                file_paths.append(f)

        # get file reader
        if file_paths[0].split('.')[-1] == 'pbtxt':
            file_reader = ddls_graph_from_pbtxt_file
        if file_paths[0].split('.')[-1] == 'txt':
            file_reader = ddls_graph_from_pipedream_txt_file
        else:
            raise Exception(f'Unsure how to read file in {file_paths[0]}')

        # create ddls graphs
        if max_files is None:
            # use all files
            ddls_computation_graphs = [file_reader(file_path, processor_type_profiled='A100', verbose=False) for file_path in file_paths]
        else:
            # only use up to max_files
            if len(file_paths) > max_files:
                ddls_computation_graphs = [file_reader(file_path, processor_type_profiled='A100', verbose=False) for file_path in file_paths[:max_files]]
            else:
                ddls_computation_graphs = [file_reader(file_path, processor_type_profiled='A100', verbose=False) for file_path in file_paths]

        # create ddls jobs
        jobs = []
        for graph in ddls_computation_graphs:
            details = {'job_name': graph.graph['graph_name']}
            jobs.append(Job(computation_graph=graph,
                            num_training_steps=num_training_steps))

        # init job sampler
        self.job_sampler = Sampler(pool=jobs, sampling_mode=job_sampling_mode, shuffle=self.shuffle_files)

        # init job interarrival time dist
        self.job_interarrival_time_dist = job_interarrival_time_dist

        # init general parameters of jobs
        self.jobs_params = self._init_jobs_params(jobs)

    def __len__(self):
        return len(self.job_sampler)

    def sample_job(self):
        return self.job_sampler.sample()

    def sample_interarrival_time(self, size: int = None):
        if len(self.job_sampler) == 0:
            # no more jobs left to sample
            return float('inf')
        else:
            return self.job_interarrival_time_dist.sample(size=size)

    def _init_jobs_params(self, jobs):
        jobs_params = defaultdict(lambda: [])

        # TODO TEMP: Assume one worker type, but should update to account for multiple worker types?
        device_type = list(jobs[0].details['job_sequential_completion_time'].keys())[0]

        for job in jobs:
            jobs_params['job_sequential_completion_times'].append(job.details['job_sequential_completion_time'][device_type])
            jobs_params['job_total_op_memory_costs'].append(job.details['job_total_op_memory_cost'])
            jobs_params['job_total_dep_sizes'].append(job.details['job_total_dep_size'])
            jobs_params['job_total_num_ops'].append(len(list(job.computation_graph.nodes())))
            jobs_params['job_total_num_deps'].append(len(list(job.computation_graph.edges())))
            jobs_params['job_num_training_steps'].append(job.num_training_steps)

        updated_jobs_params = {}
        for key, vals in jobs_params.items():
            updated_jobs_params[key] = vals
            updated_jobs_params[f'min_{key}'] = np.min(vals)
            updated_jobs_params[f'max_{key}'] = np.max(vals)
            updated_jobs_params[f'mean_{key}'] = np.mean(vals)
            updated_jobs_params[f'std_{key}'] = np.std(vals)

        return updated_jobs_params
