from ddls.utils import Sampler
from ddls.distributions.distribution import Distribution
from ddls.utils import ddls_graph_from_pbtxt_file, ddls_graph_from_pipedream_txt_file, get_class_from_path
from ddls.demands.jobs.job import Job

import glob
from typing import Union
from collections import defaultdict
import numpy as np
import copy

class JobsGenerator:
    def __init__(self, 
                 path_to_files: str, 
                 job_interarrival_time_dist: Union[Distribution, dict],
                 # job_interarrival_time_dist: Union[Distribution, str], # either a Distribution object or a path leading to the distbution path
                 max_files: int = None, # maximum number of files in path_to_files dir to use
                 replication_factor: int = 1, # number of times to replicate files in path_to_files (e.g. if path_to_files has 1 job graph profile file and replication_factor=10, will have 10 identical jobs).
                 job_sampling_mode: Union['replace', 'remove', 'remove_and_repeat'] = 'remove_and_repeat',
                 shuffle_files: bool = False, # whether or not to shuffle loaded file order when re-load files
                 num_training_steps: int = 1,
                 max_partitions_per_op_in_observation: int = 1, # use to set max possible nodes and edges for normalising observation feats. N.B. set to None if max num edges and nodes in obs is just the size of the original graph rather than the partitioned graph (i.e. if doing partitioning with gym agent rather than as part of env)
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
        for _ in range(replication_factor):
            for graph in ddls_computation_graphs:
                # # set model name as graph.txt file's parent folder
                # details = {'model': graph.graph['file_path'].split('/')[-2]}
                # set model name as <model>.txt file name
                details = {'model': graph.graph['file_path'].split('/')[-1].replace('.txt', '')}
                jobs.append(Job(computation_graph=graph,
                                num_training_steps=num_training_steps,
                                details=details))

        # init job sampler
        self.job_sampler = Sampler(pool=jobs, sampling_mode=job_sampling_mode, shuffle=self.shuffle_files)

        # init job interarrival time dist
        if isinstance(job_interarrival_time_dist, dict):
            # need to instantiate Distribution object from dict of kwargs
            if '_target_' not in job_interarrival_time_dist:
                raise Exception(f'job_interarrival_time_dist specified as dict, therefore expecting dict of kwargs, but require _target_ kwarg giving path to Distribution class so can instantiate.')
            kwargs = {kwarg: val for kwarg, val in job_interarrival_time_dist.items() if kwarg != '_target_'} 
            self.job_interarrival_time_dist = get_class_from_path(job_interarrival_time_dist['_target_'])(**kwargs)
        else:
            # Distribution object already provided
            self.job_interarrival_time_dist = job_interarrival_time_dist

        # init general parameters of jobs
        self.max_partitions_per_op_in_observation = max_partitions_per_op_in_observation
        self.jobs_params = self._init_jobs_params(jobs, max_partitions_per_op_in_observation=self.max_partitions_per_op_in_observation)

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

    def _init_jobs_params(self, jobs, max_partitions_per_op_in_observation=1):
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
            if key in {'job_total_num_ops', 'job_total_num_deps', 'job_total_dep_sizes'}:
                if key == 'job_total_num_ops':
                    updated_jobs_params[f'max_{key}'] = int(np.max(vals) * max_partitions_per_op_in_observation) # each op can be split up to max partition degree times
                elif key == 'job_total_num_deps':
                    # updated_jobs_params[f'max_{key}'] = int( ((np.max(vals) / 2) * max_partitions_per_op_in_observation) + ((np.max(vals) * max_partitions_per_op_in_observation * 2) ) # each forward edge can be split, each backward edge can be split AND be bidirectional
                    # QUESTION: Need to add (num_edges) / 2 extra edges since backward pass edges can be bidirectional when partitioned?
                    max_forward_edges = int((np.max(vals) / 2) * max_partitions_per_op_in_observation * 2) # each edge has a parent and a child, both of which can be split, therefore max edges = # edges x max partition degree x 2
                    max_backward_edges = int(max_forward_edges * 2) # backward edges can be bidirectional
                    updated_jobs_params[f'max_{key}'] = max_forward_edges + max_backward_edges
                elif key == 'job_total_dep_sizes':
                    # # OLD
                    # # edges in backward pass (i.e. 50% of overall edges) can be made bidirectinal -> total dep size doubles for these edges -> increase total dep size by 50%
                    # updated_jobs_params[f'max_{key}'] = np.max(vals) * 1.5
                    # # PROBLEM: If edges unequally weighted in backward pass, then cannot just increase total by 50%

                    # # NEW
                    # # SOLUTION: Just multiply by 2 (i.e. assumes forward edges can be bidirectional too) -> not perfect since will not normalise 0-1 (will be e.g. 0-0.6) but more simple than having to account for each edge
                    # updated_jobs_params[f'max_{key}'] = np.max(vals) * 2
                    # # PROBLEM: Still does not work since some graphs exceed this. This is because more edges are added than just sync/bidirectional edges

                    # NEW NEW
                    # SOLUTION: Just assume graph can become fully connected
                    max_nodes = np.max(jobs_params['job_total_num_ops']) * max_partitions_per_op_in_observation
                    fully_connected_edges = int(max_nodes * (max_nodes - 1) / 2)
                    updated_jobs_params[f'max_{key}'] = np.max(vals) * fully_connected_edges
                else:
                    raise Exception(f'Handling param {key} not implemented.')
            else:
                updated_jobs_params[f'max_{key}'] = np.max(vals)
            # updated_jobs_params[f'mean_{key}'] = np.mean(vals)
            # updated_jobs_params[f'std_{key}'] = np.std(vals)

        return updated_jobs_params
