from ddls.environments.cluster.job_queue import JobQueue
from ddls.utils import Sampler, seed_stochastic_modules_globally
from ddls.topologies.topology import Topology
from ddls.topologies.torus import Torus
from ddls.demands.jobs.job import Job
from ddls.demands.jobs.jobs_generator import JobsGenerator
from ddls.distributions.distribution import Distribution
from ddls.utils import seed_stochastic_modules_globally, Stopwatch, get_class_from_path

from typing import Any, Union
from collections import defaultdict
import copy
import math
import numpy as np

import threading

import pathlib
import glob
from sqlitedict import SqliteDict
import gzip
import pickle
import time


class RampClusterEnvironment:
    def __init__(self,
                 topology_config: dict,
                 node_config: dict,
                 name: str = 'ramp_cluster',
                 path_to_save: str = None,
                 save_freq: int = 1,
                 use_sqlite_database: bool = False):
        '''
        In Ramp clusters, so long as the Ramp rules are followed, there is no contention
        in the network. We therefore need a separate Ramp cluster environment which 
        ensures these rules are adhered to. Since one of these rules is that the op(s)
        of >1 job can be on the same worker, once a job has begun execution on the
        Ramp cluster, it will not be interrupted. This allows the computation time
        of jobs mounted on Ramp clusters to be computed in a lookahead manner when
        the job is first mounted (i.e. before the job has actually been simulated),
        which is different from a 'normal' cluster where execution times must be
        simulated dynamically.

        Number of nodes resulting from topology_config must be equal to the total number
        of nodes specified in the node_config 
        
        The 'worker' in the node_config dict should be an **uninnstantiated** ddls processor.

        Args:
            path_to_save: Where to save data.
            save_freq: Step frequency with which to update saved data.
            use_sqlite_database: If True, will save logs to a .sqlite file to reduce
                RAM memory usage. If False, will hold logs in memory and save to a .pkl file.
        '''
        self.topology_config = topology_config
        self.node_config = node_config

        self.name = name

        self.path_to_save = path_to_save
        self.use_sqlite_database = use_sqlite_database
        if self.path_to_save is not None:
            self.path_to_save = self._init_save_dir(path=self.path_to_save, use_sqlite_database=self.use_sqlite_database)
        self.save_freq = save_freq

        # init topology
        self.topology = self._init_topology(topology_config)
        self._check_topology_node_configs_valid(self.topology, node_config)
        
        # populate topology with nodes specified by node_config
        self._populate_topology(self.topology, node_config)

        self.stopwatch = Stopwatch()

        self.reset_counter = 0

    def _init_save_dir(self, path: str = '.', use_sqlite_database: bool = False):
        # init highest level cluster folder
        _path = path + f'/{self.name}/'
        pathlib.Path(_path).mkdir(parents=True, exist_ok=True)

        # init folder for this sim
        path_items = glob.glob(_path+'*')
        ids = sorted([int(el.split('_')[-1]) for el in path_items])
        if len(ids) > 0:
            _id = ids[-1] + 1
        else:
            _id = 0
        # foldername = f'{self.name}_{_id}/reset_{self.reset_counter}/'
        foldername = f'{self.name}_{_id}/'
        pathlib.Path(_path+foldername).mkdir(parents=True, exist_ok=False)

        return _path + foldername

    def _init_topology(self, topology_config):
        if topology_config['type'] == 'torus':
            topology = Torus(**topology_config['kwargs'])
        else:
            raise Exception(f'Unrecognised topology type {topology_config["type"]}')
        return topology

    def _check_topology_node_configs_valid(self, topology, node_config):
        num_node_config_nodes = sum([node_config[node_type]['num_nodes'] for node_type in node_config])
        if num_node_config_nodes != len(topology.graph.nodes):
            raise Exception(f'topology_config generated a topology with {len(topology.graph.nodes)} nodes, but node_config specified a total of {num_node_config_nodes} nodes, which is incompatible.')

    def _populate_topology(self, topology, node_config):
        node_ids = iter(list(topology.graph.nodes))
        topology.graph.graph['worker_to_node'] = dict()
        topology.graph.graph['worker_to_type'] = dict()
        topology.graph.graph['worker_types'] = set()
        topology.graph.graph['num_workers'] = 0
        for node_type in node_config.keys():
            for _ in range(node_config[node_type]['num_nodes']):
                node_id = next(node_ids)
                topology.graph.nodes[node_id]['workers'] = dict()
                for worker_config in node_config[node_type]['workers_config']:
                    for _ in range(worker_config['num_workers']):
                        # instantiate a worker and add to this node/server
                        if isinstance(worker_config['worker'], str):
                            # get worker class from str
                            Worker = get_class_from_path(worker_config['worker'])
                        else:
                            # is already a class
                            Worker = worker_config['worker']
                        # instantiate class as an object
                        worker = Worker()
                        # update topology details
                        topology.graph.nodes[node_id]['workers'][worker.processor_id] = worker
                        topology.graph.graph['worker_to_node'][worker.processor_id] = node_id
                        topology.graph.graph['worker_to_type'][worker.processor_id] = worker.device_type
                        topology.graph.graph['num_workers'] += 1
                        if worker.device_type not in topology.graph.graph['worker_types']:
                            topology.graph.graph['worker_types'].add(worker.device_type)

    def reset(self,
              jobs_config: dict,
              max_simulation_run_time: Union[int, float] = float('inf'),
              job_queue_capacity: int = 10,
              seed: int = None,
              verbose=False):
        self.reset_counter += 1
        if self.path_to_save is not None:
            pathlib.Path(self.path_to_save + f'reset_{self.reset_counter}/').mkdir(parents=True, exist_ok=False)
            print(f'Initialised folder {self.path_to_save}reset_{self.reset_counter}')
        else:
            self.path_to_save = None

        self.jobs_generator = JobsGenerator(**jobs_config)

        self.max_simulation_run_time = max_simulation_run_time 
        self.seed = seed
        if seed is not None:
            seed_stochastic_modules_globally(seed)

        # reset loggers
        self.save_thread = None
        self._reset_sim_log()
        self._reset_steps_log()

        # reset processors
        for node_id in self.topology.graph.nodes:
            for worker in self.topology.graph.nodes[node_id]['workers'].values():
                worker.reset()

        # initialise job queue
        self.job_queue = JobQueue(queue_capacity=job_queue_capacity)

        # initialise trackers
        self.num_jobs_arrived = 0
        self.num_mounted_ops = 0
        self.num_active_workers = 0
        self.jobs_running = {}
        self.jobs_completed = {}
        self.jobs_blocked = {}

        self.job_op_to_worker = {}

        self.job_idx_to_job_id = {}
        self.job_id_to_job_idx = {}

        self.stopwatch.reset()
        self.step_counter = 0

        # add first job to queue
        self.time_next_job_to_arrive = 0
        self.job_queue.add(self._get_next_job())

        # initialise current job placement (which job ops are on which workers). Maps job id -> op id -> worker id
        self.placement = defaultdict(dict)

        obs = None

        if verbose:
            print(f'Reset cluster environment.')
            print(f'Max sim run time: {self.max_simulation_run_time}')

        return obs

    def _reset_sim_log(self):
        self.sim_log = defaultdict(list)

    def _reset_steps_log(self):
        self.steps_log = defaultdict(list)

    def _get_next_job(self):
        '''Returns next job.'''
        job = self.jobs_generator.sample_job()
        job.register_job_arrived(time_arrived=self.stopwatch.time(), 
                                 job_idx=self.num_jobs_arrived)
        self.time_last_job_arrived = copy.deepcopy(self.stopwatch.time())
        self.time_next_job_to_arrive += self.jobs_generator.sample_interarrival_time(size=None)
        self.num_jobs_arrived += 1
        return job




















