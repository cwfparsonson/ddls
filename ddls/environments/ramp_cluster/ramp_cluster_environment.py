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

    def _init_step_stats(self):
        step_stats = defaultdict(lambda: 0)
        step_stats['step_counter'] = copy.deepcopy(self.step_counter)
        step_stats['step_start_time'] = copy.deepcopy(self.stopwatch.time())
        step_stats['mean_num_active_workers'] = []

        # need to init following manually to ensure they're recorded in saved results
        step_stats['num_jobs_completed'] = 0
        step_stats['num_jobs_running'] = 0
        step_stats['num_jobs_arrived'] = 0
        step_stats['num_jobs_blocked'] = 0

        return step_stats

    def _get_next_job(self):
        '''Returns next job.'''
        job = self.jobs_generator.sample_job()
        job.register_job_arrived(time_arrived=self.stopwatch.time(), 
                                 job_idx=self.num_jobs_arrived)
        self.time_last_job_arrived = copy.deepcopy(self.stopwatch.time())
        self.time_next_job_to_arrive += self.jobs_generator.sample_interarrival_time(size=None)
        self.job_idx_to_job_id[job.details['job_idx']] = job.job_id
        self.job_id_to_job_idx[job.job_id] = job.details['job_idx']
        self.num_jobs_arrived += 1
        return job

    def _perform_lookahead_job_completion_time(self, job_placement, verbose=False):
        # do a lookahead to see how long each placed job will take to complete, and update job details accordingly
        if verbose:
            if len(job_placement) > 0:
                print('New job(s) to perform job completion time lookahead for. Performing lookahead...')
            else:
                print(f'No new jobs to perform lookahead for.')
        for job_id in actions['job_placement']:
            job = self.jobs_running.job_id_to_job_idx[job_id]
            if verbose:
                print(f'Job ID: {job_id} | Job idx: {job.details["job_idx"]} | Time arrived: {job.details["time_arrived"]}')

            # do internal lookahead simulation until job is completed
            tmp_stopwatch = Stopwatch()
            tmp_stopwatch.reset()
            step_done = False
            while not step_done:
                # run step until an op and/or a flow is completed

                # COMPUTATION
                # find: 1) highest priority op on each worker for this job; and 2) the shortest remaining run time of each highest priority op on all workers for this job
                shortest_remaining_run_time = float('inf')
                worker_to_priority_job_op = {}
                for worker_id, node_id in self.topology.graph.graph['worker_to_node'].items():
                    worker = self.topology.graph.nodes[node_id]['workers'][worker_id]
                    if job.details['job_idx'] in worker.mounted_job_idx_to_ops.keys():
                        # get highest priority ready op on this worker
                        priority_job_op = self._get_highest_priority_job_op(worker=worker)
                        if priority_job_op is not None:
                            # record priority op for this worker
                            worker_to_priority_job_op[worker_id] = priority_job_op
                            # check if should update shortest remaining run time of all priority job ops
                            job_idx, job_id, op_id = [int(i) for i in priority_job_op.split('_')]
                            job = self.jobs_running[job_idx]
                            if job.computation_graph.nodes[op_id]['remaining_run_time'] < shortest_remaining_run_time:
                                # update shortest_remaining_run_time
                                shortest_remaining_run_time = job.computation_graph.nodes[op_id]['remaining_run_time']
                    else:
                        # this job has no op(s) mounted on this worker
                        pass

                # COMMUNICATION
                # TODO find: 1) highest priority flow on each link for this job; and 2) the shortest remaining communication time of each highest priority flow on all links for this job
                shortest_remaining_communication_time = float('inf')
                link_to_priority_job_flow = {}
                


                
                tick = min(shortest_remaining_run_time, shortest_remaining_communication_time)


                # TODO: tick highest priority ops and flows by amount <tick> and track which op(s) and/or flow(s) are completed










    def step(self,
             actions,
             verbose=False):
        # TODO
        if self.path_to_save is not None and self.use_sqlite_database and self.step_counter % self.save_freq == 0:
            # saved logs at end of previous step, can reset in-memory logs for this step
            self._reset_sim_log()
            self._reset_steps_log()

        self.step_stats = self._init_step_stats()
        if verbose:
            print('')
            print('-'*80)
            print(f'Step: {self.step_counter}')

        # execute control plane decisions
        # self._prioritise_jobs() # TODO
        # self._partition_jobs() # TODO
        self._place_jobs(actions['job_placement'],
                         verbose=verbose)
        self._schedule_jobs(actions['job_schedule'],
                            verbose=verbose)
        self._place_deps(actions['dep_placement'],
                         verbose=verbose)
        self._schedule_deps(actions['dep_schedule'],
                            verbose=verbose)

        # given control plane decisions, perform job completion time lookahead
        self._perform_lookahead_job_completion_time(actions['job_placement'])






        # # run step until next job arrives, a job is complete, or the simulation is done
        # step_done = False
        # self.step_stats['num_jobs_running'] = len(self.jobs_running)
        # while not step_done:
            # if verbose:
                # print('-'*80)
                # print(f'Performing cluster tick. Stopwatch time at start of tick: {self.stopwatch.time()}')





    def _check_ramp_placement_rules_broken(self, job, op_id, worker):
        '''Checks whether an op placement obeys the rules of Ramp.'''
        rules_broken = []

        # Ramp Rule 1: No worker can have ops from more than one job.
        if job.details['job_idx'] not in worker.mounted_job_idx_to_ops:
            if len(worker.mounted_job_idx_to_ops.keys()) > 0:
                # already have another job mounted on this worker
                rules_broken.append('one_job_per_worker')

        return rules_broken

    def _place_jobs(self, job_placement, verbose=False):
        if verbose:
            if len(job_placement) > 0:
                print('New job(s) to place on cluster. Placing...')
            else:
                print(f'No new jobs to place on cluster.')
        for job_id in job_placement:
            job = self.job_queue.jobs[job_id]
            if verbose:
                print(f'Job ID: {job_id} | Job idx: {job.details["job_idx"]} | Time arrived: {job.details["time_arrived"]}')
            for op_id in job_placement[job_id]:
                worker_id = job_placement[job_id][op_id]
                node_id = self.topology.graph.graph['worker_to_node'][worker_id]
                worker = self.topology.graph.nodes[node_id]['workers'][worker_id]
                rules_broken = self._check_ramp_placement_rules_broken(job, op_id, worker)
                if len(rules_broken) > 0:
                    raise Exception(f'Placement for job index {job.details["job_idx"]} job ID {job_id} op ID {op_id} worker ID {worker_id} breaks the following Ramp rules: {rules_broken}')
                else:
                    worker.mount(job=job, op_id=op_id)
                    self.num_mounted_ops += 1
                    job.reset_op_remaining_run_time(op_id, device_type=self.topology.graph.nodes[node_id]['workers'][worker_id].device_type)
                    self.job_op_to_worker[f'{job.details["job_idx"]}_{job.job_id}_{op_id}'] = worker_id
                    if verbose:
                        print(f'Op ID {op_id} of job index {job.details["job_idx"]} placed on node ID {node_id} worker ID {worker_id}')
            self._register_running_job(job)
            # update cluster tracking of current job placement
            self.placement[job_id] = job_placement[job_id]

    def _schedule_jobs(self, job_schedule, verbose=False):
        '''Sets scheduling priority for mounted ops on each worker.'''
        for worker_id in job_schedule.keys():
            node_id = self.topology.graph.graph['worker_to_node'][worker_id]
            worker = self.topology.graph.nodes[node_id]['workers'][worker_id]
            for job_idx in worker.mounted_job_idx_to_ops.keys():
                job = self.jobs_running[job_idx]
                for op_id in worker.mounted_job_idx_to_ops[job_idx]:
                    worker.mounted_job_op_to_priority[f'{job_idx}_{job.job_id}_{op_id}'] = job_schedule[worker_id][job.job_id][op_id]

    def _register_running_job(self, job):
        job.register_job_running(time_started=self.stopwatch.time())
        self.jobs_running[job.details['job_idx']] = job
        self.job_queue.remove(job)

    def _register_completed_job(self, job):
        # record completion time
        job.register_job_completed(time_completed=self.stopwatch.time())

        # update loggers
        self.jobs_completed[job.details['job_idx']] = job
        self.step_stats['num_jobs_completed'] += 1
        self.sim_log['job_completion_time'].append(job.details['time_completed'] - job.details['time_arrived'])
        self.sim_log['jobs_completed_num_nodes'].append(len(job.computation_graph.nodes))
        self.sim_log['jobs_completed_num_edges'].append(len(job.computation_graph.edges))
        self.sim_log['jobs_completed_total_operation_memory_cost'].append(job.job_total_operation_memory_cost)
        self.sim_log['jobs_completed_total_dependency_size'].append(job.job_total_dependency_size)

        # update sim
        del self.jobs_running[job.details['job_idx']]
        for op_id in job.computation_graph.nodes:
            worker_id = self.job_op_to_worker[f'{job.details["job_idx"]}_{job.job_id}_{op_id}']
            node_id = self.topology.graph.graph['worker_to_node'][worker_id]
            worker = self.topology.graph.nodes[node_id]['workers'][worker_id].unmount(job=job, op_id=op_id)
            self.num_mounted_ops -= 1
            del self.job_op_to_worker[f'{job.details["job_idx"]}_{job.job_id}_{op_id}']
        # clear job from current cluster placement tracker
        del self.placement[job.job_id]
        self.step_stats['num_jobs_completed'] += 1
            
    def _register_blocked_job(self, job):
        self.jobs_blocked[job.details['job_idx']] = job

        # update loggers
        self.step_stats['num_jobs_blocked'] += 1
        self.sim_log['jobs_blocked_num_nodes'].append(len(job.computation_graph.nodes))
        self.sim_log['jobs_blocked_num_edges'].append(len(job.computation_graph.edges))
        self.sim_log['jobs_blocked_total_operation_memory_cost'].append(job.job_total_operation_memory_cost)
        self.sim_log['jobs_blocked_total_dependency_size'].append(job.job_total_dependency_size)

    def __str__(self):
        descr = f'Cluster {type(self)}'
        descr += f' | Topology: {type(self.topology)} with {len(self.topology.graph.nodes)} nodes and {len(self.topology.graph.edges)}'
        descr += f' | Topology config: {self.topology_config}'
        descr += f' | Node config: {self.node_config}'
        return descr














