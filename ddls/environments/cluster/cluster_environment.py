from ddls.environments.cluster.job_queue import JobQueue
from ddls.utils import Sampler, seed_stochastic_modules_globally
from ddls.topologies.topology import Topology
from ddls.topologies.torus import Torus
from ddls.demands.jobs.job import Job
from ddls.distributions.distribution import Distribution
from ddls.utils import seed_stochastic_modules_globally, Stopwatch

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




class ClusterEnvironment:
    def __init__(self,
                 topology_config: dict,
                 node_config: dict,
                 name: str = 'cluster',
                 path_to_save: str = None,
                 save_freq: int = 1,
                 use_sqlite_database: bool = False):
        '''
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
        for node_type in node_config.keys():
            for _ in range(node_config[node_type]['num_nodes']):
                node_id = next(node_ids)
                topology.graph.nodes[node_id]['workers'] = dict()
                for worker_config in node_config[node_type]['workers_config']:
                    for _ in range(worker_config['num_workers']):
                        # instantiate a worker and add to this node/server
                        worker = worker_config['worker']()
                        topology.graph.nodes[node_id]['workers'][worker.processor_id] = worker
                        topology.graph.graph['worker_to_node'][worker.processor_id] = node_id
                        topology.graph.graph['worker_to_type'][worker.processor_id] = worker.device_type

    def reset(self,
              jobs: list[Job], 
              job_interarrival_time_dist: Distribution,
              max_simulation_run_time: Union[int, float] = float('inf'),
              job_sampling_mode: str = 'remove_and_repeat',
              job_queue_capacity: int = 10,
              seed: int = None,
              verbose=False):
        self.reset_counter += 1
        if self.path_to_save is not None:
            pathlib.Path(self.path_to_save + f'reset_{self.reset_counter}/').mkdir(parents=True, exist_ok=False)
            print(f'Initialised folder {self.path_to_save}reset_{self.reset_counter}')
        else:
            self.path_to_save = None

        self.job_sampler = Sampler(pool=jobs, sampling_mode=job_sampling_mode)
        self.job_interarrival_time_dist = job_interarrival_time_dist 
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
        action_set = None
        reward = None
        done = False
        info = None

        if verbose:
            print(f'Reset cluster environment.')
            print(f'Job interarrival time dist: {self.job_interarrival_time_dist}')
            print(f'Job sampler: {self.job_sampler}')
            print(f'Max sim run time: {self.max_simulation_run_time}')

        return obs, action_set, reward, done, info

    def _get_next_job(self):
        '''Returns next job.'''
        job = self.job_sampler.sample()
        job.register_job_arrived(time_arrived=self.stopwatch.time(), 
                                 job_idx=self.num_jobs_arrived)
        self.time_last_job_arrived = copy.deepcopy(self.stopwatch.time())
        self.time_next_job_to_arrive += self.job_interarrival_time_dist.sample(size=None)
        self.num_jobs_arrived += 1
        return job

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

    def _init_step_stats(self):
        step_stats = defaultdict(lambda: 0)
        step_stats['step_counter'] = copy.deepcopy(self.step_counter)
        step_stats['step_start_time'] = copy.deepcopy(self.stopwatch.time())
        step_stats['mean_num_active_workers'] = []
        step_stats['num_jobs_completed'] = 0
        step_stats['num_jobs_running'] = 0
        step_stats['num_jobs_arrived'] = 0
        step_stats['num_jobs_blocked'] = 0
        return step_stats

    def step(self, 
             actions,
             verbose=False):
        self.step_stats = self._init_step_stats()
        if verbose:
            print('')
            print('-'*80)
            print(f'Step: {self.step_counter}')

        self._prioritise_jobs()
        self._partition_jobs()
        self._place_jobs(actions['job_placement'],
                         verbose=verbose)
        self._schedule_jobs(actions['job_schedule'],
                            verbose=verbose)

        # run step until next job arrives, a job is complete, or the simulation is done
        step_done = False
        self.step_stats['num_jobs_running'] = len(self.jobs_running)
        while not step_done:
            if verbose:
                print('-'*80)
                print(f'Performing cluster tick. Stopwatch time at start of tick: {self.stopwatch.time()}')

            # run ops which have been placed on workers and records which op(s) completed
            max_tick = min(self.time_next_job_to_arrive - self.stopwatch.time(), self.max_simulation_run_time - self.stopwatch.time())
            job_idx_to_completed_op_ids = self._tick_workers(actions['job_schedule'], max_tick=max_tick, verbose=verbose)
            if verbose:
                print(f'job_idx_to_completed_op_ids: {job_idx_to_completed_op_ids}')

            # TODO: 
            # 1. Take completed ops and (1) do any communication necessary (2) update parent_dependencies_satisfied
            # 2. Implement more efficient way of tracking and updating shortest_remaining_run_time of mounted ops that doesn't need to iterate every job op each time

            # TEMPORARY: Assume no network communication overhead -> child dependencies of completed ops immediately satisfied
            for job_idx, op_ids in job_idx_to_completed_op_ids.items():
                job = self.jobs_running[job_idx]
                for op_id in op_ids:
                    for child_dependency in job.computation_graph.out_edges(op_id):
                        job.register_satisfied_dependency(child_dependency)

            # register any completed job training steps or completed jobs
            for job_idx in job_idx_to_completed_op_ids.keys():
                job = self.jobs_running[job_idx]
                if job.is_training_step_complete():
                    # reset job ready for another training step
                    job.reset()
                    if verbose:
                        print(f'Job with job_idx {job_idx} completed training step {job.training_step_counter} of {job.num_training_steps}')
                if job.is_job_complete():
                    self._register_completed_job(job)
                    step_done = True
                    if verbose:
                        print(f'Job with job_idx {job_idx} completed. Time arrived: {job.details["time_arrived"]} | Time completed: {job.details["time_completed"]}')

            # check if next job should arrive
            if len(self.job_sampler) > 0:
                if verbose:
                    print(f'Time next job due to arrive: {self.time_next_job_to_arrive}')
                if self.stopwatch.time() > self.time_next_job_to_arrive:
                    raise Exception(f'Stopwatch time is {self.stopwatch.time()} but next job should have arrived at {self.time_next_job_to_arrive}')
                elif self.stopwatch.time() == self.time_next_job_to_arrive:
                    next_job = self._get_next_job()
                    self.step_stats['num_jobs_arrived'] += 1
                    if verbose:
                        print(f'Next job with job_idx {next_job.details["job_idx"]} arrived. Added to queue.')
                    if self.job_queue.can_fit(next_job):
                        self.job_queue.add(next_job)
                    else:
                        self._register_blocked_job(next_job)
                    step_done = True
                else:
                    pass
            else:
                # no more jobs to sample
                self.time_next_job_to_arrive = float('inf')

            # check if simulation finished
            if self._is_done():
                step_done = True

            if verbose:
                print(f'Finished cluster tick. Stopwatch time at end of tick: {self.stopwatch.time()}')

        self._communicate_jobs()

        # log step-level data
        self.step_stats['step_end_time'] = self.stopwatch.time()
        self.step_stats['mean_num_active_workers'] = np.mean(self.step_stats['mean_num_active_workers'])
        self.step_stats['job_queue_length'] = len(self.job_queue)
        self._update_steps_log(copy.deepcopy(self.step_stats))

        # move to next step
        self.step_counter += 1

        # save logs
        if self.path_to_save is not None:
            if self.step_counter % self.save_freq == 0 or self._is_done():
                self.save()
                if self._is_done():
                    self.save_thread.join()

        # TEMPORARY
        obs, action_set, reward, done, info = None, None, None, self._is_done(verbose), None

        return obs, action_set, reward, done, info

    def _is_done(self, verbose=False):
        '''Checks if simulation has finished.'''
        done = False

        if self.max_simulation_run_time is not None:
            if self.stopwatch.time() >= self.max_simulation_run_time:
                done = True
                if verbose:
                    print(f'Maximum simulation run time reached -> done.')

        if len(self.job_sampler) == 0 and len(self.jobs_running) == 0 and len(self.job_queue) == 0:
            done = True
            if verbose:
                print(f'No more jobs running, in queue, or left to sample -> done.')

        return done

    def _tick_workers(self, job_schedule, max_tick=None, verbose=False):
        '''
        Ticks all operations on workers by shortest remaining run time of all 
        mounted ops (up to max_tick if max_tick is not None). After performing 
        this tick, at least one operation in the cluster will have been completed.

        Will also tick the cluster's stopwatch.

        Args:
            job_schedule: Dict mapping worker_id -> job_id -> op_id -> priority,
                where the ops should be scheduled in order of highest priority value.

        Returns dict mapping job_idx -> completed op ids
        '''
        # find: 1) highest priority op on each worker; and 2) the shortest remaining run time of each highest priority op on all workers
        worker_to_priority_job_op = {}
        shortest_remaining_run_time = float('inf')
        for worker_id, node_id in self.topology.graph.graph['worker_to_node'].items():
            # get highest priority ready op on this worker
            # priority_job_op = self._get_highest_priority_job_op(worker=self.topology.graph.nodes[node_id]['workers'][worker_id], job_schedule=job_schedule)
            priority_job_op = self._get_highest_priority_job_op(worker=self.topology.graph.nodes[node_id]['workers'][worker_id])
            if priority_job_op is not None:
                # record priority op for this worker
                worker_to_priority_job_op[worker_id] = priority_job_op
                # check if should update shortest remaining run time of all priority job ops
                job_idx, job_id, op_id = [int(i) for i in priority_job_op.split('_')]
                job = self.jobs_running[job_idx]
                if job.computation_graph.nodes[op_id]['remaining_run_time'] < shortest_remaining_run_time:
                    shortest_remaining_run_time = job.computation_graph.nodes[op_id]['remaining_run_time']
        if max_tick is not None:
            tick = min(shortest_remaining_run_time, max_tick)
        else:
            tick = shortest_remaining_run_time

        # tick highest priority mounted ready op on each worker by shortest_remaining_run_time (or max_tick) and track which op(s) completed
        job_idx_to_completed_op_ids = defaultdict(list)
        num_active_workers = 0
        for worker_id, priority_job_op in worker_to_priority_job_op.items():
            if priority_job_op is not None:
                num_active_workers += 1
                node_id = self.topology.graph.graph['worker_to_node'][worker_id]
                worker = self.topology.graph.nodes[node_id]['workers'][worker_id]
                job_idx, job_id, op_id = [int(i) for i in priority_job_op.split('_')]
                job = self.jobs_running[job_idx]
                if verbose:
                    remaining_run_time = job.computation_graph.nodes[op_id]['remaining_run_time']
                    print(f'Ticking op {op_id} with remaining run time {remaining_run_time} of job index {job_idx} on node {node_id} worker {worker_id} by amount {tick}')
                job.tick_op(op_id, tick=tick)
                if op_id in job.computation_graph.graph['ops_completed']:
                    # op was completed
                    job_idx_to_completed_op_ids[job_idx].append(op_id)
                    if verbose:
                        print(f'Op {op_id} of job index {job_idx} completed')
        self.step_stats['mean_num_active_workers'].append(num_active_workers)

        # tick stopwatch
        self.stopwatch.tick(tick)

        return job_idx_to_completed_op_ids

    def _get_highest_priority_job_op(self, worker):
        '''
        Takes a worker processor object and returns a string identifying which operation
        which is ready to run on the worker has the highest priority. If no
        operation is available to run, will return None.
        '''
        priority_job_op = None
        for job_idx in worker.mounted_job_idx_to_ops.keys():
            job = self.jobs_running[job_idx]
            for op_id in worker.mounted_job_idx_to_ops[job_idx]:
                if op_id in job.computation_graph.graph['ops_ready']:
                    # op is ready to run
                    job_op = f'{job_idx}_{job.job_id}_{op_id}'
                    if priority_job_op is None:
                        # not yet considered any other ops, set this op as priority op
                        priority_job_op = job_op
                    else:
                        # check if op has higher priority than current highest priority op found so far
                        if worker.mounted_job_op_to_priority[job_op] > worker.mounted_job_op_to_priority[priority_job_op]:
                            # op has higher priority, update priority job op
                            priority_job_op = job_op
                else:
                    # op not yet ready to run
                    pass
        return priority_job_op

    def _prioritise_jobs(self):
        pass

    def _partition_jobs(self):
        pass

    def _place_jobs(self, job_placement, verbose):
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
                self.topology.graph.nodes[node_id]['workers'][worker_id].mount(job=job, op_id=op_id)
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

    def _communicate_jobs(self):
        pass

    def render(self):
        pass

    def seed(self, seed):
        pass

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

    def _reset_sim_log(self):
        self.sim_log = defaultdict(list)

    def _reset_steps_log(self):
        self.steps_log = defaultdict(list)

    def _update_steps_log(self, step_stats):
        for key, val in step_stats.items():
            self.steps_log[key].append(val)

    def _save_logs(self, logs: dict):
        start_time = time.time_ns()
        for log_name, log in logs.items():
            log_path = self.path_to_save + f'reset_{self.reset_counter}/{log_name}'
            if self.use_sqlite_database:
                # update log sqlite database under database folder
                with SqliteDict(log_path + '.sqlite') as _log:
                    for key, val in log.items():
                        if key in _log and type(val) == list:
                            # extend vals list
                            _log[key] += val
                        else:
                            # create val
                            _log[key] = val
                    _log.commit()
                    _log.close()
            else:
                # save log as pkl
                with gzip.open(log_path + '.pkl', 'wb') as f:
                    pickle.dump(log, f)
        print(f'Saved logs to {self.path_to_save}reset_{self.reset_counter}/ in {(time.time_ns() - start_time)*1e-9:.4f} s.')

    def save(self):
        if self.save_thread is not None:
            self.save_thread.join()
        self.save_thread = threading.Thread(target=self._save_logs, 
                                            args=({'sim_log': copy.deepcopy(self.sim_log),
                                                   'steps_log': copy.deepcopy(self.steps_log)},))
        self.save_thread.start()

        if self.use_sqlite_database:
            # reset in-memory logs
            self._reset_sim_log()
            self._reset_steps_log()
























