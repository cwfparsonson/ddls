from ddls.environments.cluster.job_queue import JobQueue
from ddls.utils import Sampler, seed_stochastic_modules_globally, gen_job_dep_str, load_job_dep_str
from ddls.topologies.topology import Topology
from ddls.topologies.torus import Torus
from ddls.topologies.ramp import Ramp
from ddls.demands.jobs.job import Job
from ddls.demands.jobs.jobs_generator import JobsGenerator
from ddls.distributions.distribution import Distribution
from ddls.utils import seed_stochastic_modules_globally, Stopwatch, get_class_from_path
from ddls.environments.ramp_cluster.ramp_rules import check_if_ramp_op_placement_rules_broken, check_if_ramp_dep_placement_rules_broken
from ddls.environments.ramp_cluster.actions.action import Action

from typing import Any, Union
from collections import defaultdict
import copy
import math
import json
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
                 use_sqlite_database: bool = False,
                 machine_epsilon=1e-7):
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
            machine_epsilon (float): An upper bound on the relative approximation error due to rounding
                in python's floating point arithmetic. Note that the limit of the simulation's
                time resolution will be the machine epsilon (i.e. cannot measure time with 
                resolution better than machine_epsilon). Use machine_epsilon to prevent
                floating point arithmetic errors during simulation.
        '''
        self.topology_config = topology_config
        self.node_config = node_config

        self.name = name

        self.path_to_save = path_to_save
        self.use_sqlite_database = use_sqlite_database
        if self.path_to_save is not None:
            self.path_to_save = self._init_save_dir(path=self.path_to_save, use_sqlite_database=self.use_sqlite_database)
        self.save_freq = save_freq

        self.machine_epsilon = machine_epsilon

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
        elif topology_config['type'] == 'ramp':
            topology = Ramp(**topology_config['kwargs'])
        else:
            raise Exception(f'Unrecognised topology type {topology_config["type"]}. May need to implement here.')
        return topology

    def _check_topology_node_configs_valid(self, topology, node_config):
        num_node_config_nodes = sum([node_config[node_type]['num_nodes'] for node_type in node_config])
        if num_node_config_nodes != len(topology.graph.nodes):
            raise Exception(f'topology_config generated a topology with {len(topology.graph.nodes)} nodes, but node_config specified a total of {num_node_config_nodes} nodes, which is incompatible. Please either change the number of nodes in node_config OR the implied number of nodes in topology_config so that they are compatible.')

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
                    if worker_config['num_workers'] > 1:
                        raise Exception(f'ERROR: Current RAMP implementation only supports 1 worker per server. Set worker_config["num_workers"] = 1.')
                    for i in range(worker_config['num_workers']):
                        # instantiate a worker and add to this node/server
                        if isinstance(worker_config['worker'], str):
                            # get worker class from str
                            Worker = get_class_from_path(worker_config['worker'])
                        else:
                            # is already a class
                            Worker = worker_config['worker']
                        # instantiate class as an object
                        worker = Worker(processor_id=f'node_{node_id}_worker_{i}')
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

        # seed environment
        self.seed = seed
        if seed is not None:
            seed_stochastic_modules_globally(seed)

        # reset stopwatch
        self.stopwatch.reset()

        # load jobs
        self.jobs_generator = JobsGenerator(**jobs_config)

        # set max sim time
        self.max_simulation_run_time = max_simulation_run_time 

        # reset loggers
        self.save_thread = None
        self._reset_steps_log()
        self._reset_sim_log()
        self.episode_stats = self._init_episode_stats()

        # reset computation workers
        for node_id in self.topology.graph.nodes:
            for worker in self.topology.graph.nodes[node_id]['workers'].values():
                worker.reset()

        # reset communication channels
        for channel in self.topology.channel_id_to_channel.values():
            channel.reset()

        # initialise job queue
        self.job_queue = JobQueue(queue_capacity=job_queue_capacity)

        # initialise trackers
        self.num_jobs_arrived = 0
        self.num_mounted_ops = 0
        self.num_mounted_deps = 0
        # self.num_active_workers = 0
        self.mounted_workers = set()
        self.mounted_channels = set()
        self.jobs_running = {}
        self.jobs_completed = {}
        self.jobs_blocked = {}

        self.job_op_to_worker = {}
        self.job_dep_to_channels = defaultdict(set)

        self.job_idx_to_job_id = {}
        self.job_id_to_job_idx = {}

        self.step_counter = 0

        self.action = None

        # add first job to queue
        self.time_next_job_to_arrive = 0
        self.job_queue.add(self._get_next_job())

        # initialise current job op placement (which job ops are on which workers). Maps job id -> op id -> worker id
        self.job_op_placement = {}

        # initialise current job dep placement (which job deps are on which channels)> Maps job id -> dep id -> worked id
        self.job_dep_placement = {}

        obs = None

        if verbose:
            print(f'Reset cluster environment.')
            print(f'Max sim run time: {self.max_simulation_run_time}')

        return obs

    def _reset_sim_log(self):
        self.sim_log = defaultdict(list)

    def _reset_steps_log(self):
        self.steps_log = defaultdict(list)

    def _reset_steps_log(self):
        self.steps_log = defaultdict(list)

    def _init_step_stats(self):
        step_stats = defaultdict(lambda: 0)

        step_stats['step_counter'] = copy.deepcopy(self.step_counter)
        step_stats['step_start_time'] = copy.deepcopy(self.stopwatch.time())

        step_stats['mean_num_mounted_workers'] = []
        step_stats['mean_num_mounted_channels'] = []

        step_stats['mean_compute_throughput'] = 0
        step_stats['mean_dep_throughput'] = 0
        step_stats['mean_cluster_throughput'] = 0

        step_stats['mean_demand_compute_throughput'] = 0
        step_stats['mean_demand_dep_throughput'] = 0
        step_stats['mean_demand_total_throughput'] = 0
        
        step_stats['mean_compute_overhead_frac'] = []
        step_stats['mean_communication_overhead_frac'] = []

        step_stats['mean_mounted_worker_utilisation_frac'] = []
        # step_stats['mean_mounted_channel_utilisation_frac'] = []

        step_stats['mean_cluster_worker_utilisation_frac'] = []

        # need to init following manually to ensure they're recorded in saved results
        step_stats['num_jobs_completed'] = 0
        # step_stats['num_jobs_running'] = 0
        step_stats['mean_num_jobs_running'] = []
        step_stats['num_jobs_arrived'] = 0
        step_stats['num_jobs_blocked'] = 0

        return step_stats

    def _init_episode_stats(self):
        episode_stats = defaultdict(list)

        episode_stats['num_jobs_arrived'] = 0
        episode_stats['num_jobs_completed'] = 0
        episode_stats['num_jobs_blocked'] = 0

        episode_stats['episode_start_time'] = copy.deepcopy(self.stopwatch.time())

        return episode_stats

    def _get_next_job(self):
        '''Returns next job.'''
        job = self.jobs_generator.sample_job()
        job.register_job_arrived(time_arrived=self.stopwatch.time(), 
                                 job_idx=self.num_jobs_arrived)
        # job.job_id = job.details['job_idx']
        self.time_last_job_arrived = copy.deepcopy(self.stopwatch.time())
        self.time_next_job_to_arrive += self.jobs_generator.sample_interarrival_time(size=None)
        self.job_idx_to_job_id[job.details['job_idx']] = job.job_id
        self.job_id_to_job_idx[job.job_id] = job.details['job_idx']
        self.num_jobs_arrived += 1
        self.last_job_arrived_job_idx = job.details['job_idx']
        self.episode_stats['num_jobs_arrived'] += 1
        return job

    def _perform_lookahead_job_completion_time(self, action, verbose=False):
        # do a lookahead to see how long each placed job will take to complete, and update job details accordingly
        if verbose:
            if len(action.job_ids) > 0:
                print('\nNew job(s) to perform job completion time lookahead for. Performing lookahead...')
            else:
                print(f'No new jobs to perform lookahead for.')

        for job_id in action.job_ids:
            job_idx = self.job_id_to_job_idx[job_id]
            job = self.jobs_running[job_idx]
            if verbose:
                print(f'Job ID: {job_id} | Job idx: {job.details["job_idx"]} | Time arrived: {job.details["time_arrived"]}')

            tmp_stopwatch = Stopwatch()
            tmp_stopwatch.reset()
            # do internal lookahead simulation until job is completed
            lookahead_tick_counter = 1
            tick_counter_to_active_workers_tick_size = defaultdict(list) # e.g. tick_counter_to_active_workers_tick_size = {1: [3, 25], 2: [2, 25]} means at tick 1, 3 workers were active for 25 time units; at tick 2, 2 workers were active for 25 time units
            # tick_counter_to_active_channels_tick_size = defaultdict(list)
            while True:
                # run step tick until an op and/or a dep is completed
                if verbose:
                    print('-'*80)
                    print(f'Performing lookahead tick {lookahead_tick_counter}. Temporary stopwatch time at start of tick: {tmp_stopwatch.time()}')

                # initialise trackers for this tick
                tick_counter_to_active_workers_tick_size[lookahead_tick_counter] = [0, 0]
                # tick_counter_to_active_channels_tick_size[lookahead_tick_counter] = [0, 0]

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
                            job_idx, job_id, op_id = load_job_dep_str(priority_job_op)
                            job = self.jobs_running[job_idx]
                            if job.computation_graph.nodes[op_id]['remaining_run_time'] < shortest_remaining_run_time:
                                # update shortest_remaining_run_time
                                shortest_remaining_run_time = job.computation_graph.nodes[op_id]['remaining_run_time']
                        else:
                            # no op(s) ready or mounted on this worker
                            pass
                    else:
                        # this job has no op(s) mounted on this worker
                        pass

                # NON-FLOW DEPENDENCIES
                # find any ready deps which never became flows and therefore have 0 run time
                # print(f'deps ready: {job.computation_graph.graph["deps_ready"]}')
                non_flow_deps = self.gather_job_ready_non_flow_deps(job)

                # COMMUNICATION
                priority_job_deps = set()
                channel_to_priority_job_dep = {}
                priority_job_dep_to_priority = {}
                priority_job_dep_to_channels = defaultdict(set)
                if len(non_flow_deps) == 0:
                    # no non-flow deps to tick -> need to consider communication overhead this tick
                    # 1) Find highest priority flow on each channel for this job
                    for channel_id, channel in self.topology.channel_id_to_channel.items():
                        if job.details['job_idx'] in channel.mounted_job_idx_to_deps.keys():
                            # get highest priority ready dep on this channel
                            priority_job_dep = self._get_highest_priority_job_dep(channel=channel)
                            if priority_job_dep is not None:
                                # record priority dep and corresponding priority for this channel
                                priority_job_deps.add(priority_job_dep)
                                # if priority_job_dep not in channel.mounted_job_dep_to_priority:
                                    # # TODO TEMP DEBUGGING
                                    # print(f'ERROR')
                                    # print(f'priority_job_dep {priority_job_dep} not found in channel {channel_id} mounted_job_dep_to_priority, this job dep must not have been mounted or has been unmounted or not mean to be trying to place this job.')
                                    # print(f'channel mounted_job_dep_to_priority: {channel.mounted_job_dep_to_priority}')
                                    # print(f'job_id: {job_id}')
                                    # print(f'job_idx: {job_idx}')
                                    # print(f'jobs_running: {self.jobs_running.keys()}')
                                    # print(f'jobs_completed: {self.jobs_completed.keys()}')
                                    # print(f'jobs_blocked: {self.jobs_blocked.keys()}')
                                    # print(f'jobs queued: {self.job_queue.jobs.keys()}')
                                    # import pdb; pdb.set_trace()
                                    # raise Exception()
                                channel_to_priority_job_dep[channel_id] = priority_job_dep
                                priority_job_dep_to_priority[priority_job_dep] = channel.mounted_job_dep_to_priority[priority_job_dep]
                                priority_job_dep_to_channels[priority_job_dep].add(channel_id)
                            else:
                                # no dep(s) ready or mounted on this channel
                                pass
                        else:
                            # this job has no dep(s) mounted on this channel
                            pass

                    # 2) Find any highest priority flows contending for same channel(s) -> only use highest priority flows
                    for job_dep in priority_job_deps:
                        # check if any of this dep's channels are contending with other dep(s)
                        contending_deps = set([job_dep])
                        for channel_id in priority_job_dep_to_channels[job_dep]:
                            _job_dep = channel_to_priority_job_dep[channel_id]
                            if _job_dep != job_dep:
                                # contention found
                                contending_deps.add(_job_dep)
                        if len(contending_deps) > 1:
                            # contention(s) found, resolve
                            winner = max(priority_job_dep_to_priority, key=priority_job_dep_to_priority.get)
                            losers = [contender for contender in contending_deps if contender != winner]
                            # remove losers
                            for loser in losers:
                                loser_channels = priority_job_dep_to_channels[loser]
                                for loser_channel in loser_channels:
                                    del channel_to_priority_job_dep[loser_channel]
                                del priority_job_dep_to_priority[loser]
                                del priority_job_dep_to_channels[loser]
                    
                    # 3) Find the shortest remaining communication time of each remaining highest priority dep on all channels for this job
                    shortest_remaining_communication_time = float('inf')
                    for job_dep in channel_to_priority_job_dep.values():
                        # check if should update shortest remaining communication time of all priority job deps
                        job_idx, job_id, dep_id = load_job_dep_str(job_dep)
                        job = self.jobs_running[job_idx]
                        u, v, k = dep_id
                        if job.computation_graph[u][v][k]['remaining_run_time'] < shortest_remaining_communication_time:
                            # update shortest_remaining_communication_time
                            shortest_remaining_communication_time = job.computation_graph[u][v][k]['remaining_run_time']
                else:
                    # have non-flow dependencies to tick which have 0 communication overhead -> no need to consider overheads this tick
                    shortest_remaining_communication_time = 0

                # PERFORM TICK
                # tick highest priority mounted ready ops and deps on each worker and channel and record any ops or deps which are completed
                tick = min(shortest_remaining_run_time, shortest_remaining_communication_time)

                # record deps ready before op is ticked so that do not tick future ready deps one step early (since job.computation_graph.graph['deps_ready'] is automatically updated by Jobs class when an op is ticked and completed)
                deps_ready = copy.deepcopy(job.computation_graph.graph['deps_ready'])

                # tick ops mounted on workers
                job_idx_to_completed_op_ids = defaultdict(list)
                # self.num_active_workers = 0
                ticked_ops = False
                for worker_id in sorted(worker_to_priority_job_op.keys()):
                    priority_job_op = worker_to_priority_job_op[worker_id]
                    if priority_job_op is not None:
                        # self.num_active_workers += 1
                        node_id = self.topology.graph.graph['worker_to_node'][worker_id]
                        worker = self.topology.graph.nodes[node_id]['workers'][worker_id]
                        # job_idx, job_id, op_id = [int(i) for i in priority_job_op.split('_')]
                        job_idx, job_id, op_id = load_job_dep_str(priority_job_op)
                        job = self.jobs_running[job_idx]
                        if verbose:
                            remaining_run_time = job.computation_graph.nodes[op_id]['remaining_run_time']
                            print(f'Ticking op {op_id} with remaining run time {remaining_run_time} of job index {job_idx} on node {node_id} worker {worker_id} by amount {tick}')
                        job.tick_op(op_id, tick=tick)
                        ticked_ops = True
                        tick_counter_to_active_workers_tick_size[lookahead_tick_counter][0] += 1
                        if op_id in job.computation_graph.graph['ops_completed']:
                            # op was completed
                            job_idx_to_completed_op_ids[job_idx].append(op_id)
                            if verbose:
                                print(f'Op {op_id} of job index {job_idx} completed')
                tick_counter_to_active_workers_tick_size[lookahead_tick_counter][1] = tick

                # tick non-flow deps
                for dep_id in sorted(non_flow_deps):
                    u, v, k = dep_id
                    if verbose:
                        remaining_run_time = job.computation_graph[u][v][k]['remaining_run_time']
                        print(f'Ticking non-flow dep {dep_id} with remaining run time {remaining_run_time} of job index {job_idx} by amount {tick}')
                    job.tick_dep(dep_id, tick=tick)
                    if dep_id in job.computation_graph.graph['deps_completed']:
                        # dep was completed
                        job_idx_to_completed_dep_ids[job_idx].append(dep_id)
                        if verbose:
                            print(f'Non-flow dep {dep_id} of job index {job_idx} completed')

                # tick flows mounted on channels
                ticked_flows = False

                # # OLD: Only tick one dep at a time on each channel according to scheduling priority
                # job_idx_to_completed_dep_ids = defaultdict(list)
                # for channel_id in sorted(channel_to_priority_job_dep.keys()):
                    # priority_job_dep = channel_to_priority_job_dep[channel_id]
                    # if priority_job_dep is not None:
                        # channel = self.topology.channel_id_to_channel[channel_id]
                        # job_idx, job_id, dep_id = load_job_dep_str(priority_job_dep)
                        # job = self.jobs_running[job_idx]
                        # u, v, k = dep_id
                        # if verbose:
                            # remaining_run_time = job.computation_graph[u][v][k]['remaining_run_time']
                            # print(f'Ticking dep {dep_id} with remaining run time {remaining_run_time} of job index {job_idx} on channel {channel_id} by amount {tick}')
                        # job.tick_dep(dep_id, tick=tick)
                        # ticked_flows = True
                        # if dep_id in job.computation_graph.graph['deps_completed']:
                            # # dep was completed
                            # job_idx_to_completed_dep_ids[job_idx].append(dep_id)
                            # if verbose:
                                # print(f'Dep {dep_id} of job index {job_idx} completed')

                # TODO NEW TEMP HACK: Tick all ready deps on each channel regardless of scheduling order (i.e. assume can transfer flows in parallel -> ignore need for scheduling)
                job_idx_to_completed_dep_ids = defaultdict(list)
                # for dep_id in sorted(job.computation_graph.graph['deps_ready']):
                for dep_id in sorted(deps_ready):
                    if dep_id not in non_flow_deps:
                        u, v, k = dep_id
                        if verbose:
                            remaining_run_time = job.computation_graph[u][v][k]['remaining_run_time']
                            print(f'Ticking flow dep {dep_id} with remaining run time {remaining_run_time} of job index {job_idx} by amount {tick}')
                        job.tick_dep(dep_id, tick=tick)
                        ticked_flows = True
                        # tick_counter_to_active_channels_tick_size[lookahead_tick_counter][0] += 1
                        if dep_id in job.computation_graph.graph['deps_completed']:
                            # dep was completed
                            job_idx_to_completed_dep_ids[job_idx].append(dep_id)
                            if verbose:
                                print(f'Flow dep {dep_id} of job index {job_idx} completed')
                # tick_counter_to_active_channels_tick_size[lookahead_tick_counter][1] = tick

                # # record any communicaiton vs. computation bottleneck/overhead time

                # OLD: How I think overhead should be calculated
                # if ticked_ops and ticked_flows:
                    # # neither communication or computation are bottleneck this tick
                    # pass
                # elif ticked_flows and not ticked_ops:
                    # job.details['communication_overhead_time'] += tick
                # elif ticked_ops:
                    # job.details['computation_overhead_time'] += tick

                # NEW: How most papers calc overhead
                if ticked_ops and ticked_flows:
                    job.details['communication_overhead_time'] += tick
                    job.details['computation_overhead_time'] += tick
                    if verbose:
                        print(f'Both communication and computation conducted this tick.')
                elif ticked_flows and not ticked_ops:
                    job.details['communication_overhead_time'] += tick
                    if verbose:
                        print(f'Only communication conducted this tick.')
                elif ticked_ops and not ticked_flows:
                    job.details['computation_overhead_time'] += tick
                    if verbose:
                        print(f'Only computation conducted this tick.')

                # tick stopwatch
                tmp_stopwatch.tick(tick)

                if job.is_training_step_complete():
                    # finished lookahead
                    if verbose:
                        print(f'Lookahead completed -> Job ID {job_id} Job idx {job.details["job_idx"]} lookahead training step time: {tmp_stopwatch.time() * job.num_training_steps}')

                    # TODO HACK TEMP: Assume all workers have same device_type
                    device_type = list(self.topology.graph.graph['worker_types'])[0]

                    if tmp_stopwatch.time() * job.num_training_steps > job.details['max_acceptable_job_completion_time'][device_type]:
                        # maximum acceptable job completion time requirement not met, job blocked
                        if verbose:
                            print(f'Job completion time ({tmp_stopwatch.time() * job.num_training_steps}) exceeds maximum acceptable job completion time ({job.details["max_acceptable_job_completion_time"]}), job blocked.')
                        # register stats of original job with job blocked stats
                        self._register_blocked_job(job.original_job)
                        # remove partitioned job from workers, channels, queue, etc. where necessary
                        self._remove_job_from_cluster(job)
                    else:
                        # calc overall average utilisation of the mounted workers and channels for this job
                        # print(f'\nEvaluating worker utilisation...')
                        # print(f'tick_counter_to_active_workers_tick_size: {tick_counter_to_active_workers_tick_size}')
                        mean_mounted_worker_utilisation_frac = 0
                        for num_active_workers, tick_size in tick_counter_to_active_workers_tick_size.values():
                            mean_mounted_worker_utilisation_frac += ( (num_active_workers / len(job.details['mounted_workers'])) * (tick_size / tmp_stopwatch.time()) )
                            # print(f'num_active_workers: {num_active_workers} | tick_size: {tick_size} | mounted workers: {len(job.details["mounted_workers"])} | stopwatch time: {tmp_stopwatch.time()} -> mean_mounted_worker_utilisation_frac: {mean_mounted_worker_utilisation_frac}')
                        # # print(f'\nEvaluating channel utilisation...')
                        # mean_mounted_channel_utilisation_frac = 0
                        # for num_active_channels, tick_size in tick_counter_to_active_channels_tick_size.values():
                            # mean_mounted_channel_utilisation_frac += ( (num_active_channels / len(job.details['mounted_channels'])) * (tick_size / tmp_stopwatch.time()) )
                            # # print(f'num_active_channels: {num_active_channels} | tick_size: {tick_size} | mounted channels: {len(job.details["mounted_channels"])} | stopwatch time: {tmp_stopwatch.time()} -> mean_mounted_channel_utilisation_frac: {mean_mounted_channel_utilisation_frac}')

                        # reset whole job ready for actual simulation and record lookahead job completion time
                        job.reset_job(details={
                                                'lookahead_job_completion_time': tmp_stopwatch.time() * job.num_training_steps,
                                                'communication_overhead_time': copy.deepcopy(job.details['communication_overhead_time']) * job.num_training_steps,
                                                'computation_overhead_time': copy.deepcopy(job.details['computation_overhead_time']) * job.num_training_steps,
                                                'mounted_workers': job.details['mounted_workers'],
                                                'mounted_channels': job.details['mounted_channels'],
                                                'mean_mounted_worker_utilisation_frac': mean_mounted_worker_utilisation_frac,
                                                # 'mean_mounted_channel_utilisation_frac': mean_mounted_channel_utilisation_frac,
                                                })
                        job.details['job_total_flow_size'] = 0 # track info size of deps which became flows
                        for dep_id in job.computation_graph.edges:
                            run_time = self.set_dep_init_run_time(job, dep_id)
                            if run_time != 0:
                                # dep became a flow, record flow size
                                u, v, k = dep_id
                                job.details['job_total_flow_size'] += job.computation_graph[u][v][k]['size']

                        # # record metrics
                        # self.step_stats['mean_num_mounted_workers'].append(len(self.mounted_workers))
                        # self.step_stats['mean_num_mounted_channels'].append(len(self.mounted_channels))

                    break

                else:
                    # not yet finished training step lookahead, continue
                    pass

                if verbose:
                    print(f'Finished lookahead tick. Temporary stopwatch time at end of tick: {tmp_stopwatch.time()}')

                if math.isinf(tick):
                    raise Exception(f'ERROR: Last tick was infinite, a bug has occurred somewhere.')

                lookahead_tick_counter += 1

            if verbose:
                print(f'Finished all new job lookaheads.')

    def gather_job_ready_non_flow_deps(self, job):
        job_idx = self.job_id_to_job_idx[job.job_id]
        non_flow_deps = set()
        for dep_id in job.computation_graph.graph['deps_ready']:
            u, v, k = dep_id
            src_job_op = gen_job_dep_str(job_idx, job.job_id, u)
            dst_job_op = gen_job_dep_str(job_idx, job.job_id, v)
            src_worker = self.job_op_to_worker[src_job_op]
            dst_worker = self.job_op_to_worker[dst_job_op]
            # print(f'dep_id {dep_id} | size: {job.computation_graph[u][v][k]["size"]} | src_worker: {src_worker} | dst_worker: {dst_worker}')
            # print(f'Considering dep {dep_id} with src {self.topology.graph.graph["worker_to_node"][src_worker]} dst {self.topology.graph.graph["worker_to_node"][dst_worker]} size {job.computation_graph[u][v][k]["size"]}') # DEBUG
            if job.computation_graph[u][v][k]['size'] == 0:
                # 0 data transferred -> not a flow
                non_flow_deps.add(dep_id)
            elif (self.topology.graph.graph['worker_to_node'][src_worker] == self.topology.graph.graph['worker_to_node'][dst_worker]):
                # src == dst server node -> not a flow
                non_flow_deps.add(dep_id)
            else:
                # is a flow
                pass
        return non_flow_deps

    def set_dep_init_run_time(self, job, dep_id):
        u, v, k = dep_id
        job_idx = self.job_id_to_job_idx[job.job_id]
        # src_job_op = f'{job_idx}_{job.job_id}_{u}'
        # dst_job_op = f'{job_idx}_{job.job_id}_{v}'
        src_job_op = gen_job_dep_str(job_idx, job.job_id, u)
        dst_job_op = gen_job_dep_str(job_idx, job.job_id, v)
        src_worker = self.job_op_to_worker[src_job_op]
        dst_worker = self.job_op_to_worker[dst_job_op]
        if self.topology.graph.graph['worker_to_node'][src_worker] == self.topology.graph.graph['worker_to_node'][dst_worker]:
            # src == dst server node -> not a flow
            run_time = 0
        elif job.computation_graph[u][v][k]['size'] == 0:
            # 0 data transferred -> not a flow
            run_time = 0
        else:
            run_time = job.computation_graph[u][v][k]['init_run_time']
        job.set_dep_init_run_time(dep_id, run_time)
        return run_time

    def _get_highest_priority_job_op(self, worker):
        '''
        Takes a worker processor object and returns a string identifying which operation
        which is ready to run on the worker has the highest priority. If no
        operation is available to run, will return None.
        '''
        priority_job_op = None
        for job_idx in sorted(worker.mounted_job_idx_to_ops.keys()):
            job = self.jobs_running[job_idx]
            for op_id in sorted(worker.mounted_job_idx_to_ops[job_idx]):
                if op_id in job.computation_graph.graph['ops_ready']:
                    # op is ready to run
                    # job_op = f'{job_idx}_{job.job_id}_{op_id}'
                    # job_op = json.dumps(job_idx) + '_' + json.dumps(job.job_id) + '_' + json.dumps(op_id)
                    job_op = gen_job_dep_str(job_idx, job.job_id, op_id)
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

    def _get_highest_priority_job_dep(self, channel):
        '''
        Takes a channel object and returns a string identifying which dependency
        which is ready to run on the channel has the highest priority. If no
        channel is available to run, will return None.
        '''
        priority_job_dep = None
        for job_idx in sorted(channel.mounted_job_idx_to_deps.keys()):
            job = self.jobs_running[job_idx]
            for dep_id in sorted(channel.mounted_job_idx_to_deps[job_idx]):
                if dep_id in job.computation_graph.graph['deps_ready']:
                    # dep is ready to run
                    job_dep = gen_job_dep_str(job_idx, job.job_id, dep_id)
                    if priority_job_dep is None:
                        # not yet considered any other deps, set this dep as priority dep
                        priority_job_dep = job_dep
                    else:
                        # check if dep has higher priority than current highest priority dep found so far
                        if channel.mounted_job_dep_to_priority[job_dep] > channel.mounted_job_dep_to_priority[priority_job_dep]:
                            # dep has higher priority, update priority job dep
                            priority_job_dep = job_dep
                else:
                    # dep not yet ready to run
                    pass
        return priority_job_dep

    def step(self,
             action: Action,
             verbose: bool = False):
        # verbose = True # DEBUG

        self.action = action
        # if action.actions['op_placement'] is None and action.actions['op_schedule'] is None and action.actions['dep_placement'] is None and action.actions['dep_schedule'] is None:
            # raise Exception(f'>=1 action must != None.')
        if self.path_to_save is not None and self.use_sqlite_database and self.step_counter % self.save_freq == 0:
            # saved logs at end of previous step, can reset in-memory logs for this step
            self._reset_sim_log()
            self._reset_steps_log()

        self.step_stats = self._init_step_stats()

        if verbose:
            print('')
            print('-'*80)
            print(f'Step: {self.step_counter}')

        # register any blocked jobs (jobs which were in queue at last step but not handled by action)
        for job_id, job in self.job_queue.jobs.items():
            if job_id not in action.job_ids:
                self._register_blocked_job(job)
                if verbose:
                    print(f'Job with job_idx {job.details["job_idx"]} was blocked.')

        # execute control plane decisions
        # self._prioritise_jobs() # TODO
        if action.actions['op_partition'] is not None:
            self._partition_ops(action.actions['op_partition'],
                                verbose=verbose)
        if action.actions['op_placement'] is not None:
            self._place_ops(action.actions['op_placement'],
                             verbose=verbose)
        if action.actions['op_schedule'] is not None:
            self._schedule_ops(action.actions['op_schedule'],
                                verbose=verbose)
        if action.actions['dep_placement'] is not None:
            self._place_deps(action.actions['dep_placement'],
                             verbose=verbose)
        if action.actions['dep_schedule'] is not None:
            self._schedule_deps(action.actions['dep_schedule'],
                                verbose=verbose)

        # given control plane decisions, perform job completion time lookahead for new mounted jobs
        self._perform_lookahead_job_completion_time(action, verbose=verbose)

        # run step until next job arrives, a job is completed, or the simulation is completed
        step_done = False
        while not step_done:
            if verbose:
                print('-'*80)
                print(f'Performing cluster tick. Stopwatch time at start of tick: {self.stopwatch.time()}')

            # tick simulator stopwatch until a job is completed, a new job arrives, or the simulation finishes
            tick = min(self.time_next_job_to_arrive - self.stopwatch.time(), self.max_simulation_run_time - self.stopwatch.time())
            for job in self.jobs_running.values():
                elapsed_run_time = self.stopwatch.time() - job.details['time_started']
                remaining_run_time = job.details['lookahead_job_completion_time'] - elapsed_run_time
                tick = min(tick, remaining_run_time)

            # record stats this tick
            compute_info_processed, dep_info_processed, flow_info_processed, cluster_info_processed = 0, 0, 0, 0
            demand_compute_info_processed, demand_dep_info_processed, demand_total_info_processed = 0, 0, 0
            self.mounted_workers, self.mounted_channels = set(), set()
            # mounted_worker_utilisation, mounted_channel_utilisation = [], []
            mounted_worker_utilisation = []
            for job in self.jobs_running.values():

                frac_job_completed_this_tick = tick / job.details['lookahead_job_completion_time']

                self.step_stats['compute_info_processed'] += (job.details['job_total_op_memory_cost'] * frac_job_completed_this_tick)
                self.step_stats['dep_info_processed'] += (job.details['job_total_dep_size'] * frac_job_completed_this_tick)
                self.step_stats['flow_info_processed'] += (job.details['job_total_flow_size'] * frac_job_completed_this_tick)
                self.step_stats['cluster_info_processed'] += ((job.details['job_total_op_memory_cost'] + job.details['job_total_dep_size']) * frac_job_completed_this_tick)

                self.step_stats['demand_compute_info_processed'] += (job.original_job.details['job_total_op_memory_cost'] * frac_job_completed_this_tick)
                self.step_stats['demand_dep_info_processed'] += (job.original_job.details['job_total_dep_size'] * frac_job_completed_this_tick)
                self.step_stats['demand_total_info_processed'] += ((job.original_job.details['job_total_op_memory_cost'] + job.original_job.details['job_total_dep_size']) * frac_job_completed_this_tick)

                self.step_stats['mean_compute_overhead_frac'].append(job.details['computation_overhead_time'] / job.details['lookahead_job_completion_time'])
                self.step_stats['mean_communication_overhead_frac'].append(job.details['communication_overhead_time'] / job.details['lookahead_job_completion_time'])

                self.mounted_workers.update(job.details['mounted_workers'])
                self.mounted_channels.update(job.details['mounted_channels'])

                mounted_worker_utilisation.append(job.details['mean_mounted_worker_utilisation_frac'])
                # mounted_channel_utilisation.append(job.details['mean_mounted_channel_utilisation_frac'])

            self.step_stats['mean_num_jobs_running'].append(len(self.jobs_running))

            self.step_stats['mean_num_mounted_workers'].append(len(self.mounted_workers))
            self.step_stats['mean_num_mounted_channels'].append(len(self.mounted_channels))

            if len(mounted_worker_utilisation) > 0:
                self.step_stats['mean_mounted_worker_utilisation_frac'].append(np.mean(mounted_worker_utilisation))
                self.step_stats['mean_cluster_worker_utilisation_frac'].append( ( len(self.mounted_workers) / self.topology.graph.graph['num_workers'] ) * np.mean(mounted_worker_utilisation) )
            else:
                self.step_stats['mean_mounted_worker_utilisation_frac'].append(0)
                self.step_stats['mean_cluster_worker_utilisation_frac'].append(0)
            # self.step_stats['mean_mounted_channel_utilisation_frac'].append(np.mean(mounted_channel_utilisation))

            # perform tick
            self.stopwatch.tick(tick)

            if verbose:
                print(f'Ticked cluster by amount {tick}. Stopwatch time at end of tick: {self.stopwatch.time()}')

            # register any jobs completed this tick
            jobs_completed = []
            for job in self.jobs_running.values():
                elapsed_run_time = self.stopwatch.time() - job.details['time_started']
                remaining_run_time = (job.details['lookahead_job_completion_time'] - elapsed_run_time) - self.machine_epsilon
                if verbose:
                    print(f'Running job_idx {job.details["job_idx"]} job_id {job.job_id} remaining run time: {remaining_run_time}')
                if remaining_run_time <= 0:
                    # job was completed this tick
                    jobs_completed.append(job)
                    step_done = True
            for job in jobs_completed:
                self._register_completed_job(job)
                if verbose:
                    print(f'Job with job_idx {job.details["job_idx"]} completed. Time arrived: {job.details["time_arrived"]} | Time completed: {job.details["time_completed"]}')

            # check if next job should arrive
            if len(self.jobs_generator) > 0:
                if verbose:
                    print(f'Time next job due to arrive: {self.time_next_job_to_arrive}')
                # if self.stopwatch.time() > self.time_next_job_to_arrive:
                    # raise Exception(f'Stopwatch time is {self.stopwatch.time()} but next job should have arrived at {self.time_next_job_to_arrive}')
                if (self.stopwatch.time() + self.machine_epsilon) >= self.time_next_job_to_arrive:
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
            if self.is_done(verbose=verbose):
                step_done = True

        # log step-level data
        self.step_stats['step_end_time'] = self.stopwatch.time()

        for metric in [
                       'mean_num_jobs_running',

                       'mean_num_mounted_workers',
                       'mean_num_mounted_channels',

                       'mean_compute_overhead_frac',
                       'mean_communication_overhead_frac',
                       ]:
            if len(self.step_stats[metric]) > 0:
                self.step_stats[metric] = np.mean(self.step_stats[metric])
            else:
                self.step_stats[metric] = 0

        self.step_stats['step_time'] = self.step_stats['step_end_time'] - self.step_stats['step_start_time']
        for throughput_metric, info_processed in {
                'mean_compute_throughput': 'compute_info_processed',
                'mean_dep_throughput': 'dep_info_processed',
                'mean_flow_throughput': 'flow_info_processed',
                'mean_cluster_throughput': 'cluster_info_processed',

                'mean_demand_compute_throughput': 'demand_compute_info_processed',
                'mean_demand_dep_throughput': 'demand_dep_info_processed',
                'mean_demand_total_throughput': 'demand_total_info_processed',
                }.items():
            if self.step_stats[info_processed] != 0 and self.step_stats['step_time'] != 0:
                self.step_stats[throughput_metric] = self.step_stats[info_processed] / self.step_stats['step_time'] 
            else:
                self.step_stats[throughput_metric] = 0

        # self.step_stats['mean_num_mounted_workers'] = len(self.mounted_workers)
        # self.step_stats['mean_num_mounted_channels'] = len(self.mounted_channels)
        # self.step_stats['mean_worker_compute_utilisation'] = self.step_stats['mean_num_queued_workers'] / len(list(self.topology.graph.graph['worker_to_node']))
        self.step_stats['job_queue_length'] = len(self.job_queue)
        self._update_steps_log(copy.deepcopy(self.step_stats))

        # log episode-level data
        for metric in [
                    'compute_info_processed',
                    'dep_info_processed',
                    'flow_info_processed',
                    'cluster_info_processed',

                    'demand_compute_info_processed',
                    'demand_dep_info_processed',
                    'demand_total_info_processed',
                    ]:
            self.episode_stats[metric].append(self.step_stats[metric])

        # move to next step
        self.step_counter += 1

        if self.is_done():
            # register any jobs currently running as having been blocked
            blocked_jobs = []
            for job in self.jobs_running.values():
                blocked_jobs.append(job)
            for job in blocked_jobs:
                # register the original job as having been blocked
                self._register_blocked_job(job.original_job)
                # remove partitioned job from workers, channels, queue, etc. where necessary
                self._remove_job_from_cluster(job)

            # update episode-level data as necessary

            self.episode_stats['episode_end_time'] = copy.deepcopy(self.stopwatch.time())

            self.episode_stats['episode_time'] = self.episode_stats['episode_end_time'] - self.episode_stats['episode_start_time']

            for throughput_metric, info_processed in {
                        'mean_compute_throughput': 'compute_info_processed',
                        'mean_dep_throughput': 'dep_info_processed',
                        'mean_flow_throughput': 'flow_info_processed',
                        'mean_cluster_throughput': 'cluster_info_processed',

                        'mean_demand_compute_throughput': 'demand_compute_info_processed',
                        'mean_demand_dep_throughput': 'demand_dep_info_processed',
                        'mean_demand_total_throughput': 'demand_total_info_processed',
                        }.items():
                self.episode_stats[info_processed] = np.sum(self.episode_stats[info_processed])
                if self.episode_stats[info_processed] != 0 and self.episode_stats['episode_time'] != 0:
                    self.episode_stats[throughput_metric] = self.episode_stats[info_processed] / self.episode_stats['episode_time'] 
                else:
                    self.episode_stats[throughput_metric] = 0

        # save logs
        if self.path_to_save is not None:
            if self.step_counter % self.save_freq == 0 or self.is_done():
                self.save()
                if self.is_done():
                    self.save_thread.join()

        obs, action_set, reward, done, info = None, None, None, self.is_done(), None


        return obs, action_set, reward, done, info

    def _update_flow_run_times(self, job):
        pass

    def _partition_ops(self, action, verbose=False):
        op_partition = action
        if verbose:
            if len(op_partition) > 0:
                print('New job op(s) to partition. Partitioning...')
            else:
                print(f'No new job ops to partition.')
        for job_id in op_partition.action:
            # update job in queue with partitioned job
            orig_job = self.job_queue.jobs[job_id]
            if verbose:
                print(f'Job ID: {job_id} | Job idx: {orig_job.details["job_idx"]} | Time arrived: {orig_job.details["time_arrived"]}')
                for op_id in orig_job.computation_graph.nodes:
                    num_partitions = op_partition.action[job_id][op_id]
                    if num_partitions > 1:
                        print(f'Op ID {op_id} partitioned into {num_partitions} sub-ops')
                    else:
                        print(f'Op ID {op_id} not partitioned.')
            self.job_queue.jobs[job_id] = op_partition.partitioned_jobs[job_id]

    def _place_ops(self, action, verbose=False):
        # # DEBUG
        # verbose = True
        # print(f'Placing ops for action {action}') 

        op_placement = action.action
        if verbose:
            if len(op_placement) > 0:
                print('New job op(s) to place on cluster. Placing...')
            else:
                print(f'No new job ops to place on cluster.')
        for job_id in op_placement:
            job = self.job_queue.jobs[job_id]
            if verbose:
                print(f'Job ID: {job_id} | Job idx: {job.details["job_idx"]} | Time arrived: {job.details["time_arrived"]}')

            # 1. place ops for this job
            for op_id in op_placement[job_id]:
                worker_id = op_placement[job_id][op_id]
                node_id = self.topology.graph.graph['worker_to_node'][worker_id]
                worker = self.topology.graph.nodes[node_id]['workers'][worker_id]
                rules_broken = check_if_ramp_op_placement_rules_broken(worker, job)
                if len(rules_broken) > 0:
                    raise Exception(f'Placement for job index {job.details["job_idx"]} job ID {job_id} op ID {op_id} worker ID {worker_id} breaks the following Ramp rules: {rules_broken}')
                else:
                    worker.mount(job=job, op_id=op_id)
                    job.details['mounted_workers'].add(worker_id)
                    # self.mounted_workers.add(worker_id)
                    self.num_mounted_ops += 1
                    job.reset_op_remaining_run_time(op_id, device_type=self.topology.graph.nodes[node_id]['workers'][worker_id].device_type)
                    # self.job_op_to_worker[f'{job.details["job_idx"]}_{job.job_id}_{op_id}'] = worker_id
                    self.job_op_to_worker[gen_job_dep_str(job.details['job_idx'], job.job_id, op_id)] = worker_id
                    if verbose:
                        print(f'Op ID {op_id} of job index {job.details["job_idx"]} placed on node ID {node_id} worker ID {worker_id}')

            # 2. update the dependency (flow) communication times for this job based on the placement
            self._update_flow_run_times(job)
            
            # 3. register the job as having began to start timers etc
            self._register_running_job(job)

            # 4. update cluster tracking of current job placement
            self.job_op_placement[job_id] = op_placement[job_id]

    def _place_deps(self, action, verbose=False):
        # verbose = True # DEBUG
        dep_placement = action.action
        if verbose:
            if len(dep_placement) > 0:
                print('New job dep(s) to place on cluster. Placing...')
            else:
                print(f'No new job deps to place on cluster.')
        for job_id in dep_placement:
            job_idx = self.job_id_to_job_idx[job_id]
            job = self.jobs_running[job_idx]
            if verbose:
                print(f'Job ID: {job_id} | Job idx: {job.details["job_idx"]} | Time arrived: {job.details["time_arrived"]}')
            for dep_id in dep_placement[job_id].keys():
                for channel_id in dep_placement[job_id][dep_id]:
                    if channel_id is not None:
                        # dep was placed on a channel
                        channel = self.topology.channel_id_to_channel[channel_id]
                        rules_broken = check_if_ramp_dep_placement_rules_broken(channel, job)
                        if len(rules_broken) > 0:
                            raise Exception(f'Dep placement for job index {job.details["job_idx"]} job ID {job_id} dep ID {dep_id} channel ID {channel_id} breaks the following Ramp rules: {rules_broken}')
                        else:
                            channel.mount(job, dep_id)
                            job.details['mounted_channels'].add(channel_id)
                            # self.mounted_channels.add(channel_id)
                            self.num_mounted_deps += 1
                            job.reset_dep_remaining_run_time(dep_id)
                            self.job_dep_to_channels[gen_job_dep_str(job_idx, job.job_id, dep_id)].add(channel_id)
                            if verbose:
                                print(f'Dep ID {dep_id} of job index {job.details["job_idx"]} placed on channel ID {channel_id}')
                    else:
                        if verbose:
                            print(f'Dep ID {dep_id} of job index {job.details["job_idx"]} not placed on any channel since not a flow')
                        pass

            # update cluster tracking of current job placement
            self.job_dep_placement[job_id] = dep_placement[job_id]

    def _schedule_ops(self, action, verbose=False):
        '''Sets scheduling priority for mounted ops on each worker.'''
        op_schedule = action.action
        for worker_id in op_schedule.keys():
            node_id = self.topology.graph.graph['worker_to_node'][worker_id]
            worker = self.topology.graph.nodes[node_id]['workers'][worker_id]
            for job_idx in sorted(worker.mounted_job_idx_to_ops.keys()):
                job = self.jobs_running[job_idx]
                for op_id in sorted(worker.mounted_job_idx_to_ops[job_idx]):
                    # worker.mounted_job_op_to_priority[f'{job_idx}_{job.job_id}_{op_id}'] = op_schedule[worker_id][job.job_id][op_id]
                    worker.mounted_job_op_to_priority[gen_job_dep_str(job_idx, job.job_id, op_id)] = op_schedule[worker_id][job.job_id][op_id]

    def _schedule_deps(self, action, verbose=False):
        '''Sets scheduling priority for mounted deps on each channel.'''
        # verbose = True # DEBUG
        dep_schedule = action.action
        # print(f'dep_schedule: {dep_schedule}')
        for channel_id in dep_schedule.keys():
            if channel_id is not None:
                # dep was placed on a channel
                channel = self.topology.channel_id_to_channel[channel_id]
                # print(f'channel_id: {channel_id} | channel mounted_job_idx_to_deps: {channel.mounted_job_idx_to_deps}')
                for job_idx in sorted(channel.mounted_job_idx_to_deps.keys()):
                    job = self.jobs_running[job_idx]
                    for dep_id in sorted(channel.mounted_job_idx_to_deps[job_idx]):
                        # if dep_id not in dep_schedule[channel_id][job.job_id]:
                            # print(f'ERROR')
                            # print(f'dep_id {dep_id} not found in dep_schedule[{channel_id}][{job.job_id}] {dep_schedule[channel_id][job.job_id]}')
                            # print(f'channel_id: {channel_id}')
                            # print(f'job_id: {job.job_id}')
                            # print(f'job_idx: {job_idx}')
                            # print(f'jobs_running: {self.jobs_running.keys()}')
                            # print(f'jobs_completed: {self.jobs_completed.keys()}')
                            # print(f'jobs_blocked: {self.jobs_blocked.keys()}')
                            # print(f'jobs queued: {self.job_queue.jobs.keys()}')
                            # import pdb; pdb.set_trace()
                            # raise Exception()
                        channel.mounted_job_dep_to_priority[gen_job_dep_str(job_idx, job.job_id, dep_id)] = dep_schedule[channel_id][job.job_id][dep_id]
            else:
                # dep not a flow so not placed on channel so no need to schedule
                pass

    def _register_running_job(self, job):
        job.register_job_running(time_started=self.stopwatch.time())
        self.jobs_running[job.details['job_idx']] = job
        self.job_queue.remove(job)
        # set dependency run times
        for dep_id in job.computation_graph.edges:
            self.set_dep_init_run_time(job, dep_id)

    def _remove_job_from_cluster(self, job):
        # print(f'REMOVING JOB ID {job.job_id} JOB IDX {job.details["job_idx"]} FROM CLUSTER!!!') # DEBUG
        if job.job_id in self.job_queue.jobs:
            # job currently in queue, remove
            self.job_queue.remove(job)

        if job.details['job_idx'] in self.jobs_running.keys():
            # job currently running, remove
            del self.jobs_running[job.details['job_idx']]

        # unmount any ops which were previously mounted
        for op_id in job.computation_graph.nodes:
            job_dep_str = gen_job_dep_str(job.details['job_idx'], job.job_id, op_id)
            if job_dep_str in self.job_op_to_worker:
                # op was mounted onto a worker, need to unmount
                worker_id = self.job_op_to_worker[gen_job_dep_str(job.details['job_idx'], job.job_id, op_id)]
                node_id = self.topology.graph.graph['worker_to_node'][worker_id]
                worker = self.topology.graph.nodes[node_id]['workers'][worker_id]
                worker.unmount(job=job, op_id=op_id)
                self.num_mounted_ops -= 1
                del self.job_op_to_worker[gen_job_dep_str(job.details['job_idx'], job.job_id, op_id)]

        # unmount any deps which were previously mounted
        for dep_id in job.computation_graph.edges:
            job_idx = job.details['job_idx']
            job_dep = gen_job_dep_str(job_idx, job.job_id, dep_id)
            if job_dep in self.job_dep_to_channels:
                # dep was mounted onto a channel, need to unmount
                channel_ids = self.job_dep_to_channels[job_dep]
                for channel_id in channel_ids:
                    channel = self.topology.channel_id_to_channel[channel_id]
                    channel.unmount(job, dep_id)
                    self.num_mounted_deps -= 1
                del self.job_dep_to_channels[job_dep]

        # clear job from current cluster placement tracker
        if job.job_id in self.job_op_placement:
            del self.job_op_placement[job.job_id]
        if job.job_id in self.job_dep_placement:
            del self.job_dep_placement[job.job_id]

    def _register_completed_job(self, job):
        # record completion time
        job.register_job_completed(time_completed=self.stopwatch.time())

        # update loggers
        self.jobs_completed[job.details['job_idx']] = job
        self.step_stats['num_jobs_completed'] += 1

        self.episode_stats['num_jobs_completed'] += 1

        # TODO HACK TEMP: Assume all workers have same device_type
        device_type = list(self.topology.graph.graph['worker_types'])[0]

        self.episode_stats['job_completion_time'].append(job.details['time_completed'] - job.details['time_arrived'])
        self.episode_stats['job_completion_time_speedup'].append(job.details['job_sequential_completion_time'][device_type] / (job.details['time_completed'] - job.details['time_arrived']))
        self.episode_stats['job_communication_overhead_time'].append(job.details['communication_overhead_time'])
        self.episode_stats['job_computation_overhead_time'].append(job.details['computation_overhead_time'])

        self.episode_stats['jobs_completed_num_nodes'].append(len(job.computation_graph.nodes))
        self.episode_stats['jobs_completed_num_edges'].append(len(job.computation_graph.edges))
        self.episode_stats['jobs_completed_total_operation_memory_cost'].append(job.job_total_operation_memory_cost)
        self.episode_stats['jobs_completed_total_dependency_size'].append(job.job_total_dependency_size)
        self.episode_stats['jobs_completed_max_partitions_per_op'].append(job.details['max_partitions_per_op'])
        self.episode_stats['jobs_completed_job_sequential_completion_time'].append(job.details['job_sequential_completion_time'][device_type])
        self.episode_stats['jobs_completed_max_acceptable_job_completion_time_frac'].append(job.max_acceptable_job_completion_time_frac)
        self.episode_stats['jobs_completed_max_acceptable_job_completion_time'].append(job.details['max_acceptable_job_completion_time'][device_type])
        self.episode_stats['jobs_completed_num_mounted_workers'].append(len(job.details['mounted_workers']))
        self.episode_stats['jobs_completed_num_mounted_channels'].append(len(job.details['mounted_channels']))
        self.episode_stats['jobs_completed_mean_mounted_worker_utilisation_frac'].append(job.details['mean_mounted_worker_utilisation_frac'])

        self.episode_stats['jobs_completed_original_demand_num_nodes'].append(len(job.original_job.computation_graph.nodes))
        self.episode_stats['jobs_completed_original_demand_num_edges'].append(len(job.original_job.computation_graph.edges))
        self.episode_stats['jobs_completed_original_demand_total_operation_memory_cost'].append(job.original_job.job_total_operation_memory_cost)
        self.episode_stats['jobs_completed_original_demand_total_dependency_size'].append(job.original_job.job_total_dependency_size)

        # update simulator workers and channels
        self._remove_job_from_cluster(job)
            
    def _register_blocked_job(self, job):
        if job.job_id in self.job_queue.jobs:
            # job currently in queue, remove
            self.job_queue.remove(job)

        if job.details['job_idx'] in self.jobs_running.keys():
            # job currently running, remove
            del self.jobs_running[job.details['job_idx']]

        if job.details['job_idx'] in self.jobs_blocked:
            # job has already been registered as blocked, no need to re-register
            pass
        else:
            self.jobs_blocked[job.details['job_idx']] = job

            # update loggers
            self.step_stats['num_jobs_blocked'] += 1
            # TODO: Record cause_of_block
            # self.step_stats['jobs_blocked_cause'] = cause_of_block

            # TODO HACK TEMP: Assume all workers have same device_type
            device_type = list(self.topology.graph.graph['worker_types'])[0]

            self.episode_stats['num_jobs_blocked'] += 1

            self.episode_stats['jobs_blocked_num_nodes'].append(len(job.computation_graph.nodes))
            self.episode_stats['jobs_blocked_num_edges'].append(len(job.computation_graph.edges))
            self.episode_stats['jobs_blocked_total_operation_memory_cost'].append(job.job_total_operation_memory_cost)
            self.episode_stats['jobs_blocked_total_dependency_size'].append(job.job_total_dependency_size)
            self.episode_stats['jobs_blocked_job_sequential_completion_time'].append(job.details['job_sequential_completion_time'][device_type])
            self.episode_stats['jobs_blocked_max_acceptable_job_completion_time_frac'].append(job.max_acceptable_job_completion_time_frac)
            self.episode_stats['jobs_blocked_max_acceptable_job_completion_time'].append(job.details['max_acceptable_job_completion_time'][device_type])

            self.episode_stats['jobs_blocked_original_demand_num_nodes'].append(len(job.original_job.computation_graph.nodes))
            self.episode_stats['jobs_blocked_original_demand_num_edges'].append(len(job.original_job.computation_graph.edges))
            self.episode_stats['jobs_blocked_original_demand_total_operation_memory_cost'].append(job.original_job.job_total_operation_memory_cost)
            self.episode_stats['jobs_blocked_original_demand_total_dependency_size'].append(job.original_job.job_total_dependency_size)

    def is_done(self, verbose=False):
        '''Checks if simulation has finished.'''
        done = False

        if self.max_simulation_run_time is not None:
            if self.stopwatch.time() >= self.max_simulation_run_time:
                done = True
                if verbose:
                    print(f'Maximum simulation run time reached -> done.')

        if len(self.jobs_generator) == 0 and len(self.jobs_running) == 0 and len(self.job_queue) == 0:
            done = True
            if verbose:
                print(f'No more jobs running, in queue, or left to sample -> done.')

        return done

    def __str__(self):
        descr = f'Cluster {type(self)}'
        descr += f' | Topology: {type(self.topology)} with {len(self.topology.graph.nodes)} nodes and {len(self.topology.graph.edges)} edges'
        descr += f' | Topology config: {self.topology_config}'
        descr += f' | Node config: {self.node_config}'
        return descr

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
