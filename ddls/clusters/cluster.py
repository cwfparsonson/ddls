from ddls.clusters.job_queue import JobQueue
from ddls.utils import Sampler
from ddls.topologies.topology import Topology
from ddls.topologies.torus import Torus
from ddls.demands.jobs.job import Job
from ddls.distributions.distribution import Distribution
from ddls.utils import seed_stochastic_modules_globally, Stopwatch

from typing import Any, Union
import copy
import math










class Cluster:
    def __init__(self,
                 topology_config: dict,
                 node_config: dict):
        '''
        Number of nodes resulting from topology_config must be equal to the total number
        of nodes specified in the node_config 
        
        The 'worker' in the node_config dict should be an **uninnstantiated** ddls processor.
        '''
        self.topology_config = topology_config
        self.node_config = node_config

        # init topology
        self.topology = self._init_topology(topology_config)
        self._check_topology_node_configs_valid(self.topology, node_config)
        
        # populate topology with nodes specified by node_config
        self._populate_topology(self.topology, node_config)

        self.stopwatch = Stopwatch()
        
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
              max_simulation_run_time: Union[int, float],
              job_sampling_mode: str = 'remove_and_repeat',
              job_queue_capacity: int = 10,
              verbose=False):
        self.job_sampler = Sampler(pool=jobs, sampling_mode=job_sampling_mode)
        self.job_interarrival_time_dist = job_interarrival_time_dist 
        self.max_simulation_run_time = max_simulation_run_time 

        self.job_queue = JobQueue(queue_capacity=job_queue_capacity)

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
        job.details['time_arrived'] = copy.deepcopy(self.stopwatch.time())
        job.details['time_started'] = None
        job.details['time_completed'] = None
        job.details['job_idx'] = copy.deepcopy(self.num_jobs_arrived)
        self.time_last_job_arrived = copy.deepcopy(self.stopwatch.time())
        self.time_next_job_to_arrive += self.job_interarrival_time_dist.sample(size=None)
        self.num_jobs_arrived += 1
        return job

    def _register_running_job(self, job):
        job.details['time_started'] = copy.deepcopy(self.stopwatch.time())
        self.jobs_running[job.details['job_idx']] = job
        self.job_queue.remove(job)

    def _register_completed_job(self, job):
        job.details['time_completed'] = copy.deepcopy(self.stopwatch.time())
        self.jobs_completed[job.details['job_idx']] = job
        del self.jobs_running[job.details['job_idx']]
        for op_id in job.computation_graph.nodes:
            worker_id = self.job_op_to_worker[f'{job.details["job_idx"]}_{job.job_id}_{op_id}']
            node_id = self.worker_to_node[worker_id]
            worker = self.topology.graph.nodes[node_id]['workers'][worker_id].unmount(job=job, op_id=op_id)
            del self.job_op_to_worker[f'{job.details["job_idx"]}_{job.job_id}_{op_id}']
            
    def _register_blocked_job(self, job):
        self.jobs_blocked[job.details['job_idx']] = job

    def __str__(self):
        descr = f'Cluster {type(self)}'
        descr += f' | Topology: {type(self.topology)} with {len(self.topology.graph.nodes)} nodes and {len(self.topology.graph.edges)}'
        descr += f' | Topology config: {self.topology_config}'
        descr += f' | Node config: {self.node_config}'
        return descr
    
    def step(self, 
             actions,
             verbose=False):
        if verbose:
            print('')
            print('-'*48)
            print(f'Step: {self.step_counter}')

        self._prioritise_jobs()
        self._partition_jobs()
        self._place_jobs(actions['job_placement'],
                         verbose=verbose)
        self._schedule_jobs()

        # run step until either next job arrives or until there are no more mounted ops left to run
        step_done = False
        while not step_done:
            if verbose:
                print(f'Stopwatch time: {self.stopwatch.time()}')

            # check if next job should arrive
            if self.stopwatch.time() > self.time_next_job_to_arrive:
                raise Exception(f'Stopwatch time is {self.stopwatch.time()} but next job should have arrived at {self.time_next_job_to_arrive}')
            elif self.stopwatch.time() == self.time_next_job_to_arrive:
                next_job = self._get_next_job()
                if verbose:
                    print(f'Next job with job_idx {next_job.details["job_idx"]} arrived. Added to queue.')
                if self.job_queue.can_fit(next_job):
                    self.job_queue.add(next_job)
                else:
                    self._register_blocked_job(next_job)
                step_done = True
            else:
                pass

            # keep running step until: max sim time reached; next job arrives; or 
            if not step_done:
                # run ops which have been placed on workers and records which op(s) completed
                max_tick = min(self.time_next_job_to_arrive - self.stopwatch.time(), self.max_simulation_run_time - self.stopwatch.time())
                job_idx_to_completed_op_ids = self._tick_workers(actions['job_schedule'], max_tick=max_tick)
                if verbose:
                    print(f'job_idx_to_completed_op_ids: {job_idx_to_completed_op_ids}')

                # TODO: 
                # 1. Take completed ops and (1) do any communication necessary (2) update parent_dependencies_satisfied
                # 2. Implement more efficient way of tracking and updating shortest_remaining_run_time of mounted ops that doesn't need to iterate every job op each time
                # 3. Implement notion of training steps where must keep executing job graph until all training steps completed (i.e. need to count training steps and not unmount until all training steps completed)

                # TEMPORARY: Assume no network communication overhead -> child dependencies of completed ops immediately satisfied
                for job_idx, op_ids in job_idx_to_completed_op_ids.items():
                    job = self.jobs_running[job_idx]
                    for op_id in op_ids:
                        for child_dependency in job.computation_graph.out_edges(op_id):
                            job.register_satisfied_dependency(child_dependency)

                # register any completed jobs
                for job_idx in job_idx_to_completed_op_ids.keys():
                    job = self.jobs_running[job_idx]
                    if job.is_completed():
                        self._register_completed_job(job)
                        if verbose:
                            print(f'Job with job_idx {job_idx} completed. Time arrived: {job.details["time_arrived"]} | Time completed: {job.details["time_completed"]}')

                if len(job_idx_to_completed_op_ids) == 0 or self.stopwatch.time() >= self.max_simulation_run_time:
                    # no ops left to run or reached max sim time
                    step_done = True

        self._communicate_jobs()

        self.step_counter += 1

    def _tick_workers(self, job_schedule, max_tick=None):
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
        # # find shortest remaining run time of mounted operations
        # tick = self._get_shortest_remaining_run_time_of_mounted_ops()

        # find: 1) highest priority op on each worker; and 2) the shortest remaining run time of each highest priority op on all workers
        worker_to_priority_job_op = {}
        shortest_remaining_run_time = float('inf')
        for worker_id, node_id in self.topology.graph.graph['worker_to_node'].items():
            # record priority job op
            priority_job_op = self._get_highest_priority_job_op(worker=topology.graph.nodes[node_id]['workers'][worker_id], job_schedule=job_schedule)
            if priority_job_op is not None:
                worker_to_priority_job_op[worker_id] = priority_job_op
                # update shortest remaining run time of all priority job ops
                job_idx, job_id, op_id = priority_job_op.split('_')
                job = self.jobs_running[job_idx]
                if job.computation_graph.nodes[op_id]['remaining_run_time'] < shortest_remaining_run_time:
                    shortest_remaining_run_time = job.computation_graph.nodes[op_id]['remaining_run_time']
        if max_tick is not None:
            tick = min(shortest_remaining_run_time, max_tick)
        else:
            tick = shortest_remaining_run_time

        # tick highest priority mounted ready op on each worker by shortest_remaining_run_time (or max_tick) and track which op(s) completed
        job_idx_to_completed_op_ids = defaultdict(list)
        for worker_id, priority_job_op in worker_to_priority_job_op.items():
            if priority_job_op is not None:
                node_id = topology.graph.graph['worker_to_node'][worker_id]
                worker = topology.graph.nodes[node_id]['workers'][worker_id]
                job_idx, job_id, op_id = priority_job_op.split('_')
                job = self.jobs_running[job_idx]
                job.tick_op(op_id, tick=tick)
                if op_id in job.computation_graph.graph['ops_completed']:
                    # op was completed
                    job_idx_to_completed_op_ids[job_idx].append(op_id)

        # tick stopwatch
        self.stopwatch.tick(tick)

        return job_idx_to_completed_op_ids

    def _get_highest_priority_job_op(self, worker, job_schedule):
        '''
        Takes a worker processor object and returns a string identifying which operation
        which is ready to run on the worker has the highest priority. If no
        operation is available to run, will return None.
        '''
        priority_job_op = None
        for job_idx in worker.mounted_job_idx_to_ops.keys():
            job = self.jobs_running[job_idx]
            for op_id in job.computation_graph.graph['ops_ready']:
                if priority_job_op is None:
                    # not yet considered any other ops, set this op as priority op
                    priority_job_op = f'{job_idx}_{job.job_id}_{op_id}'
                else:
                    if job_schedule[worker_id][job.job_id][op_id] > job_placement[worker_id][int(priority_job_op.split('_')[1])][int(priority_job_op.split('_')[2])]:
                        # op has higher priority, update priority job op
                        priority_job_op = f'{job_idx}_{job.job_id}_{op_id}'
        return priority_job_op

    def _prioritise_jobs(self):
        pass

    def _partition_jobs(self):
        pass

    def _place_jobs(self, job_placement, verbose):
        if verbose:
            print('-'*48)
            print('Placing job ops onto workers...')
        for job_id in job_placement:
            job = self.job_queue.jobs[job_idx]
            if verbose:
                print(f'Job ID: {job_id} | Job idx: {job.details["job_idx"]} | Time arrived: {job.details["time_arrived"]}')
            for op_id in job_placement[job_id]:
                worker_id = job_placement[job_id][op_id]
                node_id = self.topology.graph.graph['worker_to_node'][worker_id]
                self.topology.graph.nodes[node_id]['workers'][worker_id].mount(job=job, op_id=op_id)
                job.reset_op_remaining_run_time(op_id, device_type=self.topology.graph.nodes[node_id]['workers'][worker_id].device_type)
                self.job_op_to_worker[f'{job.details["job_idx"]}_{job.job_id}_{op_id}'] = worker_id
                if verbose:
                    print(f'Op ID {op_id} placed on node ID {node_id} worker ID {worker_id}')
            self._register_running_job(job)

    def _schedule_jobs(self):
        pass

    def _communicate_jobs(self):
        pass

    def render(self):
        pass

    def seed(self, seed):
        pass










# class OldCluster:
    # def __init__(self,
                 # topology: Topology,
                 # scheduler: Any,
                 # placer: Any,
                 # queue_capacity: int = int(100e12),
                 # verbose: bool = True,
                 # **kwargs):
        # '''
        # Args:
            
        # '''
        # self.topology = topology
        # self.scheduler = scheduler
        # self.placer = placer
        
        # self.queue_capacity = queue_capacity

        # self.verbose = verbose
                
        # self.kwargs = self._init_default_kwargs(kwargs)
        
    # def _init_default_kwargs(self, kwargs):
        # if 'max_simulation_time' not in kwargs:
            # kwargs['max_simulation_time'] = 1e12
        # if 'simulation_time_resolution' not in kwargs:
            # kwargs['simulation_time_resolution'] = 1
        # return kwargs
        
    # def reset(self, seed: int = 0):
        # self.seed = seed
        # seed_stochastic_modules_globally(self.seed)
        
        # self.wallclock_time, self.time_last_job_arrived = 0, 0
        # self.jobs_arrived = {}
        # self.jobs_completed = {}
        # self.jobs_blocked = {}
        
        # self.job_queue = JobQueue(queue_capacity=self.queue_capacity)
        
    # def run(self,
            # mode: str,
            # jobs: list[Job], 
            # job_interarrival_times: Union[int, float, list] = 1,
            # job_sampling_mode: str = 'remove_and_repeat'):
        # '''
        # Args:
            # mode ('time_driven', 'event_driven')
        # '''
        # self.job_sampler = Sampler(pool=jobs, sampling_mode=job_sampling_mode)
        # self.job_interarrival_times = job_interarrival_times
        
        # if mode == 'time_driven':
            # self.run_time_driven_mode()
        # elif mode == 'event_driven':
            # self.run_event_driven_mode()
        # else:
            # raise Exception(f'Unrecognised run mode {mode}')
        
    # def run_time_driven_mode(self):
        # while not self.check_if_done():
            # self.step_cluster()
    
    # def run_event_driven_mode(self):
        # raise Exception('Not implemented.')
    
    # def check_if_done(self):
        # pass
    
    # def step_job_arrival(self, arrival_time):
        # '''Add the next job to the queue.'''
        # # get the next job
        # job = self.job_sampler.sample()
        # job.details['time_arrived'] = arrival_time
        # self.time_last_job_arrived = copy.deepcopy(arrival_time)
        
        # if self.job_queue.can_fit(job):
            # # add job to queue
            # self.job_queue.add(job)
        # else:
            # # no space for job in queue
            # self.jobs_blocked[f'{job.job_type}_{job.job_id}'] = job
            
    # def step_wallclock_time(self):
        # self.prev_wallclock_time = copy.deepcopy(self.wallclock_time)
        # self.wallclock_time += self.kwargs['simulation_time_resolution']
        
    # def step_job_arrivals(self):
        # '''Get next job(s) which arrive during cluster step.'''
        # if type(self.job_interarrival_times) is int or type(self.job_interarrival_times) is float:
            # # jobs arriving at fixed time interval
            # while self.wallclock_time - self.time_last_job_arrived >= self.job_interarrival_times:
                # self.step_job_arrival(arrival_time=self.time_last_job_arrived+self.job_interarrival_times)
        # elif type(self.job_interarrival_times) is list:
            # raise Exception('Not implemented.')
        # else:
            # raise Exception(f'Unrecognised job_interarrival_times type {type(self.job_interarrival_times)}')
            
    # def step_cluster_managers(self):
        # '''Schedule order in which to service jobs and place job workloads across cluster.'''
        # # use scheduler to assign scheduling priority to 
        # prioritised_jobs = self.scheduler.prioritise_jobs(self.job_queue.jobs)
        
        # # while have space, parallelise the jobs into workloads and place the workloads across the cluster
        # self.workload_to_workloads_manager, self.workloads_managers = {}, []
        # for job in prioritised_jobs:
            # workloads_manager = self.placer.place_job(job, self)
            # if workloads_manager.node_to_workloads is None:
                # # could not place job on cluster
                # break
            # else:
                # # mount workloads onto cluster devices
                # self.workloads_managers.append(workloads_manager)
                # for node, workloads in workloads_manager.node_to_workloads.items():
                    # for workload in workloads:
                        # self.topology.topology.nodes[node]['device'].mount(workload)
                        # self.workload_to_workloads_manager[id(workload)] = workloads_manager
                        
    # def step_cluster_processors(self):
        # '''Step the cluster devices.'''
        # for node in self.topology.topology.nodes:
            # self.topology.topology.nodes[node]['device'].step(time=self.kwargs['simulation_time_resolution'])

    # def register_completed_workloads(self):
        # completed_workloads = []
        # for node in self.topology.topology.nodes:
            # device = self.topology.topology.nodes[node]['device']
            # for workload in device.mounted_workloads.values():
                # if workload.completed:
                    # completed_workloads.append(workload)
                    # self.workload_to_workloads_manager[id(workload)].register_completed_workload()
                    # device.unmount(workload)

    # def step_workloads_managers(self):
        # completed_managers = []
        # for workloads_manager in self.workloads_managers:
            # raise Exception('Not implemented.')
    
    # def step_cluster(self):
        # self.step_wallclock_time()
        # self.step_job_arrivals()
        # self.step_cluster_managers()
        # self.step_cluster_processors()
        
        # self.register_completed_workloads()
        # self.step_workloads_managers()

        # if self.verbose:
            # print(self.verbose_logger())

    # def verbose_logger(self):
        # verbose_log = f'Wallclock time: {self.wallclock_time}'
        # return verbose_log
