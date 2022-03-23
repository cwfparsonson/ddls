from ddls.clusters.job_queue import JobQueue
from ddls.utils import Sampler
from ddls.topologies.topology import Topology
from ddls.topologies.torus import Torus
from ddls.demands.jobs.job import Job
from ddls.distributions.distribution import Distribution
from ddls.utils import seed_stochastic_modules_globally

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

        self.job_arrival_times = iter(self._init_job_arrival_times())

        self.job_queue = JobQueue(queue_capacity=job_queue_capacity)

        self.jobs_arrived = {}
        self.jobs_completed = {}
        self.jobs_blocked = {}

        self.job_idx_to_job_id = {}
        self.job_id_to_job_idx = {}

        self.step_counter = 0

        # add first job to queue
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

    def _init_job_arrival_times(self, 
                                max_counter: int = 10000):
        job_arrival_times, counter = [0], 0
        while job_arrival_times[-1] < self.max_simulation_run_time:
            job_arrival_times.append(self.job_interarrival_time_dist.sample(size=None) + job_arrival_times[-1])
            counter += 1
            if counter > max_counter:
                raise Exception(f'Unable to meet max_simulation_time after sampling {counter} times. Increase max_counter or decrease max_simulation_run_time')
        return job_arrival_times

    def _get_next_job(self):
        job = self.job_sampler.sample()
        job.job_details['time_arrived'] = next(self.job_arrival_times)
        job.job_details['time_completed'] = None
        job_idx = len(list(self.jobs_arrived.keys()))
        self.jobs_arrived[job_idx] = job
        self.job_idx_to_job_id[job_idx] = job.job_id
        self.job_id_to_job_idx[job.job_id] = job_idx
        return job

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
        self._communicate_jobs()

        self.step_counter += 1


    def _prioritise_jobs(self):
        pass

    def _partition_jobs(self):
        pass

    def _place_jobs(self, job_placement, verbose):
        if verbose:
            print('-'*48)
            print('Placing job ops onto workers...')
        for job_id in job_placement:
            job_idx = self.job_id_to_job_idx[job_id]
            job = self.jobs_arrived[job_idx]
            if verbose:
                print(f'Job ID: {job_id} | Job index: {job_idx}')
            for op_id in job_placement[job_id]:
                worker_id = job_placement[job_id][op_id]
                node_id = self.topology.graph.graph['worker_to_node'][worker_id]
                self.topology.graph.nodes[node_id]['workers'][worker_id].mount(job=job, op_id=op_id)
                if verbose:
                    print(f'Op ID {op_id} placed on node ID {node_id} worker ID {worker_id}')

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
