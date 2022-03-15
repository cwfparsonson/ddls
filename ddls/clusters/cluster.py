from ddls.clusters.job_queue import JobQueue
from ddls.utils import Sampler
from ddls.topologies.topology import Topology
from ddls.demands.jobs.job import Job
from ddls.utils import seed_stochastic_modules_globally

from typing import Any, Union
import copy
import math


class Cluster:
    def __init__(self,
                 topology: Topology,
                 scheduler: Any,
                 placer: Any,
                 queue_capacity: int = int(100e12),
                 verbose: bool = True,
                 **kwargs):
        '''
        Args:
            
        '''
        self.topology = topology
        self.scheduler = scheduler
        self.placer = placer
        
        self.queue_capacity = queue_capacity

        self.verbose = verbose
                
        self.kwargs = self._init_default_kwargs(kwargs)
        
    def _init_default_kwargs(self, kwargs):
        if 'max_simulation_time' not in kwargs:
            kwargs['max_simulation_time'] = 1e12
        if 'simulation_time_resolution' not in kwargs:
            kwargs['simulation_time_resolution'] = 1
        return kwargs
        
    def reset(self, seed: int = 0):
        self.seed = seed
        seed_stochastic_modules_globally(self.seed)
        
        self.wallclock_time, self.time_last_job_arrived = 0, 0
        self.jobs_arrived = {}
        self.jobs_completed = {}
        self.jobs_blocked = {}
        
        self.job_queue = JobQueue(queue_capacity=self.queue_capacity)
        
    def run(self,
            mode: str,
            jobs: list[Job, ...],
            job_interarrival_times: Union[int, float, list] = 1,
            job_sampling_mode: str = 'remove_and_repeat'):
        '''
        Args:
            mode ('time_driven', 'event_driven')
        '''
        self.job_sampler = Sampler(pool=jobs, sampling_mode=job_sampling_mode)
        self.job_interarrival_times = job_interarrival_times
        
        if mode == 'time_driven':
            self.run_time_driven_mode()
        elif mode == 'event_driven':
            self.run_event_driven_mode()
        else:
            raise Exception(f'Unrecognised run mode {mode}')
        
    def run_time_driven_mode(self):
        while not self.check_if_done():
            self.step_cluster()
    
    def run_event_driven_mode(self):
        raise Exception('Not implemented.')
    
    def check_if_done(self):
        pass
    
    def step_job_arrival(self, arrival_time):
        '''Add the next job to the queue.'''
        # get the next job
        job = self.job_sampler.sample()
        job.details['time_arrived'] = arrival_time
        self.time_last_job_arrived = copy.deepcopy(arrival_time)
        
        if self.job_queue.can_fit(job):
            # add job to queue
            self.job_queue.add(job)
        else:
            # no space for job in queue
            self.jobs_blocked[f'{job.job_type}_{job.job_id}'] = job
            
    def step_wallclock_time(self):
        self.prev_wallclock_time = copy.deepcopy(self.wallclock_time)
        self.wallclock_time += self.kwargs['simulation_time_resolution']
        
    def step_job_arrivals(self):
        '''Get next job(s) which arrive during cluster step.'''
        if type(self.job_interarrival_times) is int or type(self.job_interarrival_times) is float:
            # jobs arriving at fixed time interval
            while self.wallclock_time - self.time_last_job_arrived >= self.job_interarrival_times:
                self.step_job_arrival(arrival_time=self.time_last_job_arrived+self.job_interarrival_times)
        elif type(self.job_interarrival_times) is list:
            raise Exception('Not implemented.')
        else:
            raise Exception(f'Unrecognised job_interarrival_times type {type(self.job_interarrival_times)}')
            
    def step_cluster_managers(self):
        '''Schedule order in which to service jobs and place job workloads across cluster.'''
        # use scheduler to assign scheduling priority to 
        prioritised_jobs = self.scheduler.prioritise_jobs(self.job_queue.jobs)
        
        # while have space, parallelise the jobs into workloads and place the workloads across the cluster
        self.workload_to_workloads_manager, self.workloads_managers = {}, []
        for job in prioritised_jobs:
            workloads_manager = self.placer.place_job(job, self)
            if workloads_manager.node_to_workloads is None:
                # could not place job on cluster
                break
            else:
                # mount workloads onto cluster devices
                self.workloads_managers.append(workloads_manager)
                for node, workloads in workloads_manager.node_to_workloads.items():
                    for workload in workloads:
                        self.topology.topology.nodes[node]['device'].mount(workload)
                        self.workload_to_workloads_manager[id(workload)] = workloads_manager
                        
    def step_cluster_processors(self):
        '''Step the cluster devices.'''
        for node in self.topology.topology.nodes:
            self.topology.topology.nodes[node]['device'].step(time=self.kwargs['simulation_time_resolution'])

    def register_completed_workloads(self):
        completed_workloads = []
        for node in self.topology.topology.nodes:
            device = self.topology.topology.nodes[node]['device']
            for workload in device.mounted_workloads.values():
                if workload.completed:
                    completed_workloads.append(workload)
                    self.workload_to_workloads_manager[id(workload)].register_completed_workload()
                    device.unmount(workload)

    def step_workloads_managers(self):
        completed_managers = []
        for workloads_manager in self.workloads_managers:
            raise Exception('Not implemented.')

    
    def step_cluster(self):
        self.step_wallclock_time()
        self.step_job_arrivals()
        self.step_cluster_managers()
        self.step_cluster_processors()
        
        self.register_completed_workloads()
        self.step_workloads_managers()

        if self.verbose:
            print(self.verbose_logger())

    def verbose_logger(self):
        verbose_log = f'Wallclock time: {self.wallclock_time}'
        return verbose_log
