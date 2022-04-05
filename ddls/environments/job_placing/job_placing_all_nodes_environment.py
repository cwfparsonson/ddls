from ddls.environments.cluster.cluster_environment import ClusterEnvironment
from ddls.environments.job_placing.observations.job_placing_all_nodes_observation import JobPlacingAllNodesObservation
from ddls.environments.job_placing.rewards.worker_compute_utilisation import WorkerComputeUtilisation
from ddls.demands.jobs.job import Job
from ddls.distributions.distribution import Distribution
from ddls.utils import flatten_list

from typing import Union
import numpy as np
import pprint


class JobPlacingAllNodesEnvironment:
    def __init__(self,
                 topology_config: dict,
                 node_config: dict,
                 jobs: list[Job], 
                 job_interarrival_time_dist: Distribution,
                 continuous_action_mode: bool = False,
                 worker_selection: Union['random'] = 'random',
                 op_allocation: Union['random', 'sequential'] = 'sequential',
                 observation_function: str = 'job_placing_all_nodes_observation',
                 information_function: str = 'default',
                 reward_function: str = 'worker_compute_utilisation',
                 max_cluster_simulation_run_time: Union[int, float] = float('inf'),
                 job_sampling_mode: str = 'remove_and_repeat',
                 job_queue_capacity: int = 10,
                 seed: int = None,
                 name: str = 'job_placing',
                 cluster_name: str = 'cluster',
                 path_to_save: str = None,
                 save_cluster_data: bool = False,
                 save_freq: int = 1,
                 use_sqlite_database: bool = False):
        '''
        The cluster has $$n$$ servers, each server $$s_i$$ has $$m$$ workers,
        therefore there are $$n \times m$$ workers in total.

        Args:
            continuous_action_mode: Whether or not to accept continuous float actions
                or discrete int actions to specify number of workers to use.
            worker_selection: Given a number of workers to use in the cluster,
                how to select the set of workers.
            op_allocation: Given a set of workers to use, how to allocate the ops
                in the job across some workers.
                    sequential: Loops through each op in order of index and each worker
                        and allocates sequentially. When reach end of workers, starts
                        at first worker again. Repeat until all ops allocated.
                    random: Choose random workers from set for each op.

        MDP:
            state: Job computation graph which is requesting to be placed on the
                cluster.
            action: 
                if continuous_action_mode:
                    Float 0 <= action <= 1 specifying fraction of the $$n \times m$$
                        workers in the cluster to use for the job.
                else:
                    Int 0 <= action <= $$n \times m$$ specifying the number of workers
                        in the cluster to use for the job.
            transition: When next job arrives, will transition to next state.
        '''
        self.topology_config = topology_config
        self.node_config = node_config
        self.jobs = jobs
        self.job_interarrival_time_dist = job_interarrival_time_dist
        self.continuous_action_mode = continuous_action_mode
        self.worker_selection = worker_selection
        self.op_allocation = op_allocation
        self.max_cluster_simulation_run_time = max_cluster_simulation_run_time
        self.job_sampling_mode = job_sampling_mode
        self.job_interarrival_time_dist = job_interarrival_time_dist
        self.max_cluster_simulation_run_time = max_cluster_simulation_run_time
        self.job_sampling_mode = job_sampling_mode
        self.job_queue_capacity = job_queue_capacity
        self.seed = seed
        self.cluster_name = cluster_name
        self.path_to_save = path_to_save
        self.save_cluster_data = save_cluster_data
        self.save_freq = save_freq
        self.use_sqlite_database = use_sqlite_database

        self.observation_function_str = observation_function
        if observation_function == 'job_placing_all_nodes_observation':
            self.observation_function = JobPlacingAllNodesObservation()
        else:
            raise Exception(f'Unrecognised observation_function {self.observation_function_str}')
        self.information_function_str = information_function
        if information_function == 'default':
            # TODO: Not implemented
            pass
        else:
            raise Exception(f'Unrecognised information_function {self.information_function_str}')
        self.reward_function_str = reward_function
        if reward_function == 'worker_compute_utilisation':
            self.reward_function = {'worker_compute_utilisation': WorkerComputeUtilisation()}
        else:
            raise Exception(f'Unrecognised reward_function {self.reward_function_str}')

        self._init_cluster()

    def _init_cluster(self):
        self.cluster =  ClusterEnvironment(topology_config=self.topology_config,
                                           node_config=self.node_config,
                                           name=self.cluster_name,
                                           path_to_save=self.path_to_save if self.save_cluster_data else None,
                                           save_freq=self.save_freq,
                                           use_sqlite_database=self.use_sqlite_database)

    def _reset_cluster(self):
        _ = self.cluster.reset(jobs=self.jobs,
                               job_sampling_mode=self.job_sampling_mode,
                               job_interarrival_time_dist=self.job_interarrival_time_dist,
                               max_simulation_run_time=self.max_cluster_simulation_run_time,
                               job_queue_capacity=self.job_queue_capacity,
                               seed=self.seed,
                               verbose=False)

    def is_done(self):
        return self.cluster.is_done()

    def reset(self):
        # reset the cluster environment
        self._reset_cluster()

        # reset the reward and observation function
        self.observation_function.reset()
        for reward_function in self.reward_function.values():
            reward_function.reset()

        # extract MDP info for this step
        done = self.is_done()
        obs = self.observation_function.extract(cluster=self.cluster, done=done) # encoded obs of job to place
        action_set = self._get_action_set()
        action_mask = self._get_action_mask()
        reward = {reward: self.reward_function[reward].extract(cluster=self.cluster, done=done) for reward in self.reward_function.keys()}
        info = self._get_info()

        return obs, action_set, action_mask, reward, done, info

    def step(self, action):
        pass

    def _get_action_set(self):
        return [action for action in range(len(self.cluster.topology.graph.graph['worker_to_node']) + 1)]

    def _get_action_mask(self):
        # TEMPORARY: Just assume placing 1st job in queue
        # TODO: Implement where get given job and do per-job encoding?
        job = list(self.cluster.job_queue.jobs.values())[0] # assume event-driven where only ever have one job to queue

        # check how much memory is available on each worker
        worker_to_available_memory = self._get_workers_available_memory(self.cluster, sort=True)
        workers = list(worker_to_available_memory.keys())

        # initialise action mask
        action_mask = [1] # choosing 0 workers (i.e. do not place job) is always valid

        # check which action(s) valid
        for action in range(1, len(workers)):
            if sum(worker_to_available_memory[workers[i]] for i in range(action)) >= job.job_total_operation_memory_cost:
                # can fit job on cluster if use 'action' many workers -> action is valid
                action_mask.append(1)
            else:
                # cannot fit job on cluster if use 'action' many workers -> action is invalid
                action_mask.append(0)

        return action_mask 

    def _get_info(self):
        return None

    def _get_workers_available_memory(self, 
                                      cluster: ClusterEnvironment, 
                                      sort: bool = True):
        '''
        Maps worker ids to available memory. 

        Args:
            sort: If true, returned dict is in order of memory available,
                with the worker with the most memory available first, etc.
        '''
        worker_to_available_memory = dict()
        for worker_id, node_id in cluster.topology.graph.graph['worker_to_node'].items():
            node_id = cluster.topology.graph.graph['worker_to_node'][worker_id]
            worker = cluster.topology.graph.nodes[node_id]['workers'][worker_id]
            worker_to_available_memory[worker_id] = worker.memory_capacity - worker.memory_occupied
        if sort:
            worker_to_available_memory = dict(sorted(worker_to_available_memory.items(), key=lambda x:x[1], reverse=True))
        return worker_to_available_memory

























