from ddls.environments.cluster.cluster_environment import ClusterEnvironment
from ddls.environments.job_placing.observations.job_placing_all_nodes_observation import JobPlacingAllNodesObservation
from ddls.environments.job_placing.rewards.worker_compute_utilisation import WorkerComputeUtilisation
from ddls.environments.job_placing.rewards.mean_job_completion_time import MeanJobCompletionTime 
from ddls.environments.job_placing.rewards.total_job_completion_time import TotalJobCompletionTime 
from ddls.managers.schedulers.srpt_job_scheduler import SRPTJobScheduler
from ddls.demands.jobs.job import Job
from ddls.distributions.distribution import Distribution
from ddls.utils import flatten_list

import gym
import numpy as np

from typing import Union
from itertools import cycle
import pprint


class JobPlacingAllNodesEnvironment(gym.Env):
    def __init__(self,
                 topology_config: dict,
                 node_config: dict,
                 # jobs: list[Job], 
                 jobs_config: dict,
                 continuous_action_mode: bool = False,
                 worker_selection: Union['random'] = 'random',
                 op_allocation: Union['random', 'sequential'] = 'sequential',
                 job_scheduler: Union['srpt_job_scheduler'] = 'srpt_job_scheduler',
                 pad_obs_kwargs: dict = None,
                 observation_function: Union['default'] = 'default',
                 information_function: Union['default'] = 'default',
                 reward_function: Union['worker_compute_utilisation', 'mean_job_completion_time', 'total_job_completion_time'] = 'mean_job_completion_time',
                 max_cluster_simulation_run_time: Union[int, float] = float('inf'),
                 job_queue_capacity: int = 10,
                 name: str = 'job_placing_all_nodes',
                 cluster_name: str = 'cluster',
                 path_to_save: str = None,
                 save_cluster_data: bool = False,
                 save_freq: int = 1,
                 use_sqlite_database: bool = False):
        '''
        The cluster has $$n$$ servers, each server $$s_i$$ has $$m$$ workers,
        therefore there are $$n \times m$$ workers in total.

        Args:
            continuous_action_mode: If True, $$0 <= \text{action} <= 1$$ specifies 
                the fraction of cluster workers to use. If False, $$0 <= \text{action} <= \text{num_workers}$$
                specifies the integer number of workers to use.
            worker_selection: Given a number of workers to use in the cluster,
                how to select the set of workers.
            op_allocation: Given a set of workers to use, how to allocate the ops
                in the job across some workers.
                    sequential: Loops through each op in order of index and each worker
                        and allocates sequentially. When reach end of workers, starts
                        at first worker again. Repeat until all ops allocated.
                    random: Choose random workers from set for each op.
            pad_obs_kwargs: If not None will look at jobs_config, get max number of nodes and
                edges across all jobs, and pad each obs to ensure dimensionality of
                each obs is consistent even for observations with varying graph sizes.
                pad_obs_kwargs must be dict of {'max_nodes': <int>, 'max_edges': <int>}
                UPDATE: Only needs to be {'max_nodes': <int>}, will then calc max edges
                by assuming fully connected graph of max_nodes.

        MDP:
            state: Job computation graph which is requesting to be placed on the
                cluster.
            action: 
                if using continuous actions:
                    Float 0 <= action <= 1 specifying fraction of the $$n \times m$$
                        workers in the cluster to use for the job.
                else:
                    Int 0 <= action <= $$n \times m$$ specifying the number of workers
                        in the cluster to use for the job.
            transition: When next job arrives, will transition to next state.
        '''
        self.topology_config = topology_config
        self.node_config = node_config
        self.jobs_config = jobs_config
        self.continuous_action_mode = continuous_action_mode
        self.worker_selection = worker_selection
        self.op_allocation = op_allocation
        self.max_cluster_simulation_run_time = max_cluster_simulation_run_time
        self.max_cluster_simulation_run_time = max_cluster_simulation_run_time
        self.job_queue_capacity = job_queue_capacity
        self.cluster_name = cluster_name
        self.path_to_save = path_to_save
        self.save_cluster_data = save_cluster_data
        self.save_freq = save_freq
        self.use_sqlite_database = use_sqlite_database
        self.pad_obs_kwargs = pad_obs_kwargs

        # init ddls cluster simulator
        self.cluster = self._init_cluster()

        # init obs
        self.observation_function_str = observation_function
        if observation_function == 'default':
            self.observation_function = JobPlacingAllNodesObservation(pad_obs_kwargs=self.pad_obs_kwargs)
        else:
            raise Exception(f'Unrecognised observation_function {self.observation_function_str}')

        # init action space
        if self.continuous_action_mode:
            self.action_space = gym.spaces.Box(low=0, high=1, dtype=np.float32)
        else:
            # self.action_space = gym.spaces.Discrete(self.cluster.topology.graph.graph['num_workers'] + 1)
            self.action_space = gym.spaces.Discrete(self.cluster.topology.graph.graph['num_workers'])

        # init info
        self.information_function_str = information_function
        if information_function == 'default':
            # TODO: Not implemented
            pass
        else:
            raise Exception(f'Unrecognised information_function {self.information_function_str}')

        # init reward
        self.reward_function_str = reward_function
        if reward_function == 'worker_compute_utilisation':
            self.reward_function = WorkerComputeUtilisation()
        elif reward_function == 'mean_job_completion_time':
            self.reward_function = MeanJobCompletionTime()
        elif reward_function == 'total_job_completion_time':
            self.reward_function = TotalJobCompletionTime()
        else:
            raise Exception(f'Unrecognised reward_function {self.reward_function_str}')

        self.job_scheduler_str = job_scheduler
        if job_scheduler == 'srpt_job_scheduler':
            self.job_scheduler = SRPTJobScheduler()
        else:
            raise Exception(f'Unrecognised job_scheduler {self.job_scheduler_str}')



    def _init_cluster(self):
        return ClusterEnvironment(topology_config=self.topology_config,
                                   node_config=self.node_config,
                                   name=self.cluster_name,
                                   path_to_save=self.path_to_save if self.save_cluster_data else None,
                                   save_freq=self.save_freq,
                                   use_sqlite_database=self.use_sqlite_database)

    def _reset_cluster(self, seed: int = None):
        _ = self.cluster.reset(jobs_config=self.jobs_config,
                               max_simulation_run_time=self.max_cluster_simulation_run_time,
                               job_queue_capacity=self.job_queue_capacity,
                               seed=seed,
                               verbose=False)

    def _is_done(self):
        return self.cluster.is_done()

    def reset(self, seed: int = None):
        # reset the cluster environment
        self._reset_cluster(seed=seed)

        # reset the observation function
        self.observation_function.reset(self.cluster)
        self.observation_space = self.observation_function.observation_space

        # reset the reward function
        self.reward_function.reset(self.cluster)

        # extract current MDP info and save so can access for next env.step() call
        self.obs = self._get_observation() # encoded obs of job to place

        return self.obs

    def _conv_frac_workers_to_int(self, frac):
        '''Converts a fraction of cluster works to an integer number of workers.'''
        return round(frac * self.cluster.topology.graph.graph['num_workers'])

    def _get_worker_set(self, num_workers):
        cluster_workers = self.cluster.topology.graph.graph['worker_to_node']
        if self.worker_selection == 'random':
            workers = np.random.choice(list(cluster_workers), size=num_workers, replace=False)
        else:
            raise Exception(f'Unrecognised worker_selection {self.worker_selection}')
        return workers

    def _get_op_to_worker(self, job, workers):
        op_to_worker, iterable_workers = {}, cycle(workers)
        for op in job.computation_graph.nodes:
            op_to_worker[op] = next(iterable_workers)
        return op_to_worker

    def step(self, action: int):
        # check the action passed to env
        if isinstance(action, float):
            if action > 1:
                raise Exception(f'Float action must be 0 <= action <= 1 but is {action}')
            # action is a fraction of workers -> convert to integer number of workers
            processed_action = self._conv_frac_workers_to_int(action)
        else:
            # action already an integer number of workers, convert from 0-num_workers idx to a valid non-zero action
            processed_action = action + 1
        # if processed_action not in self.obs.action_set[self.obs.action_mask]:
            # raise Exception(f'Action {action} (processed as {processed_action}) is invalid for observation with valid actions {self.obs.action_set[self.obs.action_mask]}')

        control_plane = {}
        if processed_action != 0:
            # select a set of processed_action workers to use in the cluster
            workers = self._get_worker_set(processed_action)

            # get job to place
            # TEMPORARY: Just assume placing 1st job in queue
            job = list(self.cluster.job_queue.jobs.values())[0] # assume event-driven where only ever have one job to queue

            # generate per-op placement on the selected workers
            op_to_worker = self._get_op_to_worker(job, workers)

            # create control plane mappings for cluster simulator
            control_plane['job_placement'] = {job.job_id: op_to_worker}
            control_plane['job_schedule'] = self.job_scheduler.get_schedule(new_placements=control_plane['job_placement'], cluster=self.cluster)
        else:
            # not placing job
            control_plane['job_placement'] = {}
            control_plane['job_schedule'] = {}

        # step the cluster
        self.cluster.step(actions=control_plane)
        self.reward = self._get_reward()

        # continue stepping cluster until there is a job to place
        while len(self.cluster.job_queue) == 0:
            self.cluster.step(actions={'job_placement': {},
                                       'job_schedule': {}
                                       })
            if self.cluster.is_done():
                break

        # extract current MDP info and save so can access for next env.step() call
        self.done = self._is_done()
        if not self.done:
            self.obs = self._get_observation() # encoded obs of job to place
        else:
            # done, just return last create self.obs
            pass
        self.info = self._get_info()

        return self.obs, self.reward, self.done, self.info

    def _get_observation(self):
        # extract obs node and edge features
        obs = self.observation_function.extract(cluster=self.cluster, done=self._is_done())

        # set obs action info
        obs.action_set = self._get_action_set()
        obs.action_mask = self._get_action_mask()

        return obs

    def _get_reward(self):
        return self.reward_function.extract(cluster=self.cluster, done=self._is_done())

    def _get_action_set(self):
        return np.array([action for action in range(len(self.cluster.topology.graph.graph['worker_to_node']) + 1)])

    def _get_action_mask(self):
        # TEMPORARY: Just assume placing 1st job in queue
        job = list(self.cluster.job_queue.jobs.values())[0] # assume event-driven where only ever have one job to queue

        # check how much memory is available on each worker
        worker_to_available_memory = self._get_workers_available_memory(self.cluster, sort=True)
        workers = list(worker_to_available_memory.keys())

        # initialise action mask
        action_mask = [True] # choosing 0 workers (i.e. do not place job) is always valid

        # check which action(s) are valid
        for action in range(1, self.cluster.topology.graph.graph['num_workers'] + 1):
            if sum(worker_to_available_memory[workers[i]] for i in range(action)) >= job.job_total_operation_memory_cost:
                # can fit job on cluster if use 'action' many workers -> action is valid
                action_mask.append(True)
            else:
                # cannot fit job on cluster if use 'action' many workers -> action is invalid
                action_mask.append(False)

        return np.array(action_mask)

    def _get_info(self):
        return {}

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

























