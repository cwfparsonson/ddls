from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment

# from ddls.environments.ramp_cluster.agents.partitioners.random_op_partitioner import RandomOpPartitioner
# from ddls.environments.ramp_cluster.agents.partitioners.sip_ml_op_partitioner import SipMlOpPartitioner
from ddls.environments.ramp_cluster.agents.job_placement_shapers.ramp_random_job_placement_shaper import RampRandomJobPlacementShaper
from ddls.environments.ramp_cluster.agents.job_placement_shapers.ramp_first_fit_job_placement_shaper import RampFirstFitJobPlacementShaper
from ddls.environments.ramp_cluster.agents.placers.random_op_placer import RandomOpPlacer
from ddls.environments.ramp_cluster.agents.placers.ramp_first_fit_op_placer import RampFirstFitOpPlacer
from ddls.environments.ramp_cluster.agents.schedulers.srpt_op_scheduler import SRPTOpScheduler
from ddls.environments.ramp_cluster.agents.placers.first_fit_dep_placer import FirstFitDepPlacer
from ddls.environments.ramp_cluster.agents.schedulers.srpt_dep_scheduler import SRPTDepScheduler

from ddls.environments.ramp_job_partitioning.rewards.lookahead_job_completion_time import LookaheadJobCompletionTime
from ddls.environments.ramp_job_partitioning.rewards.job_acceptance import JobAcceptance
from ddls.environments.ramp_job_partitioning.rewards.mean_compute_throughput import MeanComputeThroughput
from ddls.environments.ramp_job_partitioning.rewards.mean_cluster_throughput import MeanClusterThroughput
from ddls.environments.ramp_job_partitioning.rewards.mean_demand_total_throughput import MeanDemandTotalThroughput
from ddls.environments.ramp_job_partitioning.rewards.multi_objective_jct_blocking import MultiObjectiveJCTBlocking

from ddls.environments.ramp_job_partitioning.observations.ramp_job_partitioning_observation import RampJobPartitioningObservation

# from ddls.environments.ramp_cluster.actions.job_placement_shape import JobPlacementShape
from ddls.environments.ramp_cluster.actions.op_partition import OpPartition
from ddls.environments.ramp_cluster.actions.action import Action

from ddls.utils import get_forward_graph



import gym

import numpy as np
import math

from collections import defaultdict

from typing import Union

import copy


class RampJobPartitioningEnvironment(gym.Env):
    def __init__(self,
                 topology_config: dict,
                 node_config: dict,
                 jobs_config: dict,
                 # op_partitioner: Union['random_op_partitioner', 'sip_ml_op_partitioner'] = 'sip_ml_op_partitioner',
                 # op_partitioner_kwargs: dict = None,
                 max_partitions_per_op: int = None,
                 min_op_run_time_quantum: Union[float, int] = 0.000006,
                 job_placement_shaper: Union['ramp_random_job_placement_shaper', 'first_fit_job_placement_shaper'] = 'ramp_random_job_placement_shaper',
                 job_placement_shaper_kwargs: dict = None,
                 op_placer: Union['ramp_first_fit_op_placer'] = 'ramp_first_fit_op_placer',
                 op_placer_kwargs: dict = None,
                 op_scheduler: Union['srpt_op_scheduler'] = 'srpt_op_scheduler',
                 op_scheduler_kwargs: dict = None,
                 dep_placer: Union['first_fit_dep_placer'] = 'first_fit_dep_placer',
                 dep_placer_kwargs: dict = None,
                 dep_scheduler: Union['srpt_dep_scheduler'] = 'srpt_dep_scheduler',
                 dep_scheduler_kwargs: dict = None,
                 observation_function: Union['ramp_job_partitioning_observation'] = 'ramp_job_partitioning_observation',
                 pad_obs_kwargs: dict = None,
                 information_function: Union['default'] = 'default',
                 reward_function: Union['lookahead_job_completion_time', 'job_acceptance', 'mean_compute_throughput', 'mean_cluster_throughput', 'mean_demand_total_throughput', 'multi_objective_jct_blocking'] = 'lookahead_job_completion_time',
                 reward_function_kwargs: dict = None,
                 max_simulation_run_time: Union[int, float] = None,
                 job_queue_capacity: int = 10,
                 name: str = 'ramp_job_partitioning',
                 path_to_save: str = None,
                 save_cluster_data: bool = False,
                 save_freq: int = 1,
                 use_sqlite_database: bool = False):
        self.topology_config = topology_config
        self.node_config = node_config
        self.jobs_config = jobs_config

        if max_simulation_run_time is None:
            self.max_simulation_run_time = float('inf')
        else:
            self.max_simulation_run_time = max_simulation_run_time
        self.job_queue_capacity = job_queue_capacity

        self.name = name
        self.pad_obs_kwargs = pad_obs_kwargs
        self.path_to_save = path_to_save
        self.save_cluster_data = save_cluster_data
        self.save_freq = save_freq
        self.use_sqlite_database = use_sqlite_database

        # init ddls ramp cluster
        self.cluster = self._init_cluster()

        if max_partitions_per_op is None:
            self.max_partitions_per_op = self.cluster.topology.graph.graph['num_workers']
        else:
            self.max_partitions_per_op = max_partitions_per_op
        self.min_op_run_time_quantum = min_op_run_time_quantum

        # init obs
        self.observation_function_str = observation_function
        if observation_function == 'ramp_job_partitioning_observation':
            self.observation_function = RampJobPartitioningObservation(self.max_partitions_per_op, pad_obs_kwargs=self.pad_obs_kwargs)
        else:
            raise Exception(f'Unrecognised observation_function {self.observation_function_str}')

        # init action space
        # self.action_space = gym.spaces.Discrete(int((self.cluster.topology.num_communication_groups * self.cluster.topology.num_racks_per_communication_group * self.cluster.topology.num_servers_per_rack) + 1))
        # self.action_to_job_placement_shape = self._get_action_to_job_placement_shape()
        self.action_set = [action for action in range(self.max_partitions_per_op+1)] # 0 -> corresponds to do not place job
        self.action_space = gym.spaces.Discrete(int(len(self.action_set)))

        # init observation space
        self.observation_space = gym.spaces.Dict({})

        # init info
        self.information_function_str = information_function
        if information_function == 'default':
            # TODO: Not implemented
            pass
        else:
            raise Exception(f'Unrecognised information_function {self.information_function_str}')

        # init reward
        if reward_function_kwargs is None:
            reward_function_kwargs = {}
        self.reward_function_str = reward_function
        if reward_function == 'lookahead_job_completion_time':
            self.reward_function = LookaheadJobCompletionTime(**reward_function_kwargs)
        elif reward_function == 'job_acceptance':
            self.reward_function = JobAcceptance(**reward_function_kwargs)
        elif reward_function == 'mean_compute_throughput':
            self.reward_function = MeanComputeThroughput(**reward_function_kwargs)
        elif reward_function == 'mean_cluster_throughput':
            self.reward_function = MeanClusterThroughput(**reward_function_kwargs)
        elif reward_function == 'mean_demand_total_throughput':
            self.reward_function = MeanDemandTotalThroughput(**reward_function_kwargs)
        elif reward_function == 'multi_objective_jct_blocking':
            self.reward_function = MultiObjectiveJCTBlocking(**reward_function_kwargs)
        else:
            raise Exception(f'Unrecognised reward_function {self.reward_function_str}')

        # init cluster environment managers
        # if op_partitioner_kwargs is not None:
            # self.op_partitioner_kwargs = op_partitioner_kwargs
        # else:
            # self.op_partitioner_kwargs = {}
        if job_placement_shaper_kwargs is not None:
            self.job_placement_shaper_kwargs = job_placement_shaper_kwargs
        else:
            self.job_placement_shaper_kwargs = {}
        if op_placer_kwargs is not None:
            self.op_placer_kwargs = op_placer_kwargs
        else:
            self.op_placer_kwargs = {}
        if op_scheduler_kwargs is not None:
            self.op_scheduler_kwargs = op_scheduler_kwargs
        else:
            self.op_scheduler_kwargs = {}
        if dep_placer_kwargs is not None:
            self.dep_placer_kwargs = dep_placer_kwargs
        else:
            self.dep_placer_kwargs = {}
        if dep_scheduler_kwargs is not None:
            self.dep_scheduler_kwargs= dep_scheduler_kwargs
        else:
            self.dep_scheduler_kwargs = {}

        # self.op_partitioner_str = op_partitioner 
        self.job_placement_shaper_str = job_placement_shaper
        self.op_placer_str = op_placer
        self.op_scheduler_str = op_scheduler
        self.dep_placer_str = dep_placer
        self.dep_scheduler_str = dep_scheduler

        # self.op_partitioner, self.op_placer, self.op_scheduler, self.dep_placer, self.dep_scheduler = self._init_cluster_managers()
        self.job_placement_shaper, self.op_placer, self.op_scheduler, self.dep_placer, self.dep_scheduler = self._init_cluster_managers()

        # TODO: Is this really needed? NEW
        self.reset()

    def _get_action_to_job_placement_shape(self):
        '''Returns a mapping of action (int) -> job_placement_shape (tuple).'''
        action_to_job_placement_shape, action = {0: None}, 1
        for c in range(1, self.cluster.topology.num_communication_groups+1):
            for r in range(1, self.cluster.topology.num_racks_per_communication_group+1):
                for s in range(1, self.cluster.topology.num_servers_per_rack+1):
                    action_to_job_placement_shape[action] = (c, r, s)
                    action += 1
        return action_to_job_placement_shape

    def _init_cluster(self):
        return RampClusterEnvironment(topology_config=self.topology_config,
                                      node_config=self.node_config,
                                      path_to_save=self.path_to_save if self.save_cluster_data else None,
                                      save_freq=self.save_freq,
                                      use_sqlite_database=self.use_sqlite_database)

    def _init_cluster_managers(self):
        # if self.op_partitioner_str == 'random_op_partitioner':
            # op_partitioner = RandomOpPartitioner(**self.op_partitioner_kwargs)
        # elif self.op_partitioner_str == 'sip_ml_op_partitioner':
            # op_partitioner = SipMlOpPartitioner(**self.op_partitioner_kwargs)
        # else:
            # raise Exception(f'Unrecognised op_partitioner {self.op_partitioner_str}')
        if self.job_placement_shaper_str == 'ramp_random_job_placement_shaper':
            job_placement_shaper = RampRandomJobPlacementShaper(**self.job_placement_shaper_kwargs)
        elif self.job_placement_shaper_str == 'ramp_first_fit_job_placement_shaper':
            job_placement_shaper = RampFirstFitJobPlacementShaper(**self.job_placement_shaper_kwargs)
        else:
            raise Exception(f'Unrecognised job_placement_shaper {self.job_placement_shaper_str}')

        if self.op_placer_str == 'ramp_first_fit_op_placer':
            op_placer = RampFirstFitOpPlacer(**self.op_placer_kwargs)
        else:
            raise Exception(f'Unrecognised op_placer {self.op_placer_str}')

        if self.op_scheduler_str == 'srpt_op_scheduler':
            op_scheduler = SRPTOpScheduler(**self.op_scheduler_kwargs)
        else:
            raise Exception(f'Unrecognised op_scheduler {self.op_scheduler_str}')

        if self.dep_placer_str == 'first_fit_dep_placer':
            dep_placer = FirstFitDepPlacer(**self.dep_placer_kwargs)
        else:
            raise Exception(f'Unrecognised dep_placer {self.dep_placer_str}')

        if self.dep_scheduler_str == 'srpt_dep_scheduler':
            dep_scheduler = SRPTDepScheduler(**self.dep_scheduler_kwargs)
        else:
            raise Exception(f'Unrecognised dep_scheduler {self.dep_scheduler_str}')

        # return op_partitioner, op_placer, op_scheduler, dep_placer, dep_scheduler
        return job_placement_shaper, op_placer, op_scheduler, dep_placer, dep_scheduler

    def reset(self,
              seed: int = None,
              verbose=False):

        self.step_counter = 0

        # init env decisions
        # self.op_partition = None
        self.job_placement_shape = None
        self.op_placement = None
        self.op_schedule = None
        self.dep_placement = None
        self.dep_schedule = None

        # reset the cluster environment
        self._reset_cluster(seed=seed, verbose=verbose)

        # # update op partition ready for next job placement shape decision by agent
        # max_partitions_per_op = self.cluster.jobs_generator.max_partitions_per_op_in_observation
        # self.op_partition = self.op_partitioner.get(cluster=self.cluster, max_partitions_per_op=max_partitions_per_op)

        # reset the observation function
        self.observation_function.reset(self)
        self.observation_space = self.observation_function.observation_space

        # reset the reward function
        self.reward_function.reset(env=self)

        # extract current MDP info and save so can access for next env.step() call
        self.obs = self._get_observation() # encoded obs of job to place

        return self.obs

    def _reset_cluster(self, seed: int = None, verbose: bool = False):
        _ = self.cluster.reset(jobs_config=self.jobs_config,
                               max_simulation_run_time=self.max_simulation_run_time,
                               job_queue_capacity=self.job_queue_capacity,
                               seed=seed,
                               verbose=verbose)

    def _is_done(self):
        return self.cluster.is_done()

    def _get_observation(self):
        # extract obs node and edge features
        return self.observation_function.extract(env=self, done=self._is_done())

    def _get_info(self):
        return {}

    def _step_cluster(self, action, verbose=False):
        # step cluster
        self.cluster.step(action=action, verbose=False)

        # update cluster step stats tracker
        self.cluster_step_stats[self.cluster.step_counter] = self.cluster.step_stats

    def step(self, action: int, verbose=False):
        # verbose = True # DEBUG

        # action = 32 # DEBUG
        # action = 4 # DEBUG
        # action = 0 # DEBUG


        if verbose:
            print(f'\n~~~~~~~~~~~~~~~~~~~ Step {self.step_counter} ~~~~~~~~~~~~~~~~~~~~~')

        # init cluster step stats tracker for this step
        self.cluster_step_stats = {}

        # # PROCESS AGENT DECISION
        if action not in self.obs['action_set']:
            raise Exception(f'Action {action} not in action set {self.obs["action_set"]}')
        # TODO TEMP: Checking if RLlib pre-checks are crashing scripts
        if not self.obs['action_mask'][action]:
            raise Exception(f'Action {action} is invalid given action mask {self.obs["action_mask"]} for action set {self.obs["action_set"]}')

        if action != 0:
            job_id, job = list(self.cluster.job_queue.jobs.keys())[0], list(self.cluster.job_queue.jobs.values())[0]
            job_id_to_op_id_to_num_partitions = defaultdict(lambda: defaultdict(lambda: 1))

            # collapse mirrored graph into only forward pass nodes
            forward_graph = get_forward_graph(job.computation_graph)

            max_partitions_per_op = action
            for forward_op_id in forward_graph.nodes:
                # choose an EVEN number of times to partition this op
                # HACK: assume worker type is A100
                worker_type = 'A100'
                num_partitions = int(max(1, min(math.ceil(math.ceil(forward_graph.nodes[forward_op_id]['compute_cost'][worker_type] / self.min_op_run_time_quantum) / 2) * 2, max_partitions_per_op)))

                # partition this forward op
                job_id_to_op_id_to_num_partitions[job_id][forward_op_id] = num_partitions

                # apply same partitioning to the backward op
                backward_op_id = job.computation_graph.nodes[forward_op_id]['backward_node_id']
                job_id_to_op_id_to_num_partitions[job_id][backward_op_id] = num_partitions

            self.op_partition = OpPartition(job_id_to_op_id_to_num_partitions, cluster=self.cluster)

        else:
            # job was not placed
            self.op_partition = OpPartition({}, cluster=self.cluster)

        if verbose:
            print(f'Agent action: {action} -> Action set: {self.obs["action_set"]} | Action mask: {self.obs["action_mask"]}')

        # get env decisions
        # self.job_placement_shape = self.job_placement_shaper.get(op_partition=self.op_partition, cluster=self.cluster)
        # # print(f'job_placement_shape: {self.job_placement_shape}')
        # self.op_placement = self.op_placer.get(op_partition=self.op_partition, job_placement_shape=self.job_placement_shape, cluster=self.cluster)
        self.op_placement = self.op_placer.get(op_partition=self.op_partition, cluster=self.cluster)
        # print(f'op_placement: {self.op_placement}')
        self.op_schedule = self.op_scheduler.get(op_partition=self.op_partition, op_placement=self.op_placement, cluster=self.cluster)      
        # print(f'op_schedule: {self.op_schedule}')
        self.dep_placement = self.dep_placer.get(op_partition=self.op_partition, op_placement=self.op_placement, cluster=self.cluster)      
        # print(f'dep_placement: {self.dep_placement}')
        self.dep_schedule = self.dep_scheduler.get(op_partition=self.op_partition, dep_placement=self.dep_placement, cluster=self.cluster)
        # print(f'dep_schedule: {self.dep_schedule}')

        # syncronise decisions into a valid ClusterEnvironment action
        self.action = Action(op_partition=self.op_partition,
                             # job_placement_shape=self.job_placement_shape,
                             op_placement=self.op_placement,
                             op_schedule=self.op_schedule,
                             dep_placement=self.dep_placement,
                             dep_schedule=self.dep_schedule)

        # record job idx of most recently arrived job
        self.last_job_arrived_job_idx = copy.deepcopy(self.cluster.last_job_arrived_job_idx)

        # step the cluster
        self._step_cluster(action=self.action, verbose=False)

        # check which jobs were placed (useful for accessing externally with e.g. reward functions)
        self.placed_job_idxs, jobs_blocked = self.action.job_idxs, []
        for job_idx in self.placed_job_idxs:
            if job_idx in self.cluster.jobs_blocked:
                # job was blocked due to exceeding its maximum acceptable job completion time
                jobs_blocked.append(job_idx)
        for job_idx in jobs_blocked:
            self.placed_job_idxs.remove(job_idx)

        # get the reward
        self.reward = self._get_reward()

        # continue stepping cluster until there is a job to place or until sim is completed
        while len(self.cluster.job_queue) == 0 and not self.cluster.is_done():
            self._step_cluster(action=Action(), verbose=False)

        # extract current MDP info and save so can access for next env.step() call
        self.done = self._is_done()
        if not self.done:
            self.obs = self._get_observation() # encoded obs of job to place
        else:
            # done, just return last created self.obs
            pass
        self.info = self._get_info()

        # if not self.done:
            # # update op partition ready for next job placement shape decision by agent
            # max_partitions_per_op = self.cluster.jobs_generator.max_partitions_per_op_in_observation
            # self.op_partition = self.op_partitioner.get(cluster=self.cluster, max_partitions_per_op=max_partitions_per_op)

        if verbose:
            print(f'Reward: {self.reward} | Done: {self.done}')

        
        self.step_counter += 1

        # # DEBUG
        # if self.step_counter == 2:
            # raise Exception()

        return self.obs, self.reward, self.done, self.info

    def _get_reward(self):
        return self.reward_function.extract(env=self, done=self._is_done())
