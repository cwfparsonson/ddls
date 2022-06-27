from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment

from ddls.environments.ramp_cluster.agents.partitioners.random_op_partitioner import RandomOpPartitioner
from ddls.environments.ramp_cluster.agents.partitioners.sip_ml_op_partitioner import SipMlOpPartitioner
from ddls.environments.ramp_cluster.agents.job_placement_shapers.ramp_random_job_placement_shaper import RampRandomJobPlacementShaper
from ddls.environments.ramp_cluster.agents.placers.random_op_placer import RandomOpPlacer
from ddls.environments.ramp_cluster.agents.placers.ramp_random_op_placer import RampRandomOpPlacer
from ddls.environments.ramp_cluster.agents.schedulers.srpt_op_scheduler import SRPTOpScheduler
from ddls.environments.ramp_cluster.agents.placers.first_fit_dep_placer import FirstFitDepPlacer
from ddls.environments.ramp_cluster.agents.schedulers.srpt_dep_scheduler import SRPTDepScheduler

from ddls.environments.ramp_job_placement_shaping.rewards.lookahead_job_completion_time import LookaheadJobCompletionTime

from ddls.environments.ramp_job_placement_shaping.observations.ramp_job_placement_shaping_observation import RampJobPlacementShapingObservation

from ddls.environments.ramp_cluster.actions.job_placement_shape import JobPlacementShape
from ddls.environments.ramp_cluster.actions.action import Action


import gym
import numpy as np

from typing import Union


class RampJobPlacementShapingEnvironment(gym.Env):
    def __init__(self,
                 topology_config: dict,
                 node_config: dict,
                 jobs_config: dict,
                 op_partitioner: Union['random_op_partitioner', 'sip_ml_op_partitioner'] = 'sip_ml_op_partitioner',
                 op_partitioner_kwargs: dict = None,
                 op_placer: Union['ramp_random_op_placer'] = 'ramp_random_op_placer',
                 op_placer_kwargs: dict = None,
                 op_scheduler: Union['srpt_op_scheduler'] = 'srpt_op_scheduler',
                 op_scheduler_kwargs: dict = None,
                 dep_placer: Union['first_fit_dep_placer'] = 'first_fit_dep_placer',
                 dep_placer_kwargs: dict = None,
                 dep_scheduler: Union['srpt_dep_scheduler'] = 'srpt_dep_scheduler',
                 dep_scheduler_kwargs: dict = None,
                 observation_function: Union['ramp_job_placement_shaping_observation'] = 'ramp_job_placement_shaping_observation',
                 pad_obs_kwargs: dict = None,
                 information_function: Union['default'] = 'default',
                 reward_function: Union['lookahead_job_completion_time'] = 'lookahead_job_completion_time',
                 max_simulation_run_time: Union[int, float] = float('inf'),
                 job_queue_capacity: int = 10,
                 name: str = 'ramp_job_placement_shaping',
                 path_to_save: str = None,
                 save_cluster_data: bool = False,
                 save_freq: int = 1,
                 use_sqlite_database: bool = False):
        self.topology_config = topology_config
        self.node_config = node_config
        self.jobs_config = jobs_config

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

        # init obs
        self.observation_function_str = observation_function
        if observation_function == 'ramp_job_placement_shaping_observation':
            self.observation_function = RampJobPlacementShapingObservation(pad_obs_kwargs=self.pad_obs_kwargs)
        else:
            raise Exception(f'Unrecognised observation_function {self.observation_function_str}')

        # init action space
        self.action_space = gym.spaces.Discrete(int(self.cluster.topology.num_communication_groups * self.cluster.topology.num_racks_per_communication_group * self.cluster.topology.num_servers_per_rack))
        self.action_to_job_placement_shape = self._get_action_to_job_placement_shape()

        # init info
        self.information_function_str = information_function
        if information_function == 'default':
            # TODO: Not implemented
            pass
        else:
            raise Exception(f'Unrecognised information_function {self.information_function_str}')

        # init reward
        self.reward_function_str = reward_function
        if reward_function == 'lookahead_job_completion_time':
            self.reward_function = LookaheadJobCompletionTime()
        else:
            raise Exception(f'Unrecognised reward_function {self.reward_function_str}')

        # init cluster environment managers
        if op_partitioner_kwargs is not None:
            self.op_partitioner_kwargs = op_partitioner_kwargs
        else:
            self.op_partitioner_kwargs = {}
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

        self.op_partitioner_str = op_partitioner 
        self.op_placer_str = op_placer
        self.op_scheduler_str = op_scheduler
        self.dep_placer_str = dep_placer
        self.dep_scheduler_str = dep_scheduler

        self.op_partitioner, self.op_placer, self.op_scheduler, self.dep_placer, self.dep_scheduler = self._init_cluster_managers()

    def _get_action_to_job_placement_shape(self):
        '''Returns a mapping of action (int) -> job_placement_shape (tuple).'''
        action_to_job_placement_shape, action = {}, 0
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
        if self.op_partitioner_str == 'random_op_partitioner':
            op_partitioner = RandomOpPartitioner(**self.op_partitioner_kwargs)
        elif self.op_partitioner_str == 'sip_ml_op_partitioner':
            op_partitioner = SipMlOpPartitioner(**self.op_partitioner_kwargs)
        else:
            raise Exception(f'Unrecognised op_partitioner {self.op_partitioner_str}')

        if self.op_placer_str == 'ramp_random_op_placer':
            op_placer = RampRandomOpPlacer(**self.op_placer_kwargs)
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

        return op_partitioner, op_placer, op_scheduler, dep_placer, dep_scheduler

    def reset(self,
              seed: int = None,
              verbose=False):

        # init env decisions
        self.op_partition = None
        self.op_placement = None
        self.op_schedule = None
        self.dep_placement = None
        self.dep_schedule = None

        # reset the cluster environment
        self._reset_cluster(seed=seed, verbose=verbose)

        # update op partition ready for next job placement shape decision by agent
        self.op_partition = self.op_partitioner.get(cluster=self.cluster)

        # reset the observation function
        self.observation_function.reset(self)
        self.observation_space = self.observation_function.observation_space

        # reset the reward function
        self.reward_function.reset(self.cluster)

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

    def step(self, action: int):
        # process agent decision
        if action not in self.obs['action_set']:
            raise Exception(f'Action {action} not in action set {self.obs["action_set"]}')
        if not self.obs['action_mask'][action]:
            raise Exception(f'Action {action} is invalid given action mask {self.obs["action_mask"]} for action set {self.obs["action_set"]}')
        self.job_placement_shape = JobPlacementShape({list(self.op_partition.job_ids)[0]: self.action_to_job_placement_shape[action]})

        # get env decisions
        self.op_placement = self.op_placer.get(op_partition=self.op_partition, job_placement_shape=self.job_placement_shape, cluster=self.cluster)
        self.op_schedule = self.op_scheduler.get(op_partition=self.op_partition, op_placement=self.op_placement, cluster=self.cluster)      
        self.dep_placement = self.dep_placer.get(op_partition=self.op_partition, op_placement=self.op_placement, cluster=self.cluster)      
        self.dep_schedule = self.dep_scheduler.get(op_partition=self.op_partition, dep_placement=self.dep_placement, cluster=self.cluster)

        # syncronise decisions into a valid ClusterEnvironment action
        self.action = Action(op_partition=self.op_partition,
                             job_placement_shape=self.job_placement_shape,
                             op_placement=self.op_placement,
                             op_schedule=self.op_schedule,
                             dep_placement=self.dep_placement,
                             dep_schedule=self.dep_schedule)
        self.placed_job_idxs = self.action.job_idxs # useful for external methods accessing placed job info

        # step the cluster
        self.cluster.step(self.action, verbose=False)

        # get the reward
        self.reward = self._get_reward()

        # continue stepping cluster until there is a job to place or until sim is completed
        while len(self.cluster.job_queue) == 0 and not self.cluster.is_done():
            self.cluster.step(action=Action(), verbose=False)

        # extract current MDP info and save so can access for next env.step() call
        self.done = self._is_done()
        if not self.done:
            self.obs = self._get_observation() # encoded obs of job to place
        else:
            # done, just return last created self.obs
            pass
        self.info = self._get_info()

        if not self.done:
            # update op partition ready for next job placement shape decision by agent
            self.op_partition = self.op_partitioner.get(cluster=self.cluster)

        return self.obs, self.reward, self.done, self.info

    def _get_reward(self):
        return self.reward_function.extract(env=self, done=self._is_done())
