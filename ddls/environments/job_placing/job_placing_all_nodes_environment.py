from ddls.environments.cluster.cluster_environment import ClusterEnvironment
from ddls.environments.job_placing.observations.job_placing_all_nodes_observation import JobPlacingAllNodesObservation
from ddls.environments.job_placing.rewards.job_completion_time import JobCompletionTime
from ddls.demands.jobs.job import Job
from ddls.distributions.distribution import Distribution

from typing import Union


class JobPlacingAllNodesEnvironment:
    def __init__(self,
                 topology_config: dict,
                 node_config: dict,
                 jobs: list[Job], 
                 job_interarrival_time_dist: Distribution,
                 observation_function: str = 'default',
                 information_function: str = 'default',
                 reward_function: str = 'default',
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
        At each step, place all nodes in a computation graph -> one episode
        corresponds to >= one job placement (place jobs until cluster env is done).

        This is as opposed to JobPlacingPerNodeEnvironment, where at each step
        only place one operation -> one episode corresponds to one job placement.
        '''
        self.topology_config = topology_config
        self.node_config = node_config
        self.jobs = jobs
        self.job_interarrival_time_dist = job_interarrival_time_dist
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
        if observation_function == 'default':
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
        if reward_function == 'default':
            self.reward_function = {'job_completion_time': JobCompletionTime()}
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

        # extract observation
        obs = self.observation_function.extract(cluster=self.cluster, done=self.is_done())









    def step(self, action):
        pass





















