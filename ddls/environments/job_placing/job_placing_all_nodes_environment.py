from ddls.environments.cluster.cluster_environment import ClusterEnvironment
from ddls.demands.jobs.job import Job
from ddls.distributions.distribution import Distribution

class JobPlacingAllNodesEnvironment:
    def __init__(self,
                 topology_config: dict,
                 node_config: dict,
                 jobs: list[Job], 
                 job_interarrival_time_dist: Distribution,
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
        corresponds to >= one job placement.

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

        self.cluster = self._init_cluster()
        self._reset_cluster()

    def _init_cluster(self):
        return ClusterEnvironment(topology_config=self.topology_config,
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

    def reset(self):
        pass

    def step(self, action):
        pass





















