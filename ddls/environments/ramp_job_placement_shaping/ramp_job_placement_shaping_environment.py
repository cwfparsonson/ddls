
import gym


class RampJobPlacementShapingEnvironment(gym.Env):
    def __init__(self,
                 topology_config: dict,
                 node_config: dict,
                 name: str = 'ramp_job_placement_shaping',
                 path_to_save: str = None,
                 save_freq: int = 1,
                 use_sqlite_database: bool = False):
        pass

    def reset(self,
              jobs_config: dict,
              max_simulation_run_time: Union[int, float] = float('inf'),
              job_queue_capacity: int = 10,
              seed: int = None,
              verbose=False):
        pass
