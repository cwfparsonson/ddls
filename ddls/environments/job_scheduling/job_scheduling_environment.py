



class JobSchedulingEnvironment:
    def __init__(self,
                 topology_config: dict,
                 node_config: dict,
                 name: str = 'job_scheduling',
                 path_to_save: str = None,
                 save_freq: int = 1,
                 use_sqlite_database: bool = False):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass
