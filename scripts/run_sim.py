from ddls.devices.processors.gpus.A100 import A100
from ddls.clusters.cluster import Cluster
from ddls.utils import ddls_graph_from_pbtxt_file
from ddls.plotting.plotting import plot_computation_graph
from ddls.demands.jobs.job import Job
from ddls.managers.placers.random_job_placer import RandomJobPlacer
from ddls.managers.schedulers.srpt_job_scheduler import SRPTJobScheduler
from ddls.managers.schedulers.random_job_scheduler import RandomJobScheduler
from ddls.distributions.uniform import Uniform
from ddls.utils import seed_stochastic_modules_globally

import glob
import time




# initialise cluster configs
node_config = {'type_1':
                  {
                      'num_nodes': 16,
                      'workers_config': 
                          [
                              {
                               'num_workers': 4,
                               'worker': A100
                              }
                          ]
                  }
              }

topology_config = {'type':
                      'torus',
                   'kwargs':
                      {
                          'x_dims': 4,
                          'y_dims': 4
                      }
                  }

# initialise deep learning jobs
num_graphs = 100
path_to_files = '/scratch/datasets/ddls/jobs/tensorflow_synthetic_graphs/valid'
file_paths = glob.glob(path_to_files + '/*')
ddls_computation_graphs = [ddls_graph_from_pbtxt_file(file_path, 
                                                      processor_type_profiled='A100', 
                                                      verbose=False) 
                                    for file_path in file_paths[:num_graphs]]
jobs = [Job(computation_graph=graph, num_training_steps=2) for graph in ddls_computation_graphs]

# initialise decision-making agents
control_plane = {
    'job_placer': RandomJobPlacer(),
    'job_scheduler': SRPTJobScheduler()
    # 'job_scheduler': RandomJobScheduler()
    }

# initilise cluster environment
env = Cluster(topology_config=topology_config,
              node_config=node_config,
              path_to_save='/scratch/datasets/ddls/sims',
              save_freq=100,
              use_sqlite_database=True)





if __name__ == '__main__':
    # run the simulations
    seeds = [0, 1, 2]
    for seed in seeds:
        print(f'\n\n\n~~~~~~~~~~~~~~~~~~~~~~~ Seed {seed} ~~~~~~~~~~~~~~~~~~~~~~~')
        seed_stochastic_modules_globally(seed)
        obs, action_set, reward, done, info = env.reset(jobs=jobs,
                                                        job_sampling_mode='remove',
                                                        job_interarrival_time_dist=Uniform(min_val=1, max_val=1000),
                                                        max_simulation_run_time=float('inf'),
                                                        job_queue_capacity=10,
                                                        seed=seed,
                                                        verbose=True)
        
        start_time = time.time()
        while not done:
            # make decisions
            actions = {}
            actions['job_placement'] = control_plane['job_placer'].get_placement(cluster=env)
            actions['job_schedule'] = control_plane['job_scheduler'].get_schedule(new_placements=actions['job_placement'], cluster=env)

            # pass actions to cluster environment and step the cluster
            obs, action_set, reward, done, info = env.step(actions, verbose=False)

            print(f'Step {env.step_counter} | Jobs arrived: {env.num_jobs_arrived} | completed: {len(env.jobs_completed)} | blocked: {len(env.jobs_blocked)} | running: {len(env.jobs_running)} | queued: {len(env.job_queue)}')

        print(f'\nCompleted simulation in {time.time() - start_time:.3f} s')
