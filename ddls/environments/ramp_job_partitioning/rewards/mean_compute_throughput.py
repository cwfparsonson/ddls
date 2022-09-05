from ddls.environments.ddls_reward_function import DDLSRewardFunction
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment

import numpy as np
import math
from typing import Union


class MeanComputeThroughput(DDLSRewardFunction):
    def __init__(self, 
                 sign: int = 1, 
                 transform_with_log: bool = False,
                 normalise: bool = False,
                 ):
        self.sign = sign
        self.transform_with_log = transform_with_log
        self.normalise = normalise

    def reset(self, 
              env, 
              **kwargs):
        # calc max/min computation throughput (occurs where job with highest compute throughput node is placed on each machine and is being executed at same time)
        max_op_comp_throughput = env.cluster.jobs_generator.jobs_params['max_job_max_op_compute_throughputs']
        self.max_comp_throughput = max_op_comp_throughput * env.cluster.topology.graph.graph['num_workers']
        self.min_comp_throughput = 0

    def _normalise_reward(self, reward):
        return (reward - self.min_comp_throughput) / (self.max_comp_throughput - self.min_comp_throughput)

    def extract(self, 
                env, # RampJobPartitioningEnvironment, 
                done: bool):
        # get all ramp cluster environment steps' mean compute throughputs recorded since last job partitioning env step
        throughputs = [step_stats['mean_compute_throughput'] for step_stats in env.cluster_step_stats.values()]

        # use mean throughput over last cluster steps as env reward
        reward = np.mean(throughputs)

        # do any reward processing
        # print(f'reward before normalising: {Decimal(reward):.2E}')
        if self.normalise:
            reward = self._normalise_reward(reward)
        # print(f'reward after normalising: {reward}')

        if reward != 0:
            reward *= self.sign
        else:
            pass

        if self.transform_with_log:
            if reward != 0:
                sign = math.copysign(1, reward)
                reward = sign * math.log(1 + abs(reward), 10)
            else:
                pass

        return reward
