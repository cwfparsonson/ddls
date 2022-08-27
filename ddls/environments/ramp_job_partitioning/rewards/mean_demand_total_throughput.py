from ddls.environments.ddls_reward_function import DDLSRewardFunction
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment

import numpy as np
import math
from typing import Union

from decimal import Decimal


class MeanDemandTotalThroughput(DDLSRewardFunction):
    def __init__(self, 
                 sign: int = 1, 
                 transform_with_log: bool = False,
                 normalise: bool = False
                 ):
        '''
        Unlike MeanClusterThroughput, which uses the partitioned job's characteristics
        to calculate throughput and reward, this function uses the original job's
        throughput (i.e. before partitioning). This means that throughput is not
        arbitrarily increased by more partitioning (since when partition a node,
        you make copies of its edges but copied edge sizes have same size as original
        edge, so dependency info increases -> info processed increases -> higher
        cluster throughput).
        '''
        self.sign = sign
        self.transform_with_log = transform_with_log
        self.normalise = normalise

        # self.verbose = True # DEBUG
        self.verbose = False

    def reset(self, 
              env, 
              **kwargs):
        # calc max/min computation throughput (occurs where job with highest compute throughput op is placed on each machine and is being executed at same time)
        max_op_comp_throughput = env.cluster.jobs_generator.jobs_params['max_job_max_op_compute_throughputs']
        self.max_comp_throughput = max_op_comp_throughput * env.cluster.topology.graph.graph['num_workers']
        self.min_comp_throughput = 0

        # calc max/min communication throughput (occurs where all workers are transferring data along all of their links' channels)
        self.max_dep_throughput = env.cluster.topology.graph.graph['num_workers'] * env.cluster.topology.channel_bandwidth * env.cluster.topology.num_channels
        self.min_dep_throughput = 0

        # calc max/min cluster throughput
        self.max_cluster_throughput = self.max_comp_throughput + self.max_dep_throughput
        self.min_cluster_throughput = self.min_comp_throughput + self.min_dep_throughput

        if self.verbose:
            print(f'min_comp_throughput: {Decimal(self.min_comp_throughput):.2E}')
            print(f'max_comp_throughput: {Decimal(self.max_comp_throughput):.2E}')
            print(f'min_dep_throughput: {Decimal(self.min_dep_throughput):.2E}')
            print(f'max_dep_throughput: {Decimal(self.max_dep_throughput):.2E}')
            print(f'min_cluster_throughput: {Decimal(self.min_cluster_throughput):.2E}')
            print(f'max_cluster_throughput: {Decimal(self.max_cluster_throughput):.2E}')

    def _normalise_reward(self, reward):
        return (reward - self.min_cluster_throughput) / (self.max_cluster_throughput - self.min_cluster_throughput)

    def extract(self, 
                env, # RampJobPartitioningEnvironment, 
                done: bool):
        # get all ramp cluster environment steps' mean cluster throughputs recorded since last job partitioning env step
        throughputs = [step_stats['mean_demand_total_throughput'] for step_stats in env.cluster_step_stats.values()]
        if self.verbose:
            print(f'demand_total_throughputs: {throughputs}')

        # use mean throughput over last cluster steps as env reward
        reward = np.mean(throughputs)

        # do any reward processing
        if self.verbose:
            print(f'reward before normalising: {Decimal(reward):.2E}')
        if self.normalise:
            reward = self._normalise_reward(reward)
        if self.verbose:
            print(f'reward after normalising: {reward}')

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
