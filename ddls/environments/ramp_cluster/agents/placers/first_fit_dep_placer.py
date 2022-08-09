from ddls.managers.placers.placer import Placer
from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.dep_placement import DepPlacement
from ddls.environments.ramp_cluster.actions.op_placement import OpPlacement
from ddls.environments.ramp_cluster.actions.op_partition import OpPartition
from ddls.utils import gen_channel_id

import networkx as nx
import numpy as np
import copy
from collections import defaultdict
import random
import json



class FirstFitDepPlacer(Placer):
    '''Use first fit to shortest paths to decide route through network for job dependencies.'''
    def __init__(self):
        pass

    def get(self,
            op_partition: OpPartition,
            op_placement: OpPlacement,
            cluster: RampClusterEnvironment,
            verbose=False):
        new_job_op_placements = op_placement.action

        # initialise job_id -> dep_id -> channel_ids
        job_to_dep_to_channels = defaultdict(lambda: defaultdict(set))

        if len(new_job_op_placements) == 0:
            # no new placements made, flow placement is unchanged
            return DepPlacement(job_to_dep_to_channels)
        else:
            # new job(s) will be mounted on cluster, need to get new job dep placement 
            pass

        # make placements
        channel_ids_used_for_other_jobs = set() # track (src, dst, channel_num) channel ids used across jobs
        for job_id, job in op_partition.partitioned_jobs.items():
            _channel_ids_used_for_other_jobs = set() # track (src, dst, channel_num) channel ids used for this job
            if job_id in new_job_op_placements:
                if verbose:
                    print(f'Attempting to place deps for job idx {job.details["job_idx"]}...')
                for dep_id in job.computation_graph.edges:
                    parent, child, key = dep_id

                    # find which server node parent and child ops placed on
                    parent_worker = new_job_op_placements[job_id][parent]
                    parent_node = cluster.topology.graph.graph['worker_to_node'][parent_worker]
                    child_worker = new_job_op_placements[job_id][child]
                    child_node = cluster.topology.graph.graph['worker_to_node'][child_worker]

                    # get size of dependency's tensor
                    size = job.computation_graph[parent][child][key]['size']

                    if verbose:
                        print(f'Dep ID: {dep_id} | Src node: {parent_node} | Dst node: {child_node} | Size: {size}')

                    # search for valid dep placement 
                    if parent_node != child_node and size > 0:
                        # non-zero tensor with parent and child on different workers becomes a network flow -> must be placed
                        if verbose:
                            print(f'Dep is a flow, searching for valid placement...')
                        path, channel_num = self._get_valid_path_channel_num(cluster, dep_id, job, channel_ids_used_for_other_jobs, new_job_op_placements, verbose=verbose)
                        if path is None:
                            # no valid placement for this flow
                            if verbose:
                                print(f'No valid path-channel placement found, job blocked.')
                            try:
                                del job_to_dep_to_channels[job.job_id]
                            except KeyError:
                                # have not yet placed a flow for this job
                                pass
                            break
                        else:
                            if verbose:
                                print(f'Valid path-channel placement: Path {path} channel {channel_num}')
                            for idx in range(len(path) - 1):
                                src, dst = (path[idx], path[idx+1])
                                channel_id = gen_channel_id(src, dst, channel_num)
                                job_to_dep_to_channels[job_id][dep_id].add(channel_id)
                                _channel_ids_used_for_other_jobs.add(channel_id)
                    else:
                        # dependency is not a flow, no need to place
                        if verbose:
                            print(f'Dep is not a flow, no placement needed.')
                        job_to_dep_to_channels[job_id][dep_id].add(None)
                        # pass

            # update channel ids used across jobs
            for channel_id in _channel_ids_used_for_other_jobs:
                channel_ids_used_for_other_jobs.add(channel_id)

        return DepPlacement(job_to_dep_to_channels)

    def _get_valid_path_channel_num(self, cluster, dep, job, channel_ids_used_for_other_jobs, new_job_op_placements, verbose=False):
        # find which nodes in cluster parent and child ops were placed on
        job_id = job.job_id
        parent_op, child_op, _ = dep
        parent_worker = new_job_op_placements[job_id][parent_op]
        parent_node = cluster.topology.graph.graph['worker_to_node'][parent_worker]
        child_worker = new_job_op_placements[job_id][child_op]
        child_node = cluster.topology.graph.graph['worker_to_node'][child_worker]

        # get valid paths between child and parent ops' nodes
        paths = nx.all_shortest_paths(cluster.topology.graph, source=parent_node, target=child_node)
        
        # get possible channel numbers
        channel_nums = list(range(cluster.topology.num_channels))

        # randomly shuffle channel numbers so not all flows of a job are on same channel
        random.shuffle(channel_nums)

        # find first path-channel combination which is valid
        valid_path, valid_channel = None, None
        for path in paths:
            for channel_num in channel_nums:
                if verbose:
                    print(f'Trying path {path} channel {channel_num}...')
                is_valid = self._check_path_channel_valid(path, channel_num, job, dep, cluster, channel_ids_used_for_other_jobs, verbose=verbose)
                if verbose:
                    print(f'Is path-channel combination valid: {is_valid}')
                if is_valid:
                    return path, channel_num

        # if reach here, no valid path-channel combination was found
        return None, None

    def _check_path_channel_valid(self, path, channel_num, job, dep, cluster, channel_ids_used_for_other_jobs, verbose=False):
        is_valid = True

        # Ramp Rule 1: No channel can have flows from more than one job
        for idx in range(len(path) - 1):
            src, dst = (path[idx], path[idx+1])
            channel_id = gen_channel_id(src, dst, channel_num)
            channel = cluster.topology.graph[src][dst]['channels'][channel_id]
            if verbose:
                print(f'Channel ID {channel_id} mounted job indices: {list(channel.mounted_job_idx_to_deps.keys())}')
            if job.details['job_idx'] not in channel.mounted_job_idx_to_deps:
                # not yet mounted a flow from this job onto the channel
                if len(list(channel.mounted_job_idx_to_deps.keys())) > 0 or (channel_id in channel_ids_used_for_other_jobs):
                # if len(list(channel.mounted_job_idx_to_deps.keys())) > 0:
                    # already have another job's flow mounted on this channel
                    is_valid = False
                    break
            else:
                # already begun mounting deps from this job onto this channel so must be valid
                pass

        return is_valid










                    
