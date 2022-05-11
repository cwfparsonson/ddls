from ddls.managers.placers.placer import Placer
from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.dep_placement import DepPlacement
from ddls.utils import gen_channel_id

import networkx as nx
import numpy as np
import copy
from collections import defaultdict
import random
import json



class FirstFitFlowPlacer(Placer):
    '''Use first fit to shortest paths to decide route through network for job dependencies.'''
    def __init__(self):
        pass

    def get(self,
            job_placement: dict,
            cluster: RampClusterEnvironment):
        new_job_op_placements = job_placement.placement

        # initialise job -> dep -> path -> channel_num assignment
        job_to_dep_to_path_to_channel_num = dict()

        if len(new_job_op_placements) == 0:
            # no new placements made, flow placement is unchanged
            return job_to_dep_to_path_to_channel_num
        else:
            # new job(s) will be mounted on cluster, need to get new job dep placement 
            pass

        # make placements
        channel_ids_used = set() # track (src, dst, channel_num) channel ids used across jobs
        for job_id, job in cluster.job_queue.jobs.items():
            _channel_ids_used = set() # track (src, dst, channel_num) channel ids used for this job
            if job_id in new_job_op_placements:
                job_to_dep_to_path_to_channel_num[job_id] = dict()
                for dep in job.computation_graph.edges:
                    job_to_dep_to_path_to_channel_num[job_id][dep] = dict()
                    parent, child, key = dep

                    # find which server node parent and child ops placed on
                    parent_worker = new_job_op_placements[job_id][parent]
                    parent_node = cluster.topology.graph.graph['worker_to_node'][parent_worker]
                    child_worker = new_job_op_placements[job_id][child]
                    child_node = cluster.topology.graph.graph['worker_to_node'][child_worker]

                    # get size of dependency's tensor
                    size = job.computation_graph[parent][child][key]['size']

                    # search for valid dep placement 
                    if parent_node != child_node and size > 0:
                        # non-zero tensor with parent and child on different workers becomes a network flow -> must be placed
                        path, channel_num = self._get_valid_path_channel_num(cluster, dep, job, channel_ids_used, new_job_op_placements)
                        # print(f'\nValid path chosen: {path} | channel: {channel_num}')
                        if path is None:
                            # no valid placement for this flow
                            try:
                                del job_to_dep_to_path_to_channel_num[job.job_id]
                            except KeyError:
                                # have not yet placed a flow for this job
                                pass
                            break
                        else:
                            job_to_dep_to_path_to_channel_num[job_id][dep][json.dumps(path)] = channel_num
                            for idx in range(len(path) - 1):
                                src, dst = (path[idx], path[idx+1])
                                _channel_ids_used.add(gen_channel_id(src, dst, channel_num))
                    else:
                        # dependency is not a flow, no need to place
                        # print(f'dep {dep} w/ parent_node {parent_node} child_node {child_node} size {size} is not a flow')
                        pass

            # update channel ids used across jobs
            for channel_id in _channel_ids_used:
                channel_ids_used.add(channel_id)

        return DepPlacement(job_to_dep_to_path_to_channel_num)

    def _get_valid_path_channel_num(self, cluster, dep, job, channel_ids_used, new_job_op_placements):
        # find which nodes in cluster parent and child ops were placed on
        job_id = job.job_id
        parent_op, child_op, _ = dep
        parent_worker = new_job_op_placements[job_id][parent_op]
        parent_node = cluster.topology.graph.graph['worker_to_node'][parent_worker]
        child_worker = new_job_op_placements[job_id][child_op]
        child_node = cluster.topology.graph.graph['worker_to_node'][child_worker]

        # get valid paths between child and parent ops' nodes
        paths = nx.all_shortest_paths(cluster.topology.graph, source=parent_node, target=child_node)

        # find first path-channel combination which is valid
        valid_path, valid_channel = None, None
        for path in paths:
            for channel_num in range(cluster.topology.num_channels):
                is_valid = self._check_path_channel_valid(path, channel_num, job, dep, cluster, channel_ids_used)
                if is_valid:
                    return path, channel_num

        # if reach here, no valid path-channel combination was found
        return None, None

    def _check_path_channel_valid(self, path, channel_num, job, dep, cluster, channel_ids_used):
        # print(f'\nchecking if path {path} channel_num {channel_num} valid')
        # print(f'channel_ids_used: {channel_ids_used}')
        is_valid = True
        # Ramp Rule 1: No channel can have flows from more than one job
        for idx in range(len(path) - 1):
            src, dst = (path[idx], path[idx+1])
            channel_id = gen_channel_id(src, dst, channel_num)
            channel = cluster.topology.graph[src][dst]['channels'][channel_id]
            if job.details['job_idx'] not in channel.mounted_job_idx_to_deps:
                if len(list(channel.mounted_job_idx_to_deps.keys())) > 0 or (channel_id in channel_ids_used):
                    # already have another flow mounted on this channel
                    is_valid = False
                    break
        # print(f'is_valid: {is_valid}')
        return is_valid












                    
