from ddls.managers.placers.placer import Placer
from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
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

    def get_placement(self,
                      new_job_op_placements: dict,
                      cluster: RampClusterEnvironment):
        '''
        Places operations in a job onto available worker(s) in a cluster, where the clusters
        nodes are servers which contain >=1 worker(s) which may or may not have sufficient 
        memory available for a given operation. 
        
        Returns a mapping of job_id -> operation_id -> worker_id. If no valid placement for the operation
        could be found, the job will not be included in the placement mapping.
        '''
        # initialise job -> dep -> path -> channel_num assignment
        job_to_dep_to_path_to_channel_num = dict()

        if len(new_job_op_placements) == 0:
            # no new placements made, flow placement is unchanged
            return job_to_dep_to_path_to_channel_num
        else:
            # new job(s) will be mounted on cluster, need to get new job dep placement 
            pass

        # make placements
        channel_ids_used = set()
        for job_id, job in cluster.job_queue.jobs.items():
            if job_id in new_job_op_placements:
                job_to_dep_to_path_to_channel_num[job_id] = dict()
                for dep in job.computation_graph.edges:
                    job_to_dep_to_path_to_channel_num[job_id][json.dumps(dep)] = dict()
                    parent, child = dep[0], dep[1]

                    # find which worker parent and child ops placed on
                    parent_worker = new_job_op_placements[job_id][parent]
                    child_worker = new_job_op_placements[job_id][child]

                    # search for valid dep placement 
                    if parent_worker != child_worker and job.computation_graph[dep]['size'] > 0:
                        # non-zero tensor with parent and child on different workers becomes a network flow -> must be placed
                        path, channel_num = self._get_valid_path_channel_num(cluster, dep, job, channel_ids_used)
                        if path is None:
                            # no valid placement for this flow
                            try:
                                del job_to_dep_to_path_to_channel_num[job.job_id]
                            except KeyError:
                                # have not yet placed a flow for this job
                                pass
                            break
                        else:
                            job_to_dep_to_path_to_channel_num[job_id][json.dumps(dep)][json.dumps(path)] = channel_num
                            for idx in range(len(path) - 1):
                                link = [path[idx], path[idx+1]]
                                channel_ids_used.append(gen_channel_id(link, channel_num))

        return job_to_dep_to_path_to_channel_num

    def _get_valid_path_channel_num(self, cluster, dep, job, channel_ids_used):
        paths = nx.all_shortest_paths(cluster.topology.graph, source=dep[0], target=dep[1])
        valid_path, valid_channel = None, None
        for path in paths:
            for channel_num in range(cluster.topology.num_channels):
                is_valid = self._check_path_channel_valid(path, channel_num, job, dep, cluster, channel_ids_used)
                if is_valid:
                    return path, channel_num
        return None, None

    def _check_path_valid(self, path, channel_num, job, dep, cluster, channel_ids_used):
        is_valid = True
        # Ramp Rule 1: No channel can have flows from more than one job
        for idx in range(len(path) - 1):
            link = [path[idx], path[idx+1]]
            channel_id = f'link_{link}_channel_{channel_num}'
            channel = cluster.topology.graph.edges[link]['channels'][channel_id]
            if job.details['job_idx'] not in channel.mounted_job_idx_to_deps:
                if len(list(channel.mounted_job_idx_to_deps.keys())) > 0 or (channel_id in channel_ids_used):
                    # already have another flow mounted on this channel
                    is_valid = False
                    break
        return is_valid












                    
