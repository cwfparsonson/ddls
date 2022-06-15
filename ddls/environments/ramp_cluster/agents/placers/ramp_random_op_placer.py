from ddls.managers.placers.placer import Placer
from ddls.demands.jobs.job import Job
from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
from ddls.environments.ramp_cluster.actions.op_placement import OpPlacement
from ddls.environments.ramp_cluster.actions.op_partition import OpPartition
from ddls.utils import get_forward_graph

import numpy as np
import copy
from collections import defaultdict, deque
import random


def get_allocation_preamble(original_graph, mp_split_ids, mp_splits):
        
        parents, children = get_parents_and_children(original_graph)
        sequence = topo_sort(copy.deepcopy(parents), copy.deepcopy(children))
        op_server_info = {op: [] for op in original_graph.nodes()}
        splits = []
        for s in sequence:
            if s in mp_split_ids:
                idx = mp_split_ids.index(s)
                splits.append(mp_splits[idx])
                # print(f'node {s} in mp_split_ids')
            else:
                splits.append(1)
                # print(f'node {s} not in mp_split_ids')
        return sequence, splits, op_server_info, parents, children

def get_parents_and_children(original_graph):
    parents = {}
    children = {}
    tmp_graph = original_graph.copy()

    for node in original_graph.nodes():
        in_nodes = [in_edge[0] for in_edge in original_graph.in_edges(node)]
        parents[node] = in_nodes

        out_nodes = [out_edge[1] for out_edge in original_graph.out_edges(node)]
        children[node] = out_nodes

    return parents, children

def topo_sort(parents, children):
        sequence = []
        queue = deque()
        for node in parents.keys():
            if parents[node] == []:
                queue.append(node)
                sequence.append(node)
        while len(queue) > 0:
            node = queue.popleft()
            for child in children[node]:
                parents[child].remove(node)
                if parents[child] == []:
                    queue.append(child)
                    sequence.append(child)
        return sequence

def find_meta_block(ramp_topology, ramp_shape, meta_block_shape):
        '''
        Take an input of a RAMP topology and a required number of servers and 
        return a block of servers that is symmetric w.r.t. collective and where
        every server is currently empty.
        
        Returned values should be a set of server IDs, the shape and the 'origin'
        server in that set (i.e. the effective top left hand corner).
        
        meta_block_info = (meta_block, meta_block_shape, meta_block_origin) or None if empty.
        '''
        meta_block_info = ff_meta_block([meta_block_shape], ramp_shape, ramp_topology, 'meta')
        # print(f'find_meta_block: {ramp_topology.keys()}')
        return meta_block_info

def ff_meta_block(block_shapes,ramp_shape,ramp,mode,op_size=None,meta_block_origin=(0,0,0)):
        '''
        For each block shape, create the block.

        When a block has been created, check it using
        check_block to see if it's OK for allocation
        w.r.t. server-resources. If OK, return the block.
        Otherwise, return None.

        The seach will start at the point 'meta_block_origin' which
        should be the upper left hand corner of the block that has
        been allocated to the job. This function will then find a
        sub-block that can be given to a particular (partitioned) 
        op in that job.

        NOTE: currently this code is using (0,0,0) as the upper left
        hand corner of the meta-block. This will have to be an argument
        as different block throughout allocation will have different 
        positions in the RAMP network.

        NOTE: all functions here are only working in multiples of 2.
        this is because accounting for odd server-numbers means extra 
        conditions to be handled when distributing over multiple racks
        because of the RAMP symmetry rules for collectives.
        '''

        orgn_c, orgn_r, orgn_s = meta_block_origin
        for shape in block_shapes:
            #get the acceptable search ranges given how big the meta-block is
            #all shapes will already be maximum the size of the meta-block, so this doesn't have to be checked
            I = ramp_shape[0]-shape[0]+1
            J = ramp_shape[1]-shape[1]+1
            K = ramp_shape[2]-shape[2]+1
            if I <= 0 or J <= 0 or K <= 0:
                continue
            else:
                #get the size of the shape in each RAMP dimension
                C,R,S = shape
                for i in range(ramp_shape[0]):
                    # j = i
                    for j in range(ramp_shape[1]):
                        for k in range(ramp_shape[2]):
                            block = get_meta_block(C,R,S,ramp_shape,origin=((orgn_c+i),(orgn_r+j),(orgn_s+k)))

                            if check_block(ramp,block,op_size,mode):
                                if mode == 'sub':
                                    return block
                                if mode == 'meta':
                                    return (block, shape, (orgn_c+i,orgn_r+j,orgn_s+k))

        return None

def get_meta_block(C,R,S,ramp_shape,origin=(0,0,0)):
        '''
        Given an origin (i.e. a starting server in RAMP), 
        returns a set of servers that are part of a 'shape'
        that is centred at that origin. 

        NOTE: A simplification here is that in the case of
        one server-per-rack, servers of the same id are checked.
        If this is not done, then a fully exhaustive search has to
        be implemented which is infeasible (i.e. for group of racks
        with one server each, every possible combination of servers
        across racks has to be checked...).
        '''
        block = []
        i,j,k = origin

        for c in range(C):
            for r in range(R):
                for s in range(S):
                    block.append(((i+c)%ramp_shape[0],(j+r)%ramp_shape[1],(k+s)%ramp_shape[2]))
        return block

def check_block(ramp,block,op_size,mode):
        '''
        Iterate through each server in a block.

        Return True if each server has enough resource
        to support the requirement, False otherwise
        '''
        if block == []:
            return False
        for server in block:
            if mode == 'sub':
                if ramp[server]['mem'] < op_size:
                    return False
            if mode == 'meta':
                if ramp[server]['ops'] != []:
                    return False
        return True

def dummy_ramp(shape, cluster):
    '''
    Generates a dummy ramp 'network' which is actually
    just a dictionary. This is fine though, since the
    topology of ramp is not really interacted with in 
    any case.
    '''
    c, r, s = shape
    ramp = {}
    for i in range(c):
        for j in range(r):
            for k in range(s):
                node = f'{i}-{j}-{k}'
                mem, ops = 0, []
                ramp[(i, j, k)] = {'mem': 0, 'ops': []}
                for worker in cluster.topology.graph.nodes[node]['workers'].values():
                    ramp[(i, j, k)]['mem'] += (worker.memory_capacity - worker.memory_occupied)
                    ramp[(i, j, k)]['ops'].extend(list(worker.mounted_job_op_to_priority.keys()))
    return ramp

def parent_collective_placement(ramp,job_graph,op,split,meta_block_info,parents,op_server_info):
    '''
    Similar to _regular_collective_placement except it checks if the given op's
    parents are allocated already somewhere. It then checks if it can allocate
    all the sub-ops evenly across the set of servers associated with (one of) the
    parent(s). If there are more child sub-ops than servers used by the parent, then 
    they are packed in evenly across the servers. 
    
    Args:
        ramp (dict): RAMP 'topology' as a dict like {(a,b,c):attributes}
        job_graph (nx.DiGraph): un-partitioned and non-mirrored computational graph of a job
        op (str): name of an op (e.g. '11')
        split (int): how many times this op should be split
        parents (dict): information relating ops in job_graph to their parents (i.e. {'14':['12','13']} if 12 and 13 are parents of 14).
        op_server_info (dict): which ops are allocated across which servers (e.g. {'11':[(0,0,1),(0,0,2)]})
    '''
    # #can't split an un-split node over it's split parents
    # if split == 1:
    #     return None
    
    op_requirement = job_graph.nodes[op]['memory_cost']
    num_nodes = len(job_graph.nodes())
    
    #get sets of servers corresponding to each parent
    parents_servers = []
    for parent in parents[op]:
        if set(op_server_info[parent]).issubset(meta_block_info[0]):
            parents_servers.append(op_server_info[parent])

    #for each set of servers
    for servers in parents_servers:
        if split < len(servers):
            continue
        else:
            #check if ops can fit evenly across the servers
            available_resource = sum([ramp[server]['mem'] for server in servers])
            if available_resource >= op_requirement:
                i = 0
                while i < split:
                    for server in servers:
                        ramp[server]['mem'] -= op_requirement/split
                        if split > 1:
                            ramp[server]['ops'].append(str(int(op))+chr(97+i))
                            # ramp[server]['ops'].append(str(int(op)+num_nodes)+chr(97+i))
                            ramp[server]['ops'].append(str((2*num_nodes)-(int(op)-1))+chr(97+i))
                        else:
                            ramp[server]['ops'].append(op)
                            # ramp[server]['ops'].append(int(op)+num_nodes)
                            ramp[server]['ops'].append((2*num_nodes)-(int(op)-1))
                        op_server_info[op].append(server)
                        i += 1
                return ramp, op_server_info

    return None

def regular_collective_placement(ramp,ramp_shape,job_graph,op,split,meta_block_info,op_server_info):
        '''
        This function allocates a split op to a set of servers in a meta-block nd returns a dictionary 
        of which sub-ops are allocated to which servers, and another dictionary indicating across which
        servers are each op distributed (this does not refer to specific sub-ops and is used so that 
        the parent-checking allocation method can be implemented. Sub-ops are allocated one op per
        server.
        
        Args:
            ramp (dict): RAMP 'topology' as a dict like {(a,b,c):attributes}
            job_graph (nx.DiGraph): un-partitioned and non-mirrored computational graph of a job
            op (str): name of an op (e.g. '11')
            split (int): how many times this op should be split
            meta_block_info (tuple): return value of GreedyBlockAllocator.find_meta_block
            op_server_info (dict): which ops are allocated across which servers (e.g. {'11':[(0,0,1),(0,0,2)]})
                                    
        '''
        # print(f'_regular_collective_placement: {ramp.keys()}')
        num_nodes = len(job_graph.nodes())
        meta_block,meta_block_shape,meta_block_origin = meta_block_info

        num_servers = split #NOTE: ops_per_server should never be more than splits. Needs to be ensured somewhere. EDIT: since putting multiple sub-ops on the same server is trivial (should then just partition it by a smaller amount) we will keep 1 op-per-server for now.
        if num_servers > len(meta_block): #if there are fewer servers in the meta-block than asked for, no allocation
            return None

        op_size = job_graph.nodes[op]['memory_cost']/split
        meta_block = {server:ramp[server] for server in meta_block}

        block = find_sub_block(ramp,ramp_shape,meta_block_shape,meta_block_origin,num_servers,op_size)

        if not block: #if no block can be found (memory errors) then return None
            return None
        for j in range(len(block)):
            ramp[block[j]]['mem'] -= op_size
            if split > 1:
                ramp[block[j]]['ops'].append(str(int(op))+chr(97+j))
                # ramp[block[j]]['ops'].append(str(int(op)+num_nodes)+chr(97+j))
                ramp[block[j]]['ops'].append(str((2*num_nodes)-(int(op)-1))+chr(97+j))
            else:
                ramp[block[j]]['ops'].append(op)
                # ramp[block[j]]['ops'].append(int(op)+num_nodes)
                ramp[block[j]]['ops'].append((2*num_nodes)-(int(op)-1))
            op_server_info[op].append(block[j])

        #if allocation was feasible, return the updated (i.e. with server-memory reduced) RAMP topology for further allocations
        return ramp, op_server_info

def find_sub_block(ramp_topology,ramp_shape,meta_block_shape,meta_block_origin,num_servers,op_size):
        pairs = get_factor_pairs(num_servers)
        block_shapes = get_block_shapes(pairs,meta_block_shape)
        #if no possible shapes try rack and CG distributed
        block_shapes += [(num_servers,num_servers,-1),(num_servers,1,1)]
        # print(f'find_sub_block: {ramp_topology.keys()}')
        block = ff_block(block_shapes,meta_block_shape,ramp_shape,ramp_topology,'sub',op_size=op_size,meta_block_origin=meta_block_origin)
        return block

def ff_block(block_shapes,meta_shape,ramp_shape,ramp,mode,op_size=None,meta_block_origin=(0,0,0)):
        '''
        For each block shape, create the block.

        When a block has been created, check it using
        check_block to see if it's OK for allocation
        w.r.t. server-resources. If OK, return the block.
        Otherwise, return None.

        The seach will start at the point 'meta_block_origin' which
        should be the upper left hand corner of the block that has
        been allocated to the job. This function will then find a
        sub-block that can be given to a particular (partitioned) 
        op in that job.

        NOTE: currently this code is using (0,0,0) as the upper left
        hand corner of the meta-block. This will have to be an argument
        as different block throughout allocation will have different 
        positions in the RAMP network.

        NOTE: all functions here are only working in multiples of 2.
        this is because accounting for odd server-numbers means extra 
        conditions to be handled when distributing over multiple racks
        because of the RAMP symmetry rules for collectives.
        '''
        # print(f'find_sub_block: {ramp.keys()}')
        orgn_c, orgn_r, orgn_s = meta_block_origin
        for shape in block_shapes:
            #get the acceptable search ranges given how big the meta-block is
            #all shapes will already be maximum the size of the meta-block, so this doesn't have to be checked
            I = (meta_shape[0]-shape[0])+1
            J = (meta_shape[1]-shape[1])+1
            K = (meta_shape[2]-shape[2])+1
            if I <= 0 or J <= 0 or K <= 0:
                continue
            else:
                #get the size of the shape in each RAMP dimension
                C,R,S = shape

                for i in range(I):
                    # j = i
                    for j in range(J):
                        for k in range(K):
                            #get a block of shape (C,R,S) at origin (i,j,k)
                            block = get_block(C,R,S,ramp_shape,origin=(orgn_c+i,orgn_r+j,orgn_s+k))
                            if check_block(ramp,block,op_size,mode):
                                if mode == 'sub':
                                    return block
                                if mode == 'meta':
                                    return (block, shape, (orgn_c+i,orgn_r+j,orgn_s+k))

        return None

def get_factor_pairs(n):
        '''
        This function returns a list of tuples specifying
        all integer factor-pairs for a given integer, n.

        This is used for finding symmetric server-blocks 
        to be allocated for collective communication, where
        before a block can be found, the possible 'shapes' 
        of block (given a number of servers required) need to be known.
        '''
        pairs = []

        for i in range(1,n+1):

            if n % i == 0:
                pairs.append((int(n/i),i))

        return pairs

def get_block(C,R,S,ramp_shape,origin=(0,0,0)):
        '''
        Given an origin (i.e. a starting server in RAMP), 
        returns a set of servers that are part of a 'shape'
        that is centred at that origin. 

        NOTE: A simplification here is that in the case of
        one server-per-rack, servers of the same id are checked.
        If this is not done, then a fully exhaustive search has to
        be implemented which is infeasible (i.e. for group of racks
        with one server each, every possible combination of servers
        across racks has to be checked...).
        '''
        block = []
        i,j,k = origin

        if S == -1:
            for n in range(C):
                block.append(((i+n)%(ramp_shape[0]+1),(j+n)%(ramp_shape[1]+1),k))
        else:
            for c in range(C):
                for r in range(R):
                    for s in range(S):
                        block.append(((i+c)%(ramp_shape[0]),(j+r)%(ramp_shape[1]),(k+s)%(ramp_shape[2])))
        return block

def get_block_shapes(pairs,ramp_shape):
        '''
        Given a set of factor-pairs (corresponding to a 
        particular number of servers that have to be 
        allocated into a block) and the size of the full 
        ramp meta-block that is to have the job packed
        into it, return the set of acceptable 'shapes'
        of block that can fit within the size of this 
        meta-block.
        '''
        blocks = []
        # print(ramp_shape)
        for pair in pairs:
            if pair[0] > ramp_shape[0] or pair[0] > ramp_shape[1] or pair[1] > ramp_shape[2]:
                continue
            else:
                # blocks.append((pair[0],pair[0],pair[1]))
                blocks.append((pair[0],1,pair[1]))
                blocks.append((pair[0],pair[1],1))
        
        return blocks

def allocate(ramp,ramp_shape,job_graph,sequence,splits,meta_block_info,parents,op_server_info):
    # op_server_info = {op:[] for op in job_graph.nodes()}
    # print(f'allocate: {ramp.keys()}')
    for i in range(len(sequence)):
        op = sequence[i]
        split = splits[i]
        
        #try allocating on same servers as parents
        alloc = parent_collective_placement(ramp,job_graph,op,split,meta_block_info,parents,op_server_info)
        
        #if this doesn't work, try allocating somewhere else
        if not alloc:
            # print('no parents - regular allocation')
            alloc = regular_collective_placement(ramp,ramp_shape,job_graph,op,split,meta_block_info,op_server_info)
            
        #if that didn't work either, return None (this means the allocation has failed)
        if not alloc:
            return alloc
        
        #if either of the allocation attempts worked, update ramp and op_server_info and go onto the next op
        else:
            ramp, op_server_info = alloc[0], alloc[1]
            
    return ramp, op_server_info





class RampRandomOpPlacer(Placer):
    def __init__(self):
        pass

    def get(self, 
            op_partition: OpPartition,
            cluster: RampClusterEnvironment):
        '''
        Places operations in a job onto available worker(s) in a cluster, where the clusters
        nodes are servers which contain >=1 worker(s) which may or may not have sufficient 
        memory available for a given operation. 
        
        Returns a mapping of job_id -> operation_id -> worker_id. If no valid placement for the operation
        could be found, the job will not be included in the placement mapping.
        '''
        # gather jobs which are requesting to be placed
        jobs = op_partition.partitioned_jobs.values()

        # check how much memory is available on each worker
        worker_to_available_memory = self._get_workers_available_memory(cluster, sort=True)

        # get shape of ramp topology
        ramp_shape = (cluster.topology.num_communication_groups, cluster.topology.num_racks_per_communication_group, cluster.topology.num_servers_per_rack)
        ramp_topology = dummy_ramp(ramp_shape, cluster)

        job_to_operation_to_worker = defaultdict(lambda: defaultdict(lambda: None))
        for partitioned_job in jobs:
            job_id = partitioned_job.job_id

            # get original job
            original_job = cluster.job_queue.jobs[job_id]

            # collapse mirrored graph into only forward pass nodes
            forward_graph = get_forward_graph(original_job.computation_graph)
            
            # get partitioning decisions made
            mp_split_ids = op_partition.job_id_to_mp_split_forward_op_ids[job_id]
            mp_splits = op_partition.job_id_to_mp_splits[job_id]

            # specify shape of meta-block to be used for this job
            meta_shape = tuple([random.randint(1, dim) for dim in ramp_shape])
            
            # get useful info
            sequence, splits, op_server_info, parents, children = get_allocation_preamble(forward_graph, mp_split_ids, mp_splits)

            # get a meta-block of a particular shape which the heuristic allocator will try to pack the job fully into
            meta_block_info = find_meta_block(ramp_topology, ramp_shape, meta_shape)

            if meta_block_info:
                # valid meta block successfully found, try to allocate the job
                allocated = allocate(ramp_topology,ramp_shape,forward_graph,sequence,splits,meta_block_info,parents,op_server_info)
                if allocated:
                    # update the topology and op-server info for use in the next job allocation
                    ramp_topology, op_server_info = allocated
            
            for n in ramp_topology.keys():
                c, r, s = n
                node_id = f'{c}-{r}-{s}'
                # HACK: assume 1 worker per server
                worker_id = list(cluster.topology.graph.nodes[node_id]['workers'].keys())[0]
                for op_id in ramp_topology[n]['ops']:
                    # ensure op_id is string for consistency
                    job_to_operation_to_worker[job_id][str(op_id)] = worker_id

        return OpPlacement(job_to_operation_to_worker)
        


    def _get_workers_available_memory(self, 
                                      cluster: RampClusterEnvironment, 
                                      sort: bool = True):
        '''
        Maps worker ids to available memory. 

        Args:
            sort: If true, returned dict is in order of memory available,
                with the worker with the most memory available first, etc.
        '''
        worker_to_available_memory = dict()
        for worker_id, node_id in cluster.topology.graph.graph['worker_to_node'].items():
            node_id = cluster.topology.graph.graph['worker_to_node'][worker_id]
            worker = cluster.topology.graph.nodes[node_id]['workers'][worker_id]
            worker_to_available_memory[worker_id] = worker.memory_capacity - worker.memory_occupied
        if sort:
            worker_to_available_memory = dict(sorted(worker_to_available_memory.items(), key=lambda x:x[1], reverse=True))
        return worker_to_available_memory
