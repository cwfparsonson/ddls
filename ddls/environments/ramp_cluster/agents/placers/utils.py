from ddls.environments.ramp_cluster.ramp_cluster_environment import RampClusterEnvironment
import numpy as np
import copy
from collections import defaultdict, deque
import random
import math


# VERBOSE = True # DEBUG
VERBOSE = False


def check_meta_block_valid(c, r, s, ramp_topology, ramp_shape, job_max_partition_degree, num_available_workers):
    is_valid = False
    # if job_max_partition_degree <= c * r * s <= num_available_workers: # OLD TODO CHECK NEW IS VALID WITH ZAK
    if job_max_partition_degree <= c * r * s <= min(num_available_workers, job_max_partition_degree): # NEW
        if c * r * s == job_max_partition_degree:
            # must be able to pack job ops evenly across racks and comm groups
            if c == r:
                # check meta block is valid with first fit search
                # if find_meta_block(ramp_topology, ramp_shape, (c, r, s), job_max_partition_degree) is not None:
                if find_meta_block(ramp_topology, ramp_shape, (c, r, s)) is not None:
                    is_valid = True
        else:
            # check meta block is valid with first fit search
            # if find_meta_block(ramp_topology, ramp_shape, (c, r, s), job_max_partition_degree) is not None:
            if find_meta_block(ramp_topology, ramp_shape, (c, r, s)) is not None:
                # valid meta block found
                is_valid = True
    return is_valid

def get_partitioned_job_valid_meta_block_shapes(cluster: RampClusterEnvironment, job_max_partition_degree: int):
    '''
    Returns action_set, action_mask indicating which meta block shapes are valid
    for a partitioned job given its maximum partition degree.

    action_set is a numpy array of tuples of all meta block shapes

    action_mask is a bool mask indicating which meta block shapes are valid and which
    are not.
    '''
    # get shape of ramp topology
    ramp_shape = (cluster.topology.num_communication_groups, cluster.topology.num_racks_per_communication_group, cluster.topology.num_servers_per_rack)
    ramp_topology = dummy_ramp(ramp_shape, cluster)

    # find which meta block shape(s) valid for this job
    action_set, action_mask = [], []
    for c in range(1, cluster.topology.num_communication_groups+1):
        for r in range(1, cluster.topology.num_racks_per_communication_group+1):
            for s in range(1, cluster.topology.num_servers_per_rack+1):
                # store action
                action = (c, r, s)
                action_set.append(copy.deepcopy(action))

                # check if action is valid
                num_available_workers = cluster.topology.graph.graph['num_workers'] - len(cluster.mounted_workers)
                is_valid = check_meta_block_valid(c, r, s, ramp_topology, ramp_shape, job_max_partition_degree, num_available_workers)

                # store action validity
                action_mask.append(is_valid)

                if VERBOSE:
                    print(f'action {action} | c={c} r={r} s={s} -> {c * r * s} | job partition degree: {job_max_partition_degree} | num workers available: {cluster.topology.graph.graph["num_workers"] - len(cluster.mounted_workers)} | is_valid: {is_valid}')

    return np.array(action_set), np.array(action_mask).astype(bool)


def get_allocation_preamble(original_graph, mp_split_ids, mp_splits):
        
        parents, children = get_parents_and_children(original_graph)
        sequence = topo_sort(copy.deepcopy(parents), copy.deepcopy(children))
        op_server_info = {op: [] for op in original_graph.nodes()}
        splits = []
        for s in sequence:
            if s in mp_split_ids:
                idx = mp_split_ids.index(s)
                splits.append(mp_splits[idx])
                if VERBOSE:
                    print(f'node {s} in mp_split_ids')
            else:
                splits.append(1)
                if VERBOSE:
                    print(f'node {s} not in mp_split_ids')
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

# def find_meta_block(ramp_topology, ramp_shape, meta_block_shape, job_max_partition_degree):
def find_meta_block(ramp_topology, ramp_shape, meta_block_shape):
        '''
        Take an input of a RAMP topology and a required number of servers and 
        return a block of servers that is symmetric w.r.t. collective and where
        every server is currently empty.
        
        Returned values should be a set of server IDs, the shape and the 'origin'
        server in that set (i.e. the effective top left hand corner).
        
        meta_block_info = (meta_block, meta_block_shape, meta_block_origin) or None if empty.
        '''
        # meta_block_info = ff_meta_block([meta_block_shape], ramp_shape, ramp_topology, 'meta', job_max_partition_degree)
        meta_block_info = ff_meta_block([meta_block_shape], ramp_shape, ramp_topology, 'meta')
        return meta_block_info

# def ff_meta_block(block_shapes,ramp_shape,ramp,mode,job_max_partition_degree,op_size=None,meta_block_origin=(0,0,0)):
def ff_meta_block(block_shapes,ramp_shape,ramp,mode,op_size=None,meta_block_origin=(0,0,0)):
        '''
        For each block shape, create the block.

        When a block has been created, check it using
        check block to see if it's OK for allocation
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

                if VERBOSE:
                    print(f'Meta block shape valid. Checking for valid meta blocks...')

                #get the size of the shape in each RAMP dimension
                C,R,S = shape
                for i in range(ramp_shape[0]):
                    # j = i
                    for j in range(ramp_shape[1]):
                        for k in range(ramp_shape[2]):
                            block = get_meta_block(C,R,S,ramp_shape,origin=((orgn_c+i),(orgn_r+j),(orgn_s+k)))
                            if VERBOSE:
                                print(f'block: {block}')

                            # check block
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

def check_block(ramp,block,op_size,job_idx):
        '''
        Iterate through each server in a block.

        Return True if each server has enough resource
        to support the requirement, False otherwise
        '''
        if block == []:
            return False
        for server in block:
            # if ramp[server]['ops'] != []:
            # if ramp[server]['occupied'] or ramp[server]['mem'] < op_size:
            if len(ramp[server]['job_idxs']) != 0:
                if job_idx not in ramp[server]['job_idxs']:
                    # another job is already mounted on this server, cannot mount this job
                    return False
            if ramp[server]['mem'] < op_size:
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
                ramp[(i, j, k)] = {'mem': 0, 'ops': [], 'job_idxs': set()}
                for worker in cluster.topology.graph.nodes[node]['workers'].values():
                    ramp[(i, j, k)]['mem'] += (worker.memory_capacity - worker.memory_occupied)
                    # if len(list(worker.mounted_job_op_to_priority.keys())) != 0:
                    if len(list(worker.mounted_job_idx_to_ops.keys())) != 0:
                        # worker already occupied, update with job_idx occupying worker
                        ramp[(i, j, k)]['job_idxs'] = set(list(worker.mounted_job_idx_to_ops.keys()))
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
    backward_op_id = get_backward_op_id(op, num_nodes)
    
    #get sets of servers corresponding to each parent
    parents_servers = []
    for parent in parents[op]:
        if set(op_server_info[parent]).issubset(meta_block_info[0]):
            parents_servers.append(op_server_info[parent])

    #for each set of servers
    for servers in parents_servers:
        # if split < len(servers):
            # continue
        if split != len(servers):
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
                            # ramp[server]['ops'].append(str(int(op))+chr(97+i))
                            # ramp[server]['ops'].append(str((2*num_nodes)-(int(op)-1))+chr(97+i))
                            ramp[server]['ops'].append(get_partitioned_op_id(op, i))
                            ramp[server]['ops'].append(get_partitioned_op_id(backward_op_id, i))
                        else:
                            ramp[server]['ops'].append(op)
                            # ramp[server]['ops'].append((2*num_nodes)-(int(op)-1))
                            ramp[server]['ops'].append(backward_op_id)
                        op_server_info[op].append(server)
                        i += 1
                return ramp, op_server_info
    return None

def get_backward_op_id(forward_op_id, num_nodes):
    op_id = str((2*num_nodes)-(int(forward_op_id)-1))
    # try:
        # op_id = int(op_id)
    # except ValueError:
        # pass
    return op_id

def get_partitioned_op_id(op_id, split_id):
    op_id = str(int(op_id))+chr(97+split_id)
    # try:
        # op_id = int(op_id)
    # except ValueError:
        # pass
    return op_id


def regular_collective_placement(ramp,ramp_shape,job_graph,op,split,meta_block_info,op_server_info,job_idx):
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
    num_nodes = len(job_graph.nodes())
    meta_block,meta_block_shape,meta_block_origin = meta_block_info
    backward_op_id = get_backward_op_id(op, num_nodes)

    num_servers = split #NOTE: ops_per_server should never be more than splits. Needs to be ensured somewhere. EDIT: since putting multiple sub-ops on the same server is trivial (should then just partition it by a smaller amount) we will keep 1 op-per-server for now.
    if num_servers > len(meta_block): #if there are fewer servers in the meta-block than asked for, no allocation
        if VERBOSE:
            print(f'There are fewer servers in the meta block ({len(meta_block)}) than asked for ({num_servers}), cannot allocate.')
        return None

    op_size = job_graph.nodes[op]['memory_cost']/split
    meta_block = {server:ramp[server] for server in meta_block}

    block = find_sub_block(ramp,ramp_shape,meta_block_shape,meta_block_origin,num_servers,op_size,job_idx)

    if not block: #if no block can be found (memory errors) then return None
        if VERBOSE:
            print(f'No block found (memory errors), cannot allocate.')
        return None
    for j in range(len(block)):
        ramp[block[j]]['mem'] -= op_size
        if split > 1:
            # ramp[block[j]]['ops'].append(str(int(op))+chr(97+j))
            # ramp[block[j]]['ops'].append(str((2*num_nodes)-(int(op)-1))+chr(97+j))
            ramp[block[j]]['ops'].append(get_partitioned_op_id(op, j))
            ramp[block[j]]['ops'].append(get_partitioned_op_id(backward_op_id, j))
        else:
            ramp[block[j]]['ops'].append(op)
            # ramp[block[j]]['ops'].append((2*num_nodes)-(int(op)-1))
            ramp[block[j]]['ops'].append(backward_op_id)
        op_server_info[op].append(block[j])

    #if allocation was feasible, return the updated (i.e. with server-memory reduced) RAMP topology for further allocations
    return ramp, op_server_info

def find_sub_block(ramp_topology,ramp_shape,meta_block_shape,meta_block_origin,num_servers,op_size,job_idx):
        pairs = get_factor_pairs(num_servers)
        # pairs = get_factor_pairs(meta_block_shape[0]*meta_block_shape[1]*meta_block_shape[2])
        block_shapes = get_block_shapes(pairs,meta_block_shape)
        #if no possible shapes try rack and CG distributed
        block_shapes += [(num_servers,num_servers,-1),(num_servers,1,1)]
        block = ff_block(block_shapes,meta_block_shape,ramp_shape,ramp_topology,'sub',job_idx,op_size=op_size)
        return block

def ff_block(block_shapes,meta_shape,ramp_shape,ramp,mode,job_idx,op_size=None,meta_block_origin=(0,0,0)):
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
                            if VERBOSE:
                                print(f'sub block: {block}')
                            if check_block(ramp,block,op_size,job_idx):
                                return block

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
                # block.append(((i+n)%(ramp_shape[0]+1),(j+n)%(ramp_shape[1]+1),k))
                block.append(((i+n)%(ramp_shape[0]+1),(j+n)%(ramp_shape[1]+1),k%ramp_shape[2]))
        else:
            for c in range(C):
                for r in range(R):
                    for s in range(S):
                        block.append(((i+c)%(ramp_shape[0]),(j+r)%(ramp_shape[1]),(k+s)%(ramp_shape[2])))
        return block

def get_block_shapes(pairs,meta_block_shape):
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
        if VERBOSE:
            print(f'getting block shapes...')
            print(f'pairs: {pairs}')
            print(f'meta_block_shape: {meta_block_shape}')
        for pair in pairs:
            var = math.sqrt(pair[0])
            if VERBOSE:
                print(f'\tChecking pair {pair} with var {var}')
            if (var % 1 == 0) and (var <= meta_block_shape[0] and var <= meta_block_shape[1] and pair[1] <= meta_block_shape[2]):
                blocks.append((int(var), int(var), pair[1]))
                if VERBOSE:
                    print(f'Var is valid. Adding block {(int(var), int(var), pair[1])}')
            else:
                if VERBOSE:
                    if not (var % 1 == 0):
                        print(f'Pair is invalid since sqrt(pair[0]) % 1 != 0 ({math.sqrt(pair[0])} % 1 = {math.sqrt(pair[0]) % 1})')
                    else:
                        print(f'Pair is invalid since need var ({var}) <= meta_block_shape[0] ({meta_block_shape[0]}) and var ({var}) <= meta_block_shape[1] ({meta_block_shape[1]}) and pair[1] ({pair[1]}) <= meta_block_shape[2] ({meta_block_shape[2]})')
            if pair[0] > meta_block_shape[0] or pair[0] > meta_block_shape[1] or pair[1] > meta_block_shape[2]:
                if VERBOSE:
                    print(f'Pair is invalid since need pair[0] ({pair[0]}) < meta_block_shape[0] ({meta_block_shape[0]}) and pair[0] ({pair[0]}) < meta_block_shape[1] ({meta_block_shape[1]}) and pair[1] ({pair[1]}) < meta_block_shape[2] ({meta_block_shape[2]})')
                continue
            else:
                blocks.append((pair[0],1,pair[1]))
                blocks.append((pair[0],pair[1],1))
                if VERBOSE:
                    print(f'Pair is valid. Adding blocks {(pair[0],1,pair[1])} and {(pair[0],pair[1],1)}')
                
        return blocks

def allocate(ramp,ramp_shape,job_graph,sequence,splits,meta_block_info,parents,op_server_info,job_idx):
    # op_server_info = {op:[] for op in job_graph.nodes()}
    if VERBOSE:
        print(f'-------------------- Performing job ops allocation ------------------------')
        servers, _, _ = meta_block_info
        servers_occupied = [] 
        for server in servers:
            if len(ramp[server]['job_idxs']) != 0:
                servers_occupied.append(server)
        print(f'{len(servers_occupied)} of {len(servers)} servers occupied before allocation begun')
        print(f'Occupied servers: {servers_occupied}')

    for i in range(len(sequence)):
        op = sequence[i]
        split = splits[i]
        if VERBOSE:
            print(f'\top {op} split={split}')
        
        #try allocating on same servers as parents
        if VERBOSE:
            print(f'> Attempting parent collective allocation <')
        alloc = parent_collective_placement(ramp,job_graph,op,split,meta_block_info,parents,op_server_info)
        
        #if this doesn't work, try allocating somewhere else
        if not alloc:
            if VERBOSE:
                print('Parent collective allocation unsuccessful.')
                print('> Attempting regular collective allocation <')
            alloc = regular_collective_placement(ramp,ramp_shape,job_graph,op,split,meta_block_info,op_server_info,job_idx)
            
        #if that didn't work either, return None (this means the allocation has failed)
        if not alloc:
            if VERBOSE:
                print(f'Regular collective allocation unsuccessful.')
            return alloc
        
        #if either of the allocation attempts worked, update ramp and op_server_info and go onto the next op
        else:
            if VERBOSE:
                print(f'Collective allocation successful!')
            ramp, op_server_info = alloc[0], alloc[1]

    if VERBOSE:
        print(f'Final allocation op_server_info: {op_server_info}')
        allocated_workers = set()
        for op in op_server_info:
            for worker in op_server_info[op]:
                allocated_workers.add(worker)
        print(f'workers job allocated to ({len(allocated_workers)}): {allocated_workers}')
            
    return ramp, op_server_info
