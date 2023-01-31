from ddls.utils import get_forward_graph
from ddls.environments.ramp_cluster.agents.placers.utils import get_backward_op_id, get_partitioned_op_id

import numpy as np
from typing import Union
from collections import defaultdict
import math





def update_dep_run_times(cluster,
                         op_partition, 
                         op_placement, 
                         verbose=False):
    '''Updates the run times of the partitioned jobs' dependencies given the op placements, dependency sizes, and the network's processor and link parameters.'''
    # verbose = True # DEBUG

    if verbose:
        print(f'\nUpdating job run times\nJobs: {op_placement.job_ids}\nOp placements: {op_placement}')
    if len(op_placement.job_ids) > 0:
        for original_job, partitioned_job in zip(op_partition.original_jobs.values(), op_partition.partitioned_jobs.values()):
            if verbose:
                print(f'\nUpdating job run times for job ID {original_job.job_id}...')
            # go through graph and group edge dependencies into collective and one-to-one communications
            collectives, one_to_one_deps = group_deps_into_collective_and_one_to_one_communications(original_job, partitioned_job, op_partition=op_partition, op_placement=op_placement, verbose=verbose)
            
            # set collective dependency run times
            for collective in collectives:
                set_collective_dep_run_time(partitioned_job, collective, op_placement, cluster, verbose=verbose)
                    
            # set one-to-one dependency run times
            for dep in one_to_one_deps:
                set_one_to_one_dep_run_time(partitioned_job, dep, op_placement, cluster, verbose=verbose)
    else:
        # none of the jobs could be placed
        if verbose:
            print(f'Did not find a placement for any job, cannot update dependency run times.')
        pass
            
def calc_ramp_all_reduce_collective_communication_run_time(
    message_size: Union[int, float],
    node_ids: int,
    racks: int,
    cgs: int,
    cont_racks: int = 1, # number of contending racks (number of nodes with the same node ID and communication group ID)
    # collective: str = "allreduce",
    x: int = 32, # number of communication groups in the whole network
    # λ: int = 64,
    # J: int = 32, # number of racks per communicaiton group
    DATA_RATE=1.6e12, # I/O GPU bandwidth (DATA_RATE / x = network link bandwidth) # BYTES
    # CLK=1455e6,
    MEM_FRQ=2e12, # device memory frequency
    latency=1.25e-6, # intra GPU internal data propagation latency through the network (1.25us assumes worst case for very large network, will be ~50 ns for <1024 nodes)
    π=130e12, # device peak computational power
    # comp=True,
    # circ_rec_time=0,
    bytes_per_comp=2,
    IO_latency=100e-9, # intra-GPU in-out latency for reading/writing info from/to memory
):
    data_per_tx = DATA_RATE / x
    subgroup_size = [cgs, min(cgs, node_ids), racks, np.ceil(node_ids / x)]
    effect_bw = [
        effective_trx_per_comm(cg=x, d=devices, J=cont_racks) * data_per_tx
        for devices in subgroup_size
    ]
    msg_size = [np.ceil(message_size / subgroup_size[0])]
    for i in subgroup_size[1:]:
        msg_size.append(np.ceil(msg_size[-1] / i))
    comm_time: float = 0.0
    comp_time: float = 0.0
    for step, sub in enumerate(subgroup_size):
        if sub > 1:
            comp_time += get_parallel_add_comp_time(
                msg_size[step] * sub,
                devices=sub,
                MEM_FRQ=MEM_FRQ,
                π=π,
                bytes_per_comp=bytes_per_comp,
            )
            comm_time += latency + 2 * IO_latency + msg_size[step] / effect_bw[step]
    total_time = 2 * comm_time + comp_time # x2 since all-reduce is made up of 2 collectives: reduce-scatter and all-gather

    if math.isinf(total_time):
        raise Exception(f'Infinite ramp all reduce collective dependency run time found.')

    return total_time # units of seconds

def calc_one_to_one_communication_run_time(message_size: Union[int, float],
                                           DATA_RATE=1.6e12,
                                           latency=1.25e-6,
                                           IO_latency=100e-9):
    run_time = latency + 2 * IO_latency + message_size / DATA_RATE

    if math.isinf(run_time):
        raise Exception(f'Infinite one-to-one dependency run time found.')

    return run_time

def effective_trx_per_comm(cg=32, d=32, J=1):
    if d ==1:
        return 0
    trx_per_comm = 1
    spare = min(cg//J, cg//(d-1))-1
    return trx_per_comm + spare

def get_parallel_add_comp_time_single(data_sz,
                                        devices=32,
                                        MEM_FRQ=2e12,
                                        π=130e12,
                                        bytes_per_comp=2):
    n_op = np.ceil(np.log2(devices))
    n_bytes = (devices + 1) * bytes_per_comp
    AI = n_op / n_bytes
    total_ops = n_op * (data_sz / devices) / bytes_per_comp
    return total_ops / np.min([MEM_FRQ * AI, π])

def get_parallel_add_comp_time(data_sz, devices, **kwargs):
    if not isinstance(devices, (list, np.ndarray)):
        return get_parallel_add_comp_time_single(data_sz, devices, **kwargs)
    return np.array(
        [get_parallel_add_comp_time_single(data_sz, dev, **kwargs) for dev in devices]
    )

def set_collective_dep_run_time(partitioned_job, collective, op_placement, cluster, verbose=False):
    # get collective info
    communication_groups, racks, nodes, servers, message_size, cont_racks = get_collective_info(partitioned_job, collective, op_placement, verbose=verbose)
    
    # calc collective run time
    if len(servers) == 1:
        # all dependencies of collective placed on same server, no communication overhead for the collective
        collective_run_time = 0
    else:
        collective_run_time = calc_ramp_all_reduce_collective_communication_run_time(message_size=message_size,
                                                                                      node_ids=len(nodes),
                                                                                      racks=len(racks),
                                                                                      cgs=len(communication_groups),
                                                                                      cont_racks=cont_racks,
                                                                                      x=cluster.topology.num_communication_groups,
                                                                                      DATA_RATE=cluster.topology.channel_bandwidth,
                                                                                      latency=cluster.topology.intra_gpu_propagation_latency,
                                                                                      IO_latency=cluster.topology.worker_io_latency)
    if verbose:
        print(f'collective_run_time: {collective_run_time}')
    for dep in collective:
        partitioned_job.set_dep_init_run_time(dep, collective_run_time)

def set_one_to_one_dep_run_time(partitioned_job, dep, op_placement, cluster, verbose=False):
    u, v, k = dep
    src_server, dst_server = op_placement.action[partitioned_job.job_id][u], op_placement.action[partitioned_job.job_id][v]
    if src_server == dst_server:
        # src == dst, no communication overhead
        dep_run_time = 0
    elif partitioned_job.computation_graph[u][v][k]['size'] == 0:
        # 0-sized tensor exchanged
        dep_run_time = 0
    else:
        dep_run_time = calc_one_to_one_communication_run_time(partitioned_job.computation_graph[u][v][k]['size'],
                                                              DATA_RATE=cluster.topology.channel_bandwidth,
                                                              latency=cluster.topology.intra_gpu_propagation_latency,
                                                              IO_latency=cluster.topology.worker_io_latency)
    if verbose:
        print(f'\none-to-one dep: {dep}')
        print(f'src server: {src_server} | dst server: {dst_server}')
        print(f'one-to-one dep run time: {dep_run_time}')
    partitioned_job.set_dep_init_run_time(dep, dep_run_time)

def get_collective_info(partitioned_job, collective, op_placement, verbose=False):
    job_id = partitioned_job.job_id

    # count number of communication groups, racks, and servers used by this collective, and the total message size
    communication_groups, racks, nodes, servers, message_size = set(), set(), set(), set(), 0
    # rack_to_src_cg_to_dst_cg_to_node_id = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    ids = set()
    for dep in collective:
        u, v, k = dep
        
        src_server = op_placement.action[job_id][u]
        src_communication_group = src_server.split('node_')[1].split('_worker')[0].split('-')[0]
        src_rack = src_cg = src_server.split('node_')[1].split('_worker')[0].split('-')[1]
        src_node = src_cg = src_server.split('node_')[1].split('_worker')[0].split('-')[2]
        communication_groups.add(src_communication_group)
        racks.add(src_rack)
        nodes.add(src_node)
        servers.add(src_server)
        ids.add((src_communication_group, src_rack, src_server))
        
        dst_server = op_placement.action[job_id][v]
        dst_communication_group = dst_server.split('node_')[1].split('_worker')[0].split('-')[0]
        dst_rack = src_cg = dst_server.split('node_')[1].split('_worker')[0].split('-')[1]
        dst_node = src_cg = dst_server.split('node_')[1].split('_worker')[0].split('-')[2]
        communication_groups.add(dst_communication_group)
        racks.add(dst_rack)
        nodes.add(dst_node)
        servers.add(dst_server)
        ids.add((dst_communication_group, dst_rack, dst_server))
        
        message_size += partitioned_job.computation_graph[u][v][k]['size']

        # rack_to_src_cg_to_dst_cg_to_node_id[src_rack][src_communication_group][dst_communication_group].add(src_node)
        # rack_to_src_cg_to_dst_cg_to_node_id[dst_rack][src_communication_group][dst_communication_group].add(dst_node)

    # count number of contending racks (number of nodes with the same node ID and communication group ID)
    # # OLD
    # cont_racks = 1
    # for src_communication_group in communication_groups:
        # for dst_communication_group in communication_groups:
            # for rack in racks:
                # # check if any other rack with this src-dst comm group uses same node ids assigned to this rack
                # for node_id in rack_to_src_cg_to_dst_cg_to_node_id[src_communication_group][dst_communication_group][rack]:
                    # for _rack in racks:
                        # if _rack != rack:
                            # if node_id in rack_to_src_cg_to_dst_cg_to_node_id[src_communication_group][dst_communication_group][_rack]:
                                # cont_racks += 1

    # # NEW
    # cont_racks = len(racks)

    # NEW NEW
    cont_racks, node_to_cg = 1, defaultdict(set)
    for _id in ids:
        c, r, s = _id
        if s in node_to_cg:
            # already used this node ID, check if used this comm group for this node ID
            if c in node_to_cg[s]:
                # node and comm group ID conflict, contending rack found
                cont_racks += 1
            else:
                node_to_cg[s].add(c)
        else:
            node_to_cg[s].add(c)
    
        
    if verbose:
        print(f'\ncollective: {collective}')
        print(f'servers communicating: {servers}')
        print(f'num_communication_groups: {len(communication_groups)}')
        print(f'num_racks: {len(racks)}')
        print(f'num_nodes: {len(nodes)}')
        print(f'message_size: {message_size}')
        print(f'node_to_cg: {node_to_cg}')
        print(f'cont_racks: {cont_racks}')

    return communication_groups, racks, nodes, servers, message_size, cont_racks

def group_deps_into_collective_and_one_to_one_communications(original_job, partitioned_job, op_partition, op_placement, verbose=False):
    job_id = original_job.job_id

    # get unpartitioned forward pass graph
    orig_forward_graph = get_forward_graph(original_job.computation_graph)

    # go through graph and group edge dependencies into collective and one-to-one communications
    collectives, collective_deps, one_to_one_deps = [], set(), set()

    # # debug
    # OPS_CONSIDERED = set()
    # PARTITIONED_OPS_CONSIDERED = set()

    for forward_op_id in orig_forward_graph.nodes():
        backward_op_id = get_backward_op_id(forward_op_id, len(list(orig_forward_graph.nodes())))

        # # debug
        # OPS_CONSIDERED.add(forward_op_id)
        # OPS_CONSIDERED.add(backward_op_id)

        if forward_op_id in op_partition.job_id_to_mp_split_forward_op_ids[job_id]:
            # op was partitioned, gather partitioned dependencies
            if verbose:
                print(f'\nforward op {forward_op_id} (backward op {backward_op_id}) was partitioned. Gathering partitioned deps...')
            partitioned_forward_deps, partitioned_backward_deps, partitioned_sync_deps = [], [], []
            sync_pairs_added = set()
            for split_id in range(op_partition.job_id_to_forward_op_id_to_mp_splits[job_id][forward_op_id]):
                partitioned_forward_op_id = get_partitioned_op_id(forward_op_id, split_id)
                for partitioned_dep in partitioned_job.computation_graph.out_edges(partitioned_forward_op_id):
                    partitioned_forward_deps.append((partitioned_dep[0], partitioned_dep[1], 0))
                partitioned_backward_op_id = get_partitioned_op_id(backward_op_id, split_id)
                for partitioned_dep in partitioned_job.computation_graph.in_edges(partitioned_backward_op_id):
                    parent_id, child_id = partitioned_dep[0], partitioned_dep[1]
                    if parent_id in partitioned_job.computation_graph.successors(child_id):
                        # bidirectional sync edge found
                        if (parent_id, child_id) not in sync_pairs_added and (child_id, parent_id) not in sync_pairs_added:
                            partitioned_sync_deps.append((parent_id, child_id, 0))
                            partitioned_sync_deps.append((child_id, parent_id, 0))
                            sync_pairs_added.add((parent_id, child_id))
                    else:
                        partitioned_backward_deps.append((partitioned_dep[0], partitioned_dep[1], 0))

                # # debug  
                # PARTITIONED_OPS_CONSIDERED.add(partitioned_forward_op_id)
                # PARTITIONED_OPS_CONSIDERED.add(partitioned_backward_op_id)

            if verbose:
                print(f'forward op {forward_op_id} partitioned forward deps: {partitioned_forward_deps}')
                print(f'backward op {backward_op_id} partitioned backward deps: {partitioned_backward_deps}')
                print(f'backward op {backward_op_id} partitioned sync deps: {partitioned_sync_deps}')


                
            # check if partitioned dependencies form a collective
            # collective type 1: edges' parent op servers == edges' child op servers
            # check forward op placements
            parent_servers, child_servers = [], []
            for partitioned_dep in partitioned_forward_deps:
                parent_id, child_id = partitioned_dep[0], partitioned_dep[1]
                parent_servers.append(op_placement.action[job_id][parent_id])
                child_servers.append(op_placement.action[job_id][child_id])
                # print(f'partitioned_dep: {partitioned_dep}')
                # print(f'parent_id: {parent_id}')
                # print(f'child_id: {child_id}')
                # print(f'parent server: {op_placement.action[job_id][parent_id]}')
                # print(f'child server: {op_placement.action[job_id][child_id]}')
            if sorted(parent_servers) == sorted(child_servers):
                # symmetric parent-child placements are a collective
                if verbose:
                    print(f'forward edges {partitioned_forward_deps} are a parent-child collective')
                collectives.extend([partitioned_forward_deps])
                for dep in partitioned_forward_deps:
                    collective_deps.add(dep)
            else:
                # not a collective
                for dep in partitioned_forward_deps:
                    one_to_one_deps.add(dep)
                
            # check backward op placements
            parent_servers, child_servers = [], []
            for partitioned_dep in partitioned_backward_deps:
                parent_id, child_id = partitioned_dep[0], partitioned_dep[1]
                parent_servers.append(op_placement.action[job_id][parent_id])
                child_servers.append(op_placement.action[job_id][child_id])
            if sorted(parent_servers) == sorted(child_servers):
                # symmetric parent-child placements are a collective
                if verbose:
                    print(f'backward edges {partitioned_backward_deps} are a parent-child collective')
                collectives.extend([partitioned_backward_deps])
                for dep in partitioned_backward_deps:
                    collective_deps.add(dep)
            else:
                # not a collective
                for dep in partitioned_backward_deps:
                    one_to_one_deps.add(dep)
                
            # collective type 2: sync in backward pass
            # check if any edges are sync edges and therefore form collectives
            for idx in range(0, len(partitioned_sync_deps), 2):
                partitioned_dep = partitioned_sync_deps[idx]
                parent_id, child_id = partitioned_dep[0], partitioned_dep[1]
                if verbose:
                    print(f'sync edges {[(parent_id, child_id, 0), (child_id, parent_id, 0)]} are a sync collective')
                collectives.append([(parent_id, child_id, 0), (child_id, parent_id, 0)])
                collective_deps.add((parent_id, child_id, 0))
                collective_deps.add((child_id, parent_id, 0))
        else:
            # op was not partitioned, gather one-to-one communication dependencies of the op
            if verbose:
                print(f'\nforward op {forward_op_id} (backward op {backward_op_id}) was not partitioned. Gathering deps...')
            for dep in partitioned_job.computation_graph.out_edges(str(forward_op_id)):
                one_to_one_deps.add((dep[0], dep[1], 0))
                if verbose:
                    print(f'forward op {forward_op_id} forward deps: {(dep[0], dep[1], 0)}')
            for dep in partitioned_job.computation_graph.in_edges(str(backward_op_id)):
                one_to_one_deps.add((dep[0], dep[1], 0))
                if verbose:
                    print(f'forward op {forward_op_id} backward deps: {(dep[0], dep[1], 0)}')

            # # debug
            # PARTITIONED_OPS_CONSIDERED.add(str(forward_op_id))
            # PARTITIONED_OPS_CONSIDERED.add(str(backward_op_id))
                
    # # debug
    # print(f'\noriginal job ops considered: {len(OPS_CONSIDERED)} {OPS_CONSIDERED}')
    # print(f'num ops in original graph: {len(list(original_job.computation_graph.nodes()))}')
    # print(f'partitioned job ops considered: {len(PARTITIONED_OPS_CONSIDERED)} {PARTITIONED_OPS_CONSIDERED}')
    # print(f'num ops in parititoned graph: {len(list(partitioned_job.computation_graph.nodes()))}')
    # num_deps_missing = 0
    # for partitioned_job_op in PARTITIONED_OPS_CONSIDERED:
        # partitioned_job_op_edges = partitioned_job.computation_graph.edges(partitioned_job_op)
        # for _edge in partitioned_job_op_edges:
            # edge = (_edge[0], _edge[1], 0)
            # if edge not in collective_deps and edge not in one_to_one_deps:
                # print(f'edge {edge} of op {partitioned_job_op} was not found in collective or one to one deps')
                # num_deps_missing += 1
    # print(f'num_deps_missing: {num_deps_missing}')

    if verbose:
        print(f'\ncollectives: {collectives}')
        print(f'\ncollective deps: {collective_deps}')
        print(f'\none-to-one deps: {one_to_one_deps}')

    if len(list(partitioned_job.computation_graph.edges())) != len(collective_deps) + len(one_to_one_deps):
        raise Exception(f'ERROR: Partitioned graph contains {len(list(partitioned_job.computation_graph.edges()))} edges, but found {len(collective_deps)} collective dependencies and {len(one_to_one_deps)} one-to-one dependencies. A bug has occurred somewhere.')

    return collectives, one_to_one_deps

