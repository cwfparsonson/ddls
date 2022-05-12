
def check_if_ramp_op_placement_rules_broken(worker, job):
    '''Checks whether an op placement obeys the rules of Ramp.'''
    rules_broken = []

    # Ramp Rule 1: No worker can have ops from more than one job.
    if job.details['job_idx'] not in worker.mounted_job_idx_to_ops:
        # not yet mounted an op from this job onto the worker
        if len(worker.mounted_job_idx_to_ops.keys()) > 0:
            # already have another job mounted on this worker
            rules_broken.append('one_job_per_worker')
    else:
        # already begun mounting deps from this job onto this worker so must be valid
        pass

    return rules_broken

def check_if_ramp_dep_placement_rules_broken(channel, job):
    '''
    Args:
        channel: Channel onto which dependency is about to be mounted.
        job: Job which dependency is part of.
    '''
    rules_broken = []

    # Ramp Rule 1: No channel can have flows from more than one job
    if job.details['job_idx'] not in channel.mounted_job_idx_to_deps:
        # not yet mounted a flow from this job onto the channel
        if len(list(channel.mounted_job_idx_to_deps.keys())) > 0:
            # already have another job's flow mounted on this channel
            rules_broken.append('one_job_per_channel')
    else:
        # already begun mounting deps from this job onto this channel so must be valid
        pass

    return rules_broken
