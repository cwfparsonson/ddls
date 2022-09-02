import math

class AcceptableJCT:
    def __init__(self, name: str = 'acceptable_jct', **kwargs):
        '''
        Partitions the job according to get the completion time as close to 
        the maximum acceptable job completion time as possible.
        
        E.g. If a job has a sequential job completion time of 10 with a maximum
        acceptable job completion time of 5, then AcceptableJCT will partition the
        job job_sequential_completion_time / max_acceptable_job_completion_time = 2
        times. This is approximately the number of times the job needs to be partitioned
        in order to satisfy its maximum acceptable job completion time requirement,
        but it does NOT account for communication time overheads or the additional
        read/write overheads to/from GPUs incurred by more partitioning. I.e. it is
        an approximation, but not perfect.
        '''
        self.name = name

    def compute_action(self, obs, job_to_place, *args, **kwargs):
        valid_actions = obs['action_set'][obs['action_mask'].astype(bool)]
        if len(valid_actions) > 1:
            # partition up to max_partitions_per_op (env will partition less if min quantum is different)
            max_allowed_partitions_per_op = valid_actions[-1]

            # get approximate number of times job must be partitioned to satisfy maximum acceptable job completion time constraint
            device_type = list(job_to_place.details['job_sequential_completion_time'].keys())[0] # TODO HACK assume one device type
            acceptable_partitions = int(math.ceil(job_to_place.details['job_sequential_completion_time'][device_type] / job_to_place.details['max_acceptable_job_completion_time'][device_type]))

            # get the partition degree which is valid and most closely matches this target acceptable_partitions
            for action in valid_actions:
                if action == acceptable_partitions:
                    # target number of acceptable partitions is available, use
                    break
                elif action > acceptable_partitions:
                    # this is the valid action which most closley matches the target acceptable_partitions
                    break
                else:
                    # keep looking for a valid action until reach the last available valid action (i.e. the maximum allowed number of partitions per op); if reach this, then have to use
                    pass
        else:
            # only action = 0 (do not place job) is available
            action = valid_actions[0]
        return action
