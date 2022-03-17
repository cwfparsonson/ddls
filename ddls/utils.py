from tkinter import W
import numpy as np
import random
import copy

def seed_stochastic_modules_globally(default_seed=0, 
                                     numpy_seed=None, 
                                     random_seed=None):
    '''Seeds any stochastic modules so get reproducible results.'''
    if numpy_seed is None:
        numpy_seed = default_seed
    if random_seed is None:
        random_seed = default_seed
    
    np.random.seed(numpy_seed)
    random.seed(random_seed)

class Sampler:
    def __init__(self, 
                 pool: list,
                 sampling_mode: str):
        '''
        Args:
            sampling_mode ('replace', 'remove', 'remove_and_repeat')
        '''
        self.original_pool = pool
        self.sample_pool = copy.deepcopy(self.original_pool)
        self.sampling_mode = sampling_mode
    
    def sample(self):
        idx = np.random.randint(low=0, high=len(self.sample_pool))
        datum = self.sample_pool[idx]
        
        if self.sampling_mode == 'replace':
            pass
        elif self.sampling_mode == 'remove':
            self.sample_pool.pop(idx)
        elif self.sampling_mode == 'remove_and_repeat':
            self.sample_pool.pop(idx)
            if len(self.sample_pool) == 0:
                self.sample_pool = copy.deepcopy(self.original_pool)
            
        return datum