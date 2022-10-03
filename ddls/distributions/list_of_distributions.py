from ddls.distributions.distribution import Distribution
from ddls.utils import get_class_from_path

from typing import Union
import numpy as np
import random


class ListOfDistributions(Distribution):
    def __init__(self,
                 name_to_cls_to_kwargs: dict,
                 ):
        '''
        When call sample, will randomly sample a distribution and instantiate it.
        This can be useful during RL training where you may wish to sample different
        distributions for various parameters when generating an environment to
        help the RL agent to learn to generalise.

        Args:
            name_to_cls_to_kwargs: Dict mapping an arbitrary distribution name (can be anything)
                to a string path to a ddls.distribution.Distribution class to
                its corresponding kwargs.
                E.g. name_to_cls_to_kwargs = {
                                             'dist_1':
                                                'ddls.distributions.uniform.Uniform':
                                                    'min_val: 1',
                                                    'max_val: 10'
                                             'dist_2':
                                                'ddls.distributions.uniform.Uniform':
                                                    'min_val: 5',
                                                    'max_val: 50'
                                            }
        '''
        self.dist_names = list(name_to_cls_to_kwargs.keys())
        self.name_to_cls_to_kwargs = name_to_cls_to_kwargs 

    def sample(self):
        dist_name = random.choice(self.dist_names)
        path_to_cls = list(self.name_to_cls_to_kwargs[dist_name].keys())[0]
        kwargs = self.name_to_cls_to_kwargs[dist_name][path_to_cls]
        return get_class_from_path(path_to_cls)(**kwargs)

    def __str__(self):
        descr = f'Distribution type: {type(self)}'
        descr += f' | name_to_cls_to_kwargs: {self.name_to_cls_to_kwargs}'
        return descr
