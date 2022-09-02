from ddls.distributions.distribution import Distribution

from typing import Union
import numpy as np


class ProbabilityMassFunction(Distribution):
    def __init__(self,
                 probability_mass_function: dict):
        '''
        Args:
            probability_mass_function: Mapping of random variable values to their
                corresponding probability of being sampled.
                E.g. probability_mass_function = {5: 1} means 100% probability
                of sampling the value 5.
        '''
        self.probability_mass_function = probability_mass_function

        self.random_var_values = np.array(list(self.probability_mass_function.keys()))
        self.random_var_probs = np.array(list(self.probability_mass_function.values()))

        if np.sum(self.random_var_probs) != 1:
            raise Exception(f'probability_mass_function must sum to 1, but sums to {np.sum(self.random_var_probs)}')

    def sample(self,
               size: Union[None, int, tuple[int, ...]] = None,
               replace: bool = True):
        return np.random.choice(
                        self.random_var_values,
                        p=self.random_var_probs,
                        size=size,
                        replace=replace
                )

    def __str__(self):
        descr = f'Distribution type: {type(self)}'
        descr += f' | PMF: {self.probability_mass_function}'
        return descr
