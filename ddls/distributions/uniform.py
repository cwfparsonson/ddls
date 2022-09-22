from ddls.distributions.distribution import Distribution

from typing import Union
import numpy as np


class Uniform(Distribution):
    def __init__(self,
                 min_val: int,
                 max_val: int,
                 decimals: int = 10):
        self.min_val = min_val
        self.max_val = max_val
        if decimals > 0:
            # each subsequent random var value has min interval 0.<decimals>
            self.interval = 1 / (10**decimals)
        elif decimal < 0:
            # each subsequent random var value has min interval 10^(abs(decimals))
            self.interval = 10**(abs(decimals))
        else:
            # each subsequent random var value has min interval 1
            self.interval = 1
        self.decimals = decimals

        self.random_var_vals = np.around(np.arange(self.min_val, self.max_val+self.interval, self.interval), decimals=self.decimals)
        self.random_var_probs = np.ones(len(self.random_var_vals)) / len(self.random_var_vals)

    def sample(self,
               size: Union[None, int, tuple[int, ...]] = None,
               replace: bool = True):
        return np.random.choice(
                        self.random_var_vals, 
                        p=self.random_var_probs,
                        size=size, 
                        replace=replace
                        )

    def __str__(self):
        descr = f'Distribution type: {type(self)}'
        descr += f' | min_val: {self.min_val} | max_val: {self.max_val}'
        descr += f' | decimals: {self.decimals}'
        return descr
