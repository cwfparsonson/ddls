from ddls.distributions.distribution import Distribution

from typing import Union
import numpy as np


class Uniform(Distribution):
    def __init__(self,
                 min_val: int,
                 max_val: int,
                 interval: Union[int, float] = 1,
                 decimals: int = 10):
        self.min_val = min_val
        self.max_val = max_val
        self.interval = interval
        self.decimals = decimals

    def sample(self,
               size: Union[None, int, tuple[int, ...]] = None,
               replace: bool = True):
        return np.random.choice(
                        np.around(
                            np.arange(self.min_val, self.max_val+self.interval, self.interval), 
                            decimals=self.decimals), 
                        size=size, replace=replace)

    def __str__(self):
        descr = f'Distribution type: {type(self)}'
        descr += f' | min_val: {self.min_val} | max_val: {self.max_val}'
        descr += f' | interval: {self.interval} | decimals: {self.decimals}'
        return descr
