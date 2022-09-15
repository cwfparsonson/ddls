from ddls.distributions.distribution import Distribution

from typing import Union
import numpy as np


class Fixed(Distribution):
    def __init__(self,
                 val: Union[int, float]):
        self.val = val

    def sample(self,
               size: Union[None, int, tuple[int, ...]] = None,
               replace: bool = True):
        return self.val

    def __str__(self):
        descr = f'Distribution type: {type(self)}'
        descr += f' | val: {self.val}'
        return descr
