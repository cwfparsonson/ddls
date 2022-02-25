from ddls.distributions.distribution import Distribution


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
               size: Union[int, tuple[int, ...]] = 1,
               replace: bool = True):
        return np.random.choice(
                        np.around(
                            np.arange(self.min_val, self.max_val, self.interval), 
                            decimals=self.decimals), 
                        size=size, replace=replace)
