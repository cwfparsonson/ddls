import time
from collections import defaultdict

class EpochLoop:
    def __init__(self,
                 env_loop):
        self.env_loop = env_loop

    def run(self, batch_size: int = 1):
        '''Run one epoch.'''
        start_time = time.time()

        results = {}
        results['epoch_stats'] = self._init_epoch_stats()
        results['episode_stats'] = self._init_episode_stats()
        for batch in range(batch_size):
            _episode_stats = self.env_loop.run_episode()['episode_stats']

            # batched episode finished, update trackers
            results['episode_stats']['batch'].append(batch)
            for key, val in _episode_stats.items():
                if isinstance(val, list):
                    results['episode_stats'][key].extend(val)
                else:
                    results['episode_stats'][key].append(val)

        # epoch finished, update trackers
        for key, val in results['episode_stats'].items():
            if key not in {'batch'}:
                results['epoch_stats'][f'mean_{key}'] = np.mean(val)
        results['epoch_stats']['run_time'] = time.time() - start_time

        return results
        
    def _init_epoch_stats(self):
        return defaultdict(lambda: 0)

    def _init_episode_stats(self):
        return defaultdict(lambda: [])
