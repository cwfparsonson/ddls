from ddls.utils import get_class_from_path, seed_stochastic_modules_globally, recursively_update_nested_dict

from collections import defaultdict
import numpy as np

class RLlibEvalLoop:
    def __init__(self,
                 path_to_rllib_trainer_cls: str,
                 path_to_env_cls: str,
                 rllib_config: dict,
                 **kwargs):
        self.rllib_config = rllib_config
        self.actor = get_class_from_path(path_to_rllib_trainer_cls)(config=self.rllib_config)
        self.env = get_class_from_path(path_to_env_cls)(**self.rllib_config['env_config'])

    def run(self,
            checkpoint_path: str):
        results = {'step_stats': defaultdict(list), 'episode_stats': {}}

        if 'seed' in self.rllib_config:
            if self.rllib_config['seed'] is not None:
                seed_stochastic_modules_globally(self.rllib_config['seed'])

        self.actor.restore(checkpoint_path)

        obs, done = self.env.reset(), False
        while not done:
            action = self.actor.compute_action(obs) 
            obs, reward, done, info = self.env.step(action)

            results['step_stats']['action'].append(action)
            results['step_stats']['reward'].append(reward)
            for key, val in self.env.cluster.step_stats.items():
                results['step_stats'][key].append(val)

        for key, val in self.env.cluster.episode_stats.items():
            results['episode_stats'][key] = np.mean(list(val))
        # print(f'\nstep stats: {results["step_stats"]}')
        # print(f'\nepisode stats: {results["episode_stats"]}\n')

        return results

