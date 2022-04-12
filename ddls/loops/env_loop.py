import time
from collections import defaultdict

class EnvLoop:
    def __init__(self,
                 env,
                 actor):
        self.env = env
        self.actor = actor

    def run(self):
        '''Run one episode.'''
        start_time = time.time()

        results = {}
        results['episode_stats'] = self._init_episode_stats()
        obs = self.env.reset()
        done = False
        while not done:
            # get action from actor
            action = self.actor.select_action(obs)

            # perform action in environment
            prev_obs = obs # save
            obs, reward, done, info = self.env.step(action)
            print(f'\nStep {results["episode_stats"]["num_actor_steps"]}\nObs: {prev_obs}\nAction: {action}\nReward: {reward}\nDone: {done}\nInfo: {info}')

            # step finished, update trackers
            results['episode_stats']['num_actor_steps'] += 1
            results['episode_stats']['return'] += reward

        # episode finished, update trackers
        results['episode_stats']['run_time'] = time.time() - start_time

        return results

    def _init_episode_stats(self):
        return defaultdict(lambda: 0)
