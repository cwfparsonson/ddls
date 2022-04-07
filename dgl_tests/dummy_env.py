import numpy as np
import gym
from gym.spaces import Discrete, Box, Dict


class DummyEnv(gym.Env):

    def __init__(self, env_config):

        super(DummyEnv, self).__init__()

        self.observation_space = Dict({
            "obs_0":Box(-1,1,shape=(5,)),
            "obs_1":Discrete(10)
        })

        self.action_space = Discrete(8)


        self.dummy_obs = {
            "obs_0":self.observation_space['obs_0'].sample(),
            "obs_1":self.observation_space['obs_1'].sample()
        }
    
    def reset(self):
        
        return self.dummy_obs

    def step(self,action):
        
        return self.dummy_obs, 1, False, {}

if __name__ == '__main__':


    env_config = {
                'obs_0':{
                    'upper':-1,
                    'lower':1,
                    'dim':5
                },
                'obs_1':{
                    'dim':10
                }                        
    }

    env = DummyEnv(env_config)
