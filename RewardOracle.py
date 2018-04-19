import numpy as np
import gym 
import copy

class RewardOracle():
    
    def __init__(self, env):
        self.env = copy.deepcopy(env).unwrapped
        
    def _reward(self, state, action):
        self.env.reset()
        state = np.array(state, dtype = 'float32')
        state = np.clip(state, self.env.observation_space.low, self.env.observation_space.high)
        self.env.state = state
        _, r, _, _ = self.env.step(action)
        return r
    
    def reward(self, state, action, repeats = 1):
        """
        Returns r(s, a)
            
        If the reward is stochastic then repeats can be set to a higher value 
         and the reward will be estimated
        
        TODO: add warning if state/action outside of allowed env ranges
        """
        return np.mean([self._reward(state, action) for _ in range(repeats)])
        
if __name__ == '__main__':
    env = gym.make('NChain-v0')
    observation = env.reset()
    oracle = RewardOracle(env)
    print(oracle.reward(4, 1, 100))