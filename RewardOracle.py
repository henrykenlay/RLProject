import numpy as np
import gym 
import copy
from multiprocessing import Pool

class RewardOracle():
    
    def __init__(self, env):
        self.env = copy.deepcopy(env).unwrapped
        self.obs_ndims = len(env.observation_space.shape)
        
        
    def _reward(self, state, action):
        self.env.reset()
        state = np.array(state, dtype = 'float32')
        state = np.clip(state, self.env.observation_space.low, self.env.observation_space.high)
        self.env.state = state
        _, r, _, _ = self.env.step(action)
        return r
    
    def _batch_reward1(self, state, action):
        rewards = []
        for s, a in zip(state, action):
            rewards.append(self._reward(s, a))
        return rewards
    
    def _batch_reward2(self, state, action):
        return [self._reward(s, a) for s, a in zip(state, action)]
    
    def _batch_reward3(self, states, actions):
        # this only works for 1d statespaces...
        inputs = np.concatenate([states, actions], 1)
        outputs = np.apply_along_axis(self._vector_reward, 1, inputs)
        return outputs
    
    def _batch_reward4(self, states, actions):
        inputs = np.concatenate([states, actions], 1)
        with Pool() as p:
            rewards = p.map(self._vector_reward, inputs)
        return rewards
    
    def _vector_reward(self, x):
        obs_dim = self.env.observation_space.shape[0]
        return self._reward(x[:obs_dim], x[obs_dim:])
    
    def reward(self, state, action, repeats = 1):
        if state.ndim > self.obs_ndims:
            return self._batch_reward3(state, action)
        else:
            return self._reward(state, action)
        
if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    states = np.random.random((512, 2))
    actions = np.random.random((512, 1))
    oracle = RewardOracle(env)
    from time import time
    times = []
    for _ in range(100):
        start = time()
        rewards = oracle.reward(states, actions)
        times.append(time() - start)
    print(np.mean(times), np.std(times))