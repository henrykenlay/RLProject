import numpy as np
from dm_control import suite
import copy

class RewardOracle():
    
    def __init__(self, env):
        self.env = copy.deepcopy(env)  
        
    def _reward(self, state, action):
        self.env.reset()
        state = np.array(state, dtype = 'float32')
        with self.env.physics.reset_context():
            self.env.physics.set_state(state)
        timestep = self.env.step(action)
        return timestep.reward
    
    def _batch_reward(self, state, action):
        return [self._reward(s, a) for s, a in zip(state, action)]
    
    def reward(self, state, action, repeats = 1):
        if state.ndim > 1:
            return self._batch_reward(state, action)
        else:
            return self._reward(state, action)
        
if __name__ == '__main__':
    env = suite.load('cheetah', 'run')
    states = np.random.random((32, 18))
    actions = np.random.random((32, 6))
    oracle = RewardOracle(env)
    from time import time
    times = []
    for _ in range(5):
        start = time()
        rewards = oracle.reward(states, actions)
        print(rewards[0:5])
        times.append(time() - start)
    print(np.mean(times), np.std(times))