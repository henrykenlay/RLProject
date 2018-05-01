import gym
import numpy as np
import dask
from functools import partial

# Make environments
envs = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Swimmer-v2']
envs = {env : gym.make(env) for env in envs}

# reset env
for env in envs:
    envs[env].reset()
    
# reward functions
@dask.delayed
def reward(envname, state, action):
    env = envs[envname]
    state = np.clip(state, env.observation_space.low, env.observation_space.high)
    env.state = state
    _, r, _, _ = env.step(action)
    return r

def batch_reward(states, actions, envname):
    return [reward(envname, state,action) for (state, action) in zip(states, actions)]
 
def get_reward_function(envname):
    return partial(batch_reward, envname=envname)

# time comparisons
if __name__ == '__main__':
    from time import time
    from RewardOracle import RewardOracle
    state = np.random.random((512, 17))
    action = np.random.random((512, 6))
    n = 30
    
    times = []
    for _ in range(n):
        start = time()
        batch_reward(state, action, 'HalfCheetah-v2')
        times.append(time() - start)
    print(np.mean(times), np.std(times))
    
    cheetah_reward = get_reward_function('HalfCheetah-v2')
    times = []
    for _ in range(n):
        start = time()
        cheetah_reward(state, action)
        times.append(time() - start)
    print(np.mean(times), np.std(times))
    
    
    rewardoracle = RewardOracle(envs['HalfCheetah-v2'])
    times = []
    for _ in range(n):
        start = time()
        rewardoracle.reward(state, action)
        times.append(time() - start)
    print(np.mean(times), np.std(times))
    