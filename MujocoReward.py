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
def reward_dask(envname, state, action):
    env = envs[envname]
    state = np.clip(state, env.observation_space.low, env.observation_space.high)
    env.state = state
    _, r, _, _ = env.step(action)
    return r    
    
def reward(envname, state, action):
    env = envs[envname]
    state = np.clip(state, env.observation_space.low, env.observation_space.high)
    env.state = state
    _, r, _, _ = env.step(action)
    return r

def batch_reward(states, actions, envname, parallel = False):
    if parallel:
        rewards = [reward_dask(envname, state,action) for (state, action) in zip(states, actions)]
        rewards = dask.compute(rewards)
    else:
        rewards = [reward(envname, state,action) for (state, action) in zip(states, actions)]
    return rewards
 
def get_reward_function(envname, parallel = False):
    return partial(batch_reward, envname=envname, parallel = parallel)

for env in envs:
    print(envs[env].observation_space.shape)

# time comparisons
if __name__ == '__main__':
    from time import time
    from RewardOracle import RewardOracle
    state = np.random.random((512, 17))
    action = np.random.random((512, 6))
    n = 30
    
    # using batch_reward function
    times = []
    for _ in range(n):
        start = time()
        batch_reward(state, action, 'HalfCheetah-v2')
        times.append(time() - start)
    print(np.mean(times), np.std(times))
    
    # using partial version
    cheetah_reward = get_reward_function('HalfCheetah-v2')
    times = []
    for _ in range(n):
        start = time()
        temp = cheetah_reward(state, action)
        times.append(time() - start)
    print(np.mean(times), np.std(times))
    
    # using parallel partial version
    cheetah_reward = get_reward_function('HalfCheetah-v2', parallel = True)
    times = []
    for _ in range(n):
        start = time()
        temp = cheetah_reward(state, action)
        times.append(time() - start)
    print(np.mean(times), np.std(times))
    
    # using reward oracle
    rewardoracle = RewardOracle(envs['HalfCheetah-v2'])
    times = []
    for _ in range(n):
        start = time()
        rewardoracle.reward(state, action)
        times.append(time() - start)
    print(np.mean(times), np.std(times))
    