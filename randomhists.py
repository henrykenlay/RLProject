import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import norm


def make_plot(env_name, episode_length = 1000, repeats = 1000):
    plt.clf()
    env = gym.make(env_name)
    total_reward = []
    for i in tqdm(range(repeats)):
        episode_reward = 0
        state = env.reset()
        for j in range(episode_length):
            state, r, _, _ = env.step(env.action_space.sample())
            episode_reward += r
        total_reward.append(episode_reward)
    plt.hist(total_reward, 50, density=True)
    mu, std = norm.fit(total_reward)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "{}: mu = {:.3},  std = {:.3}".format(env_name, mu, std)
    plt.title(title)
    plt.savefig('figures/rand-hist-{}.pdf'.format(env_name), bbox_inches='tight')
    
env_names = ['Swimmer-v2', 'HalfCheetah-v2']
for env_name in env_names:
    make_plot(env_name)