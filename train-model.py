from Data import Data, get_random_data
from Agent import Agent
from tqdm import tqdm
import gym

env = gym.make('MountainCarContinuous-v0') 

# Make D_rand of random trajectories
D_rand = get_random_data(env)
D_RL = Data(capacity = len(D_rand)*9)

# Create model
agent = Agent(env)

# Main loops
state = env.reset()
for iteration in range(500):
    agent.train(D_rand + D_RL)
    for t in tqdm(range(250)):  
        action = agent.choose_action(state)
        new_state, reward, done, info = env.step(action)
        D_RL.pushTrajectory([state, action, new_state])
        state = new_state
        env.render()
        if done:
            env.reset()            
    
    print(len(D_rand), len(D_RL))