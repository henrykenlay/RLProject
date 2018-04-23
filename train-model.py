from Data import Data, get_random_data
from Agent import Agent
from tqdm import tqdm
import gym
import argparse

parser = argparse.ArgumentParser(description='Train a model using MPC')
parser.add_argument('--environment', default='MountainCarContinuous-v0',
                   help='Environment to train the agent on')
parser.add_argument('--model-suffix', default='',
                   help='Will save the model as [envname]-[suffix] if given')
parser.add_argument("--no-render", default=False, action="store_true" , help="Flag to do something")
args = parser.parse_args()

env = gym.make(args.environment) 

# Make D_rand of random trajectories
D_rand = get_random_data(env)
D_RL = Data(capacity = len(D_rand)*9)

# Create model
agent = Agent(env, softmax=True)

# Main loops
state = env.reset()
for iteration in range(500):
    agent.train(D_rand + D_RL)
    for t in tqdm(range(250)):  
        action = agent.choose_action(state)
        new_state, reward, done, info = env.step(action)
        D_RL.pushTrajectory([state, action, new_state])
        state = new_state
        if not args.no_render:
            env.render()
        if done:
            env.reset()            
    
    print(len(D_rand), len(D_RL))
    
if args.model_suffix == '':
    model_name = args.environment
else:
    model_name = '{}-{}'.format(args.environment, args.model_suffix)