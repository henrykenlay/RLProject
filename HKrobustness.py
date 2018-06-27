from Agent import Agent
from tqdm import tqdm
from dm_control import suite
import argparse
import numpy as np
import json

parser = argparse.ArgumentParser(description='Finetune a model trained using MPC with reinforce')
parser.add_argument('--environment', default='cheetah-run', help='Environment to train the agent on')
parser.add_argument('--experiment-name', default=None, help='Will save the model as [envname]-[suffix] if given')
parser.add_argument("--softmax", default=False, action="store_true",  help="Use softmax MPC")
parser.add_argument("--temperature", default=1, type=int,  help="Softmax temp")
parser.add_argument("--traj-length", default=1000, type=int, help="Length of rollouts")
parser.add_argument("--num-epochs", default=60, type=int, help="Number of epochs to train model")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--reinforce-lr", default=10e-8, type=float, help="Learning rate for REINFORCE")
parser.add_argument("--H", default=20, type=int, help="Horizon of the MPC controller")
parser.add_argument("--K", default=1000, type=int, help="Number of random rollouts of the MPC controller")
parser.add_argument("--hidden-units", default=500, type=int, help="Number of neurons in hidden layers")
parser.add_argument("--predict-reward", default=True, action="store_true", help="Use model to predict reward")
args = parser.parse_args()

assert args.experiment_name is not None

env = suite.load(*args.environment.split('-')) 

agent = Agent(env = env, H = args.H, K = args.K, traj_length = args.traj_length, 
              softmax = args.softmax, predict_rewards = args.predict_reward, 
              reinforce = False, lr = args.lr, 
              temperature = args.temperature, reinforce_lr = args.reinforce_lr,
              hidden_units = args.hidden_units)

agent.loadbest(args.experiment_name)

def testHK(agent, H, K, n=10):
    agent.H = H
    agent.K = K
    total_rewards = []
    for iteration in range(n):
        timestep = env.reset()
        rewards = []
        for t in tqdm(range(args.traj_length), desc='Generating episode'):
            state = env.physics.state()
            action = agent.choose_action(state)
            timestep = env.step(action)
            reward = timestep.reward
            rewards.append(reward)
        print('Trajectory done. Total reward: {}'.format(sum(rewards)))
        total_rewards.append(sum(rewards))
    return np.mean(total_rewards)

data = {}
for H in [1,2,4,8,16,32]:
    for K in [125,250,500,1000,2000]:
        key = '{}-{}'.format(H,K)
        data[key] = testHK(agent, H, K)
        print(data)
        
with open('project/results/HKrobust.json', 'w') as outfile:
    json.dump(data, outfile)
