from Agent import MPCAgent
from tqdm import tqdm
import gym
import os
import sys
import argparse
import pickle

parser = argparse.ArgumentParser(description='Train a model using MPC')
parser.add_argument('--environment', default='MountainCarContinuous-v0', help='Environment to train the agent on')
parser.add_argument('--experiment-name', default='', help='Will save the model as [envname]-[suffix] if given')
parser.add_argument("--render", default=False, action="store_true",  help="Render environment")
parser.add_argument("--softmax", default=False, action="store_true",  help="Use softmax MPC")
parser.add_argument("--agg-iters", default=9, type=int, help="Aggregation iterations")
parser.add_argument("--traj-per-agg", default=9, type=int, help="Number of rollouts per aggregation iterations")
parser.add_argument("--traj-length", default=1000, type=int, help="Length of rollouts")
parser.add_argument("--H", default=20, type=int, help="Horizon of the MPC controller")
parser.add_argument("--K", default=250, type=int, help="Number of random rollouts of the MPC controller")
args = parser.parse_args()

env = gym.make(args.environment) 

# Create model
agent = MPCAgent(env = env, H = args.H, K = args.K, traj_length = args.traj_length, softmax = args.softmax)

# Main loops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
assets_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'))

for iteration in range(args.agg_iters):
    agent.train(num_epochs = 60)
    if not args.experiment_name == '':
        pickle.dump((agent.D_RL, agent.D_rand, agent.model), open(os.path.join(assets_dir, 'learned_models/{}-model.p'.format(args.environment)), 'wb'))
    state = env.reset()
    for _ in range(args.traj_per_agg):
        total_reward = 0
        for t in tqdm(range(args.traj_length)):  
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            agent.D_RL.pushTrajectory([state, action, new_state])
            state = new_state
            if args.render:
                env.render()
            if done:
                state = env.reset()  
                print('Episode done. Total reward: {}'.format(total_reward))
                break
            total_reward += reward
        print('Trajectory done. Total reward: {}'.format(total_reward))



