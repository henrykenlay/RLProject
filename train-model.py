from Agent import MPCAgent, RandomAgent
from tqdm import tqdm
import gym
import os
import sys
import argparse
import pickle
from tensorboardX import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser(description='Train a model using MPC')
parser.add_argument('--environment', default='HalfCheetah-v2', help='Environment to train the agent on')
parser.add_argument('--experiment-name', default='', help='Will save the model as [envname]-[suffix] if given')
parser.add_argument("--render", default=False, action="store_true",  help="Render environment")
parser.add_argument("--softmax", default=True, action="store_true",  help="Use softmax MPC")
parser.add_argument("--agg-iters", default=1000, type=int, help="Aggregation iterations")
parser.add_argument("--traj-per-agg", default=5, type=int, help="Number of rollouts per aggregation iterations")
parser.add_argument("--traj-length", default=1000, type=int, help="Length of rollouts")
parser.add_argument("--num-epochs", default=0, type=int, help="Number of epochs to train model")
parser.add_argument("--H", default=100, type=int, help="Horizon of the MPC controller")
parser.add_argument("--K", default=25, type=int, help="Number of random rollouts of the MPC controller")
parser.add_argument("--predict-reward", default=True, action="store_true", help="Use model to predict reward")
parser.add_argument('--log-dir', default=None, metavar='LD', help='directory to output TensorBoard event file (default: runs/<DATETIME>)')

args = parser.parse_args()

# create logs
if args.log_dir is None:
    args.log_dir = os.path.join('logs', datetime.now().strftime('%b%d_%H-%M-%S'))
writer = SummaryWriter(log_dir=args.log_dir)

# create env
env = gym.make(args.environment) 

# Create model
agent = MPCAgent(env = env, H = args.H, K = args.K, traj_length = args.traj_length, softmax = args.softmax, predict_rewards = args.predict_reward, writer=writer)

# Main loops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
assets_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'))
counter = 0
for iteration in range(args.agg_iters):
    agent.train(args.num_epochs, iteration)
    if not args.experiment_name == '':
        pickle.dump((agent.D_RL, agent.D_rand, agent.model), open(os.path.join(assets_dir, 'learned_models/{}-model.p'.format(args.environment)), 'wb'))
    for traj in range(args.traj_per_agg):
        state = env.reset()
        agent.reset_reinforce()
        trajectory = []
        total_reward = 0
        for t in tqdm(range(args.traj_length)):
            action = agent.choose_action(state, iteration, traj, t)
            new_state, reward, done, info = env.step(action)
            if done:
                reward = -1
            trajectory += [state, action, reward]
            #agent.D_RL.pushTrajectory([state, action, reward, new_state])
            state = new_state
            total_reward += reward
            writer.add_scalar('total_reward/{}-{}'.format(iteration, traj), total_reward, t)
            if args.render:
                env.render()
            if done:
                break
        agent.REINFORCE(trajectory)
        print('Trajectory done. Total reward: {}'.format(total_reward))
        writer.add_scalar('final_total_reward', total_reward, counter)
        counter+=1


