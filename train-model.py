from Agent import MPCAgent
from tqdm import tqdm
from dm_control import suite
import os
import sys
import argparse
from tensorboardX import SummaryWriter
from datetime import datetime
from Recorder import Recorder

parser = argparse.ArgumentParser(description='Train a model using MPC')
parser.add_argument('--environment', default='cheetah-run', help='Environment to train the agent on')
parser.add_argument('--experiment-name', default=None, help='Will save the model as [envname]-[suffix] if given')
parser.add_argument("--softmax", default=False, action="store_true",  help="Use softmax MPC")
parser.add_argument("--agg-iters", default=100, type=int, help="Aggregation iterations")
parser.add_argument("--traj-per-agg", default=5, type=int, help="Number of rollouts per aggregation iterations")
parser.add_argument("--traj-length", default=1000, type=int, help="Length of rollouts")
parser.add_argument("--num-epochs", default=50, type=int, help="Number of epochs to train model")
parser.add_argument("--H", default=10, type=int, help="Horizon of the MPC controller")
parser.add_argument("--K", default=1000, type=int, help="Number of random rollouts of the MPC controller")
parser.add_argument("--predict-reward", default=True, action="store_true", help="Use model to predict reward")
args = parser.parse_args()

# create logs
if args.experiment_name is None:
    args.experiment_name = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir=os.path.join('logs', args.experiment_name))



# create env
env = suite.load(*args.environment.split('-')) 

# Create model
agent = MPCAgent(env = env, H = args.H, K = args.K, traj_length = args.traj_length, softmax = args.softmax, predict_rewards = args.predict_reward, writer=writer)

# Main loops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
assets_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'))

count = 0
for iteration in range(args.agg_iters):
    agent.train(args.num_epochs, iteration)
    #if not args.experiment_name == '':
        # TODO change for DM
    #    pickle.dump((agent.D_RL, agent.D_rand, agent.model), open(os.path.join(assets_dir, 'learned_models/{}-model.p'.format(args.environment)), 'wb'))
    for traj in range(args.traj_per_agg):
        timestep = env.reset()
        total_reward = 0
        actions = [0,]
        recorder = Recorder(args.experiment_name, count)
        for t in tqdm(range(args.traj_length)):
            recorder.record_frame(env.physics.render(camera_id=0), t)
            state = env.physics.state()
            action = agent.choose_action(state)
            timestep = env.step(action)
            new_state, reward = env.physics.state(), timestep.reward
            agent.D_RL.pushTrajectory([state, action, reward, new_state])
            total_reward += reward
        recorder.make_movie()
        print('Trajectory done. Total reward: {}'.format(total_reward))
        writer.add_scalar('total_reward', total_reward, count)
        count += 1
        



