from Agent import MPCAgent
from tqdm import tqdm
from dm_control import suite
import os
import argparse
from tensorboardX import SummaryWriter
from datetime import datetime
from Recorder import Recorder
from Logger import Logger

parser = argparse.ArgumentParser(description='Train a model using MPC')
parser.add_argument('--environment', default='cheetah-run', help='Environment to train the agent on')
parser.add_argument('--experiment-name', default=None, help='Will save the model as [envname]-[suffix] if given')
parser.add_argument("--softmax", default=False, action="store_true",  help="Use softmax MPC")
parser.add_argument("--temperature", default=1, type=int,  help="Softmax temp")
parser.add_argument("--learntemp", default=False, action="store_true", help="Make temp a parameter")
parser.add_argument("--agg-iters", default=100, type=int, help="Aggregation iterations")
parser.add_argument("--traj-per-agg", default=1, type=int, help="Number of rollouts per aggregation iterations")
parser.add_argument("--traj-length", default=1000, type=int, help="Length of rollouts")
parser.add_argument("--num-epochs", default=10, type=int, help="Number of epochs to train model")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--H", default=10, type=int, help="Horizon of the MPC controller")
parser.add_argument("--K", default=100, type=int, help="Number of random rollouts of the MPC controller")
parser.add_argument("--predict-reward", default=True, action="store_true", help="Use model to predict reward")
parser.add_argument("--reinforce", default=False, action="store_true", help="Use model to predict reward")
parser.add_argument("--record", default=False, action="store_true", help="Make movies of agent")


args = parser.parse_args()

# create logs
if args.experiment_name is None:
    args.experiment_name = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir=os.path.join('project/logs', args.experiment_name))
logger = Logger('project/logs', args.experiment_name)
# create env
env = suite.load(*args.environment.split('-')) 

# Create model
agent = MPCAgent(env = env, H = args.H, K = args.K, traj_length = args.traj_length, softmax = args.softmax, predict_rewards = args.predict_reward, writer=writer, reinforce = args.reinforce, lr = args.lr, temperature = args.temperature, learntemp=args.learntemp)

count = 0
for iteration in range(args.agg_iters):
    agent.train(args.num_epochs, iteration)
    for traj in range(args.traj_per_agg):
        timestep = env.reset()
        rewards = []
        if args.record:
            recorder = Recorder(args.experiment_name, count)
        for t in tqdm(range(args.traj_length), desc='Generating episode'):
            if args.record:
                recorder.record_frame(env.physics.render(camera_id=0), t)
            state = env.physics.state()
            action = agent.choose_action(state)
            timestep = env.step(action)
            new_state, reward = env.physics.state(), timestep.reward
            agent.D_RL.pushTrajectory([state, action, reward, new_state])
            rewards.append(reward)
        print('Trajectory done. Total reward: {}'.format(sum(rewards)))
        writer.add_scalar('total_reward', sum(rewards), count)
        val_loss = agent.validation_loss()
        logger.log(count, sum(rewards), val_loss[0], val_loss[1])
        print(count, sum(rewards), val_loss[0], val_loss[1])
        if args.reinforce:
            agent.REINFORCE(rewards)
        if args.record:
            recorder.make_movie()
        count += 1
        



