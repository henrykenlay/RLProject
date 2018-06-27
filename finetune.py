from Agent import Agent
from tqdm import tqdm
from dm_control import suite
from Logger import Logger
import argparse

parser = argparse.ArgumentParser(description='Finetune a model trained using MPC with reinforce')
parser.add_argument('--environment', default='cheetah-run', help='Environment to train the agent on')
parser.add_argument('--modelname', default=None, help='Model to load')
parser.add_argument('--experiment-name', default=None, help='Will save the model as [envname]-[suffix]')
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
parser.add_argument("--reinforce", default=True, action="store_true", help="Use model to predict reward")
parser.add_argument("--reinforce-batchsize", default=1, type=int, help="Batch size of REINFORCE updates")
parser.add_argument("--reinforce-shuffle", default=False, action="store_true", help="Shuffle REINFORCE samples from episode")
parser.add_argument("--iterations", default=1000000, type=int, help="Extra epochs of finetuning")
args = parser.parse_args()

assert args.experiment_name is not None
assert args.modelname is not None

# create logs
logger = Logger('project/logs', 'finetune-{}'.format(args.experiment_name))

env = suite.load(*args.environment.split('-')) 

agent = Agent(env = env, H = args.H, K = args.K, traj_length = args.traj_length, 
              softmax = args.softmax, predict_rewards = args.predict_reward, 
              reinforce = args.reinforce, lr = args.lr, 
              temperature = args.temperature, reinforce_lr = args.reinforce_lr,
              hidden_units = args.hidden_units, batch_size = args.reinforce_batchsize,
              shuffle_gradients = args.reinforce_shuffle)

agent.loadbest(args.modelname)

for iteration in range(args.iterations):
    timestep = env.reset()
    rewards = []
    for t in tqdm(range(args.traj_length), desc='Generating episode'):
        state = env.physics.state()
        action = agent.choose_action(state)
        timestep = env.step(action)
        new_state, reward = env.physics.state(), timestep.reward
        agent.D_RL.pushTrajectory([state, action, reward, new_state])
        rewards.append(reward)
    print('Trajectory done. Total reward: {}'.format(sum(rewards)))
    val_loss = agent.validation_loss()
    logger.log([iteration, sum(rewards), val_loss[0], val_loss[1]])
    print(iteration, sum(rewards), val_loss[0], val_loss[1])
    agent.REINFORCE(rewards)
