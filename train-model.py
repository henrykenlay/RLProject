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

agg_iters = 7
traj_per_agg = 9
traj_length = 100
controller_H = 8
controller_K = 100
# Make D_rand of random trajectories
D_rand = get_random_data(env, num_rolls=10, max_roll_length=traj_length)
D_RL = Data(capacity = len(D_rand)*9)

# Create model
<<<<<<< HEAD
agent = Agent(env, controller_K=controller_K, controller_H=controller_H)

# Main loops
state = env.reset()
for iteration in range(agg_iters):
	agent.train(D_rand + D_RL, num_iters=50)
	for i in range(traj_per_agg):
		episode_reward = 0
		for t in tqdm(range(traj_length)):
				action = agent.choose_action(state)
				new_state, reward, done, info = env.step(action)
				D_RL.pushTrajectory([state, action, new_state])
				state = new_state
				episode_reward += reward
				#env.render()
				if done:
					env.reset()
		print('Trajectory ', i, 'reward: ', episode_reward)

=======
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
>>>>>>> origin/master
