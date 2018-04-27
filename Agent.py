import torch
import numpy as np
import copy
from tqdm import tqdm
from Data import Data
from Model import Model
from torch.utils.data import DataLoader
from torch.autograd import Variable
from RewardOracle import RewardOracle

class Agent():
    
    def __init__(self, env, traj_length = 100, num_rolls = 10, predict_rewards = False):
        self.env = copy.deepcopy(env)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = 1 if len(env.action_space.shape) == 0 else env.action_space.shape[0]
        self.model = Model(self.observation_dim, self.action_dim, predict_rewards)
        self.predict_rewards = predict_rewards
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.criterion = torch.nn.MSELoss()
        self.state = env.reset()
        self.D_RL = Data()
        self.D_rand = self.get_random_data(num_rolls, traj_length)

    def get_random_data(self, num_rolls, traj_length):
        D = Data()
        print('Generating D_rand')
        for i in tqdm(range(num_rolls)):
            s0 = self.env.reset()
            trajectory = [s0,]
            for i in range(traj_length):
                action = self.env.action_space.sample()
                trajectory.append(action)
                observation, reward, done, _ = self.env.step(action)
                trajectory.append(reward)
                trajectory.append(observation)
                if done:
                    break
            D.pushTrajectory(trajectory)    
        print('Generated {} samples'.format(len(D)))
        return D 
            
class MPCAgent(Agent):
    
    def __init__(self, K=10, H=15, softmax = False, temperature = 1, **kwargs):
        super(MPCAgent, self).__init__(**kwargs)
        self.K = K
        self.H = H
        self.temperature = temperature
        self.softmax = softmax
        self.reward_oracle = RewardOracle(self.env)
        
    def aggregate_data(self):
        self.D = self.D_rand + self.D_RL
        self.D.calculate_statistics()
        
    def choose_action(self, state):
        trajectories = self.generate_trajectories(state)
        trajectory_scores = self.score_trajectories(trajectories)
        if self.softmax:
            probabilities = self._softmax(trajectory_scores)
            chosen_trajectory_idx = np.random.choice(list(range(self.K)), p=probabilities.data)
            #print('Action chosen with probability', probabilities[chosen_trajectory_idx].data[0])
            # Logging the entropy of probabilities may be an interesting metric...
        else:
            chosen_trajectory_idx = np.argmax(trajectory_scores)
        action = trajectories[1][chosen_trajectory_idx]
        return action
        
    def normalise_state(self, state):
        state = (state - self.D.means[0])/self.D.stds[0]
        return state
    
    def unnormalise_state(self, state):
        state = state*self.D.stds[0] + self.D.means[0]
        return state
    
    def normalise_action(self, action):
        action = (action - self.D.means[1])/self.D.stds[1]
        return action
    
    def unnormalise_action(self, action):
        action = action*self.D.stds[1] + self.D.means[1]
        return action

    def generate_trajectories(self, state):
        states = np.expand_dims(self.normalise_state(state), 0).repeat(self.K, 0) # matrix of size K * state_dimensions, each row is the state
        trajectories = [states, ]
        for _ in range(self.H):
            # sample actions
            actions = np.stack([self.normalise_action(self.env.action_space.sample()) for _ in range(self.K)])
            
            # infer with model
            state_vars = Variable(torch.from_numpy(states).float())
            action_vars = Variable(torch.from_numpy(actions).float())
            if self.predict_rewards:
                s_diff, rewards = self.model(state_vars, action_vars)
                s_diff, rewards = s_diff.data.numpy(), rewards.data.numpy().squeeze()
            else:
                rewards = self.reward_oracle.reward(self.unnormalise_state(states), self.unnormalise_action(actions))
                s_diff = self.model(state_vars, action_vars).data.numpy()

            # update trajectory
            states = states + s_diff
            trajectories.append(actions)
            trajectories.append(rewards)
            trajectories.append(states)

        return trajectories

    def score_trajectories(self, trajectories):
        rewards = np.zeros((self.K,))
        for reward_idx in range(2, len(trajectories), 3):
            rewards += trajectories[reward_idx]
        return rewards
    
    def _softmax(self, inputs):
        inputs = self.temperature*np.array(inputs)
        return torch.nn.functional.softmax(Variable(torch.from_numpy(inputs)), 0)
    
    def train(self, num_epochs):
        self.aggregate_data()
        train_data = DataLoader(self.D, batch_size = 512, shuffle=True)
        
        for epoch in range(num_epochs):
            running_loss_state, running_loss_reward = 0, 0
            for i, (state, action, reward, state_diff) in enumerate(train_data):
                state = Variable(state)
                action = Variable(action)
                state_diff = Variable(state_diff, requires_grad=False)
                reward = Variable(reward.float(), requires_grad=False)
                if self.predict_rewards:
                    state_diff_hat, reward_hat = self.model(state, action)
                else:
                    state_diff_hat = self.model(state, action)
                state_loss = self.criterion(state_diff_hat, state_diff)
                
                if self.predict_rewards:
                    reward_loss = self.criterion(reward_hat, reward)
                    loss = state_loss + reward_loss
                    running_loss_reward += reward_loss.data[0]
                else:
                    running_loss_reward = 0
                    loss = state_loss
                running_loss_state += state_loss.data[0]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if (epoch % 10 == 0):
                # print("Calculated MSE: ", np.mean((state_diff.data.numpy() - state_diff_hat.data.numpy())**2, axis=0))
                # print("Average Calculated MSE: ", np.mean((state_diff.data.numpy() - state_diff_hat.data.numpy())**2))
                # print("MSE: ", loss.data[0])
                print("Train loss on epoch ", epoch, " = ", running_loss_state, running_loss_reward)
        
                