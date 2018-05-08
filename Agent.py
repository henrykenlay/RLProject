import torch
import numpy as np
import copy
from tqdm import tqdm
from Data import Data, AggregatedData
from Model import Model
from torch.utils.data import DataLoader
from torch.autograd import Variable
from RewardOracle import RewardOracle
from scipy.stats import entropy
from torch.distributions.categorical import Categorical


class Agent():
    
    def __init__(self, env, traj_length = 100, num_rolls = 10, predict_rewards = False, writer = None):
        self.env = copy.deepcopy(env)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = 1 if len(env.action_space.shape) == 0 else env.action_space.shape[0]
        self.model = Model(self.observation_dim, self.action_dim, predict_rewards)
        self.predict_rewards = predict_rewards
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.1)
        self.criterion = torch.nn.MSELoss()
        self.reinforce_criterion = torch.nn.CrossEntropyLoss(reduce=False)
        self.state = env.reset()
        self.D_RL = Data()
        self.D_rand = self.get_random_data(10, 1000)
        self.writer = writer

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
        
    def reset_reinforce(self):
        self.logits = []
        self.action_idx = []
        
    def aggregate_data(self):
        if len(self.D_RL) > 0:
            self.D = AggregatedData([self.D_rand, self.D_RL], probabilities = [0.1, 0.9])
        else:
            self.D = AggregatedData([self.D_rand,])
        
    def choose_action(self, state, iteration, traj, t):
        state = torch.tensor(state, requires_grad=True).float()
        trajectories = self.generate_trajectories(state)
        trajectory_scores = self.score_trajectories(trajectories)
        probabilities = self._softmax(trajectory_scores)
        if self.writer is not None:
            pass
            #TODO fix
            #self.writer.add_scalar('entropy/{}-{}'.format(iteration, traj), entropy(probabilities), t)
        if self.softmax:
            #chosen_trajectory_idx = np.random.choice(list(range(self.K)), p=probabilities.data)
            chosen_trajectory_idx = Categorical(probabilities.squeeze()).sample()
        else:
            chosen_trajectory_idx = np.argmax(probabilities)
        self.logits.append(probabilities)
        self.action_idx.append(chosen_trajectory_idx)
        action = trajectories[1][chosen_trajectory_idx].detach().numpy()
        return self.unnormalise_action(action)
        
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
    
    def normalise_reward(self, reward):
        reward = (reward - self.D.means[2])/self.D.stds[2]
        return reward
    
    def unnormalise_reward(self, reward):
        reward = reward*self.D.stds[2] + self.D.means[2]
        return reward

    def generate_trajectories(self, state):
        states = state.unsqueeze(0).repeat(self.K, 1) # matrix of size K * state_dimensions, each row is the state
        trajectories = [states, ]
        for _ in range(self.H):
            # sample actions
            actions = np.stack([self.normalise_action(self.env.action_space.sample()) for _ in range(self.K)])
            
            # infer with model
            actions = torch.tensor(actions, requires_grad=True).float()
            if self.predict_rewards:
                s_diff, rewards = self.model(states, actions)
            else:
                # TODO: this is probably broken now
                rewards = self.reward_oracle.reward(self.unnormalise_state(states), self.unnormalise_action(actions))
                rewards = self.normalise_reward(rewards)
                s_diff = self.model(state_vars, action_vars).data.numpy()

            # update trajectory
            states = states + s_diff
            trajectories.append(actions)
            trajectories.append(rewards)
            trajectories.append(states)

        return trajectories

    def score_trajectories(self, trajectories):
        rewards = torch.zeros((self.K, 1), requires_grad=True)
        for reward_idx in range(2, len(trajectories), 3):
            rewards = rewards + self.unnormalise_reward(trajectories[reward_idx])
        return rewards
    
    def _softmax(self, inputs):
        inputs = self.temperature*inputs
        return torch.nn.functional.softmax(inputs, 0)
    
    def train(self, num_epochs, iteration):
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
                    reward_loss = self.criterion(reward_hat.squeeze(), reward)
                    loss = state_loss + reward_loss
                    running_loss_reward += reward_loss.item()
                else:
                    running_loss_reward = 0
                    loss = state_loss
                running_loss_state += state_loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.writer is not None:
                self.writer.add_scalar('loss/state/{}'.format(iteration), running_loss_state, epoch)
                self.writer.add_scalar('loss/reward/{}'.format(iteration), running_loss_reward, epoch)
               
    def REINFORCE(self, trajectory):
        rewards = [trajectory[i] for i in range(2, len(trajectory), 3)]
        q_values = torch.tensor(np.cumsum(rewards[::-1])[::-1].copy()).float()
        q_values = q_values - np.mean(rewards)
        actions = torch.tensor(np.array(self.action_idx), requires_grad=False).long()
        logits = torch.stack(self.logits).float()
        weighted_loss = self.reinforce_criterion(logits.squeeze(), actions)*q_values
        loss = torch.mean(weighted_loss)    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class RandomAgent():
    
    def __init__(self, env, **kwargs):
        self.env = copy.deepcopy(env)
        
    def choose_action(self, state, *a, **kwargs):
        return self.env.action_space.sample()
    
    def REINFORCE(self, trajectory):
        pass
    
    def train(self, a, b):
        pass
    
    def reset_reinforce(self):
        pass