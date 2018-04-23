import torch
import numpy as np
import copy
from Model import Model
from torch.utils.data import DataLoader
from torch.autograd import Variable
from RewardOracle import RewardOracle

class Agent():
    
    def __init__(self, env, K=10, H=15, softmax = False, temp = 1):
        env = copy.deepcopy(env)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = 1 if len(env.action_space.shape) == 0 else env.action_space.shape[0]
        self.model = Model(self.observation_dim + self.action_dim, self.observation_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.criterion = torch.nn.functional.mse_loss
        self.state = env.reset()
        self.controller = MPCActionController(env, self.model, K, H, softmax, temp)
        
    def train(self, data):
        train_data = DataLoader(data, batch_size = 512, shuffle=True)
        for i, (X, y) in enumerate(train_data):
            X = Variable(X)
            y = Variable(y, requires_grad=False)
            yhat = self.model(X)
            loss = self.criterion(y, yhat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def choose_action(self, state):
        return self.controller.choose_action(state)
            
class MPCActionController():
    
    def __init__(self, env, model, K=10, H=15, softmax = False, temp = 1):
        self.env = env
        self.model = model
        self.K = K
        self.H = H
        self.temp = temp
        self.softmax = softmax
        self.reward_oracle = RewardOracle(env)
        
        
    def choose_action(self, state):
        trajectories = self.generate_trajectories(state)
        trajectory_scores = [self.score_trajectory(trajectory) for trajectory in trajectories]
        if self.softmax:
            probabilities = self._softmax(trajectory_scores)
            chosen_trajectory_idx = np.random.choice(list(range(len(trajectories))), p=probabilities.data)
            chosen_trajectory = trajectories[chosen_trajectory_idx]
        else:
            chosen_trajectory = trajectories[np.argmax(trajectory_scores)]
        action = chosen_trajectory[1]
        return action
        
    def generate_trajectories(self, state):
        trajectories = [self.generate_trajectory(state) for _ in range(self.K)]
        return trajectories

    def generate_trajectory(self, state):
        trajectory = [state,]
        for _ in range(self.H):
            action = self.env.action_space.sample()
            trajectory.append(action)
            model_input = np.array(np.concatenate((state, action)), dtype='float32')
            model_input = Variable(torch.from_numpy(model_input))
            state = state + self.model(model_input).data.numpy()
            trajectory.append(state)
        return trajectory

    def score_trajectory(self, trajectory):
        reward = 0
        for action_idx in range(1, len(trajectory), 2):
            action = trajectory[action_idx]
            state = trajectory[action_idx-1]
            reward += self.reward_oracle.reward(state, action)
        return reward
    
    def _softmax(self, inputs):
        inputs = self.temp*np.array(inputs)
        return torch.nn.functional.softmax(Variable(torch.from_numpy(inputs)), 0)
        