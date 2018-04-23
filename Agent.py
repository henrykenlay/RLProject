import torch
import numpy as np
import copy
#from tqdm import tqdm
from Model import Model
from torch.utils.data import DataLoader
from torch.autograd import Variable
from RewardOracle import RewardOracle

class Agent():
    
    def __init__(self, env, controller_K, controller_H):
        env = copy.deepcopy(env)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = 1 if len(env.action_space.shape) == 0 else env.action_space.shape[0]
        self.model = Model(self.observation_dim + self.action_dim, self.observation_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.criterion = torch.nn.functional.mse_loss
        self.state = env.reset()
        self.controller = MPCActionController(env, self.model, controller_K, controller_H)
        
    def train(self, data, num_iters):
        train_data = DataLoader(data, batch_size = 512, shuffle=True)
        for epoch in range(num_iters):
            running_loss = 0.0
            for i, (X, y) in enumerate(train_data):
                X = Variable(X)
                y = Variable(y, requires_grad=False)
                yhat = self.model(X)
                loss = self.criterion(y, yhat)
                running_loss += loss.data[0]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if (epoch % 10 == 0):
                print("Train loss on epoch ", epoch, " = ", running_loss)
            
    def choose_action(self, state):
        return self.controller.choose_action(state)
            
class MPCActionController():
    
    def __init__(self, env, model, K=10, H=15):
        self.env = env
        self.model = model
        self.K = K
        self.H = H
        self.reward_oracle = RewardOracle(env)
        
    def choose_action(self, state):
        trajectories = self.generate_trajectories(state)
        best_trajectory = self.choose_best_trajectory(trajectories)
        action = best_trajectory[1]
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
        
    def choose_best_trajectory(self, trajectories):
        best_trajectory, best_score = None, -np.inf
        for trajectory in trajectories:
            score = self.score_trajectory(trajectory)
            if score > best_score:
                best_score = score
                best_trajectory = trajectory
        return best_trajectory
        
    def score_trajectory(self, trajectory):
        reward = 0
        for action_idx in range(1, len(trajectory), 2):
            action = trajectory[action_idx]
            state = trajectory[action_idx-1]
            reward += self.reward_oracle.reward(state, action)
        return reward
        