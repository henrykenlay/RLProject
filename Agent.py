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
    
    def __init__(self, env, traj_length = 100, num_rolls = 10):
        self.env = copy.deepcopy(env)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = 1 if len(env.action_space.shape) == 0 else env.action_space.shape[0]
        self.model = Model(self.observation_dim + self.action_dim, self.observation_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.criterion = torch.nn.functional.mse_loss
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
        trajectory_scores = [self.score_trajectory(trajectory) for trajectory in trajectories]
        if self.softmax:
            probabilities = self._softmax(trajectory_scores)
            chosen_trajectory_idx = np.random.choice(list(range(len(trajectories))), p=probabilities.data)
            chosen_trajectory = trajectories[chosen_trajectory_idx]
        else:
            chosen_trajectory = trajectories[np.argmax(trajectory_scores)]
        action = chosen_trajectory[1]
        return action
        
    def normalise_state(self, state):
        state = (state - self.D.X_mean[:self.observation_dim])/self.D.X_std[:self.observation_dim]
        return state
    
    def unnormalise_state(self, state):
        state = state*self.D.X_std[:self.observation_dim] + self.D.X_mean[:self.observation_dim]
        return state
    
    def normalise_action(self, action):
        action = (action - self.D.X_mean[self.observation_dim:])/self.D.X_std[self.observation_dim:]
        return action
    
    def unnormalise_action(self, action):
        action = action*self.D.X_std[self.observation_dim:] + self.D.X_mean[self.observation_dim:]
        return action
    
    def generate_trajectories(self, state):
        state = self.normalise_state(state)
        trajectories = [self.generate_trajectory(state) for _ in range(self.K)]
        return trajectories

    def generate_trajectory(self, state):
        trajectory = [state,]
        for _ in range(self.H):
            action = self.normalise_action(self.env.action_space.sample())
            trajectory.append(action)
            model_input = np.array(np.concatenate((state, action)), dtype='float32')
            model_input = Variable(torch.from_numpy(model_input))
            state = state + self.model(model_input).data.numpy()
            trajectory.append(state)
        return trajectory

    def score_trajectory(self, trajectory):
        reward = 0
        for action_idx in range(1, len(trajectory), 2):
            action = self.unnormalise_action(trajectory[action_idx])
            state = self.unnormalise_state(trajectory[action_idx-1])
            reward += self.reward_oracle.reward(state, action)
        return reward
    
    def _softmax(self, inputs):
        inputs = self.temperature*np.array(inputs)
        return torch.nn.functional.softmax(Variable(torch.from_numpy(inputs)), 0)
    
    def train(self, num_epochs):
        self.aggregate_data()
        train_data = DataLoader(self.D, batch_size = 512, shuffle=True)
        for epoch in range(num_epochs):
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
        