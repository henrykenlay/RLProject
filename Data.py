import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

class Data(Dataset):

    def __init__(self, env, rollout_length, num_rollouts, add_noise=True):
        super(Data, self).__init__()
        self.env = env
        self.rollout_length = rollout_length
        self.num_rollouts = num_rollouts
        self.add_noise = True
        self.populate_data()
        
    def populate_data(self):
        self.sample_all_trajectories()
        action = np.expand_dims(self.action, 1)
        state = self.state
        self.X = np.concatenate((state, action), 1)
        self.y = self.state_next - state
        self.normalise_data()
        self.noise_data()
        self.typecast_data()
        
    def sample_all_trajectories(self):
        trajectories = []
        for i in range(self.num_rollouts):
            trajectories.append(self.sample_trajectory())
        state = []
        action = []
        state_next = []    
        for trajectory in trajectories:
            for i in range(1, int((len(trajectory)-1)/2), 2): # indexes will land on actions
                state.append(trajectory[i-1])
                action.append(trajectory[i])
                state_next.append(trajectory[i+1])
        self.state = np.array(state, dtype = 'float32')
        self.action = np.array(action, dtype = 'float32')
        self.state_next = np.array(state_next, dtype = 'float32')
        
    def sample_trajectory(self):
        s0 = self.env.reset()
        trajectory = [s0,]
        for i in range(self.rollout_length-1):
            #TODO: change action to be sampled like in paper
            action = self.env.action_space.sample() #a_i 
            trajectory.append(action)
            next_state, _, done, _ = self.env.step(action) # s_{i+1}
            trajectory.append(next_state)
            if done:
                break
        return trajectory
    
    def normalise_data(self):
        self.X = (self.X - np.mean(self.X, 0))/np.std(self.X, 0)
        self.y = (self.y - np.mean(self.y, 0))/np.std(self.y, 0)
        
    def noise_data(self):
        if self.add_noise:
            self.X = self.X + np.random.normal(0, 0.001, self.X.shape)
            self.y = self.y + np.random.normal(0, 0.001, self.y.shape)
    
    def typecast_data(self):
        self.X = torch.from_numpy(np.array(self.X, dtype = 'float32'))
        self.y = torch.from_numpy(np.array(self.y, dtype = 'float32'))
    
    def __len__(self):
        return self.action.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def save(self, fname):
        np.savez('data/{}.npz'.format(fname), X = self.X.numpy(), y = self.y.numpy())
        
    def load(self, fname):
        npzfile = np.load('data/{}.npz'.format(fname))
        self.X = npzfile['X']
        self.y = npzfile['y']