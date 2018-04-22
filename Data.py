from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import copy


class Data(Dataset):

    def __init__(self, noise=0.001, capacity = np.inf):
        super(Data, self).__init__()
        self.noise = noise
        self.capacity = capacity
        self.data_pushed = 0
        self.X = [] 
        self.y = []        
        self.statistics_calculated = False
        
    def pushTrajectory(self, trajectory):
        # trajectory is [s0, a0, s1, a1...,a_{T-2}, s_{T-1}] 
        X, y = [], []
        action_idx = list(range(1, len(trajectory), 2))
        
        for i in action_idx:
            s = trajectory[i-1]
            a = trajectory[i]
            s_next = trajectory[i+1]
            X.append(np.concatenate((s, a)))
            y.append(s_next - s)
        self.push(np.stack(X), np.stack(y))
        
    def push(self, X, y):
        X, y = X.astype('float32'), y.astype('float32')
        for X_item, y_item in zip(X, y):
            self._push(X_item, y_item)
            
    def _push(self, X, y):
        if self.data_pushed < self.capacity:
            self.X.append(X)
            self.y.append(y)
        else:
            idx = int(self.data_pushed % self.capacity)
            self.X[idx] = X
            self.y[idx] = y
        self.data_pushed += 1
        self.statistics_calculated = False
            
    def calculate_statistics(self):
        if not self.statistics_calculated:
            self.X_mean, self.X_std = np.mean(self.X, 0), np.std(self.X, 0)
            self.y_mean, self.y_std = np.mean(self.y, 0), np.std(self.y, 0)
            self.statistics_calculated = True
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if idx >= self.__len__():
            return
        
        X, y = self.X[idx], self.y[idx]
        # add noise
        X = X + np.random.normal(0, self.noise, X.shape)
        y = y + np.random.normal(0, self.noise, y.shape)
        # normalise
        self.calculate_statistics()
        X = (X - self.X_mean)/self.X_std, 
        y = (y - self.y_mean)/self.y_std
        # typecast
        X = torch.from_numpy(np.array(X, dtype = 'float32'))
        y = torch.from_numpy(np.array(y, dtype = 'float32')) 
        return X, y
    
    def __add__(self, data):
        new_data = Data(capacity = data.capacity + self.capacity)
        new_data.data_pushed = data.data_pushed + self.data_pushed
        new_data.X = data.X + self.X
        new_data.y = data.y + self.y
        return new_data
        
    
def get_random_data(env, num_rolls = 25, max_roll_length = 50):
    env = copy.deepcopy(env)
    D = Data()
    print('Generating D_rand')
    for i in tqdm(range(num_rolls)):
        s0 = env.reset()
        trajectory = [s0,]
        for i in range(max_roll_length):
            action = env.action_space.sample()
            trajectory.append(action)
            observation, reward, done, _ = env.step(action)
            trajectory.append(observation)
            if done:
                break
        D.pushTrajectory(trajectory)    
    print('Generated {} samples'.format(len(D)))
    return D
        
    
if __name__ == '__main__':
    X = np.random.random((10, 5))
    X2 = np.random.random((8, 5))
    y = np.random.random((10, 2))
    y2 = np.random.random((8, 2))
    data = Data(capacity = 5)
    data.push(X, y)
    get_array = lambda x : np.random.random((x,))
    trajectory = [get_array(2), get_array(3), get_array(2), get_array(3), get_array(2), get_array(3), get_array(2)]
    data.pushTrajectory(trajectory)
    data2 = Data(capacity = 5)
    data2.push(X2, y2)
    data + data2