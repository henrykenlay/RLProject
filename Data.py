from torch.utils.data import Dataset
from Statistics import RunningStatistics
from tqdm import tqdm
import numpy as np
import torch
import copy


class Data(Dataset):

    def __init__(self, X=None, y=None, fname=None, noise=0.001):
        super(Data, self).__init__()
        self.noise = noise
        self.reset()
        
        if fname is not None:
            self.load(fname)
        elif X is not None:
            self.push(X, y)
        
        
    def pushTrajectory(self, trajectory):
        # trajectory is [s0, a0, s1, a1...,a_{T-2}, s_{T-1}] 
        X, y = [], []
        action_idx = list(range(1, len(trajectory), 2))
        for i in action_idx:
            s = trajectory[i-1]
            a = trajectory[i]
            s_next = trajectory[i+1]
            try:
                X.append(np.concatenate((s, a)))
            except:
                pass
            y.append(s_next - s)
        self.push(np.stack(X), np.stack(y))
        
    def push(self, X, y):
        X, y = X.astype('float32'), y.astype('float32')
        if self.X is None:
            self.X, self.y = X, y
            self.Xstat = RunningStatistics(X.shape[1:])
            self.ystat = RunningStatistics(y.shape[1:])
        else:
            try:
                self.X = np.concatenate((self.X, X))
            except:
                pass
            self.y = np.concatenate((self.y, y))
        self.update_stats(X, y)
    
    def update_stats(self, X, y):
        for row in X:
            self.Xstat.push(row)
        for row in y:
            self.ystat.push(row)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if idx >= self.__len__():
            return
        X, y = self.X[idx], self.y[idx]
        # add noise
        X = X + np.random.normal(0, self.noise, X.shape)
        y = y + np.random.normal(0, self.noise, y.shape)
        # normalise
        X = (X - self.Xstat.mean)/self.Xstat.std
        y = (y - self.ystat.mean)/self.ystat.std
        # typecast
        X = torch.from_numpy(np.array(X, dtype = 'float32'))
        y = torch.from_numpy(np.array(y, dtype = 'float32')) 
        return X, y
    
    def save(self, fname):
        np.savez('data/{}.npz'.format(fname), X = self.X, y = self.y)
        
    def load(self, fname):
        npzfile = np.load('data/{}.npz'.format(fname))
        self.reset()
        self.push(npzfile['X'], npzfile['y'])
        
    def reset(self):
        self.X, self.y, self.Xstat, self.ystat = None, None, None, None
        
    def __add__(self, data):
        new_data = Data(noise = np.mean([self.noise, data.noise]))
        new_data.push(self.X, self.y)
        new_data.push(data.X, data.y)
        return new_data
    
def get_random_data(env, num_rolls = 2, max_roll_length = 20):
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
    data = Data()
    data.push(X, y)
    data2 = Data(X2, y2)
    data3 = data + data2
    get_array = lambda x : np.random.random((x,))
    trajectory = [get_array(2), get_array(3), get_array(2), get_array(3), get_array(2), get_array(3), get_array(2)]
    data3.pushTrajectory(trajectory)