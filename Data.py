from torch.utils.data import Dataset
import numpy as np

class Data(Dataset):

    def __init__(self, noise=0.001):
        super(Data, self).__init__()
        self.noise = noise
        self.transitions = []
    
    def pushTrajectory(self, trajectory):
        # trajectory is s0, a0, r0, s1, a1, r1.... r_{T-1}, S_T
        for state_idx in range(3, len(trajectory), 3):
            state = np.array(trajectory[state_idx-3], dtype = 'float32')
            action = np.array(trajectory[state_idx-2], dtype = 'float32')
            reward = float(trajectory[state_idx - 1])
            next_state = np.array(trajectory[state_idx], dtype = 'float32')
            transition = np.array([state, action, reward, next_state-state])
            self.pushTransition(transition)
    
    def pushTransition(self, transition):
        # transition is [state, action, reward, next_state]
        self.transitions.append(transition)

    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None
        transition = self.transitions[idx]
        transition = self.add_noise(transition)
        transition = self.typecast(transition)
        return [*transition]
    
    def add_noise(self, transition):
        for i in [0, 1, 3]:
            transition[i] = transition[i] + np.random.normal(0, self.noise, transition[i].shape)
        return transition
    
    def typecast(self, transition):
        for i in [0, 1, 3]:
            transition[i] = np.array(transition[i], dtype='float32')
        return transition
    
    def __add__(self, data):
        new_data = Data(self.noise)
        new_data.transitions = self.transitions + data.transitions
        return new_data
    
class AggregatedData(Dataset):
    
    def __init__(self, D_rand, D_RL, probabilities = (0.1, 0.9)):
        super(AggregatedData, self).__init__()
        self.D_rand = D_rand
        self.D_RL = D_RL
        self.probabilities = probabilities

    def __len__(self):
        return len(self.D_rand) + len(self.D_RL)
    
    def __getitem__(self, idx):
        if np.random.random() < self.probabilities[0]:
            idx = np.random.randint(0, len(self.D_rand))
            return self.D_rand[idx]
        else:
            idx = np.random.randint(0, len(self.D_RL))
            return self.D_RL[idx]
