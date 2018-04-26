"""See second half of appendix.A for architecture"""
import torch

class Model(torch.nn.Module):
     
    def __init__(self, state_dim, action_dim, return_reward_est = False):
        super(Model, self).__init__()
        self.return_reward_est = return_reward_est
        self.linear1 = torch.nn.Linear(state_dim + action_dim, 500)
        self.linear2 = torch.nn.Linear(500, 500)
        self.linear3 = torch.nn.Linear(500, state_dim)
        if return_reward_est:
            self.linear4 = torch.nn.Linear(500, 1)
        self.activation_fn = torch.nn.ReLU()
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.activation_fn(self.linear1(x))
        x = self.activation_fn(self.linear2(x))
        s_diff = self.linear3(x)
        if self.return_reward_est:
            reward = self.linear4(x)
            return s_diff, reward
        else:
            return s_diff
    