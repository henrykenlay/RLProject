"""See second half of appendix.A for architecture"""
import torch

class Model(torch.nn.Module):
     
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 500)
        self.linear2 = torch.nn.Linear(500, 500)
        self.linear3 = torch.nn.Linear(500, output_dim)
        self.activation_fn = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.activation_fn(self.linear1(x))
        x = self.activation_fn(self.linear2(x))
        x = self.linear3(x)
        return x
    