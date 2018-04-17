from Data import Data
from Model import Model
from torch.utils.data import DataLoader
from torch.autograd import Variable
import gym
import torch
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v1')

model_input_dim = env.observation_space.shape[0] + 1
model_output_dim = env.observation_space.shape[0]
model = Model(model_input_dim, model_output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

validation_data = Data(env, 100, 500)
X_valid, y_valid = Variable(validation_data.X), Variable(validation_data.y)
train_data = Data(env, 100, 500)
dataloader = DataLoader(train_data, batch_size = 512, shuffle=True)


training_loss = []
validation_loss = []
best_loss, best_weights = np.inf, None
plt.clf()
fig, ax = plt.subplots()
for epoch in range(1000):
    print(epoch)
    for X, y in dataloader:    
        yhat = model(Variable(X))
        loss = torch.nn.functional.mse_loss(Variable(y, requires_grad=False), yhat)
        optimizer.zero_grad
        loss.backward()
        optimizer.step()
    validation_loss = (torch.nn.functional.mse_loss(y_valid, model(X_valid)).data[0])
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_weights = model.state_dict()
        
torch.save(best_weights, 'models/initial-model')