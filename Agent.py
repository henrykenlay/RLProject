import torch
import numpy as np
import copy
from tqdm import tqdm
from Data import Data, AggregatedData
from Model import Model
from torch.utils.data import DataLoader
from torch.autograd import Variable
from RewardOracle import RewardOracle

class Agent():
    
    def __init__(self, env, traj_length = 100, num_rolls = 10, predict_rewards = False, writer = None):
        self.env = copy.deepcopy(env)
        self.action_spec = env.action_spec()
        state_dim = env.physics.state().shape[0]
        action_dim = self.action_spec.shape[0]
        self.model = Model(state_dim, action_dim, predict_rewards)
        self.predict_rewards = predict_rewards
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.criterion = torch.nn.MSELoss()
        self.D_RL = Data()
        self.D_rand = self.get_random_data(num_rolls, traj_length)
        self.writer = writer

    def get_random_data(self, num_rolls, traj_length):
        D = Data()
        print('Generating D_rand')
        for i in tqdm(range(num_rolls)):
            self.env.reset()
            s0 = self.env.physics.state()
            trajectory = [s0,]
            for i in range(traj_length):
                action = self.sample_single_action()
                trajectory.append(action)
                timestep = self.env.step(action)
                trajectory.append(timestep.reward)
                trajectory.append(self.env.physics.state())
                if timestep.last():
                    break
            D.pushTrajectory(trajectory)    
        print('Generated {} samples'.format(len(D)))
        return D 
    
    def sample_single_action(self):
        action = np.random.uniform(self.action_spec.minimum, self.action_spec.maximum , self.action_spec.shape[0])
        return action 
    
    def sample_batch_action(self, n):
        action = np.random.uniform(self.action_spec.minimum, self.action_spec.maximum , (n, self.action_spec.shape[0]))
        return action
    
class MPCAgent(Agent):
    
    def __init__(self, K=10, H=15, softmax = False, temperature = 1, **kwargs):
        super(MPCAgent, self).__init__(**kwargs)
        self.K = K
        self.H = H
        self.temperature = temperature
        self.softmax = softmax
        self.reward_oracle = RewardOracle(self.env)
        
    def aggregate_data(self):
        if len(self.D_RL) > 0:
            self.D = AggregatedData(self.D_rand, self.D_RL)
        else:
            self.D = self.D_rand
        
    def choose_action(self, state):
        trajectories = self.generate_trajectories(state)
        trajectory_scores = self.score_trajectories(trajectories)
        probabilities = self._softmax(trajectory_scores)
        if self.softmax:
            chosen_trajectory_idx = np.random.choice(list(range(self.K)), p=probabilities.data)
        else:
            chosen_trajectory_idx = np.argmax(probabilities)
        action = trajectories[1][chosen_trajectory_idx]
        return action

    def generate_trajectories(self, state):
        states = np.expand_dims(state, 0).repeat(self.K, 0) # matrix of size K * state_dimensions, each row is the state
        trajectories = [states, ]
        for _ in range(self.H):
            # sample actions
            actions = self.sample_batch_action(self.K)
            
            # infer with model
            state_vars = Variable(torch.from_numpy(states).float())
            action_vars = Variable(torch.from_numpy(actions).float())
            if self.predict_rewards:
                s_diff, rewards = self.model(state_vars, action_vars)
                s_diff, rewards = s_diff.data.numpy(), rewards.data.numpy().squeeze()
            else:
                rewards = self.reward_oracle.reward(states, actions)
                s_diff = self.model(state_vars, action_vars).data.numpy()

            # update trajectory
            states = states + s_diff
            trajectories.append(actions)
            trajectories.append(rewards)
            trajectories.append(states)

        return trajectories

    def score_trajectories(self, trajectories):
        rewards = np.zeros((self.K,))
        for reward_idx in range(2, len(trajectories), 3):
            rewards += trajectories[reward_idx]
        return rewards
    
    def _softmax(self, inputs):
        inputs = self.temperature*np.array(inputs)
        return torch.nn.functional.softmax(Variable(torch.from_numpy(inputs)), 0)
    
    def train(self, num_epochs, iteration):
        self.aggregate_data()
        train_data = DataLoader(self.D, batch_size = 512, shuffle=True)
        
        for epoch in tqdm(range(num_epochs)):
            running_loss_state, running_loss_reward = 0, 0
            for i, (state, action, reward, state_diff) in enumerate(train_data):
                state = Variable(state)
                action = Variable(action)
                state_diff = Variable(state_diff, requires_grad=False)
                reward = Variable(reward.float(), requires_grad=False)
                if self.predict_rewards:
                    state_diff_hat, reward_hat = self.model(state, action)
                else:
                    state_diff_hat = self.model(state, action)
                state_loss = self.criterion(state_diff_hat, state_diff)
                
                if self.predict_rewards:
                    reward_loss = self.criterion(reward_hat.squeeze(), reward)
                    loss = state_loss + reward_loss
                    running_loss_reward += reward_loss.item()
                else:
                    running_loss_reward = 0
                    loss = state_loss
                running_loss_state += state_loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.writer is not None:
                self.writer.add_scalar('loss/state/{}'.format(iteration), running_loss_state, epoch)
                self.writer.add_scalar('loss/reward/{}'.format(iteration), running_loss_reward, epoch)
                validation_loss = self.validation_loss()
                self.writer.add_scalar('loss/state_eq3/{}'.format(iteration), validation_loss[0], epoch)
                self.writer.add_scalar('loss/reward_eq3/{}'.format(iteration), validation_loss[1], epoch)
                
                
    def validation_loss(self):
        self.env.reset()
        state_scores = []
        reward_scores = []
        for _ in range(100):
            trajectories = self.generate_trajectories(self.env.physics.state())
            trajectory_scores = self.score_trajectories(trajectories)
            probabilities = self._softmax(trajectory_scores)
            if self.softmax:
                chosen_trajectory_idx = np.random.choice(list(range(self.K)), p=probabilities.data)
            else:
                chosen_trajectory_idx = np.argmax(probabilities)
            actions = [trajectories[i][chosen_trajectory_idx] for i in range(1, len(trajectories), 3)]
            predicted_states = [trajectories[i][chosen_trajectory_idx] for i in range(0, len(trajectories), 3)][1:]
            predicted_rewards = [trajectories[i][chosen_trajectory_idx] for i in range(2, len(trajectories), 3)]
            states = []
            rewards = []
            for action in actions:
                timestep = self.env.step(action)
                rewards.append(timestep.reward)
                states.append(self.env.physics.state())
            state_scores.append(self.compare_trajectories(states, predicted_states))
            reward_scores.append(np.mean(np.square(np.array(rewards)-np.array(predicted_rewards))))
        return np.mean(state_scores), np.mean(reward_scores)
            
    def compare_trajectories(self, states, predicted_states):
        score = np.array(predicted_states) - np.array(states)
        score = np.mean(np.sum(np.square(score), 0)/2)
        return score
            
if __name__ == '__main__':
    from dm_control import suite
    env = suite.load('cheetah', 'run')
    agent = MPCAgent(env = env)
    agent.validation_loss()
    