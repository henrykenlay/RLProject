import torch
import numpy as np
import copy
from tqdm import tqdm
from Data import Data, AggregatedData
from Model import Model
from torch.utils.data import DataLoader
from RewardOracle import RewardOracle
from torch.distributions.categorical import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_

eps = np.finfo(np.float32).eps.item()
if torch.cuda.is_available():
    print('USING GPU')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    using_gpu = True
else:
    using_gpu = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, env, traj_length = 100, num_rolls = 10, predict_rewards = False, writer = None):
        self.env = copy.deepcopy(env)
        self.action_spec = env.action_spec()
        state_dim = env.physics.state().shape[0]
        action_dim = self.action_spec.shape[0]
        self.model = Model(state_dim, action_dim, predict_rewards)
        if using_gpu:
            self.model = self.model.to(device)
        self.predict_rewards = predict_rewards
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.01)
        self.optimizer_reinforce = torch.optim.Adam(self.model.parameters(), lr = 0.00001)
        self.criterion = torch.nn.MSELoss()
        self.D_RL = Data()
        self.D_rand = self.get_random_data(num_rolls, traj_length)
        self.writer = writer

    def get_random_data(self, num_rolls, traj_length):
        D = Data()
        for i in tqdm(range(num_rolls), desc='Generating D_rand'):
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
    
    def __init__(self, K=10, H=15, softmax = False, temperature = 10, reinforce = False, **kwargs):
        super(MPCAgent, self).__init__(**kwargs)
        self.K = K
        self.H = H
        self.temperature = temperature
        self.softmax = softmax
        self.reward_oracle = RewardOracle(self.env)
        self.reinforce = reinforce
        if reinforce:
            self.log_probs = []

        
    def aggregate_data(self):
        if len(self.D_RL) > 0:
            self.D = AggregatedData(self.D_rand, self.D_RL)
        else:
            self.D = self.D_rand
        
    def choose_action(self, state):
        state = torch.tensor(state, requires_grad=True).float()
        trajectories = self.generate_trajectories(state)
        trajectory_scores = self.score_trajectories(trajectories).squeeze()
        probabilities = self._softmax(trajectory_scores)
        m = Categorical(probabilities.squeeze())
        if self.softmax:
            chosen_trajectory_idx = m.sample()   
        else:
            chosen_trajectory_idx = torch.argmax(probabilities)
        if self.reinforce:
            self.log_probs.append(m.log_prob(chosen_trajectory_idx))
        if using_gpu:
            action = trajectories[1][chosen_trajectory_idx].cpu().detach().numpy()
        else:
            action = trajectories[1][chosen_trajectory_idx].detach().numpy()
        
        return action
    
    def generate_trajectories(self, state):
        states = state.unsqueeze(0).repeat(self.K, 1) # matrix of size K * state_dimensions, each row is the state
        trajectories = [states, ]
        for _ in range(self.H):
            # sample actions
            actions = self.sample_batch_action(self.K)
            actions = torch.tensor(actions, requires_grad=True).float()
            if using_gpu:
                actions = actions.cuda().to(device)
                states = states.cuda().to(device)

            # infer with model
            if self.predict_rewards:
                s_diff, rewards = self.model(states, actions)
                assert torch.sum(torch.isnan(s_diff)) == 0
                assert torch.sum(torch.isnan(rewards)) == 0
            else:
                s_diff = None
                # TODO: this is probably broken now
                # rewards = self.reward_oracle.reward(self.unnormalise_state(states), self.unnormalise_action(actions))
                # rewards = self.normalise_reward(rewards)
                # s_diff = self.model(state_vars, action_vars).data.numpy()
            # update trajectory
            
            states = states + s_diff
            trajectories.append(actions)
            trajectories.append(rewards)
            trajectories.append(states)
            
        return trajectories

    def score_trajectories(self, trajectories):
        rewards = torch.zeros((self.K, 1), requires_grad=True)
        for reward_idx in range(2, len(trajectories), 3):
            rewards = rewards + trajectories[reward_idx]
        return rewards
    
    def _softmax(self, inputs):
        inputs = self.temperature*inputs
        return torch.nn.functional.softmax(inputs, 0)
    
    def train(self, num_epochs, iteration):
        self.aggregate_data()
        train_data = DataLoader(self.D, batch_size = 512, shuffle=False, pin_memory=True)
        
        for epoch in tqdm(range(num_epochs), desc='Fitting NN'):
            running_loss_state, running_loss_reward = 0, 0
            for i, (state, action, reward, state_diff) in enumerate(train_data):
                
                state = torch.tensor(state)
                action = torch.tensor(action)
                state_diff = torch.tensor(state_diff, requires_grad=False)
                reward = torch.tensor(reward.float(), requires_grad=False)
                
                if using_gpu:
                    state, action, state_diff, reward = state.cuda().to(device), action.cuda().to(device), state_diff.cuda().to(device), reward.cuda().to(device)

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
        for _ in range(8):
            state = torch.tensor(self.env.physics.state()).float()
            if using_gpu:
                state.to(device)
            trajectories = self.generate_trajectories(state)
            trajectory_scores = self.score_trajectories(trajectories)
            probabilities = self._softmax(trajectory_scores)
            m = Categorical(probabilities.squeeze())
            if self.softmax:
                chosen_trajectory_idx = m.sample()     
            else:
                chosen_trajectory_idx = torch.argmax(probabilities)
            actions = [trajectories[i][chosen_trajectory_idx] for i in range(1, len(trajectories), 3)]
            predicted_states = [trajectories[i][chosen_trajectory_idx] for i in range(0, len(trajectories), 3)][1:]
            predicted_rewards = [trajectories[i][chosen_trajectory_idx] for i in range(2, len(trajectories), 3)]
            if using_gpu:
                actions = [action.cpu().detach().numpy() for action in actions]
                predicted_states = [state.cpu().detach().numpy() for state in predicted_states]
            else:
                actions = [action.detach().numpy() for action in actions]
                predicted_states = [state.detach().numpy() for state in predicted_states]
            
            predicted_rewards = [float(reward) for reward in predicted_rewards]
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
    
    def REINFORCE(self, rewards):
        assert self.reinforce
        new_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + 0.99*R
            new_rewards.insert(0, R)
        
        rewards = torch.tensor(new_rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        
        policy_loss = []
        assert len(self.log_probs) == len(rewards)

        for log_prob, reward in tqdm(zip(self.log_probs, rewards), desc='REINFORCE', total = len(rewards)):
            policy_loss = -log_prob * reward
            policy_loss.to(device)
            self.optimizer_reinforce.zero_grad()
            policy_loss.backward()
            self.optimizer_reinforce.step()          
        self.log_probs = []
            
    