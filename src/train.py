import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from collections import deque
import random
import os

class ProjectAgent:
    def __init__(self, config=None, model=None):
        self.model = model if model is not None else DqnAgent()
        self.config = config if config is not None else {
            'nb_actions': env.action_space.n,
            'learning_rate': 0.001,
            'gamma': 0.95,
            'buffer_size': 100000,
            'epsilon_min': 0.01,
            'epsilon_max': 1.,
            'epsilon_decay_period': 20000,
            'epsilon_delay_decay': 20,
            'batch_size': 512,
            'gradient_steps': 3,
            'update_target_strategy': 'ema',
            'update_target_freq': 50,
            'update_target_tau': 0.005,
            'criterion': torch.nn.SmoothL1Loss(),
            'prioritized_replay_alpha': 0.6,
            'prioritized_replay_beta': 0.4,
            'prioritized_replay_eps': 1e-6,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.memory = PrioritizedReplayBuffer(self.config['buffer_size'], self.config['prioritized_replay_alpha'], self.device)
        self.epsilon_max = self.config['epsilon_max']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay_period']
        self.epsilon_delay = self.config['epsilon_delay_decay']
        self.batch_size = self.config['batch_size']
        self.gamma = self.config['gamma']
        self.gradient_steps = self.config['gradient_steps']
        self.update_target_strategy = self.config['update_target_strategy']
        self.update_target_freq = self.config['update_target_freq']
        self.update_target_tau = self.config['update_target_tau']
        self.criterion = self.config['criterion']
        self.prioritized_replay_eps = self.config['prioritized_replay_eps']
        self.prioritized_replay_beta = self.config['prioritized_replay_beta']
        self.best_score = 0
        self.nb_episodes = 500

    def act(self, observation, use_random=False):
        if use_random:
            a = env.action_space.sample()
        else:
            device = self.device
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
                a = torch.argmax(Q).item()
        return a

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
            'config': self.config
        }, path)

    def load(self):
        path = "agent.pth"
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_score = checkpoint['best_score']
        self.config = checkpoint['config']
        self.target_model = deepcopy(self.model).to(self.device)
        self.target_model.eval()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D, weights, batch_idxes = self.memory.sample(self.batch_size, self.prioritized_replay_beta)

            # Double DQN Update
            with torch.no_grad():
                argmax_a = torch.argmax(self.model(Y), dim=1)
                update = torch.addcmul(R, 1 - D, self.target_model(Y).gather(1, argmax_a.unsqueeze(1)).squeeze(1), value=self.gamma)

            # Prediction
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1)).squeeze(1)

            # Compute TD errors and update priorities
            td_errors = (update - QXA).abs().detach().cpu().numpy()
            self.memory.update_priorities(batch_idxes, td_errors + self.prioritized_replay_eps)

            # Loss calculation with importance sampling weights
            loss = (weights * self.criterion(QXA, update)).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_score = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-((self.epsilon_max-self.epsilon_min)/self.epsilon_decay))
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = self.act(state, use_random=True)
            else:
                action = self.act(state, use_random=False)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.gradient_steps):
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                if episode_cum_reward > best_score:
                  best_score = episode_cum_reward
                  self.save('agent.pth')
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", episode return ", '{:.2e}'.format(episode_cum_reward),
                      ", best score ", '{:.2e}'.format(best_score),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return

class DqnAgent(nn.Module):
    def __init__(self):
        super(DqnAgent, self).__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)
        self.device = device
    def append(self, state, action, reward, next_state, done):
        self.data.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device),list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, device):
        self.capacity = capacity
        self.alpha = alpha
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, done):
        priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return (*map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*samples))), weights, indices)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

seed_everything(seed=42)
    
env = TimeLimit(
env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

if __name__ == "__main__":
    config = {'nb_actions': env.action_space.n,
                'learning_rate': 0.001,
                'gamma': 0.90,
                'buffer_size': 20000,
                'epsilon_min': 0.05,
                'epsilon_max': 1.0,
                'epsilon_decay_period': 70000,
                'epsilon_delay_decay': 2000,
                'batch_size': 200,
                'gradient_steps': 2,
                'update_target_strategy': 'ema',
                'update_target_freq': 60,
                'update_target_tau': 0.005,
                'criterion': torch.nn.SmoothL1Loss(),
                'prioritized_replay_alpha': 0.6,
                'prioritized_replay_beta': 0.4,
                'prioritized_replay_eps': 1e-6,
            }
    
    agent = ProjectAgent(config, DqnAgent())
    episode_return = agent.train(env, 400)