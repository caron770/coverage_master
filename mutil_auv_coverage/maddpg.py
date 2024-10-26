import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# Actor-Critic 网络定义
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # 假设动作在 [-1, 1] 范围内

class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def add(self, obs, actions, rewards, next_obs, dones):
        # 将经验保存为一个字典
        experience = {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones
        }
        self.memory.append(experience)
        # 确保缓冲区不超过其大小
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def sample(self):
        # 从 Replay Buffer 中随机采样 batch_size 个样本
        experiences = random.sample(self.memory, k=self.batch_size)

        # 获取所有可能的智能体列表
        agents = set()
        for e in experiences:
            agents.update(e["obs"].keys())

        # 转换为列表
        agents = list(agents)

        # 提取每个经验中的各个部分
        obs = {agent: [] for agent in agents}
        actions = {agent: [] for agent in agents}
        rewards = {agent: [] for agent in agents}
        next_obs = {agent: [] for agent in agents}
        dones = {agent: [] for agent in agents}

        # 遍历每个经验，将数据逐个添加到对应的字典中
        for e in experiences:
            for agent in agents:
                obs[agent].append(e["obs"][agent])
                actions[agent].append(e["actions"][agent])
                rewards[agent].append(e["rewards"][agent])
                next_obs[agent].append(e["next_obs"][agent])
                dones[agent].append(e["dones"][agent])

        # 将列表转换为 NumPy 数组，然后再转换为张量
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs = {agent: torch.tensor(np.array(obs[agent]), dtype=torch.float32).to(device) for agent in agents}
        actions = {agent: torch.tensor(np.array(actions[agent]), dtype=torch.float32).to(device) for agent in agents}
        rewards = {agent: torch.tensor(np.array(rewards[agent]), dtype=torch.float32).to(device) for agent in agents}
        next_obs = {agent: torch.tensor(np.array(next_obs[agent]), dtype=torch.float32).to(device) for agent in agents}
        dones = {agent: torch.tensor(np.array(dones[agent]), dtype=torch.float32).to(device) for agent in agents}

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.memory)


# MADDPG 算法
class MADDPG:
    def __init__(self, agents, obs_dims, action_dims, actor_lr, critic_lr, gamma, tau):
        self.agents = agents  # 智能体列表
        self.gamma = gamma
        self.tau = tau
        self.batch_size = 1024  # 您可以根据需要调整

        # 初始化 Actor 和 Critic 网络
        self.actors = {}
        self.critics = {}
        self.target_actors = {}
        self.target_critics = {}

        # 初始化优化器
        self.actor_optimizers = {}
        self.critic_optimizers = {}

        total_obs_dim = sum(obs_dims.values())
        total_action_dim = sum(action_dims.values())

        for agent in agents:
            obs_dim = obs_dims[agent]
            action_dim = action_dims[agent]

            # Actor 网络和目标网络
            self.actors[agent] = Actor(obs_dim, action_dim)
            self.target_actors[agent] = Actor(obs_dim, action_dim)
            self.target_actors[agent].load_state_dict(self.actors[agent].state_dict())

            # Critic 网络和目标网络
            self.critics[agent] = Critic(total_obs_dim, total_action_dim)
            self.target_critics[agent] = Critic(total_obs_dim, total_action_dim)
            self.target_critics[agent].load_state_dict(self.critics[agent].state_dict())

            # 优化器
            self.actor_optimizers[agent] = torch.optim.Adam(self.actors[agent].parameters(), lr=actor_lr)
            self.critic_optimizers[agent] = torch.optim.Adam(self.critics[agent].parameters(), lr=critic_lr)

    def to(self, device):
        for agent in self.agents:
            self.actors[agent] = self.actors[agent].to(device)
            self.critics[agent] = self.critics[agent].to(device)
            self.target_actors[agent] = self.target_actors[agent].to(device)
            self.target_critics[agent] = self.target_critics[agent].to(device)

    def update(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return None, None

        # 从 Replay Buffer 中采样
        obs, actions, rewards, next_obs, dones = replay_buffer.sample()

        total_actor_loss = 0
        total_critic_loss = 0

        # 对每个智能体进行更新
        for agent in self.agents:
            # 获取当前智能体的网络和优化器
            actor = self.actors[agent]
            critic = self.critics[agent]
            target_actor = self.target_actors[agent]
            target_critic = self.target_critics[agent]
            actor_optimizer = self.actor_optimizers[agent]
            critic_optimizer = self.critic_optimizers[agent]

            # 计算当前 Q 值
            all_obs = torch.cat([obs[ag] for ag in self.agents], dim=-1)
            all_actions = torch.cat([actions[ag] for ag in self.agents], dim=-1)
            current_Q = critic(all_obs, all_actions)

            # 计算目标 Q 值
            target_actions = []
            for ag in self.agents:
                target_act = target_actor if ag == agent else self.target_actors[ag]
                target_actions.append(target_act(next_obs[ag]))
            target_actions = torch.cat(target_actions, dim=-1)

            all_next_obs = torch.cat([next_obs[ag] for ag in self.agents], dim=-1)
            target_Q = target_critic(all_next_obs, target_actions)
            y = rewards[agent].view(-1, 1) + self.gamma * target_Q * (1 - dones[agent].view(-1, 1))

            # 计算 Critic 损失并更新 Critic 网络
            critic_loss = F.mse_loss(current_Q, y.detach())
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            total_critic_loss += critic_loss.item()

            # 计算 Actor 损失并更新 Actor 网络
            current_actions = []
            for ag in self.agents:
                if ag == agent:
                    current_actions.append(actor(obs[ag]))
                else:
                    current_actions.append(actions[ag].detach())
            current_actions = torch.cat(current_actions, dim=-1)

            all_obs = torch.cat([obs[ag] for ag in self.agents], dim=-1)
            actor_loss = -critic(all_obs, current_actions).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            total_actor_loss += actor_loss.item()

            # 软更新目标网络
            self.soft_update(actor, target_actor)
            self.soft_update(critic, target_critic)

        # 返回平均的 Actor 和 Critic 损失
        avg_actor_loss = total_actor_loss / len(self.agents)
        avg_critic_loss = total_critic_loss / len(self.agents)

        return avg_actor_loss, avg_critic_loss

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
