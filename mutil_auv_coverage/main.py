import torch
from pettingzoo.mpe import simple_auv_v2
from maddpg import MADDPG, ReplayBuffer
from utils import OUNoise
import numpy as np
from config import CONFIG
from visualize import plot_rewards, plot_losses

def main():
    render_mode = CONFIG["render_mode"]
    num_episodes = CONFIG["num_episodes"]
    max_steps_per_episode = CONFIG["max_steps_per_episode"]
    seed = CONFIG["seed"]

    # 检查 GPU 可用性并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化并行环境
    env = simple_auv_v2.parallel_env(render_mode=render_mode)
    obs, _ = env.reset(seed=seed)

    # 获取智能体列表
    agents = env.agents  # ['drone_0', 'drone_1', ...]

    # 获取观测和动作空间维度
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]

    # 创建观测和动作维度的字典
    obs_dims = {agent: obs_dim for agent in agents}
    action_dims = {agent: action_dim for agent in agents}

    # 初始化 MADDPG 并将模型移动到 GPU
    maddpg = MADDPG(
        agents=agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        actor_lr=CONFIG["actor_lr"],
        critic_lr=CONFIG["critic_lr"],
        gamma=CONFIG["gamma"],
        tau=CONFIG["tau"]
    )
    maddpg.to(device)  # 将 MADDPG 网络移动到 GPU

    replay_buffer = ReplayBuffer(buffer_size=CONFIG["buffer_size"], batch_size=CONFIG["batch_size"])
    noise = {agent: OUNoise(action_dim) for agent in agents}  # 为每个智能体创建噪声对象

    total_rewards = []  # 记录每个 episode 的总奖励
    actor_losses = []  # 记录每个 episode 的 actor 损失
    critic_losses = []  # 记录每个 episode 的 critic 损失

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed)
        episode_reward = 0

        for step in range(max_steps_per_episode):
            actions = {}
            for agent in agents:
                # 如果智能体已经完成，跳过
                if agent not in obs:
                    continue

                # 获取该智能体的观测数据
                observation = obs[agent]
                observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度

                # 使用 Actor 网络生成动作
                action = maddpg.actors[agent](observation_tensor).detach().cpu().numpy().squeeze(0)  # 去除批次维度

                # 使用噪声增加探索
                action += noise[agent].noise()
                # 在添加噪声后进行裁剪 确保在合法范围之内
                action = np.clip(action, 0.0, 1.0)

                # 存储动作
                actions[agent] = np.clip(action, -1, 1)  # 根据动作空间调整范围

            # 执行所有智能体的动作并获取反馈
            next_obs, rewards, dones, truncated, infos = env.step(actions)

            # 记录奖励
            episode_reward += sum(rewards.values())

            # 将经验添加到 Replay Buffer
            # 如果某个智能体在 dones 中不存在，需要补充默认值
            all_agents = set(agents) | set(obs.keys()) | set(next_obs.keys())
            obs_filled = {agent: obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
            actions_filled = {agent: actions.get(agent, np.zeros(action_dim)) for agent in all_agents}
            rewards_filled = {agent: rewards.get(agent, 0.0) for agent in all_agents}
            next_obs_filled = {agent: next_obs.get(agent, np.zeros(obs_dim)) for agent in all_agents}
            dones_filled = {agent: dones.get(agent, True) for agent in all_agents}

            replay_buffer.add(obs_filled, actions_filled, rewards_filled, next_obs_filled, dones_filled)

            # 更新观测
            obs = next_obs

            # 更新 MADDPG 网络
            actor_loss, critic_loss = maddpg.update(replay_buffer)
            if actor_loss is not None and critic_loss is not None:
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            # 检查是否所有智能体都完成
            if all([dones.get(agent, True) for agent in agents]):
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward}")

    env.close()

    # 训练结束后，绘制奖励曲线和损失曲线
    plot_rewards(total_rewards)
    plot_losses(actor_losses, critic_losses)

if __name__ == "__main__":
    main()
