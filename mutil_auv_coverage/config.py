# CONFIG = {
#     "actor_lr": 1e-3,          # Actor 网络的学习率
#     "critic_lr": 1e-3,         # Critic 网络的学习率
#     "gamma": 0.95,             # 折扣因子
#     "tau": 0.01,               # 目标网络软更新参数
#     "buffer_size": 1000000,     # Replay Buffer 大小
#     "batch_size": 512,         # Batch 大小
#     "num_episodes": 5000,      # 总训练 episode 数
#     "max_steps_per_episode":64,  # 每个 episode 的最大步数
#     "seed": 40,                # 随机种子
#     "render_mode": "None",    # 渲染模式   #human渲染
# }

CONFIG = {
    "actor_lr": 1e-3,          # Actor 网络的学习率
    "critic_lr": 1e-2,         # Critic 网络的学习率
    "gamma": 0.95,             # 折扣因子
    "tau": 0.01,               # 目标网络软更新参数
    "buffer_size": 1000000,     # Replay Buffer 大小
    "batch_size": 1024,         # Batch 大小
    "num_episodes": 10000,      # 总训练 episode 数
    "max_steps_per_episode": 128,  # 每个 episode 的最大步数
    "seed": 40,                # 随机种子
    "render_mode": "human",    # 渲染模式   #human渲染
}