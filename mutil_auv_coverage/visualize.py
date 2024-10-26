import matplotlib.pyplot as plt

def plot_rewards(rewards, save_path=None):
    """
    绘制训练过程中每个 episode 的奖励。
    
    :param rewards: list，记录每个 episode 的总奖励
    :param save_path: str，如果提供路径，将保存图像而不是显示
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards Over Episodes')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_losses(actor_losses, critic_losses, save_path=None):
    """
    绘制 actor 和 critic 的损失函数。
    
    :param actor_losses: list，记录每个 episode 的 actor 损失
    :param critic_losses: list，记录每个 episode 的 critic 损失
    :param save_path: str，如果提供路径，将保存图像而不是显示
    """
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses, label='Actor Loss')
    plt.plot(critic_losses, label='Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Actor and Critic Losses')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
