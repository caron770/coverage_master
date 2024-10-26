
import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env

from pettingzoo.utils.conversions import parallel_wrapper_fn

class MyScenario(BaseScenario):
    def make_world(self, N=5):
        world = World()
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True

        # Add agents    无人机智能体
        world.agents = [Agent() for _ in range(num_agents)] #放置智能体到一个列表里面，是一个类
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.2  # 设置机器人半径 R = 0.2m

        # Add landmarks tagret目标点
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_{i}"
            landmark.collide = False
            landmark.movable = False
        
        return world

    def reset_world(self, world, np_random):
    # 初始化每个智能体的颜色
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])  # 蓝色

    # 初始化每个地标的颜色
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])  # 灰色

    # 随机初始化智能体的位置
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, 1, world.dim_p)  #随机生成一个均匀的分布的数值，世界维度为2
            agent.state.p_vel = np.zeros(world.dim_p)  #表示初始时刻是是静止的
            agent.state.c = np.zeros(world.dim_c)  #表示智能体状态通信向量，此刻通信向量为0

    # 随机初始化地标的位置
        for landmark in world.landmarks:
            landmark.state.p_pos = np_random.uniform(-1, 1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    #奖励函数
    def reward(self, agent, world):
        rew = 0
        if agent.collide:
            for a in world.agents:
                rew -= 1.0 * (self.is_collision(a, agent) and a != agent)
        return rew

    #全局奖励函数
    def global_reward(self, world):
        rew = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        return rew
    
    #观察位置
    def observation(self, agent, world):
        # 获取所有地标和其他智能体相对于当前智能体的位置
        entity_pos = [landmark.state.p_pos - agent.state.p_pos for landmark in world.landmarks]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent]
        return np.concatenate([agent.state.p_vel, agent.state.p_pos] + entity_pos + other_pos)

class raw_env(SimpleEnv, EzPickle):
    def __init__(self, N=5, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode=None):
        #父类初始化
        EzPickle.__init__(self, N=N, local_ratio=local_ratio, max_cycles=max_cycles, continuous_actions=continuous_actions, render_mode=render_mode)
        
        # 创建场景和世界
        scenario = MyScenario()
        world = scenario.make_world(N=N)
        
        # 将所有参数一致地传递给 SimpleEnv
        SimpleEnv.__init__(self, scenario, world, max_cycles, continuous_actions, local_ratio, render_mode)
        self.metadata["name"] = "simple_auv"
env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

"""
这是第二个版本，每个无人机带有一个探测的范围，蓝色虚线
这个是用静态版本画的图像就是matplotlib画的图像
之后动态的图像还是没有相关的动态探测范围

"""

