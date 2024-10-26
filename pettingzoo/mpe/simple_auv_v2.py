"""
这个版本就是修改了获得观察，这里面直接按照原论文的东西修改的：
具体修改添加了gat_obsveration.py这个观察函数，这个就是机器人对目标点进行注意力机制，机器人和
邻居节点之间进行注意力机制
"""

import numpy as np
import matplotlib.pyplot as plt
from gymnasium.utils import EzPickle
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe.gat_observation import gat_observation
from pettingzoo.mpe.interaction_attention_observation import interaction_attention_observation

import torch
import networkx as nx

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N_drones=5,
        N_targets=10,
        coverage_radius=0.5,
        #local_ratio=0.4, #可以理解为这个参数是平衡全局奖励和单个智能体的奖励
        max_cycles=100,
        continuous_actions=True,    #定义连续或者离散动作空间
        render_mode=None,
    ):
        # 使用EzPickle来处理序列化
        EzPickle.__init__(
            self,
            N_drones=N_drones,
            N_targets=N_targets,
            coverage_radius=coverage_radius,
            #local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )

        # 检查 local_ratio 的范围
        #assert 0.0 <= local_ratio <= 1.0, "local_ratio must be a proportion between 0 and 1."

        # 正确实例化场景
        scenario = Scenario(
            N_drones=N_drones, N_targets=N_targets, coverage_radius=coverage_radius,
        )

        # 创建世界对象
        world = scenario.make_world()

        # 调用父类SimpleEnv的初始化方法，移除dynamic_rescaling参数
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            #local_ratio=local_ratio,
        )

        # 设置元数据字段
        self.metadata["name"] = "drone_coverage_env"

# 创建和封装环境
env = make_env(raw_env)  # 创建标准环境
# 将环境转换成一个并行环境（适用于并行化的多智能体强化学习训练）
parallel_env = parallel_wrapper_fn(env)

class Scenario(BaseScenario):
    def __init__(self, N_drones, N_targets, coverage_radius):
        self.N_drones = N_drones
        self.N_targets = N_targets
        self.coverage_radius = coverage_radius
        self.communication_radius = 1  # 设置无人机的通信范围
        self.is_drone_env = True    #设置一下这个是无人机的环境

    def make_world(self):
        world = World()
        # 世界属性定义
        world.dim_c = 2  # 通信维度
        world.collaborative = True  # 是否是合作型任务

        # 添加无人机（智能体）
        world.agents = [Agent() for _ in range(self.N_drones)]
        for i, agent in enumerate(world.agents):
            agent.name = f"drone_{i}"
            agent.collide = True  # 允许检测碰撞
            agent.silent = True  # 禁用通信
            agent.size = 0.1  # 无人机大小
            """
            设置无人机的速度和加速度
            """
            agent.accel=5.0     #加速度大小  这个是通过公式计算出来的
            agent.max_speed=0.5     #无人机最大速度大小

            # agent.color = np.array([0.35, 0.35, 0.85])  # 无人机颜色蓝色
            agent.color = np.array([1.0, 1.0, 0.6])  # 浅黄色

            agent.communication_radius=1   #无人机的通讯范围
            
            agent.detection_radius = 0.5  # 无人机的探测范围

        # 添加地面目标
        world.landmarks = [Landmark() for _ in range(self.N_targets)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"target_{i}"
            landmark.collide = True  # 允许目标和无人机碰撞
            landmark.movable = False  # 目标不可移动
            landmark.size = 0.05
            landmark.boundary=False
            landmark.color = np.array([0.25, 0.25, 0.25])  # 目标颜色黑色
            # landmark.color = np.array([0.35, 0.35, 0.85]) #蓝色


        return world

    # 定义global_reward函数，计算全局覆盖率的奖励
    def global_reward(self,world):
        # Step 1: 计算代数连通度
        G = nx.Graph()
        for i, agent in enumerate(world.agents):
            G.add_node(i)
        for i, agent in enumerate(world.agents):
            for j, other_agent in enumerate(world.agents):
                if i >= j:
                    continue
                dist = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
                if dist <= getattr(agent, 'communication_radius', 1.0):
                    G.add_edge(i, j)

        λ2 = nx.algebraic_connectivity(G)

        # Step 2: 计算连通性奖励 (r_c)
        if λ2 == 0:
            r_c = -10
        elif λ2 < 0.2:
            r_c = -1
        else:
            r_c = 0

        # Step 3: 计算覆盖和距离奖励 (r_s^d)
        L = len(world.landmarks)  # 地标数量
        L_c = 0  # 未被覆盖的地标数量

        # 计算地标是否被覆盖
        for landmark in world.landmarks:
            covered = False
            for agent in world.agents:
                dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                if dist <= getattr(agent, 'detection_radius', 0.5):
                    covered = True
                    break
            if not covered:
                L_c += 1

        uncovered_rate = L_c / L

        # 计算每个地标与所有智能体的最小距离
        min_distances = []
        for landmark in world.landmarks:
            distances = [np.linalg.norm(agent.state.p_pos - landmark.state.p_pos) for agent in world.agents]
            min_distances.append(min(distances))

        avg_min_distance = np.mean(min_distances)
        r_s_d = -uncovered_rate * avg_min_distance

        # Step 4: 计算复合奖励 (r^1)
        k_1 = 5  # 目覆盖权重系数，可以根据需求调整
        r_1 = r_c + k_1 * r_s_d

        return r_1

    def reward(self, agent, world):
        rew = 0

        # Step 1: 计算与地标的距离 (奖励靠近地标)
        for landmark in world.landmarks:
            dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            rew -= 0.3*dist  # 距离越小，奖励越大（负数代表靠近时减少惩罚）

        # Step 2: 计算碰撞惩罚
        if agent.collide:
            for other_agent in world.agents:
                if other_agent == agent:
                    continue
                if np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos) < (agent.size + other_agent.size):
                    rew -= 2  # 如果发生碰撞，给予惩罚

        # Step 3: 计算与邻居智能体的通信连通性（可选）
        # 可以在局部范围内，确保与邻居的通信
        communication_radius = getattr(agent, 'communication_radius', 1.0)
        connected_agents = 0
        for other_agent in world.agents:
            if other_agent == agent:
                continue
            dist = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
            if dist <= communication_radius:
                connected_agents += 1
        rew += 0.2 * connected_agents  # 每个连接的邻居给予0.2的奖励

        # Step 4: 碰撞边界的惩罚
        boundary_limit = 1.0  # 假设边界范围是从 -1.0 到 1.0
        if np.any(agent.state.p_pos < -boundary_limit) or np.any(agent.state.p_pos > boundary_limit):
            rew -= 1.0  # 如果超出边界范围，给予较大的惩罚

        return rew
    

    def reset_world(self, world, np_random):
        # 随机设置无人机和目标的初始位置

        # 固定设置 10 个目标的位置
        fixed_positions = [
            [-0.8, -0.8], [-0.9, 0.6], [-0.8, 0.2], [0.8, -0.8],
            [0, 0.8], [0.4, 0.4], [-0.8, 0], [0, -0.5],
            [-0.5, 0.5], [0.5, -0.2]
        ]

        for i, landmark in enumerate(world.landmarks):
            if i < len(fixed_positions):
                landmark.state.p_pos = np.array(fixed_positions[i])
            else:
                # 如果有多于 10 个 landmark，默认设置为零向量
                landmark.state.p_pos = np.zeros(world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)


    # 检查两个无人机是否发生碰撞
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def observation(self, agent, world):
        """
        通过 GAT 和交互注意力机制生成观测值
        自身：4维
        聚合邻居特征：32维
        聚合目标特征向量;32维
        """
         # Step 1: 使用 GAT 获取自身状态、聚合的邻居特征和目标特征
        own_state, aggregated_neighbors, aggregated_targets =gat_observation(self, agent, world)

        # 拼接自身状态、目标特征和邻居特征
        agent_index = next(i for i, a in enumerate(world.agents) if a is agent)
        combined_features = torch.tensor(np.concatenate([own_state, aggregated_targets[agent_index].detach().numpy(), aggregated_neighbors[agent_index].detach().numpy()]), dtype=torch.float)

        # Step 2: 使用交互注意力机制处理拼接后的特征
        E_i = interaction_attention_observation(combined_features)

        # Step 3: 全连接层生成最终固定长度的观测向量
        """
        combined_features.shape[0] 是 4+32+32=68，即拼接后的特征维度。
        全连接层将输入从 68 维映射到固定的 32 维，
        这意味着无论输入的邻居数量和特征如何变化，
        最终的输出维度始终是 32。
        """
        fc_layer = torch.nn.Linear(combined_features.shape[0], 32)
        final_observation = fc_layer(combined_features)

        return final_observation.detach().numpy()

    

    