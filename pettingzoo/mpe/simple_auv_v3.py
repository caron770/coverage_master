
"""
这个就是正常版本，就是没有定义gat,并且信息也是全局知道的，这里面只是修改了奖励函数
其他东西也都是按照正常的环境设置的
"""
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.utils import EzPickle
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N_drones=5,
        N_targets=10,
        coverage_radius=0.5,
        #local_ratio=0.5, #可以理解为这个参数是平衡全局奖励和单个智能体的奖励
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
           # local_ratio=local_ratio,
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
           # local_ratio=local_ratio,
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

    def make_world(self):
        world = World()
        # 世界属性定义
        world.dim_c = 2  # 通信维度
        world.collaborative = True  # 是否是合作型任务

        # 添加无人机（智能体）
        world.agents = [Agent() for _ in range(self.N_drones)]
        for i, agent in enumerate(world.agents):
            agent.name = f"drone_{i}"
            agent.collide = False  # 允许碰撞
            agent.silent = True  # 表示无人机之间为静默状态
            agent.size = 0.05  # 无人机大小

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
            landmark.size = 0.02
            landmark.boundary=False
            landmark.color = np.array([0.25, 0.25, 0.25])  # 目标颜色

        return world

    # 定义global_reward函数，计算全局覆盖率的奖励
    def global_reward(self, world):
        covered_targets = set()  # 使用集合避免重复计算覆盖的目标

        # 遍历所有无人机，计算它们覆盖的目标
        for a in world.agents:
            for lm in world.landmarks:
                dist = np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                if dist <= self.coverage_radius:  # 如果距离在覆盖半径内，认为该目标被覆盖
                    covered_targets.add(lm.name)  # 记录目标已经被覆盖

        # 计算总覆盖的目标数
        total_covered = len(covered_targets)

        # 计算覆盖率
        coverage_rate = total_covered / self.N_targets

        # 全局奖励可以基于覆盖率来定义
        return coverage_rate * 10  # 根据覆盖率返回奖励

    def reset_world(self, world, np_random):
        # 随机设置无人机和目标的初始位置
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

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

    def reward(self, agent, world):
        covered_targets = set()  # 使用集合避免重复计算覆盖的目标
        boundary_penalty = -20  # 边界惩罚
        overlap_penalty = -2  # 重复覆盖惩罚
        communication_penalty = -5  # 失去通信的惩罚
        collision_penalty = -10  # 碰撞惩罚
        reward = 0  # 初始化奖励

        # 目标覆盖奖励
        for lm in world.landmarks:
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
            if dist <= self.coverage_radius:  # 如果距离在覆盖半径内，认为该目标被覆盖
                if lm.name not in covered_targets:
                    reward += 10  # 奖励覆盖目标
                    covered_targets.add(lm.name)
                else:
                    reward += overlap_penalty  # 如果目标已经被覆盖，给予重复覆盖惩罚

        # 边界惩罚
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            reward -= bound(x) * boundary_penalty

        # 检查通信连接
        connected = False
        for a in world.agents:
            if a != agent:
                comm_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - a.state.p_pos)))
                if comm_dist <= self.communication_radius:
                    connected = True
                    break

        if not connected:
            reward += communication_penalty  # 如果失去通信连接，给予负奖励

        # 碰撞惩罚
        for other_agent in world.agents:
            if other_agent != agent:
                if self.is_collision(agent, other_agent):
                    reward += collision_penalty  # 如果与其他无人机发生碰撞，给予负奖励

        return reward

    # 检查两个无人机是否发生碰撞
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    # 获取无人机的观测值
    def observation(self, agent, world):
        entity_pos = []
        #获取目标点位置
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)  # 相对位置
        comm = []
        other_pos = []
        #获取其他智能体信息
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)  # 其他无人机的位置
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
