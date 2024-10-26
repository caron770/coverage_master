import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.mpe._mpe_utils.core import Agent
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 40,
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles, #最大循环次数
        render_mode=None,
        continuous_actions=False,
        local_ratio=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        #设置字体以及大小
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )
        """secrcode.ttf是字体文件，是矢量字体可以缩放"""

        # Set up the drawing window

        self.renderOn = False
        self._seed() #_下划线是内部属性

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)


        """
        这段代码主要定义动作空间的维度和观测空间的维度
        """
        # set spaces
        self.action_spaces = dict()     #初始化动作空间
        self.observation_spaces = dict()    #初始化观测字典空间
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:   #如果可以移动空间维度=空间的维度*2（速度和位置）+其他的附件状态
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:   #如果不能移动并且是连续空间维度
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:    #检查是否通信
                if self.continuous_actions:     #如果是连续的动作空间
                    space_dim += self.world.dim_c   #空间维度就加上通讯维度
                else:       #如果是离散动作空间
                    space_dim *= self.world.dim_c   #相乘：表示每个状态都有不同的通讯方式   

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, shape=(space_dim,)
                )
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        ).astype(np.float32)

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        """
        奖励函数相关的定义的东西
        如果local_ratio=0:则奖励函数就只和个体的奖励相关，就不和global_reward相关
        如果local_ratio!=0,则奖励函数就是只和reward相关，也是返回一个字典

        在我的这个实验中，全体的奖励函数占比比重还是稍微大一点
        我先设置全局奖励函数占比：0.6，local_ratio=0.4
        """
        global_reward = 0.0
        if self.local_ratio is not None:    #如果local_ratio(局部比例)
            global_reward = float(self.scenario.global_reward(self.world))

        #遍历所有智能体，获得相关的奖励函数
        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:    #local_ratio不为0，就计算局部与全局奖励函数综合奖励函数
                reward = (
                    global_reward * (1 - self.local_ratio)          
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            #得到一个奖励函数的字典
            self.rewards[agent.name] = reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                # Note: this ordering preserves the same movement direction as in the discrete case
                agent.action.u[0] += action[0][2] - action[0][1]
                agent.action.u[1] += action[0][4] - action[0][3]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        #这一段判断是否终结或者截至，如果有的话直接调用函数_was_dead_step(action)
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        
        #获取当前选取的智能体的名字和id
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()
        #当前智能体动作保存到列表之中
        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()
        #人类渲染模式
        if self.render_mode == "human":
            self.render()
    
    #检查渲染
    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True
    #提示没有设置渲染模式
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw() #画图
        #rgb模式
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        #画图渲染模式
        elif self.render_mode == "human":
            pygame.display.flip()   #用于刷新整个显示窗口
            self.clock.tick(self.metadata["render_fps"])    #设置渲染的帧数（每秒多少帧）
            return  #没有返回任何数据

    def draw(self):
        # 固定相机范围大小，确保画面不会缩小或者放大
        fixed_cam_range = 1.0  # 根据您的场景大小调整

        # 清屏
        self.screen.fill((255, 255, 255))

        # 创建一个字典来存储实体的屏幕坐标
        entity_screen_positions = {}

        # 更新几何和文本位置
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # 几何
            x, y = entity.state.p_pos
            y *= -1  # 翻转 y 轴

            # 将场景坐标映射到屏幕坐标
            x = (x / fixed_cam_range) * (self.width / 2)
            y = (y / fixed_cam_range) * (self.height / 2)

            # 将原点从屏幕左上角移动到屏幕中央
            x += self.width / 2
            y += self.height / 2

            # 存储屏幕坐标（不再进行坐标限制）
            entity_screen_positions[entity] = (x, y)

            # 绘制探测范围（半透明的蓝色虚线圆圈）
            if isinstance(entity, Agent):
                detection_radius = getattr(entity, 'detection_radius', 0)
                scaled_radius = (detection_radius / fixed_cam_range) * (self.width / 2)
                pygame.draw.circle(self.screen, (0, 0, 255, 50), (x, y), int(scaled_radius), 1)

            # 绘制智能体
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350
            )
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )

            # 文本（省略，与之前相同）

        # 绘制通信范围内的红线
        agents = [entity for entity in self.world.entities if isinstance(entity, Agent)]
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                # 获取翻转后的坐标
                pos1 = agent1.state.p_pos.copy()
                pos2 = agent2.state.p_pos.copy()
                pos1[1] *= -1
                pos2[1] *= -1

                # 计算距离
                distance = np.linalg.norm(pos1 - pos2)

                # 获取通信范围
                communication_range = agent1.communication_radius  # 假设通信范围相同

                if distance <= communication_range:
                    # 获取屏幕坐标
                    x1, y1 = entity_screen_positions[agent1]
                    x2, y2 = entity_screen_positions[agent2]
                    # 在两个智能体之间画一根红线
                    pygame.draw.line(self.screen, (255, 0, 0), (x1, y1), (x2, y2), 1)




    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
