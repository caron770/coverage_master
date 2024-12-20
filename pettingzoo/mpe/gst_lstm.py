import torch
import torch.nn as nn
import torch.nn.functional as F


class GST_LSTM(nn.Module):
    """
    GST-LSTM 模块：结合图卷积和标准 LSTM 捕获多智能体系统的空间结构和时间依赖。

    输入:
        - ht: 当前时间步的输入特征 (num_robots, input_dim)
        - Ht_1: 上一时间步的隐藏状态 (num_robots, hidden_dim)
        - Ct_1: 上一时间步的单元记忆 (num_robots, hidden_dim)
        - adj_matrix: 当前时间步的邻接矩阵 (num_robots, num_robots)

    输出:
        - Ht: 当前时间步的隐藏状态 (num_robots, hidden_dim)
        - Ct: 当前时间步的单元记忆 (num_robots, hidden_dim)
    """

    def __init__(self, input_dim, hidden_dim):
        super(GST_LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # 定义 LSTM 的各个门控的权重映射
        self.Wxi = nn.Linear(input_dim, hidden_dim)  # 输入门
        self.Whi = nn.Linear(hidden_dim, hidden_dim)
        self.Wxf = nn.Linear(input_dim, hidden_dim)  # 遗忘门
        self.Whf = nn.Linear(hidden_dim, hidden_dim)
        self.Wxo = nn.Linear(input_dim, hidden_dim)  # 输出门
        self.Who = nn.Linear(hidden_dim, hidden_dim)
        self.Wxc = nn.Linear(input_dim, hidden_dim)  # 单元记忆更新
        self.Whc = nn.Linear(hidden_dim, hidden_dim)

    def graph_conv(self, x, adj_matrix):
        """
        图卷积操作：将当前状态与邻居状态通过邻接矩阵进行聚合
        输入:
            - x: 节点特征 (num_robots, hidden_dim)
            - adj_matrix: 邻接矩阵 (num_robots, num_robots)
        输出:
            - 聚合后的特征 (num_robots, hidden_dim)
        """
        return torch.matmul(adj_matrix, x)  # 通过邻接矩阵聚合邻居状态

    def forward(self, ht, Ht_1, Ct_1, adj_matrix):
        """
        前向传播
        """
        # 输入门
        it = torch.sigmoid(self.Wxi(ht) + self.Whi(self.graph_conv(Ht_1, adj_matrix)))
        # 遗忘门
        ft = torch.sigmoid(self.Wxf(ht) + self.Whf(self.graph_conv(Ht_1, adj_matrix)))
        # 输出门
        ot = torch.sigmoid(self.Wxo(ht) + self.Who(self.graph_conv(Ht_1, adj_matrix)))
        # 单元记忆更新
        ut = torch.tanh(self.Wxc(ht) + self.Whc(self.graph_conv(Ht_1, adj_matrix)))
        Ct = ft * Ct_1 + it * ut  # 更新单元记忆
        # 隐藏状态更新
        Ht = ot * torch.tanh(Ct)

        return Ht, Ct
