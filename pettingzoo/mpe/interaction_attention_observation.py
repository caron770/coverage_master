import torch
import torch.nn.functional as F
import numpy as np
# from pettingzoo.mpe.gat_observation import gat_observation  # 导入之前定义的 GAT 模块

class AttentionInteraction(torch.nn.Module):
    def __init__(self, input_dim, attention_dim, num_heads=4):
        super(AttentionInteraction, self).__init__()
        self.num_heads = num_heads
        # 定义 Query、Key 和 Value 的线性变换，用于多头注意力
        self.W_q = torch.nn.Linear(input_dim, attention_dim * num_heads)
        self.W_k = torch.nn.Linear(input_dim, attention_dim * num_heads)
        self.W_v = torch.nn.Linear(input_dim, attention_dim * num_heads)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, h_neighbors):
        # 计算查询、键、值
        Q = self.W_q(h_neighbors)
        K = self.W_k(h_neighbors)
        V = self.W_v(h_neighbors)

        # 拆分为多头，每个维度是 (num_neighbors, num_heads, attention_dim)
        Q = Q.view(Q.shape[0], self.num_heads, -1)
        K = K.view(K.shape[0], self.num_heads, -1)
        V = V.view(V.shape[0], self.num_heads, -1)

        # 计算注意力系数 (num_heads, num_neighbors, num_neighbors)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.shape[-1])
        attention_weights = self.softmax(attention_scores)

        # 计算相互注意力表示 E_i (num_heads, num_neighbors, attention_dim)
        E_i = torch.matmul(attention_weights, V)

        # 合并多头 (num_neighbors, num_heads * attention_dim)
        E_i = E_i.view(E_i.shape[0], -1)
        return E_i


"""
详细说明一下这里面的输入和输出：

输入：这里面的输入是来自gat全连接输出，是一个（num_neighbors,32)维的张量
    这里面的输入是32维
输出：这里面的每个头输出维度为8，这里面使用了四个heads，因此维度被扩展为8*4=32
"""
def interaction_attention_observation(combined_features):
    """
    使用交互注意力机制处理 GAT 的输出
    输入:
    - combined_features: GAT 输出的全连接向量（包括自身状态、目标状态、邻居状态）
    输出:
    - E_i: 聚合后的交互注意力特征
    """
   # 使用交互注意力机制
    attention_interaction = AttentionInteraction(input_dim=combined_features.shape[-1], attention_dim=8, num_heads=4)
    combined_tensor = combined_features.unsqueeze(0)
    E_i = attention_interaction(combined_tensor).squeeze(0).detach().numpy()

    return E_i
