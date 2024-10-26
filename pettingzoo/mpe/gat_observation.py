import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import networkx as nx
import numpy as np


# 临时解决 OpenMP 警告（不推荐，建议优先解决库冲突问题）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class GATModule(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GATModule, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        # 检查 edge_index 是否为空或维度不正确
        if edge_index.numel() == 0 or edge_index.dim() != 2 or edge_index.size(0) != 2:
            # 返回一个全零的张量，维度与输出一致
            return torch.zeros(x.size(0), self.gat2.out_channels).to(x.device)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x


def gat_observation(self,agent, world):
    """
    使用两个 GAT 进行特征聚合的观测函数
    """
    # Step 1: 构建自身状态特征
    own_state = np.concatenate([agent.state.p_pos, agent.state.p_vel])

    # Step 2: 构建邻居和目标点的特征
    input_features_neighbors = []
    input_features_targets = []

    # 为每个智能体生成特征（位置和速度）
    for agent_in_world in world.agents:
        features = np.concatenate([agent_in_world.state.p_pos, agent_in_world.state.p_vel])
        input_features_neighbors.append(features)

    # 为每个目标点生成特征（位置）
    for landmark in world.landmarks:
        input_features_targets.append(landmark.state.p_pos)

    # 将输入特征列表转换为单一 numpy 数组
    input_features_neighbors_np = np.array(input_features_neighbors)
    input_features_targets_np = np.array(input_features_targets)

    # 将 numpy 数组转换为张量
    x_neighbors = torch.tensor(input_features_neighbors_np, dtype=torch.float)
    x_targets = torch.tensor(input_features_targets_np, dtype=torch.float)

    # Step 3: 构建邻接关系 (考虑 k-hop, k=3)
    k_hop = 3  # k-hop 跳步
    G_neighbors = nx.Graph()
    for i, agent_in_world in enumerate(world.agents):
        G_neighbors.add_node(i)
    for i, agent_in_world in enumerate(world.agents):
        for j, other_agent in enumerate(world.agents):
            if i >= j:
                continue
            dist = np.linalg.norm(agent_in_world.state.p_pos - other_agent.state.p_pos)
            communication_radius = getattr(agent_in_world, 'communication_radius', 1.0)
            if dist <= communication_radius:
                G_neighbors.add_edge(i, j)

    # 获取 k-hop 邻接矩阵
    G_k_hop = nx.Graph()
    for node in G_neighbors.nodes:
        k_hop_neighbors = nx.single_source_shortest_path_length(G_neighbors, node, cutoff=k_hop).keys()
        for neighbor in k_hop_neighbors:
            if node != neighbor:
                G_k_hop.add_edge(node, neighbor)

    edges = list(G_k_hop.edges)
    if len(edges) > 0:
        edge_index_neighbors = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index_neighbors = torch.empty((2, 0), dtype=torch.long)

    # 构建目标点的边关系，使得每个地标与每个智能体相连
    if len(world.landmarks) > 0 and len(world.agents) > 0:
        edge_index_targets = torch.tensor(
            [[i for i in range(len(world.agents))] * len(world.landmarks),
             [j for j in range(len(world.landmarks)) for _ in range(len(world.agents))]],
            dtype=torch.long
        )
    else:
        # 如果没有地标或智能体，返回一个空的 edge_index
        edge_index_targets = torch.empty((2, 0), dtype=torch.long)

    # Step 4: 使用 GAT 聚合邻居特征和目标特征
    input_dim_neighbors = x_neighbors.shape[1]  # 输入特征的维度（位置+速度）
    input_dim_targets = x_targets.shape[1]  # 输入特征的维度（位置）

    hidden_dim = 8  # GAT 的隐藏层维度
    output_dim_neighbors = 32  # GAT 1 的输出特征维度，用于邻居特征
    output_dim_targets = 32  # GAT 2 的输出特征维度，用于目标特征

    # 使用函数属性来确保 GATModule 只被实例化一次
    if not hasattr(gat_observation, "gat_neighbors"):
        gat_observation.gat_neighbors = GATModule(input_dim_neighbors, hidden_dim, output_dim_neighbors, heads=4)
    if not hasattr(gat_observation, "gat_targets"):
        gat_observation.gat_targets = GATModule(input_dim_targets, hidden_dim, output_dim_targets, heads=4)

    # 将模型移动到相同设备（假设使用CPU，如果使用GPU，需要根据实际情况调整）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gat_observation.gat_neighbors.to(device)
    gat_observation.gat_targets.to(device)

    # 将输入张量移动到设备上
    x_neighbors = x_neighbors.to(device)
    x_targets = x_targets.to(device)
    edge_index_neighbors = edge_index_neighbors.to(device)
    edge_index_targets = edge_index_targets.to(device)

    # 聚合邻居信息和目标信息
    aggregated_neighbors = gat_observation.gat_neighbors(x_neighbors, edge_index_neighbors)
    aggregated_targets = gat_observation.gat_targets(x_targets, edge_index_targets)

    """
    直接返回的是分别输出的三个向量，因为好做进一步的处理
    """

    # 将结果从设备移回CPU（如果需要）
    aggregated_neighbors = aggregated_neighbors.cpu()
    aggregated_targets = aggregated_targets.cpu()

    return own_state, aggregated_neighbors, aggregated_targets

