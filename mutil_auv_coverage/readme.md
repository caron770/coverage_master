# 代码

## 代码文件结构

相关文件结构如下图所示

```
maddpg_project/
│
├── maddpg.py         # 包含 MADDPG 算法、Actor-Critic 网络、Replay Buffer 等所有相关代码
├── mpe_env.py        # 自定义 MPE 环境的包装
├── utils.py          # 包含噪声、训练辅助函数、日志等工具
├── main.py           # 主训练脚本
└── config.py         # 配置文件
```

再加上可视化结果分析的文件
