# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据
    'd_model': 320,          # Transformer输入维度 (256→320)
    'nhead': 8,             # 注意力头数量 (4→8)
    'num_layers': 4,        # Transformer层数 (3→4)
    'dim_feedforward': 640, # 前馈网络维度 (512→640)
    'batch_size': 4,        # 排序任务batch_size可以小一些
    'num_epochs': 60,       # 增加epoch (50→60)
    'learning_rate': 1e-4,  # 峰值学习率 (Warmup目标)
    'min_learning_rate': 1e-6,  # 最小学习率
    'warmup_epochs': 5,     # Warmup epoch数
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,
    'stock_embedding_dim': 32,  # 可学习股票Embedding维度

    'pairwise_weight': 2.0, # 配对损失权重 (1→2)
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 3.0, # top-5样本权重 (2→3，增强top-k关注)

    # 多horizon标签权重 [1日, 5日]
    'label_horizon_weights': [0.3, 0.7],

    # LambdaRank 参数
    'lambda_sigma': 1.0,     # 温度系数

    # 特征筛选
    'feature_selection_topk': 80,

    'output_dir': f'./model/{sequence_length}_{feature_num}_v2',
    'data_path': './data',
}