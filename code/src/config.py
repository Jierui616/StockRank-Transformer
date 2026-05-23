# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去40个交易日的数据 (60→40，增加样本量)
    'd_model': 128,          # Transformer输入维度 (320→128，减轻过拟合)
    'nhead': 8,             # 注意力头数量
    'num_layers': 2,        # Transformer层数 (4→2)
    'dim_feedforward': 256, # 前馈网络维度 (640→256)
    'batch_size': 4,        # 排序任务batch_size
    'num_epochs': 60,       # 最大epoch数 (早停会提前终止)
    'learning_rate': 1e-4,  # 峰值学习率
    'min_learning_rate': 1e-6,  # 最小学习率
    'warmup_epochs': 5,     # Warmup epoch数
    'dropout': 0.15,        # Dropout
    'feature_num': feature_num,
    'max_grad_norm': 5.0,
    'stock_embedding_dim': 32,  # 可学习股票Embedding维度
    'use_market_encoder': False,  # 市场状态编码器 (v3 实验完毕: 0.0014)

    'pairwise_weight': 2.0, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 3.0, # top-5样本权重 (2→3，增强top-k关注)

    # 多horizon标签权重 [1日, 5日]
    'label_horizon_weights': [0.3, 0.7],

    # 标签市值中性化
    'use_label_neutralization': False,  # 已实验 (-0.0713)，关闭

    # LambdaRank / ApproxNDCG 参数
    'lambda_sigma': 0.5,     # sigmoid 温度 (0.3~5.0)
    'loss_type': 'approx_ndcg',  # 'approx_ndcg' 或 'lambdarank'
    'sigma_search': [0.3, 0.5, 1.0, 2.0, 5.0],  # 网格搜索候选值
    'run_sigma_grid_search': False,  # sigma 网格搜索已完成，最佳 sigma=0.5

    # 特征筛选
    'feature_selection_topk': 80,
    'run_topk_grid_search': False,  # Top-K 搜索因内存不足放弃，topk=80 保持
    'topk_search': [80],
    'max_preprocess_workers': 4,  # 特征工程多进程数

    # 早停
    'early_stopping_patience': 10,

    # 学习率调度
    'lr_scheduler': 'plateau',  # 'plateau' 或 'cosine'
    'lr_patience': 5,           # plateau 模式: 5轮不提升则降lr
    'lr_factor': 0.5,           # plateau 模式: lr 衰减因子

    # 梯度累积 (有效 batch_size = batch_size × accumulation_steps)
    'gradient_accumulation_steps': 4,

    # 集成训练 (单 seed 模式)
    'ensemble': False,
    'seeds': [42],
    'base_seed': 42,

    # 推理
    'inference_temperature': 1.0,
    'inference_weight_mode': 'rank_linear',  # rank_linear best for 75-sample model

    'output_dir': f'./model/{sequence_length}_{feature_num}_v2',
    'data_path': './data',
}