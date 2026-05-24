# 配置参数
sequence_length = 20
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # v4最佳: 20
    'd_model': 64,          # 128→64，减小模型容量
    'nhead': 8,             # 注意力头数量
    'num_layers': 2,        # Transformer层数
    'dim_feedforward': 128, # 256→128，配合d_model缩减
    'batch_size': 4,        # 排序任务batch_size
    'num_epochs': 60,       # 最大epoch数 (早停会提前终止)
    'learning_rate': 1e-4,  # 峰值学习率
    'min_learning_rate': 1e-6,  # 最小学习率
    'warmup_epochs': 5,     # Warmup epoch数
    'dropout': 0.30,        # v4最佳: 0.30
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
    'loss_type': 'listmle_pairwise',  # ListMLE + PairwiseTopK 混合损失
    'listmle_k': 10,         # ListMLE 的 k 参数
    'pairwise_margin': 0.1,  # top-5 vs rest 的最小分数差距
    'pairwise_loss_weight': 0.5,  # pairwise 损失权重
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
    'lr_scheduler': 'plateau',  # 'plateau' | 'cosine' | 'cyclic_cosine'
    'lr_patience': 5,           # plateau 模式: 5轮不提升则降lr
    'lr_factor': 0.5,           # plateau 模式: lr 衰减因子

    # Snapshot ensemble 相关 (当前未启用，保留备用)
    'cyclic_period': 8,
    'cyclic_cycles': 5,
    'snapshot_val_threshold': 0.05,
    'snapshot_min_count': 2,

    # 梯度累积 (有效 batch_size = batch_size × accumulation_steps)
    'gradient_accumulation_steps': 4,

    # 集成训练 (多 seed 筛选，Plateau 早停保证质量)
    'ensemble': True,
    'seeds': [42, 123, 456, 789, 1024, 2024, 555, 777, 888, 999],
    'base_seed': 42,

    # 推理
    'inference_temperature': 1.0,
    'inference_weight_mode': 'equal',  # equal: 模型top-5内部排序不可靠，等权更鲁棒 (seed456: 0.0875)
    'ensemble_mode': 'mean',  # 融合模式: 'mean' 均值分数 / 'rank' 均值排名 / 'median' 中位数 (多模型融合已证实无效，保留单模型)

    'output_dir': f'./model/{sequence_length}_{feature_num}_v10',
    'data_path': './data',
}