import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized
from utils import select_features_lgb
import joblib
import json
import os
import math
import shutil
import random
import multiprocessing as mp
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

feature_cloums_map = {
    '39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv','volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}
feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。支持市值中性化。"""
    processed = processed.copy()
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t2'] = processed.groupby('股票代码')['开盘'].shift(-2)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    # 过滤无效开盘价
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    # 计算各horizon原始收益率
    processed['ret_1d'] = (processed['open_t2'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed['ret_5d'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)

    # ---- 市值中性化 ----
    if config.get('use_label_neutralization', False):
        # 流通市值代理: 成交额 / 换手率
        processed['mkt_cap_proxy'] = processed['成交额'] / (processed['换手率'] + 1e-12)
        processed['log_mkt'] = np.log(np.abs(processed['mkt_cap_proxy']) + 1e-12)

        def _neutralize_day(grp):
            # 过滤 NaN，确保回归有效
            valid = grp[['ret_1d', 'ret_5d', 'log_mkt']].notna().all(axis=1)
            if valid.sum() < 10:
                return grp  # 样本太少，跳过
            X = np.column_stack([np.ones(valid.sum()), grp.loc[valid, 'log_mkt'].values])
            for col in ['ret_1d', 'ret_5d']:
                y = grp.loc[valid, col].values
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                # 对所有股票（含 NaN）计算残差
                X_all = np.column_stack([np.ones(len(grp)), grp['log_mkt'].fillna(grp['log_mkt'].median()).values])
                grp[col] = y_all = grp[col].values - X_all @ beta
            return grp

        processed = processed.groupby('日期', group_keys=False).apply(_neutralize_day)
        processed.drop(columns=['mkt_cap_proxy', 'log_mkt'], inplace=True)
        print("标签市值中性化完成")

    # 直接使用 raw 5-day forward return 作为 label
    processed['label'] = processed['ret_5d']

    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t1', 'open_t2', 'open_t5',
                            'ret_1d', 'ret_5d'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(config.get('max_preprocess_workers', 4), mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)

    # 添加截面特征
    from utils import add_cross_sectional_features
    processed, cs_features = add_cross_sectional_features(processed, feature_columns)
    all_features = feature_columns + cs_features

    return processed, all_features


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# ApproxNDCG 排序损失函数
# 用 sigmoid 软化排序位置，直接可微地优化 NDCG
# 参考: allRank (https://github.com/sshyran/allRank), MiM-StocR (arXiv:2509.10461)
class ApproxNDCGLoss(nn.Module):
    """
    ApproxNDCG = DCG_approx / IDCG
    soft_rank(i) = 1 + Σ_{j≠i} σ((s_j - s_i) / τ)
    直接优化排序指标，避免 LambdaRank 的 pairwise 间接近似。
    """
    def __init__(self, sigma=1.0, k=5):
        super(ApproxNDCGLoss, self).__init__()
        self.sigma = sigma  # temperature for sigmoid
        self.k = k

    def forward(self, y_pred, y_true):
        B, N = y_pred.size()
        if N < 2:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # Per-query gain normalization: map returns to [0, 1] then exp gains
        y_min = y_true.min(dim=1, keepdim=True).values
        y_max = y_true.max(dim=1, keepdim=True).values
        scale = y_max - y_min
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        y_norm = (y_true - y_min) / (scale + 1e-12)
        gains = torch.pow(2.0, y_norm) - 1.0  # [B, N]

        # Sigmoid-smoothed pairwise comparisons
        # P(j ranked before i) = σ((s_j - s_i) / τ)
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)  # [B, N, N]
        prob_j_beats_i = torch.sigmoid(pred_diff / max(self.sigma, 1e-8))  # [B, N, N]

        # Soft rank: π̂(i) = 1 + Σ_{j≠i} P(j beats i)
        soft_ranks = 1.0 + prob_j_beats_i.sum(dim=2)  # [B, N]

        # Discount: 1 / log2(rank + 1)
        discounts = 1.0 / torch.log2(soft_ranks + 1.0)  # [B, N]

        # Approx DCG
        dcg = (gains * discounts).sum(dim=1)  # [B]

        # Ideal DCG: true relevance ranking
        ideal_gains, _ = gains.sort(dim=1, descending=True)
        ideal_ranks = torch.arange(1, N + 1, device=y_pred.device).float().unsqueeze(0)
        ideal_discounts = 1.0 / torch.log2(ideal_ranks + 1.0)
        idcg = (ideal_gains * ideal_discounts).sum(dim=1)  # [B]

        ndcg = dcg / (idcg + 1e-12)
        loss = (1.0 - ndcg).mean()
        return loss


# 保留原有 LambdaRank 作为可选方案
class LambdaRankLoss(nn.Module):
    """
    LambdaRank 损失：用 ΔNDCG 对 pairwise loss 加权，直接优化排序指标。
    全批量向量化，无 Python for 循环。
    """
    def __init__(self, sigma=1.0, k=5):
        super(LambdaRankLoss, self).__init__()
        self.sigma = sigma
        self.k = k

    def forward(self, y_pred, y_true):
        B, N = y_pred.size()
        if N < 2:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        rel_min = y_true.min(dim=1, keepdim=True).values
        rel_max = y_true.max(dim=1, keepdim=True).values
        scale = rel_max - rel_min
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        rel_norm = (y_true - rel_min) / (scale + 1e-12)

        gains = torch.pow(2.0, rel_norm) - 1.0

        positions = y_pred.argsort(dim=1, descending=True).argsort(dim=1).float()
        discounts = 1.0 / torch.log2(positions + 2.0)

        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        pair_mask = (true_diff.abs() > 1e-8).float()

        gain_diff = gains.unsqueeze(2) - gains.unsqueeze(1)
        disc_diff = discounts.unsqueeze(2) - discounts.unsqueeze(1)
        delta_ndcg = (gain_diff * disc_diff).abs() * pair_mask

        pair_loss = torch.log1p(torch.exp(
            -self.sigma * pred_diff * true_diff.sign()
        ))

        query_loss = (delta_ndcg * pair_loss * pair_mask).sum(dim=(1, 2))
        num_pairs = pair_mask.sum(dim=(1, 2)).clamp(min=1)
        loss = (query_loss / num_pairs).mean()

        return loss

class ListMLELoss(nn.Module):
    """
    ListMLE (List-wise Maximum Likelihood Estimation) via Plackett-Luce model.
    L = -Σ_{i=1}^{k} log P(item at rank i | remaining items)
    直接最大化真实排序的似然，不需要 gain function，对小数据更鲁棒。
    """
    def __init__(self, k=5):
        super(ListMLELoss, self).__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        B, N = y_pred.size()
        device = y_pred.device
        k = N if self.k is None else min(self.k, N)

        # 真实排序: 按 y_true 降序的索引
        _, true_order = y_true.sort(dim=1, descending=True)  # [B, N]

        loss = torch.tensor(0.0, device=device)
        for i in range(k):
            # 第 i 个位置的正确股票索引
            idx_i = true_order[:, i]                      # [B]
            s_i = y_pred.gather(1, idx_i.unsqueeze(1)).squeeze(1)  # [B]

            # 剩余候选集: 位置 i 到 N-1
            remaining_idx = true_order[:, i:]             # [B, N-i]
            remaining_scores = y_pred.gather(1, remaining_idx)  # [B, N-i]

            # log Σ_{j>=i} exp(s_j) - s_i  (总是 >=0)
            log_sum_exp = torch.logsumexp(remaining_scores, dim=1)  # [B]
            loss = loss + (log_sum_exp - s_i).mean()

        return loss / k


class PairwiseTopKMarginLoss(nn.Module):
    """
    Top-K pairwise margin loss: 强制 top-k 股票的分数高于其余股票至少 margin。
    只关注 top-k vs rest 的区分度，忽略内部排序。
    """
    def __init__(self, k=5, margin=0.1):
        super(PairwiseTopKMarginLoss, self).__init__()
        self.k = k
        self.margin = margin

    def forward(self, y_pred, y_true):
        B, N = y_pred.size()
        if N <= self.k:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        _, sorted_idx = y_true.sort(dim=1, descending=True)
        topk_idx = sorted_idx[:, :self.k]
        rest_idx = sorted_idx[:, self.k:]

        topk_scores = y_pred.gather(1, topk_idx).unsqueeze(2)   # [B, k, 1]
        rest_scores = y_pred.gather(1, rest_idx).unsqueeze(1)   # [B, 1, N-k]

        # Hinge: max(0, margin - (s_top - s_rest))
        diff = topk_scores - rest_scores                         # [B, k, N-k]
        loss = torch.relu(self.margin - diff).mean()

        return loss


class ListMLEPairwiseLoss(nn.Module):
    """
    ListMLE + PairwiseTopKMargin 混合损失。
    ListMLE 保证排序似然，pairwise margin 确保 top-k 区分度。
    """
    def __init__(self, listmle_k=10, pairwise_k=5, margin=0.1, pairwise_weight=0.5):
        super(ListMLEPairwiseLoss, self).__init__()
        self.listmle = ListMLELoss(k=listmle_k)
        self.pairwise = PairwiseTopKMarginLoss(k=pairwise_k, margin=margin)
        self.pairwise_weight = pairwise_weight

    def forward(self, y_pred, y_true):
        loss_lmle = self.listmle(y_pred, y_true)
        loss_pair = self.pairwise(y_pred, y_true)
        return loss_lmle + self.pairwise_weight * loss_pair


def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """计算评估指标，包含与测试对齐的加权收益指标。"""
    import numpy as np
    batch_size = y_pred.size(0)

    # Metrics accumulators
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    test_aligned_list = []  # 与 score_self.py 同款：softmax 加权 Top-5 收益

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()

        if valid_indices.numel() < k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices] # This is the 5-day return

        # 1. Predicted Top 5
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()

        # 2. True Top 5 (Theoretical Max)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()

        # 3. Random 5 (Expected Value)
        random_return_sum = k * valid_true.mean().item()

        # 计算每个样本的比例与稳定化 final_score
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0

        # 4. 测试对齐指标：softmax 加权 Top-5 收益 (与 score_self.py 一致)
        top5_scores = valid_pred[pred_indices].float().cpu().numpy()
        # 数值稳定 softmax (float64)
        top5_scores_f64 = top5_scores.astype(np.float64)
        exp_scores = np.exp(top5_scores_f64 - top5_scores_f64.max())
        top5_weights = exp_scores / exp_scores.sum()
        top5_returns = valid_true[pred_indices].float().cpu().numpy()
        test_aligned = float(np.dot(top5_weights, top5_returns))

        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)
        test_aligned_list.append(test_aligned)

    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
    }

    # 比值用逐样本均值，降低极端日影响
    metrics['ratio_pred'] = np.mean(ratio_pred_list) if ratio_pred_list else 0.0
    metrics['ratio_random'] = np.mean(ratio_random_list) if ratio_random_list else 0.0
    metrics['final_score'] = np.mean(final_score_list) if final_score_list else 0.0
    metrics['test_aligned_score'] = np.mean(test_aligned_list) if test_aligned_list else 0.0

    return metrics

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),  # [num_stocks, seq_len, features]
            'targets': torch.FloatTensor(self.targets[idx]),      # [num_stocks] 真实涨跌幅
            'relevance': torch.LongTensor(self.relevance_scores[idx]),  # [num_stocks] 排序标签
            'stock_indices': torch.LongTensor(self.stock_indices[idx])  # [num_stocks] 股票索引
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []
    
    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
    
    return {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks)                      # [batch, max_stocks]
    }

# 排序训练函数
def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer, accumulation_steps=1):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    optimizer.zero_grad()
    accum_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        relevance = batch['relevance'].to(device)    # [batch, max_stocks] 预处理的相关性得分
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask

        # 模型预测
        outputs = model(sequences)  # [batch, max_stocks] 预测分数

        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks

        # 计算损失（只对有效股票计算）
        batch_loss = None
        batch_size = sequences.size(0)

        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()

            if valid_indices.numel() == 0:
                continue

            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)

            valid_pred = masked_outputs[i][valid_indices]
            valid_relevance = masked_relevance[i][valid_indices]

            if len(valid_pred) > 1:
                loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss

        if batch_loss is not None:
            batch_loss = batch_loss / (batch_size * accumulation_steps)
            batch_loss.backward()
            accum_loss += batch_loss.item() * accumulation_steps  # undo division for logging

            # 梯度累积: 每 accumulation_steps 步更新一次
            if (batch_idx + 1) % accumulation_steps == 0:
                if not config.get('drop_clip', True):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    if writer:
                        writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += accum_loss
                local_step += 1

                # 计算评估指标
                with torch.no_grad():
                    metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                    for k, v in metrics.items():
                        if k not in total_metrics:
                            total_metrics[k] = 0
                        total_metrics[k] += v

                if writer:
                    writer.add_scalar('train/loss', accum_loss, global_step=epoch*len(dataloader)+local_step)
                    for k, v in metrics.items():
                        writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)

                accum_loss = 0.0

    # 处理剩余未累积的梯度
    if (batch_idx + 1) % accumulation_steps != 0:
        if not config.get('drop_clip', True):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        optimizer.step()
        optimizer.zero_grad()
        total_loss += accum_loss
        local_step += 1

    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step

    return total_loss / max(local_step, 1), total_metrics

def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            
            # 模型预测
            outputs = model(sequences)
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            
            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                
                if len(valid_pred) > 1:
                    _, sorted_indices = torch.sort(valid_true, descending=True)
                    relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                    relevance_scores[sorted_indices] = torch.arange(len(valid_true), 0, -1, device=device, dtype=torch.float32)
                    relevance_scores = relevance_scores.detach()
                    
                    loss = criterion(valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
            
            # 计算评估指标
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()
    
    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()
    
    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []
    
    for stock_code in data['股票代码'].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = data[
            (data['股票代码'] == stock_code) & 
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)
        
        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])
    
    if len(day_sequences) == 0:
        return []
    
    # 转换为tensor
    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)  # [1, num_stocks, seq_len, features]
    
    with torch.no_grad():
        # 模型预测
        outputs = model(sequences)  # [1, num_stocks]
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]
        
        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1
            })
    
    return top_stocks

def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def split_train_val_by_last_month(df, sequence_length):
    """按最后一个月做验证集划分，并为验证集补充序列上下文。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    # 验证集需要保留前 sequence_length-1 个交易日作为序列上下文，
    # 这样第一个验证样本的窗口结束日就可以落在 val_start。
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后一个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    # 恢复为字符串，保持与原流程一致
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start

def _prepare_and_train_for_topk(topk, base_output_dir, num_stocks, features_raw,
                               device, train_data, val_data, val_start,
                               stockid2idx, config):
    """为给定 topk 执行特征筛选、标准化、数据集创建和训练，返回 best_score。"""
    import copy
    output_dir = os.path.join(base_output_dir, f'topk_{topk}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 特征筛选
    features = list(features_raw)
    if topk > 0 and topk < len(features):
        lgb_features = [f for f in features if f != 'instrument']
        selected_features = select_features_lgb(train_data, lgb_features, topk=topk)
        features = ['instrument'] + selected_features
        with open(os.path.join(output_dir, 'selected_features.json'), 'w') as f:
            json.dump(features, f, ensure_ascii=False)
        print(f'[topk={topk}] 特征筛选完成，使用 {len(features)} 个特征')
    
    # 4. 标准化
    scaler = StandardScaler()
    train_scaled = train_data.copy()
    val_scaled = val_data.copy()
    train_scaled[features] = scaler.fit_transform(train_data[features])
    val_scaled[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # 5. 创建排序数据集
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_scaled, features, config['sequence_length'],
        ranking_data_path=config.get('train_ranking_data_path'))
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_scaled, features, config['sequence_length'],
        ranking_data_path=config.get('val_ranking_data_path'),
        min_window_end_date=val_start.strftime('%Y-%m-%d'))
    print(f'[topk={topk}] 训练集: {len(train_sequences)} 样本, 验证集: {len(val_sequences)} 样本')
    
    # 6. 训练 (使用最佳 sigma=0.5)
    seed = config.get('seeds', [42])[0]
    score = _train_one_seed(
        seed, output_dir, num_stocks, features, device,
        train_sequences, train_targets, train_relevance, train_stock_indices,
        val_sequences, val_targets, val_relevance, val_stock_indices,
        override_sigma=config.get('lambda_sigma', 0.5))
    
    # 保存 features 和 score 供汇总使用
    with open(os.path.join(output_dir, 'topk_info.json'), 'w') as f:
        json.dump({'topk': topk, 'score': float(score), 'n_features': len(features)}, f)
    
    return score, features

# 单 seed 训练
def _train_one_seed(seed, output_dir, num_stocks, features, device,
                    train_sequences, train_targets, train_relevance, train_stock_indices,
                    val_sequences, val_targets, val_relevance, val_stock_indices,
                    override_sigma=None):
    """训练单个 seed 的模型，返回 best_score。"""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))
    print(f"\n{'='*60}")
    print(f"Seed {seed} 训练开始, 输出: {output_dir}")
    print(f"{'='*60}")

    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=False)

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    sigma = override_sigma if override_sigma is not None else config.get('lambda_sigma', 1.0)
    loss_type = config.get('loss_type', 'listmle')
    listmle_k = config.get('listmle_k', 5)
    if loss_type == 'listmle':
        criterion = ListMLELoss(k=listmle_k if listmle_k > 0 else None)
        k_label = f'top-{listmle_k}' if listmle_k > 0 else 'full-list'
        print(f"使用 ListMLE 损失 ({k_label} Plackett-Luce)")
    elif loss_type == 'listmle_pairwise':
        pw_weight = config.get('pairwise_loss_weight', 0.5)
        pw_margin = config.get('pairwise_margin', 0.1)
        criterion = ListMLEPairwiseLoss(
            listmle_k=listmle_k if listmle_k > 0 else None,
            pairwise_k=5, margin=pw_margin, pairwise_weight=pw_weight)
        print(f"使用 ListMLE+Pairwise 混合损失 (k={listmle_k}, margin={pw_margin}, pw_weight={pw_weight})")
    elif loss_type == 'approx_ndcg':
        criterion = ApproxNDCGLoss(sigma=sigma, k=5)
        print(f"使用 ApproxNDCG 损失 (sigma={sigma})")
    else:
        criterion = LambdaRankLoss(sigma=sigma, k=5)
        print(f"使用 LambdaRank 损失 (sigma={sigma})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    accum_steps = config.get('gradient_accumulation_steps', 1)
    effective_bs = config['batch_size'] * accum_steps
    print(f"梯度累积: {accum_steps}步, 有效 batch_size={effective_bs}")

    warmup_epochs = config.get('warmup_epochs', 5)
    total_epochs = config['num_epochs']
    peak_lr = config['learning_rate']
    min_lr = config.get('min_learning_rate', 1e-6)
    patience = config.get('early_stopping_patience', 10)
    lr_patience = config.get('lr_patience', 5)
    lr_factor = config.get('lr_factor', 0.5)
    scheduler_type = config.get('lr_scheduler', 'plateau')

    # Cyclic Cosine Annealing (snapshot ensemble)
    if scheduler_type == 'cyclic_cosine':
        cyclic_period = config.get('cyclic_period', 8)
        cyclic_cycles = config.get('cyclic_cycles', 4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cyclic_period, eta_min=min_lr)
        snapshot_dir = os.path.join(output_dir, 'snapshots')
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_scores = []  # list of (epoch, val_score, path)
        print(f"学习率调度: Cyclic Cosine (period={cyclic_period}, cycles={cyclic_cycles})")
        print(f"快照保存目录: {snapshot_dir}")
    elif scheduler_type == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=lr_factor, patience=lr_patience,
            min_lr=min_lr)
        print(f"学习率调度: Plateau (patience={lr_patience}, factor={lr_factor})")
    else:
        print(f"学习率调度: Warmup+Cosine")

    best_score = -float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    lr_epochs_no_improve = 0  # for plateau scheduler

    for epoch in range(total_epochs):
        print(f"\n=== Seed {seed} Epoch {epoch+1}/{total_epochs} ===")

        train_loss, train_metrics = train_ranking_model(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            accumulation_steps=accum_steps)
        print(f"Train Loss: {train_loss:.4f}")
        for k, v in train_metrics.items():
            print(f"Train {k}: {v:.4f}")

        eval_loss, eval_metrics = evaluate_ranking_model(
            model, val_loader, criterion, device, writer, epoch)
        print(f"Eval Loss: {eval_loss:.4f}")
        for k, v in eval_metrics.items():
            print(f"Eval {k}: {v:.4f}")

        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        if epoch < warmup_epochs:
            # Linear warmup
            current_lr = min_lr + (peak_lr - min_lr) * (epoch + 1) / max(warmup_epochs, 1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        elif scheduler_type == 'cyclic_cosine':
            # Cyclic cosine: 每个周期内 cosine 退火，周期结束时保存快照并重置 LR
            cycle_epoch = (epoch - warmup_epochs) % cyclic_period
            is_cycle_end = (cycle_epoch == cyclic_period - 1)
            if cycle_epoch == 0:
                # 新周期开始，重置 optimizer 和 scheduler
                for param_group in optimizer.param_groups:
                    param_group['lr'] = peak_lr
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cyclic_period, eta_min=min_lr)
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        elif scheduler_type == 'plateau':
            current_score = eval_metrics.get('final_score', 0.0)
            lr_scheduler.step(current_score)
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            current_lr = min_lr + (peak_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        if writer:
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)

        # 早停 + 保存最佳模型
        current_score = eval_metrics.get('final_score', 0.0)
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"保存最佳模型 - final score: {best_score:.4f}")
        else:
            epochs_no_improve += 1
            print(f"未提升 ({epochs_no_improve}/{patience})")

        # Snapshot 保存 (cyclic_cosine 模式)
        if scheduler_type == 'cyclic_cosine':
            cycle_epoch = (epoch - warmup_epochs) % cyclic_period
            is_cycle_end = (cycle_epoch == cyclic_period - 1) and epoch >= warmup_epochs
            if is_cycle_end:
                cycle_idx = (epoch - warmup_epochs) // cyclic_period + 1
                snap_path = os.path.join(
                    snapshot_dir,
                    f'snapshot_cycle{cycle_idx}_epoch{epoch+1}_val{current_score:.4f}.pth')
                torch.save(model.state_dict(), snap_path)
                snapshot_scores.append({
                    'cycle': cycle_idx,
                    'epoch': epoch + 1,
                    'val_score': float(current_score),
                    'path': os.path.basename(snap_path),
                })
                print(f"[Snapshot] cycle={cycle_idx}, val_score={current_score:.4f}")

        # 早停: cyclic_cosine 模式跑满，其他模式按 patience 早停
        if scheduler_type != 'cyclic_cosine' and epochs_no_improve >= patience:
            print(f"早停触发! Seed {seed} 在 epoch {epoch+1} 停止")
            break

    print(f"\nSeed {seed} 完成! 最佳 epoch: {best_epoch}, 最佳 final score: {best_score:.4f}")
    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Seed: {seed}\nBest epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")

    # 保存快照明细
    if scheduler_type == 'cyclic_cosine' and snapshot_scores:
        manifest_path = os.path.join(snapshot_dir, 'scores.json')
        with open(manifest_path, 'w') as f:
            json.dump(snapshot_scores, f, indent=2, ensure_ascii=False)
        print(f"快照明细已保存: {manifest_path} ({len(snapshot_scores)} 个快照)")

    writer.close()
    return best_score


def main():
    base_seed = config.get('base_seed', 42)
    set_seed(base_seed)

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    data_path = config['data_path']

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")

    # 1. 数据加载
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    print(f"股票数量: {num_stocks}")

    # 2. 特征工程与预处理 (所有 seed 共享)
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)

    features_raw = list(features)  # 保存原始特征列表供 grid search 使用

    # 3. 特征筛选 (LightGBM LambdaRank)
    topk = config.get('feature_selection_topk', 80)
    if topk > 0 and topk < len(features):
        lgb_features = [f for f in features if f != 'instrument']
        selected_features = select_features_lgb(train_data, lgb_features, topk=topk)
        features = ['instrument'] + selected_features
        with open(os.path.join(output_dir, 'selected_features.json'), 'w') as f:
            json.dump(features, f, ensure_ascii=False)
        print(f"特征筛选完成，使用 {len(features)} 个特征")

    # 4. 标准化 (所有 seed 共享)
    scaler = StandardScaler()
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # 5. 创建排序数据集 (所有 seed 共享)
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data, features, config['sequence_length'],
        ranking_data_path=config.get('train_ranking_data_path'))
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data, features, config['sequence_length'],
        ranking_data_path=config.get('val_ranking_data_path'),
        min_window_end_date=val_start.strftime('%Y-%m-%d'))
    print(f"训练集: {len(train_sequences)} 样本, 验证集: {len(val_sequences)} 样本")

    # 6. 训练
    seeds = config.get('seeds', [42])
    ensemble = config.get('ensemble', False)
    
    # --- Top-K 网格搜索 ---
    run_topk_search = config.get('run_topk_grid_search', False)
    topk_candidates = config.get('topk_search', [80])
    
    if run_topk_search and len(topk_candidates) > 1:
        print(f"\n{'='*60}")
        print(f"Top-K 网格搜索: {len(topk_candidates)} 个候选值 {topk_candidates}")
        print(f"{'='*60}")
        best_topk = None
        best_topk_score = -float('inf')
        topk_results = {}
        
        for topk_val in topk_candidates:
            print(f"\n--- Top-K = {topk_val} ---")
            score, selected_feats = _prepare_and_train_for_topk(
                topk_val, output_dir, num_stocks, features_raw,
                device, train_data, val_data, val_start,
                stockid2idx, config)
            topk_results[topk_val] = score
            if score > best_topk_score:
                best_topk_score = score
                best_topk = topk_val
        
        print(f"\n{'='*60}")
        print(f"Top-K 网格搜索汇总:")
        for k, sc in sorted(topk_results.items()):
            marker = " <-- BEST" if k == best_topk else ""
            print(f"  topk={k}: {sc:.6f}{marker}")
        print(f"最佳 topk={best_topk}, score={best_topk_score:.6f}")
        print(f"{'='*60}")
        
        # 复制最佳模型
        best_model_src = os.path.join(output_dir, f'topk_{best_topk}', 'best_model.pth')
        best_model_dst = os.path.join(output_dir, 'best_model.pth')
        if os.path.exists(best_model_src):
            shutil.copy2(best_model_src, best_model_dst)
            print(f"已复制最佳模型到: {best_model_dst}")
        
        # 写入汇总
        with open(os.path.join(output_dir, 'topk_search_results.txt'), 'w') as f:
            f.write("Top-K grid search results\n")
            f.write("="*40 + "\n")
            for k, sc in sorted(topk_results.items()):
                f.write(f"topk={k}: {sc:.6f}\n")
            f.write(f"\nBest topk={best_topk}, score={best_topk_score:.6f}\n")
        
        # 更新 config
        config['feature_selection_topk'] = best_topk
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        print("\n########## Top-K 网格搜索完成！ ##########")
        return

    # --- Sigma 网格搜索 ---
    run_sigma_search = config.get('run_sigma_grid_search', False)
    sigma_candidates = config.get('sigma_search', [0.5])

    if run_sigma_search and len(sigma_candidates) > 1:
        print(f"\n{'='*60}")
        print(f"Sigma 网格搜索: {len(sigma_candidates)} 个候选值 {sigma_candidates}")
        print(f"{'='*60}")
        best_sigma = None
        best_sigma_score = -float('inf')
        sigma_results = {}

        for sigma_val in sigma_candidates:
            sigma_dir = os.path.join(output_dir, f'sigma_{sigma_val}')
            print(f"\n--- Sigma = {sigma_val} ---")
            score = _train_one_seed(
                seeds[0], sigma_dir, num_stocks, features, device,
                train_sequences, train_targets, train_relevance, train_stock_indices,
                val_sequences, val_targets, val_relevance, val_stock_indices,
                override_sigma=sigma_val)
            sigma_results[sigma_val] = score
            if score > best_sigma_score:
                best_sigma_score = score
                best_sigma = sigma_val

        print(f"\n{'='*60}")
        print(f"Sigma 网格搜索汇总:")
        for s, sc in sorted(sigma_results.items()):
            marker = " <-- BEST" if s == best_sigma else ""
            print(f"  sigma={s}: {sc:.6f}{marker}")
        print(f"最佳 sigma={best_sigma}, score={best_sigma_score:.6f}")
        print(f"{'='*60}")

        # 复制最佳 sigma 的模型到主目录
        best_model_src = os.path.join(output_dir, f'sigma_{best_sigma}', 'best_model.pth')
        best_model_dst = os.path.join(output_dir, 'best_model.pth')
        if os.path.exists(best_model_src):
            shutil.copy2(best_model_src, best_model_dst)
            print(f"已复制最佳模型到: {best_model_dst}")

        # 写入汇总
        with open(os.path.join(output_dir, 'sigma_search_results.txt'), 'w') as f:
            f.write(f"Sigma grid search results\n")
            f.write(f"{'='*40}\n")
            for s, sc in sorted(sigma_results.items()):
                f.write(f"sigma={s}: {sc:.6f}\n")
            f.write(f"\nBest sigma={best_sigma}, score={best_sigma_score:.6f}\n")

        # 更新 config 记录最佳 sigma
        config['lambda_sigma'] = best_sigma
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    elif ensemble and len(seeds) > 1:
        print(f"\n集成训练: {len(seeds)} 个 seed {seeds}")
        scores = {}
        for seed in seeds:
            seed_dir = os.path.join(output_dir, f'seed{seed}')
            score = _train_one_seed(
                seed, seed_dir, num_stocks, features, device,
                train_sequences, train_targets, train_relevance, train_stock_indices,
                val_sequences, val_targets, val_relevance, val_stock_indices)
            scores[seed] = score
        print(f"\n集成训练汇总:")
        for s, sc in scores.items():
            print(f"  Seed {s}: {sc:.4f}")
        print(f"  均值: {np.mean(list(scores.values())):.4f}")
    else:
        _train_one_seed(seeds[0], output_dir, num_stocks, features, device,
                        train_sequences, train_targets, train_relevance, train_stock_indices,
                        val_sequences, val_targets, val_relevance, val_stock_indices)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
    print("\n########## 训练完成！ ##########")