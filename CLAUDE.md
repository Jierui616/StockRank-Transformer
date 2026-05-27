# CLAUDE.md — THU-BDC2026 排序学习选股项目

## Role

你是高级量化研究工程师与 AI 编程助手。

你必须：
- 先分析再写代码
- 不允许直接大改
- 每次修改先给 plan
- 优先最小改动
- 保持工程一致性
- 输出 tensor shape
- 注意 future leakage
- 注意 rank IC
- 所有训练必须可复现

---

## Project Overview

沪深300成分股排序学习选股方案。

- **输入**: 每只股票过去 L 天（`sequence_length=20`）的量价与技术特征序列
- **模型**: `StockTransformer` — 时序编码 + 股票间交叉注意力 + 排序打分
- **输出**: 对同一天全部候选股票打分排序，取 Top5 等权 0.2 输出
- **标签**: 未来 5 日开盘价相对收益 (`ret_5d`)

数据流: `get_stock_data.py` → `split_train_test.py` → `train.py` → `predict.py` → `score_self.py`

---

## Architecture

### 目录结构

```
├── code/src/          # 核心源码
│   ├── config.py      # 统一配置（模型/训练/数据/推理参数）
│   ├── model.py       # StockTransformer + 各子模块
│   ├── train.py       # 训练/验证/损失函数/评估指标
│   ├── predict.py     # 推理/多模型融合/Top5 输出
│   └── utils.py       # 特征工程(39/158+39)/数据集构建/截面特征/特征筛选(LGB)
├── data/              # 数据目录: train.csv, test.csv, stock_data.csv
├── model/final/       # 训练产物: best_model.pth, scaler.pkl, config.json, selected_features.json
├── output/            # 推理产物: result.csv
├── test/              # 评分脚本: score_self.py, test.py, score_docker.py
├── temp/              # 临时评分结果 tmp.csv
├── train.sh           # python code/src/train.py
├── test.sh            # python code/src/predict.py
└── GUIDE.md           # 赛事方提供的完整使用文档
```

### 模型结构 (StockTransformer)

```
Input:  [B, N, L, F]  (batch, stocks, seq_len, features)
├── Linear(d_model) + PositionalEncoding
├── TransformerEncoder (2层, nhead=8)     → 时序建模 [B*N, L, d_model]
├── FeatureAttention (softmax 聚合)        → 时序聚合 [B*N, d_model]
├── Reshape → [B, N, d_model]
├── + StockEmbedding (可学习, 32维)        → 股票先验
├── CrossStockAttention (可选 MarketEncoder) → 股票间交互
├── RankingLayers (d_model→d_model/2)
└── ScoreHead (d_model/2→d_model/4→1)     → [B, N]
```

### Tensor Shapes（关键）

| 位置 | Shape | 说明 |
|------|-------|------|
| 模型输入 | `[B, N, L, F]` | B=batch, N=当日股票数, L=20, F=特征数(~200+) |
| 模型输出 | `[B, N]` | 每只股票的排序分数 |
| 时序编码 | `[B*N, L, d_model]` | 展平股票维 |
| 特征聚合 | `[B*N, d_model]` | 时间维压缩 |
| 股票交互 | `[B, N, d_model]` | 交叉注意力前后 |
| 排序层 | `[B*N, d_model/2]` | 排序特异性变换 |
| collate_fn 后 | `[B, max_stocks, L, F]` | 同 batch 补齐到当日最大股票数 |

### 损失函数

- **ListMLE**: Plackett-Luce 排序似然最大化（主损失）
- **PairwiseTopKMargin**: Top-k vs rest 间隔惩罚（辅助损失）
- 混合: `loss = ListMLE + 0.5 * PairwiseMargin`

### 特征体系

- **39 技术指标**: SMA/EMA/MACD/RSI/KDJ/Bollinger/ATR/OBV/Volatility
- **158 Alpha 特征**: K线形态/价格动量/均线偏离/波动率/相关系数/成交量/截面排名
- **截面特征**: 每日在全体股票中计算 rank% 和 z-score（约 70 个）
- **特征筛选**: LightGBM LambdaRank 选 Top-80

---

## Coding Rules

- 使用 PyTorch，优先 `torch.nn.functional` 而非手写循环
- 使用 `typing` 标注函数签名
- 禁止重复代码，函数必须单一职责
- 修改前先阅读相关文件
- 不允许破坏现有接口
- 所有随机操作必须设置 `set_seed()`
- 数据处理使用 `numpy` + `pandas`，模型部分使用 `torch`
- 多进程入口必须用 `mp.set_start_method('spawn', force=True)`

---

## Workflow

执行任务时：
1. 先分析问题
2. 给出 step-by-step plan，**必须包含实验验证方案**（用什么指标对比、如何判断改动有效）
3. 等确认再改代码
4. 修改后解释原因
5. 给出风险点
6. 改完后必须完整跑一遍训练 + 推理 + 评分，记录指标并与改动前对比
7. 如果指标下降，必须回退改动并给出原因分析

---

## Quant Rules

这是股票排序学习项目。

必须：
- **避免未来函数**
  - `shift(-1)` 构建标签时，只能使用当天已知信息
  - 验证集切分必须保留 `sequence_length - 1` 天序列上下文
  - `create_ranking_dataset_vectorized()` 中检查未来 5 日自然日连续性以排除停牌
- **注意时间泄漏**
  - 标准化 (`StandardScaler`) 必须在训练集 `fit`，验证/测试集 `transform`
  - 特征工程按单只股票独立计算，window-based 操作不能跨期
  - `split_train_val_by_last_month()` 确保验证集不会用到未来信息
- **使用 cross-sectional normalization**
  - `add_cross_sectional_features()` 每日对全部股票计算 rank% 和 z-score
  - 标签支持市值中性化 (`use_label_neutralization`)
- **复现性**
  - `set_seed(seed=123)` 固定 Python/numpy/torch/cudnn 随机种子
  - `torch.backends.cudnn.deterministic = True`
  - 配置固化到 `config.json`
- **输出 RankIC**
- **输出 ICIR**
- **输出分层收益**

---

## Training

训练时：
- 输出 loss 曲线（TensorBoard: `model/final/log/`）
- 输出验证集指标（`final_score`, `pred_return_sum`, `ratio_pred`, `test_aligned_score`）
- 保存 best checkpoint (`best_model.pth`)
- 早停 patience=10，学习率 Plateau 调度 (patience=5, factor=0.5)
- 梯度累积 4 步，有效 batch_size = `batch_size × accumulation_steps = 16`
- 训练后记录 `final_score.txt`、`config.json`

### 实验记录与回退

每次修改代码后必须：

1. **改动前备份基线**
   - 记录当前 `model/final/final_score.txt` 中的最佳 `final_score`
   - 运行一次完整 `train.py` + `predict.py` + `score_self.py` 得到参考分数
   - 将 `model/final/` 下 `best_model.pth` 和 `config.json` 备份（如 `best_model.pth.bak`）

2. **改动后复现对比**
   - `set_seed(config['base_seed'])` 确保同 seed 可比
   - 训练完成后记录新的 `final_score`
   - 运行 `predict.py` + `score_self.py` 得到新评分

3. **判定标准**
   - `final_score` 下降 → 回退改动，分析原因
   - `final_score` 持平或提升 → 保留改动，更新备份

4. **回退方法**
   - `git checkout -- <file>` 恢复代码
   - 或 `git revert` 撤销整个 commit
   - 恢复 `best_model.pth.bak` → `best_model.pth`

验证指标说明：
| 指标 | 含义 |
|------|------|
| `pred_return_sum` | 模型预测 Top5 的真实未来收益之和 |
| `max_return_sum` | 理论最优 Top5 收益之和 |
| `ratio_pred` | `pred_return_sum / max_return_sum` |
| `final_score` | `(pred - random) / (max - random)`，归一化分数 |
| `test_aligned_score` | softmax 加权 Top5 收益，与赛事方评分对齐 |

---

## Evaluation (Score)

赛事方评分逻辑 (`score_self.py`)：
1. 读取选手 `result.csv`（最多 5 只股票，权重和为 1）
2. 取 `test.csv` 中这 5 只股票的最后 5 个交易日
3. 对每只股票计算首尾开盘价收益率
4. 加权求和作为最终得分

注意：目前预测输出 `inference_weight_mode='equal'`，每只股票 0.2 等权。

---

## Key Files Quick Reference

| 文件 | 作用 | 关键函数/类 |
|------|------|-------------|
| `code/src/config.py` | 全局配置字典 | `config`, `sequence_length`, `feature_num` |
| `code/src/model.py` | StockTransformer 定义 | `StockTransformer`, `CrossStockAttention`, `PositionalEncoding` |
| `code/src/train.py` | 训练主流程 | `main()`, `_train_one_seed()`, `ListMLEPairwiseLoss`, `calculate_ranking_metrics()` |
| `code/src/predict.py` | 推理预测 Top5 | `main()`, `build_inference_sequences()` |
| `code/src/utils.py` | 特征工程 + 数据准备 | `engineer_features_158plus39()`, `create_ranking_dataset_vectorized()`, `add_cross_sectional_features()` |
| `model/final/config.json` | 当前最佳配置快照 | 训练时自动生成 |
| `test/score_self.py` | 本地评分模拟 | `calculate_predict_weight_score()` |

---

## Docker / Submission

1. `docker buildx build --platform linux/amd64 -t bdc2026 .`
2. `docker save -o 队伍名称.tar bdc2026:latest`
3. 验证: `docker compose up` → 检查 `test/output/result.csv`

---

## Common Pitfalls

- `TA-Lib` 需系统级安装，`uv sync` 只装 Python 包
- 多进程入口必须 `mp.set_start_method('spawn', force=True)`（Windows 兼容）
- 特征工程后的 `inf`/`NaN` 必须清洗: `replace([inf, -inf], nan).fillna(0)`
- 验证集 `min_window_end_date` 过滤保证评估样本在目标时间段内
- 停牌股票处理: `create_ranking_dataset_vectorized()` 检查未来 5 日日历连续性
- 股票代码需 str 并 zfill(6)
- `config.json` 变更时需同时更新 `pyproject.toml` 依赖（若有新增）
