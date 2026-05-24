"""评估所有 seed 的单模型表现 - 直接推理版本"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path

# 添加源码路径
sys.path.insert(0, "D:/zhuomian/THU-BDC2026-main/code/src")
from config import config
from model import StockTransformer
from predict import preprocess_predict_data, build_inference_sequences

MODEL_DIR = Path("D:/zhuomian/THU-BDC2026-main/model/60_158+39_v3")
OUTPUT_DIR = Path("D:/zhuomian/THU-BDC2026-main/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
data_file = os.path.join(config['data_path'], 'train.csv')
scaler_path = MODEL_DIR / 'scaler.pkl'
raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
raw_df['日期'] = pd.to_datetime(raw_df['日期'])
latest_date = raw_df['日期'].max()
stock_ids = sorted(raw_df['股票代码'].unique())
stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

print(f"数据: {len(raw_df)} 行, {len(stock_ids)} 只股票, 最新日期: {latest_date.date()}")

# 特征工程 (只需做一次)
processed, features = preprocess_predict_data(raw_df, stockid2idx)
selected_path = MODEL_DIR / 'selected_features.json'
if selected_path.exists():
    with open(selected_path) as f:
        features = json.load(f)
    print(f"使用筛选后特征: {len(features)} 个")

processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
scaler = joblib.load(scaler_path)
processed[features] = scaler.transform(processed[features])

sequence_length = config['sequence_length']
sequences_np, sequence_stock_ids = build_inference_sequences(
    processed, features, sequence_length, stock_ids, latest_date,
)
print(f"推理序列: {sequences_np.shape}, 股票数: {len(sequence_stock_ids)}")

# 设备
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"设备: {device}")

# 逐个 seed 评估
seeds = sorted([d.name for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith("seed")])
results = {}

for seed_name in seeds:
    seed_dir = MODEL_DIR / seed_name
    model_path = seed_dir / "best_model.pth"
    if not model_path.exists():
        print(f"  [SKIP] {seed_name}: 无模型文件")
        continue

    # 读取 val score
    val_score = None
    val_score_path = seed_dir / "final_score.txt"
    if val_score_path.exists():
        with open(val_score_path) as f:
            for line in f:
                if "Best final_score" in line:
                    val_score = float(line.split(":")[-1].strip())

    # 加载模型并推理
    m = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device)
    m.eval()

    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        scores = m(x).squeeze(0).detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    ranked = [sequence_stock_ids[i] for i in order]
    top5_stocks = ranked[:5]
    top5_scores = scores[order][:5]

    # Equal 权重
    top5_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    result_df = pd.DataFrame({
        'stock_id': top5_stocks,
        'weight': top5_weights,
    })
    output_path = OUTPUT_DIR / f"result_{seed_name}.csv"
    result_df.to_csv(output_path, index=False)

    results[seed_name] = {
        "val_score": val_score,
        "top5": top5_stocks.tolist(),
        "top5_scores": top5_scores.tolist(),
    }
    print(f"  {seed_name}: val={val_score:.6f}, top5={top5_stocks.tolist()}")

# 保存结果
with open(OUTPUT_DIR / "seed_eval_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n结果已保存: {OUTPUT_DIR / 'seed_eval_results.json'}")

# 打印汇总
print("\n=== 汇总 (val_score → top5) ===")
for seed_name, r in results.items():
    print(f"  {seed_name}: val={r['val_score']:.6f} → {r['top5']}")
