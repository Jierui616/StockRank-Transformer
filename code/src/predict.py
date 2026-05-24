import os
import json
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39


feature_cloums_map = {
	'39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
		'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
		'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
		'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
		'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
		'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
		'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
		'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
		'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
		'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
		'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
		'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
}


def preprocess_predict_data(df, stockid2idx):
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = feature_cloums_map[config['feature_num']]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])

	from utils import add_cross_sectional_features
	processed, cs_features = add_cross_sectional_features(processed, feature_columns)
	all_features = feature_columns + cs_features

	return processed, all_features


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=str, default=None)
	parser.add_argument('--output', type=str, default=None)
	_args = parser.parse_args()

	data_file = os.path.join(config['data_path'], 'train.csv')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	output_path = _args.output if _args.output else os.path.join('./output/', 'result.csv')

	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

	processed, features = preprocess_predict_data(raw_df, stockid2idx)

	# 读取训练时保存的特征筛选结果
	selected_path = os.path.join(config['output_dir'], 'selected_features.json')
	if os.path.exists(selected_path):
		with open(selected_path, 'r') as f:
			features = json.load(f)
		print(f"使用筛选后特征: {len(features)} 个")

	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	scaler = joblib.load(scaler_path)
	processed[features] = scaler.transform(processed[features])

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	# 多模型融合：扫描 seed{N}/ 目录，加载所有模型
	model_paths = []
	output_dir = config['output_dir']

	# 0. 单 seed 推理 (命令行参数)
	if _args.seed:
		seed_dir = os.path.join(output_dir, f'seed{_args.seed}')
		mp = os.path.join(seed_dir, 'best_model.pth')
		if os.path.exists(mp):
			model_paths = [mp]
			print(f'单模型推理: seed{_args.seed}')
		else:
			raise FileNotFoundError(f'未找到模型: {mp}')

	# 1. Snapshot ensemble (Plan C)
	snapshot_manifest = os.path.join(output_dir, 'snapshots', 'scores.json')
	snapshot_threshold = config.get('snapshot_val_threshold', 0.05)
	snapshot_min_count = config.get('snapshot_min_count', 2)
	if os.path.exists(snapshot_manifest):
		with open(snapshot_manifest, 'r') as f:
			snapshot_info = json.load(f)
		filtered = [s for s in snapshot_info if abs(s['val_score']) < snapshot_threshold]
		if len(filtered) >= snapshot_min_count:
			for s in filtered:
				mp = os.path.join(output_dir, 'snapshots', s['path'])
				if os.path.exists(mp):
					model_paths.append(mp)
			print(f'Snapshot 融合: {len(filtered)}/{len(snapshot_info)} 个快照通过阈值筛选 (|val| < {snapshot_threshold})')
			for s in filtered:
				print(f'  cycle={s["cycle"]}, epoch={s["epoch"]}, val_score={s["val_score"]:.4f}')
		elif len(filtered) == 1:
			print(f'Snapshot: 仅 {len(filtered)} 个快照通过筛选，使用单个快照')
			model_paths.append(os.path.join(output_dir, 'snapshots', filtered[0]['path']))
		else:
			print(f'Snapshot: {len(filtered)} 个快照通过筛选 (阈值={snapshot_threshold})，回退到最佳模型')
			mp = os.path.join(output_dir, 'best_model.pth')
			if os.path.exists(mp):
				model_paths = [mp]

	# 2. Multi-seed 回退
	if not model_paths and os.path.isdir(output_dir):
		for name in sorted(os.listdir(output_dir)):
			seed_dir = os.path.join(output_dir, name)
			if name.startswith('seed') and os.path.isdir(seed_dir):
				mp = os.path.join(seed_dir, 'best_model.pth')
				if os.path.exists(mp):
					model_paths.append(mp)

	# 3. 单模型回退
	if not model_paths:
		mp = os.path.join(output_dir, 'best_model.pth')
		if os.path.exists(mp):
			model_paths = [mp]

	if not model_paths:
		raise FileNotFoundError(f'未找到任何模型文件')

	print(f'加载 {len(model_paths)} 个模型进行融合')
	if len(model_paths) > 1:
		for mp in model_paths:
			print(f'  {os.path.basename(os.path.dirname(mp))}/{os.path.basename(mp)}')

	# 加载所有模型
	models = []
	for mp in model_paths:
		m = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
		m.load_state_dict(torch.load(mp, map_location=device))
		m.to(device)
		m.eval()
		models.append(m)

	# 融合推理
	ensemble_mode = config.get('ensemble_mode', 'mean')
	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
		all_scores = []
		for m in models:
			s = m(x).squeeze(0).detach().cpu().numpy()  # [N]
			all_scores.append(s)
		all_scores = np.array(all_scores)  # [M, N]

		if len(models) == 1:
			scores = all_scores[0]
		elif ensemble_mode == 'rank':
			# Rank 融合：每个模型内转排名，均值排名 → 排序
			all_ranks = np.zeros_like(all_scores)
			for i in range(len(models)):
				all_ranks[i] = len(sequence_stock_ids) - 1 - np.argsort(np.argsort(all_scores[i]))
			avg_ranks = np.mean(all_ranks, axis=0)  # 平均排名
			scores = -avg_ranks  # 排名越小越好 → 取负后越大越好
			print(f'融合模式: rank (平均排名)')
		elif ensemble_mode == 'median':
			scores = np.median(all_scores, axis=0)
			print(f'融合模式: median (中位数分数)')
		else:  # 'mean'
			scores = np.mean(all_scores, axis=0)
			print(f'融合模式: mean (均值分数)')

	order = np.argsort(scores)[::-1]
	ranked_stock_ids = [sequence_stock_ids[i] for i in order]

	if len(ranked_stock_ids) < 5:
		raise ValueError(f'可预测股票不足5只，当前仅有 {len(ranked_stock_ids)} 只')

	# Top5 权重分配策略 (A/B test: score_linear > top1_only > rank_linear > softmax)
	top5_stocks = ranked_stock_ids[:5]
	top5_scores = scores[order][:5]
	weight_mode = config.get('inference_weight_mode', 'score_linear')
	temperature = config.get('inference_temperature', 1.0)

	if weight_mode == 'equal':
		top5_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
	elif weight_mode == 'rank_linear':
		top5_weights = np.array([5/15, 4/15, 3/15, 2/15, 1/15])
	elif weight_mode == 'score_linear':
		s_min = top5_scores.min()
		s_range = top5_scores.max() - s_min
		if s_range > 1e-8:
			top5_weights = (top5_scores - s_min) / s_range
		else:
			top5_weights = np.ones(5)
		top5_weights = top5_weights.astype(np.float64)
		top5_weights = top5_weights / top5_weights.sum()
		top5_weights = np.clip(top5_weights, 0, 1)
	else:  # softmax
		top5_scores_f64 = top5_scores.astype(np.float64)
		top5_weights = softmax(top5_scores_f64 / max(temperature, 1e-8))
		top5_weights = top5_weights / top5_weights.sum()

	output_df = pd.DataFrame({
		'stock_id': top5_stocks,
		'weight': top5_weights,
	})
	output_df.to_csv(output_path, index=False)

	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(ranked_stock_ids)}')
	if temperature != 1.0:
		print(f'Temperature: {temperature}')
	print(f'Top-5: {dict(zip(top5_stocks, top5_weights))}')
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
