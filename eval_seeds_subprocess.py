"""逐个 seed 评估 - 使用 subprocess 避免 multiprocessing spawn 问题"""
import os, sys, json, subprocess, shutil, re
from pathlib import Path

MODEL_DIR = Path("D:/zhuomian/THU-BDC2026-main/model/60_158+39_v3")
OUTPUT_DIR = Path("D:/zhuomian/THU-BDC2026-main/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CWD = "D:/zhuomian/THU-BDC2026-main"

seeds = sorted([d.name[4:] for d in MODEL_DIR.iterdir()
                if d.is_dir() and d.name.startswith("seed")])

results = {}

for seed_name in seeds:
    print(f"\n{'='*50}")
    print(f"评估 seed{seed_name}...")

    # 运行 predict.py --seed
    result = subprocess.run(
        ["uv", "run", "python", "code/src/predict.py", "--seed", seed_name,
         "--output", f"./output/result_{seed_name}.csv"],
        cwd=CWD, capture_output=True, text=True, timeout=300
    )
    print(result.stdout.strip()[-500:] if result.stdout else "(no stdout)")
    if result.stderr:
        lines = [l for l in result.stderr.split('\n')
                if 'FutureWarning' not in l and 'RuntimeWarning' not in l]
        if lines:
            print("STDERR:", ''.join(lines[-5:]))

    if result.returncode != 0:
        print(f"  [FAIL] seed{seed_name}: predict.py returned {result.returncode}")
        results[seed_name] = {"val_score": None, "test_score": None, "error": "predict failed"}
        continue

    # Copy result to result.csv for score_self.py
    seed_result = OUTPUT_DIR / f"result_{seed_name}.csv"
    main_result = OUTPUT_DIR / "result.csv"
    if seed_result.exists():
        shutil.copy(seed_result, main_result)

    # 运行 score_self.py
    score_result = subprocess.run(
        ["uv", "run", "python", "test/score_self.py"],
        cwd=CWD, capture_output=True, text=True, timeout=60
    )

    score = None
    for line in score_result.stdout.split('\n'):
        match = re.search(r'([-]?\d+\.\d+)', line)
        if match:
            score = float(match.group(1))

    # 读取 val score
    val_score_path = MODEL_DIR / f"seed{seed_name}" / "final_score.txt"
    val_score = None
    if val_score_path.exists():
        with open(val_score_path) as f:
            for line in f:
                if "Best final_score" in line:
                    val_score = float(line.split(":")[-1].strip())

    results[seed_name] = {"val_score": val_score, "test_score": score}
    print(f"  seed{seed_name}: val={val_score}, test={score}")

# 汇总
print("\n" + "="*60)
print("=== 汇总 (val vs test score) ===")
print("="*60)
for seed_name in sorted(results.keys(), key=lambda s: int(s)):
    r = results[seed_name]
    print(f"  seed{seed_name}: val={r['val_score']:>10.6f}  test={r['test_score']}")

with open(OUTPUT_DIR / "seed_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

# 计算每个 seed 的 top5 股票，检查重叠
print("\n=== Top-5 股票 ===")
all_top5 = {}
for seed_name in sorted(results.keys(), key=lambda s: int(s)):
    result_csv = OUTPUT_DIR / f"result_{seed_name}.csv"
    if result_csv.exists():
        import pandas as pd
        df = pd.read_csv(result_csv)
        stocks = df['stock_id'].tolist()
        all_top5[seed_name] = stocks
        print(f"  seed{seed_name}: {stocks}")

# 计算 pairwise 重叠
print("\n=== Top-5 重叠矩阵 ===")
seed_list = sorted(all_top5.keys(), key=lambda s: int(s))
for i, s1 in enumerate(seed_list):
    overlaps = []
    for j, s2 in enumerate(seed_list):
        if i == j:
            overlaps.append(" - ")
        else:
            common = len(set(all_top5[s1]) & set(all_top5[s2]))
            overlaps.append(f" {common} ")
    print(f"  seed{s1}: " + " | ".join(overlaps))
