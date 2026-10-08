"""
步骤 ③：用 480 维 ESM2 向量训练随机森林，预测某个异构体的产物占比

用法:
  python scripts/03_train_rf.py --label SS
  （--label 可选 RS / SR / SS / RR / 2OH，即 mutations.csv 里的标签列名）

产物:
  models/rf_<label>.joblib                    —— 全量数据训练的最终模型
  outputs/oof_predictions_<label>.csv         —— 5 折交叉验证的"留一折"预测，方便画 parity 图
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"


def load_dataset(label: str):
    """读标签 CSV + 1D_embeddings 下的向量，对齐成特征矩阵 X 和标签 y。"""
    df = pd.read_csv(DATA / "mutations.csv")

    embeddings = {}
    for pt in (OUTPUTS / "1D_embeddings").glob("*.pt"):
        obj = torch.load(pt, map_location="cpu", weights_only=False)
        embeddings[pt.stem] = obj["embedding"].numpy()

    X, y, names = [], [], []
    missing = []
    for _, row in df.iterrows():
        # strip + / 换成 _，保证和步骤②保存的文件名一致
        key = str(row["Mutations"]).strip().replace("/", "_")
        if key not in embeddings:
            missing.append(key)
            continue
        X.append(embeddings[key])
        y.append(float(row[label]))
        names.append(key)

    if missing:
        print(f"警告: {len(missing)} 个突变体找不到向量，已跳过: {missing}")

    return np.stack(X), np.array(y), names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="mutations.csv 中的目标标签列名，如 SS")
    args = parser.parse_args()

    MODELS.mkdir(exist_ok=True)
    X, y, names = load_dataset(args.label)
    print(f"特征矩阵 X: {X.shape}（样本数 × 480 维）  标签 y: {y.shape}")

    # ---------- 5 折交叉验证 ----------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros_like(y, dtype=float)  # 每个样本"没见过它的那一折"给的预测
    fold_rmses = []

    for fold, (tr, te) in enumerate(kf.split(X), 1):
        model = RandomForestRegressor(
            n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        oof_pred[te] = pred
        rmse = np.sqrt(mean_squared_error(y[te], pred))
        fold_rmses.append(rmse)
        print(f"  第 {fold} 折 RMSE = {rmse:.3f}")

    print(
        f"\n5 折 RMSE: {np.mean(fold_rmses):.3f} ± {np.std(fold_rmses):.3f}"
        f"   （OOF 整体 R² = {r2_score(y, oof_pred):.3f}）"
    )

    # 保存交叉验证预测（后续可画"预测 vs 真实"散点图）
    oof_df = pd.DataFrame({"Mutations": names, "y_true": y, "y_pred_oof": oof_pred})
    oof_path = OUTPUTS / f"oof_predictions_{args.label}.csv"
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")

    # ---------- 用全部数据训练最终模型并保存 ----------
    final = RandomForestRegressor(
        n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
    )
    final.fit(X, y)
    model_path = MODELS / f"rf_{args.label}.joblib"
    dump(final, model_path)

    print(f"OOF 预测已存: {oof_path}")
    print(f"最终模型已存: {model_path}")


if __name__ == "__main__":
    main()
