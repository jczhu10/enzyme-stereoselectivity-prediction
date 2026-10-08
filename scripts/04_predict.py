"""
步骤 ④：用训练好的随机森林模型预测新突变体的标签值

用法:
  python scripts/04_predict.py --label SS

前提:
  1. 已把测试集的 mutations.csv + wildtype.fasta 放入 data/
  2. 已运行 01_make_sequences.py 和 02_esm2_embeddings.py 生成测试集向量
  3. models/ 下有训练好的 rf_<label>.joblib

输出:
  outputs/predictions_<label>.csv —— 列: Mutations, Predicted
  （如果 mutations.csv 里有该标签列，会额外输出 y_true 和误差列，方便评估）
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="要预测的标签列名，需和训练时一致，如 SS")
    args = parser.parse_args()

    # ---------- 检查依赖 ----------
    model_path = MODELS / f"rf_{args.label}.joblib"
    emb_dir = OUTPUTS / "1D_embeddings"
    seq_csv = OUTPUTS / "sequences.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型 {model_path}，请先运行 03_train_rf.py --label {args.label}")
    if not emb_dir.exists() or not any(emb_dir.glob("*.pt")):
        raise FileNotFoundError(f"找不到 {emb_dir} 下的向量，请先运行 01 和 02 生成测试集向量")
    if not seq_csv.exists():
        raise FileNotFoundError(f"找不到 {seq_csv}，请先运行 01_make_sequences.py")

    # ---------- 加载模型 ----------
    model = load(model_path)
    print(f"已加载模型: {model_path}")

    # ---------- 读取测试集向量 + 突变名 ----------
    df_seq = pd.read_csv(seq_csv)
    df_seq = df_seq[df_seq["Status"].isin(["Success", "Wildtype"])]
    df_seq["Mutations"] = df_seq["Mutations"].fillna("WT").str.replace("/", "_", regex=False)

    X, names = [], []
    missing = []
    for _, row in df_seq.iterrows():
        pt_file = emb_dir / f"{row['Mutations']}.pt"
        if not pt_file.exists():
            missing.append(row["Mutations"])
            continue
        obj = torch.load(pt_file, map_location="cpu", weights_only=False)
        X.append(obj["embedding"].numpy())
        names.append(row["Mutations"])

    if missing:
        print(f"警告: {len(missing)} 个突变体缺少向量，已跳过: {missing}")

    X = np.stack(X)
    print(f"测试集特征矩阵: {X.shape}")

    # ---------- 预测 ----------
    preds = model.predict(X)

    out_df = pd.DataFrame({"Mutations": names, "Predicted": preds})

    # 如果测试集 CSV 里有该标签列，附上真实值和误差
    test_csv = DATA / "mutations.csv"
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        if args.label in df_test.columns:
            df_test["Mutations"] = df_test["Mutations"].astype(str).str.replace("/", "_", regex=False)
            truth_map = dict(zip(df_test["Mutations"], df_test[args.label]))
            out_df["y_true"] = out_df["Mutations"].map(truth_map)
            out_df["error"] = out_df["Predicted"] - out_df["y_true"]
            valid = out_df["y_true"].notna()
            if valid.sum() > 0:
                rmse = np.sqrt(np.mean((out_df.loc[valid, "Predicted"] - out_df.loc[valid, "y_true"]) ** 2))
                print(f"测试集 RMSE = {rmse:.3f}（基于 {valid.sum()} 个有标签样本）")

    out_path = OUTPUTS / f"predictions_{args.label}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"预测结果已保存: {out_path}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
