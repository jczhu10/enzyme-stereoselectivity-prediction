"""
步骤 ④：用训练好的随机森林模型预测新突变体的标签值

用法:
  python scripts/04_predict.py --label SS

前提:
  1. 已把测试集的 mutations-test.csv + wildtype.fasta 放入 data/
  2. 已运行 01 和 02（加 --out-dir outputs-test）生成测试集向量
  3. models/ 下有训练好的 rf_<label>*.joblib

输出:
  outputs-test/predictions_<label>.csv —— 列: Mutations, Predicted
  （如果 mutations-test.csv 里有该标签列且有数值，会额外输出 y_true 和 error）
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS_TEST = ROOT / "outputs-test"
MODELS = ROOT / "models"


def select_model(label: str) -> Path:
    """查找该 label 的所有模型，0 个报错，1 个直接用，多个交互式选择。"""
    pattern = f"rf_{label}*.joblib"
    candidates = []
    for f in MODELS.glob(pattern):
        suffix = f.stem[len(f"rf_{label}"):]
        if suffix == "" or (suffix.startswith("_") and suffix[1:].isdigit()):
            candidates.append(f)
    candidates = sorted(candidates)

    if not candidates:
        raise FileNotFoundError(
            f"找不到 {MODELS / pattern}，请先运行 03_train_rf.py --label {label}"
        )
    if len(candidates) == 1:
        return candidates[0]

    print(f"\n找到 {len(candidates)} 个 {label} 标签的模型，请选择：")
    for i, f in enumerate(candidates):
        print(f"  [{i}] {f.name}")
    while True:
        try:
            idx = int(input("输入编号: ").strip())
            if 0 <= idx < len(candidates):
                return candidates[idx]
            print(f"请输入 0 ~ {len(candidates) - 1} 之间的数字")
        except ValueError:
            print("请输入数字")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="要预测的标签列名，需和训练时一致，如 SS")
    args = parser.parse_args()

    # ---------- 检查依赖 ----------
    emb_dir = OUTPUTS_TEST / "1D_embeddings"
    seq_csv = OUTPUTS_TEST / "sequences.csv"

    if not emb_dir.exists() or not any(emb_dir.glob("*.pt")):
        raise FileNotFoundError(f"找不到 {emb_dir} 下的向量，请先运行 01 和 02（加 --out-dir outputs-test）")
    if not seq_csv.exists():
        raise FileNotFoundError(f"找不到 {seq_csv}，请先运行 01_make_sequences.py --out-dir outputs-test")

    # ---------- 选择并加载模型 ----------
    model_path = select_model(args.label)
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

    # 如果测试集 CSV 里有该标签列且包含有效数值，附上真实值和误差
    test_csv = DATA / "mutations-test.csv"
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        other_cols = [c for c in df_test.columns if c != "Mutations"]
        if args.label not in df_test.columns:
            if other_cols:
                raise ValueError(
                    f"测试集 CSV 中找不到标签列 '{args.label}'。"
                    f"可用列: {other_cols}。请用 --label 指定正确的列名。"
                )
            # 只有 Mutations 列，无标签，正常输出预测值
        else:
            df_test["Mutations"] = df_test["Mutations"].astype(str).str.replace("/", "_", regex=False)
            truth_map = dict(zip(df_test["Mutations"], pd.to_numeric(df_test[args.label], errors="coerce")))
            y_true = out_df["Mutations"].map(truth_map)
            valid = y_true.notna()
            if valid.sum() > 0:
                out_df["y_true"] = y_true
                out_df["error"] = out_df["Predicted"] - y_true
                rmse = np.sqrt(np.mean((out_df.loc[valid, "Predicted"] - y_true[valid]) ** 2))
                print(f"测试集 RMSE = {rmse:.3f}（基于 {valid.sum()} 个有标签样本）")

    out_path = OUTPUTS_TEST / f"predictions_{args.label}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"预测结果已保存: {out_path}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
