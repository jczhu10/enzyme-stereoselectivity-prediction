"""
步骤 ②：用 ESM2 把每条氨基酸序列编码成 480 维向量

做的事（对应论文 Methods）:
  序列 (251 个氨基酸)
    -> ESM2_t12_35M_UR50D 最后一层隐藏状态，形状 (1, 251, 480)
    -> 沿序列维度做均值池化 (mean pooling)
    -> 得到 480 维向量，存成 outputs/1D_embeddings/<突变名>.pt

首次运行会自动下载 ESM2 权重（约 140 MB）到 models/esm/。
国内网络自动走 hf-mirror 镜像。
"""
import argparse
import os
# 如直连 HuggingFace 不畅，可在运行前设置环境变量走国内镜像：
#   PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"  # 论文用的同款 35M 小模型，480 维
LOCAL_MODEL = MODELS / "esm2_local"           # 已下载到本地的权重目录


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs", help="输入输出目录名（默认 outputs；测试集用 outputs-test）")
    args = parser.parse_args()

    OUTPUTS = ROOT / args.out_dir
    input_csv = OUTPUTS / "sequences.csv"
    out_dir = OUTPUTS / "1D_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"找不到 {input_csv}，请先运行 01_make_sequences.py")

    df = pd.read_csv(input_csv)
    df = df[df["Status"].isin(["Success", "Wildtype"])]  # 跳过失败行
    # 文件名里不能有 /，统一换成 _
    df["Mutations"] = df["Mutations"].fillna("WT").str.replace("/", "_", regex=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}")

    # 优先用本地权重；没有则从 HuggingFace 下载
    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else MODEL_NAME
    print(f"加载模型: {model_path}")

    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path, add_pooling_layer=False).to(device)
    model.eval()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ESM2 编码", ncols=80):
        mutation, sequence = row["Mutations"], row["Sequences"]
        out_file = out_dir / f"{mutation}.pt"
        if out_file.exists():
            continue  # 已算过的跳过，中断后可重跑

        inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # 最后一层隐藏状态 (1, L, 480) 沿序列维均值池化 -> (480,)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

        torch.save(
            {"embedding": embedding, "mutation": mutation, "sequence": sequence},
            out_file,
        )

    print(f"\n完成，向量保存在: {out_dir}")


if __name__ == "__main__":
    main()
