"""
步骤 ①：由 mutations.csv + wildtype.fasta 生成每个突变体的完整氨基酸序列

输入:
  data/mutations.csv   —— 必须有 Mutations 列；WT=野生型，点突变如 A93F，多点用 / 连接
  data/wildtype.fasta  —— 野生型序列
输出:
  outputs/sequences.csv —— 列: Mutations, Sequences, Status

突变写法 A93F 的含义: 序列第 93 位（从 1 开始数）的 A 被替换成 F。
脚本会校验野生型该位置确实是 A，防止位置写错。
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")  # 20 种标准氨基酸


def read_fasta(path: Path) -> str:
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip())
    sequence = "".join(seq).upper()
    if not sequence:
        raise ValueError(f"FASTA 里没有读到序列: {path}")
    return sequence


def apply_mutations(wt_seq: str, muts: str) -> str:
    """把 'A93F/L194M' 这样的写法作用到野生型序列上，返回突变后的序列。"""
    seq = wt_seq
    for m in muts.split("/"):
        m = m.strip()
        if len(m) < 3 or not m[1:-1].isdigit():
            raise ValueError(f"突变写法不合法: {m}（应为 字母+数字+字母，如 A93F）")
        old, pos, new = m[0], int(m[1:-1]), m[-1]
        if old not in VALID_AA or new not in VALID_AA:
            raise ValueError(f"突变 {m} 中出现了非标准氨基酸字母")
        if not 1 <= pos <= len(seq):
            raise ValueError(f"突变 {m} 的位置超出序列长度 {len(seq)}")
        if seq[pos - 1] != old:
            raise ValueError(
                f"突变 {m} 对不上：野生型第 {pos} 位是 {seq[pos - 1]}，不是 {old}"
            )
        seq = seq[: pos - 1] + new + seq[pos:]
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs", help="输出目录名（默认 outputs；测试集用 outputs-test）")
    args = parser.parse_args()

    OUTPUTS = ROOT / args.out_dir
    OUTPUTS.mkdir(exist_ok=True)
    input_csv = DATA / "mutations.csv"
    fasta = DATA / "wildtype.fasta"
    output_csv = OUTPUTS / "sequences.csv"

    wt_seq = read_fasta(fasta)
    print(f"野生型序列长度: {len(wt_seq)} aa")

    rows_out, ok, bad = [], 0, 0
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "Mutations" not in reader.fieldnames:
            raise ValueError("mutations.csv 必须有 'Mutations' 列")
        for i, row in enumerate(reader, 1):
            muts = row["Mutations"].strip().upper()
            try:
                if muts == "WT":
                    rows_out.append({"Mutations": "WT", "Sequences": wt_seq, "Status": "Wildtype"})
                else:
                    mut_seq = apply_mutations(wt_seq, muts)
                    rows_out.append({"Mutations": muts, "Sequences": mut_seq, "Status": "Success"})
                ok += 1
            except Exception as e:
                bad += 1
                rows_out.append({"Mutations": muts, "Sequences": "INVALID", "Status": str(e)})
                print(f"[第 {i} 行] {e}", file=sys.stderr)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Mutations", "Sequences", "Status"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"完成: 成功 {ok} 条，失败 {bad} 条 -> {output_csv}")


if __name__ == "__main__":
    main()
