# enzyme-stereoselectivity-prediction

基于 ESM2 蛋白质语言模型 + 随机森林的酶立体选择性预测流水线。

复现自论文 *Access to all stereoisomers of chiral alcohols with multiple stereocentres enabled by machine learning-empowered protein engineering*（Nature Synthesis, 2026）中的 **BioStrucTag 序列路（sequence-only）** 部分。

## 方法概述

```
mutations.csv + wildtype.fasta
        │  ① 按突变写法生成每条突变体的完整氨基酸序列
        ▼
   ESM2（esm2_t12_35M_UR50D）  ← 蛋白质语言模型，序列 → 张量
        │  ② 最后一层隐藏状态 (L×480) 沿序列做均值池化
        ▼
   每个突变体一个 480 维向量
        │  ③ 拼接标签（异构体占比 %）
        ▼
   随机森林回归（300 树）+ 5 折交叉验证 RMSE
```

## 目录结构

```
enzyme-stereoselectivity-prediction/
├─ data/                      # 输入数据（迁移时只改这里）
│  ├─ mutations.csv           # 训练集：突变体 + 标签
│  ├─ mutations-test.csv      # 测试集：突变体（标签可选）
│  └─ wildtype.fasta          # 野生型氨基酸序列
├─ scripts/                   # 四步流水线
│  ├─ 01_make_sequences.py    # CSV+FASTA → 突变体完整序列
│  ├─ 02_esm2_embeddings.py   # 序列 → ESM2 → 480 维向量
│  ├─ 03_train_rf.py          # 480 维向量 → 随机森林回归（训练）
│  └─ 04_predict.py           # 加载模型 → 预测新突变体
├─ outputs/                   # 训练流程产物（sequences.csv、向量、OOF 预测，不入库）
├─ outputs-test/              # 测试流程产物（向量、predictions.csv，不入库）
├─ models/                    # 模型权重（不入库）
│  ├─ esm2_local/             # ESM2 预训练权重
│  └─ rf_<label>[_N].joblib   # 训练好的随机森林（同一 label 多次训练会自动编号 _1, _2...）
└─ requirements.txt           # 依赖清单
```

## 环境搭建

```powershell
conda create -n biotag python=3.10 -y
conda activate biotag
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers scikit-learn pandas "numpy<2" joblib matplotlib tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> ESM2 权重需下载到 `models/esm2_local/`，包含 `config.json`、`vocab.txt`、`tokenizer_config.json`、`special_tokens_map.json`、`model.safetensors`。脚本会优先从本地目录加载。

## 使用方法

### 训练流程

```powershell
conda activate biotag
cd C:\Users\z1595\Desktop\enzyme-stereoselectivity-prediction

# 把训练集的 mutations.csv + wildtype.fasta 放入 data/

# ① 生成突变体序列 → outputs/sequences.csv
python scripts/01_make_sequences.py

# ② ESM2 编码 → 480 维向量 → outputs/1D_embeddings/*.pt
python scripts/02_esm2_embeddings.py

# ③ 训练随机森林（--label 选标签列）→ models/rf_<label>.joblib
python scripts/03_train_rf.py --label SS
```

> **模型自动编号**：同一 label 重复训练时，新模型自动命名为 `rf_<label>_1.joblib`、`rf_<label>_2.joblib`...，不会覆盖已有模型。

### 预测流程

训练完成后，把测试集数据放入 `data/`，运行：

```powershell
# 把测试集的 mutations-test.csv + wildtype.fasta 放入 data/（FASTA 可能是另一种酶）

# ①② 对测试集生成序列和向量，输出到 outputs-test/
python scripts/01_make_sequences.py --out-dir outputs-test
python scripts/02_esm2_embeddings.py --out-dir outputs-test

# ④ 加载训练好的模型预测（不重新训练）→ outputs-test/predictions_<label>.csv
python scripts/04_predict.py --label SS
```

> `04_predict.py` 输出列为 `Mutations, Predicted`。
> 若 `mutations-test.csv` 带该标签列且有数值，会额外输出 `y_true`、`error` 两列并打印 RMSE。
> 若该 label 有多个模型，会交互式列出供选择。

## 数据格式

### 通用规则

两个文件都放在 `data/` 下：

- `wildtype.fasta`：野生型氨基酸序列（纯文本，一行或多行氨基酸字母）
- `mutations.csv`：突变体列表，第一列必须叫 `Mutations`

`Mutations` 列写法：
- `WT` = 野生型
- `A93F` = 第 93 位氨基酸由 A 突变为 F
- 多点突变用 `/` 连接，如 `A93Y/L194M/V195G`

### 训练集 mutations.csv（必须有标签列）

训练时需要真实标签，CSV 里 `Mutations` 后面跟一个或多个数值标签列：

```csv
Mutations,RS,SR,SS,RR,2OH
WT,0,0,10,90,0
A93F,0,0,24,76,0
E144A,2.8,0,87.4,9.9,0
```

- 标签列名随意（如 `SS`、`activity`、`yield`），训练时用 `--label 列名` 指定
- 数值可以是百分比、活性值等任意回归目标

### 预测集 mutations-test.csv（只需 Mutations 列）

预测时只需突变体列表，标签列可选。输入情况及对应输出（以 `--label SS` 为例）：

| 输入情况 | 输入 CSV | 输出 CSV |
|---|---|---|
| ① 仅 Mutations 列 | `Mutations` | `Mutations, Predicted` |
| ② Mutations + SS（有数值，用于评估） | `Mutations, SS`（有数值） | `Mutations, Predicted, y_true, error` |
| ③ CSV 有标签列但列名与 `--label` 不一致 | `Mutations, activity` | 终端报错，不输出文件 |

情况②中 `y_true` 为真实值、`error = Predicted - y_true`，并在终端打印 RMSE。
情况③会在终端提示可用的列名，修正 `--label` 后重跑。

情况①示例输入：

```csv
Mutations
A93F
E144A
L194F
```

## 迁移到其他实验

只需替换 `data/` 下的文件，脚本无需修改：

1. `wildtype.fasta` → 对应酶的野生型序列（训练集和测试集可不同）
2. `mutations.csv` → 训练集（必须有标签列）
3. `mutations-test.csv` → 测试集（只需 Mutations 列，标签可选）

按"训练流程"和"预测流程"分别运行即可。

## 参考

- 论文：[Nature Synthesis 2026, DOI: 10.1038/s44160-026-01144-y](https://doi.org/10.1038/s44160-026-01144-y)
- 官方代码：[zhoujiahui01/BioStrucTag](https://github.com/zhoujiahui01/BioStrucTag)
- ESM2 模型：[facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
