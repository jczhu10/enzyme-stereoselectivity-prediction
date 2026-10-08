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
│  ├─ mutations.csv           # 突变体 + 异构体占比标签
│  └─ wildtype.fasta          # 野生型氨基酸序列
├─ scripts/                   # 四步流水线
│  ├─ 01_make_sequences.py    # CSV+FASTA → 突变体完整序列
│  ├─ 02_esm2_embeddings.py   # 序列 → ESM2 → 480 维向量
│  ├─ 03_train_rf.py          # 480 维向量 → 随机森林回归（训练）
│  └─ 04_predict.py           # 加载模型 → 预测新突变体
├─ outputs/                   # 预测流程产物（predictions.csv 等，不入库）
├─ outputs_test/              # 训练流程产物（sequences.csv、向量、OOF 预测，不入库）
├─ models/                    # 模型权重（不入库）
│  ├─ esm2_local/             # ESM2 预训练权重
│  └─ rf_<label>.joblib       # 训练好的随机森林
├─ references/                # 论文参考资料（不入库）
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

# ① 生成突变体序列
python scripts/01_make_sequences.py

# ② ESM2 编码 → 480 维向量
python scripts/02_esm2_embeddings.py

# ③ 训练随机森林（--label 选标签列）→ 保存模型到 models/rf_<label>.joblib
python scripts/03_train_rf.py --label SS
```

### 预测流程

训练完成后，清空 `data/`，放入测试集数据，运行：

```powershell
# 把测试集的 mutations.csv + wildtype.fasta 放入 data/（FASTA 可能是另一种酶）

# ①② 对测试集重复序列生成和 ESM2 编码
python scripts/01_make_sequences.py
python scripts/02_esm2_embeddings.py

# ④ 加载训练好的模型预测（不重新训练）→ outputs/predictions_<label>.csv
python scripts/04_predict.py --label SS
```

> `04_predict.py` 会自动检测测试集 CSV 中是否有对应标签列，有则输出真实值和 RMSE，无则只输出预测值。

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

### 预测集 mutations.csv（只需 Mutations 列）

预测时只需突变体列表，**标签列可选**：

```csv
Mutations
A93F
E144A
L194F
```

如果预测集 CSV 里也带了标签列（用于评估模型效果），`04_predict.py` 会自动计算 RMSE 并在结果中附上真实值。

## 运行结果（示例数据）

| 项目 | 数值 |
|---|---|
| 样本数 | 74 |
| 特征维度 | 480（ESM2 均值池化） |
| 模型 | 随机森林（300 树，max_depth=20） |
| 5 折 RMSE（SS 标签） | 25.85 ± 5.62 |
| OOF R² | 0.097 |

> 序列路本身预测能力有限（结构信息才是立体选择性的关键），此处仅跑通流程。

## 迁移到其他实验

只需替换 `data/` 下的两个文件，脚本无需修改：

1. `wildtype.fasta` → 对应酶的野生型序列（训练集和测试集可不同）
2. `mutations.csv` → 按"数据格式"一节准备训练集或预测集

按"训练流程"和"预测流程"分别运行即可。

## 参考

- 论文：[Nature Synthesis 2026, DOI: 10.1038/s44160-026-01144-y](https://doi.org/10.1038/s44160-026-01144-y)
- 官方代码：[zhoujiahui01/BioStrucTag](https://github.com/zhoujiahui01/BioStrucTag)
- ESM2 模型：[facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
