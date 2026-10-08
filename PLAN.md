# BioStrucTag（序列路）复现与迁移执行方案

> 论文: *Access to all stereoisomers of chiral alcohols with multiple stereocentres enabled by
> machine learning-empowered protein engineering*, Nature Synthesis (2026),
> DOI `10.1038/s44160-026-01144-y`
> 官方代码: https://github.com/zhoujiahui01/BioStrucTag （已 clone 到 `C:\Users\z1595\Desktop\BioStrucTag`，只读参考）

## 0. 目标与范围（2026-10-07 已与导师确认）

**只做序列路（sequence-only）**，不做结构/体素/3D CNN：

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

后续迁移到其他实验时，只需替换 `data/` 里的 FASTA 和 CSV（一个突变体一行 + 数值标签列）。

## 1. 环境（Windows + RTX 5060）

```powershell
conda create -n biotag python=3.10 -y
conda activate biotag
# torch 2.10 + cu128（官方源不稳时用阿里云镜像）
pip install torch -f https://mirrors.aliyun.com/pytorch-wheels/cu128/ -i https://pypi.tuna.tsinghua.edu.cn/simple
# 其余包走清华源
pip install transformers scikit-learn pandas joblib matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

ESM2 权重从 HuggingFace 拉取；国内网络由脚本自动走 `https://hf-mirror.com`。
GPU 只用于 ESM2 推理加速（75 条短序列，CPU 也只需要几分钟）。

## 2. 目录结构（本仓库）

```
enzyme-stereoselectivity-prediction/
├─ data/
│  ├─ mutations.csv        # 突变体 + 5 个异构体占比标签（论文示例，73 突变体 + WT）
│  └─ wildtype.fasta       # LbADH 野生型序列（251 aa）
├─ scripts/
│  ├─ 01_make_sequences.py # CSV+FASTA → outputs/sequences.csv
│  ├─ 02_esm2_embeddings.py# 序列 → ESM2 张量 → 均值池化 → outputs/1D_embeddings/*.pt
│  └─ 03_train_rf.py       # 480 维特征 + 标签 → RF 训练/5 折 RMSE/存模型
├─ outputs/                # 运行产物（不入库）
├─ models/                 # 训练好的模型（不入库）
└─ requirements.txt
```

## 3. 数据格式约定

`mutations.csv`：第一列必须叫 `Mutations`，`WT` 表示野生型，点突变写法如 `A93F`
（93 位的 A 换成 F），多点突变用 `/` 连接，如 `A93Y/L194M/V195G`；
其余列都是数值标签，训练时用 `--label 列名` 选一个。

## 4. 运行

```powershell
conda activate biotag
python scripts/01_make_sequences.py
python scripts/02_esm2_embeddings.py
python scripts/03_train_rf.py --label SS
```

验收点：
- ① `outputs/sequences.csv` 74 行（WT + 73 突变体），Status 全部 Success/Wildtype
- ② `outputs/1D_embeddings/` 74 个 .pt，每个 embedding shape = `(480,)`
- ③ 打印 5-fold RMSE，保存 `models/rf_SS.joblib`

## 5. 学习地图（零基础版）

- **ESM2**：蛋白质界的小型 BERT/GPT，输入氨基酸序列，输出每个位置的 480 维上下文向量
- **均值池化**：把 251 个位置的向量取平均，压成 1 个 480 维向量，表示整条序列
- **随机森林**：300 棵决策树投票做回归，sklearn 内置；5 折交叉验证 = 数据切 5 份轮流当考卷
- 本质：**序列 → 480 个数 → 预测一个百分数**，没有图片/结构的事
