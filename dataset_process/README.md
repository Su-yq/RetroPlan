# Dataset Preprocess

本目录包含多步数据集处理有关脚本文件。这些文件负责从原始多步逆合成数据出发，经过清洗、标准化等步骤，生成供模型训练和评估使用的数据集。

---

## 文件清单

### 数据预处理与构建

| 文件 | 功能说明 |
|------|---------|
| `preprocess_multistep_retro.py` | 预处理多步逆合成数据，进行 SMILES 标准化、原子映射去除、RDKit canonicalization |
| `build_single_step_dataset.py` | 从多步路线数据中提取单步反应步骤，构建单步逆合成数据集 |
| `filter_single_step_overlaps.py` | 过滤单步数据与多步数据之间的重叠样本，确保数据独立性 |

---

## 数据处理流水线

```
原始多步路线数据
       │
       ▼
  preprocess_multistep_retro.py  ──► SMILES 标准化
       │
       ▼
  build_single_step_dataset.py   ──► 提取单步反应
       │
       ▼
  filter_single_step_overlaps.py  ──► 过滤重叠样本，避免数据泄露
```

---

## 运行指令

raw数据处理为逐步多步逆合成
```bash
nohup python preprocess_multistep_retro.py \
  --input_dir ../dataset \
  --output_dir ../dataset \
  --inner_path_as_route \
  --save_flat_route_level > preprocess.log 2>&1 &
```

构建单步数据集
```bash
nohup python build_single_step_dataset.py \
  --input_dir ../dataset \
  --output_dir ../dataset/single_step \
  --dedup_mode reaction > preprocess.log 2>&1 &
```

数据集去重
```bash
nohup python filter_single_step_overlaps.py \
  --data_dir ../dataset/single_step \
  --output_dir ../dataset/single_step_no_overlap \
  --key_type reaction > tmp.log 2>&1 &
```

## 依赖库

- **RDKit**: 化学信息学工具包，用于 SMILES 解析、分子描述符计算
- **numpy**: 数值计算
- **torch**: PyTorch 深度学习框架
- **transformers**: Hugging Face 预训练模型库
- **tqdm**: 进度条显示
