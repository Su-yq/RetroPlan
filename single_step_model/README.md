# 单步模型类 (Single-Step Model)

本目录包含所有**单步逆合成模型**相关的代码，包括模型定义、Prompt 构建、SFT/DPO 训练以及测试评估脚本。这些文件构成了从单步反应预测到多步搜索的核心模型系统。

---

## 文件清单

### 1. 单步生成模型（MolT5）

| 文件 | 功能说明 |
|------|---------|
| `train_molt5_route_context_sft.py` | 使用路线上下文对 MolT5 进行 SFT 训练 |
| `build_u_positive_sft_data.py` | 基于 U（效用）分数构建正向 SFT（监督微调）训练数据 |
| `train_molt5_route_context_positive_sft.py` | 使用路线上下文 + U 引导正样本对 MolT5 进行 SFT 训练（正向-only 训练，不使用拒绝样本） |
| `eval_molt5_topk.py` | 评估 MolT5 单步预测的 top-k 准确率，支持 Prompt 模板化输入 |


---

## 模型训练流程概览

```
initial MolT5
    │
    ├── Prompt: Route context + "Please predict the reactant of the product:\n{SMILES}"
    │
    ▼
route-context Molt5
    │
    ├── route context prompt
    ├── U-positive data
    │
    ▼
u-positive route-context Molt5
```

---

## Prompt 模板

单步模型使用统一的 Prompt 格式：

```
Please predict the reactant of the product:\n{product_SMILES}
```

- 对于路线上下文模型，会额外编码当前中间体和目标产物信息

---


---

## 运行指令
---

训练新prompt（route context）
nohup python train_molt5_route_context_sft.py \
  --data_dir ../dataset/single_step_no_overlap \
  --model_dir ../molt5 \
  --output_dir ../molt5_route_context_sft \
  --train_file train_single_step_dedup.json \
  --valid_file valid_single_step_no_train_overlap.json \
  --max_depth 14 \
  --learning_rate 5e-5 \
  --batch_size 4 \
  --epochs 40 \
  --weight_decay 0.1 \
  --save_every_epochs 10 \
  --early_stop_patience 5 \
  --early_stop_min_delta 1e-4 \
  --fp16 > train.log 2>&1 &

生成top20路径候选（为dpo偏好对构建做准备）
nohup python generate_route_context_candidates.py \
  --data_dir ../dataset/single_step_no_overlap \
  --model_dir ../molt5_route_context_sft/checkpoint-best \
  --output_dir ./dpo_candidates_route_context_sft_top20 \
  --splits train valid \
  --topk 20 \
  --batch_size 8 \
  --max_depth 14 \
  --fp16 > process.log 2>&1 &

构建偏好对
nohup python build_dpo_pairs_with_u.py \
  --candidate_dir ./dpo_candidates_route_context_sft_top20 \
  --single_step_dir ../dataset/single_step_no_overlap \
  --output_dir ./dpo_pairs_u \
  --project_root /root \
  --forward_model_path ../ReactionT5/model \
  --splits train valid \
  --forward_topk 5 \
  --forward_num_beams 5 \
  --forward_batch_size 16 \
  --forward_fp16 > build_dpo_pairs_u.log 2>&1 &

筛选构建u-positive:
python build_u_positive_sft_data.py \
  --train_json ../dataset/single_step_no_overlap/train_single_step_dedup.json \
  --valid_json ../dataset/single_step_no_overlap/valid_single_step_no_train_overlap.json \
  --train_scored_candidates ./dpo_pairs_u/train_scored_candidates.jsonl \
  --output_dir ./sft_u_positive \
  --max_pseudo_per_sample 1 \
  --max_pseudo_to_gold_ratio 0.5 \
  --min_valid_score 1.0 \
  --max_bad_action_penalty 0.0 \
  --min_forward_plausibility 0.5 \
  --min_utility 1.5

30轮dpo-sft:
nohup env CUDA_VISIBLE_DEVICES=3 python train_molt5_route_context_positive_sft.py \
  --init_model_dir ../molt5_route_context_sft/checkpoint-best \
  --train_file ./sft_u_positive/train_u_positive_sft.json \
  --valid_file ./sft_u_positive/valid_gold_sft.json \
  --output_dir ../molt5_route_context_sft_u_positive_30epoch \
  --epochs 30 \
  --lr 3e-6 \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --eval_batch_size 8 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --label_smoothing 0.02 \
  --max_source_length 512 \
  --max_target_length 256 \
  --max_depth 14 \
  --max_grad_norm 1.0 \
  --fp16 \
  --logging_steps 100 > train_u_positive_sft.log 2>&1 &

测试molt5
nohup python eval_molt5_topk.py \
  --model_dir ../molt5_route_context_sft_u_positive_30epoch/checkpoint-best \
  --data_file ../dataset/single_step_no_overlap/test_single_step_no_train_valid_overlap.json \
  --output_dir ./molt5_topk \
  --prompt_mode route_context \
  --max_depth 14 \
  --topk 10 \
  --batch_size 16 \
  --fp16 > test1.log 2>&1 &

## 依赖库

- **PyTorch**: 深度学习框架
- **transformers**: Hugging Face 预训练模型（T5 系列）
- **RDKit**: 化学信息学工具包
- **numpy**: 数值计算
- **tqdm**: 进度条显示
- **pickle**: 模型序列化
