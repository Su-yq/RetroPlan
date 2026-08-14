# generate_route_context_candidates.py

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


ATOM_MAP_RE = re.compile(r":\d+(?=\])")
TASK_PREFIX = "Please predict the reactant of the product:\n"


try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    Chem = None


def strip_atom_mapping(smi: str) -> str:
    if smi is None:
        return ""
    return ATOM_MAP_RE.sub("", str(smi).strip())


def canonicalize_smiles_with_validity(smi: str) -> Tuple[str, bool]:
    """
    返回 canonical SMILES 和 RDKit validity。
    如果 RDKit 不可用，则只做去 atom mapping，并默认 valid=True。
    """
    smi = strip_atom_mapping(smi)

    if not smi:
        return "", False

    if not HAS_RDKIT:
        return smi, True

    mol = None

    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        mol = None

    if mol is None:
        return smi, False

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    try:
        canon = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
        return canon, True
    except Exception:
        return smi, False


def canonicalize_smiles(smi: str) -> str:
    canon, _ = canonicalize_smiles_with_validity(smi)
    return canon


def normalize_reactants_text(text: str) -> Tuple[str, bool, int]:
    """
    将模型输出标准化为 canonical reactants string。

    返回：
    reactants_str: canonical sorted reactants joined by '.'
    valid_all: 所有 reactant 是否都能被 RDKit 解析
    num_reactants: reactants 数量
    """
    if text is None:
        return "", False, 0

    text = str(text).strip()

    # 兼容模型输出 product>>reactants 的情况
    if ">>" in text:
        text = text.split(">>", 1)[1]

    # T5 decode 有时会产生空格
    text = re.sub(r"\s+", "", text)

    if not text:
        return "", False, 0

    parts = [p for p in text.split(".") if p.strip()]

    if not parts:
        return "", False, 0

    clean_parts = []
    valid_flags = []

    for p in parts:
        canon, valid = canonicalize_smiles_with_validity(p)
        if canon:
            clean_parts.append(canon)
            valid_flags.append(valid)

    if not clean_parts:
        return "", False, 0

    clean_parts = sorted(clean_parts)
    valid_all = all(valid_flags) and len(valid_flags) == len(parts)

    return ".".join(clean_parts), valid_all, len(clean_parts)


def canonicalize_gold_reactants(sample: Dict[str, Any]) -> str:
    if sample.get("reactants_str"):
        reactants_str, _, _ = normalize_reactants_text(sample["reactants_str"])
        return reactants_str

    reactants = sample.get("reactants_smiles", [])

    if isinstance(reactants, list):
        clean = []
        for x in reactants:
            canon = canonicalize_smiles(x)
            if canon:
                clean.append(canon)
        return ".".join(sorted(clean))

    reactants_str, _, _ = normalize_reactants_text(str(reactants))
    return reactants_str


def build_route_context_prompt(sample: Dict[str, Any], max_depth: int) -> str:
    """
    和 route-context SFT 训练时保持一致。

    不使用：
    - materials
    - intermediates
    - depth_remaining
    - route_id
    - gold route depth

    使用：
    - target_smiles
    - current_smiles
    - current_depth = step_order - 1
    - fixed max_depth
    """
    target = canonicalize_smiles(sample.get("target_smiles", ""))

    current = sample.get("current_smiles") or sample.get("input_current") or ""
    current = canonicalize_smiles(current)

    step_order = int(sample.get("step_order", 1))
    current_depth = max(step_order - 1, 0)

    return (
        f"{TASK_PREFIX}"
        f"<target> {target}\n"
        f"<current> {current}\n"
        f"<current_depth> {current_depth}\n"
        f"<max_depth> {max_depth}\n"
        f"<goal> purchasable_starting_materials"
    )


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl_line(f, obj: Dict[str, Any]):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_split_file(data_dir: Path, split: str) -> Path:
    if split == "train":
        return data_dir / "train_single_step_dedup.json"

    if split == "valid":
        return data_dir / "valid_single_step_no_train_overlap.json"

    if split == "test":
        return data_dir / "test_single_step_no_train_valid_overlap.json"

    raise ValueError(f"Unknown split: {split}")


@torch.no_grad()
def generate_candidates_for_split(
    split: str,
    data_path: Path,
    output_jsonl: Path,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    device: torch.device,
    topk: int = 20,
    num_beams: Optional[int] = None,
    batch_size: int = 8,
    max_depth: int = 14,
    max_source_length: int = 512,
    max_target_length: int = 256,
    fp16: bool = False,
    save_prompt: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:

    samples = load_json(data_path)

    if not isinstance(samples, list):
        raise ValueError(f"{data_path} 顶层必须是 list。")

    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]

    if num_beams is None:
        num_beams = topk

    if num_beams < topk:
        raise ValueError("num_beams 必须 >= topk。")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = len(samples)
    total_unique_candidates = 0
    total_valid_unique_candidates = 0
    total_gold_in_unique_topk = 0
    total_gold_rank_sum = 0
    total_gold_rank_count = 0
    total_empty_candidate_sets = 0

    model.eval()

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for start in tqdm(range(0, total, batch_size), desc=f"Generating {split}"):
            batch = samples[start:start + batch_size]

            prompts = [
                build_route_context_prompt(sample=x, max_depth=max_depth)
                for x in batch
            ]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_source_length,
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            gen_kwargs = {
                "num_beams": num_beams,
                "num_return_sequences": topk,
                "early_stopping": True,
                "max_length": max_target_length,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            if fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    gen_out = model.generate(**enc, **gen_kwargs)
            else:
                gen_out = model.generate(**enc, **gen_kwargs)

            sequences = gen_out.sequences

            # Beam search 时一般有 sequences_scores。
            sequences_scores = getattr(gen_out, "sequences_scores", None)

            decoded = tokenizer.batch_decode(
                sequences,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            for i, sample in enumerate(batch):
                raw_start = i * topk
                raw_end = (i + 1) * topk

                raw_texts = decoded[raw_start:raw_end]

                if sequences_scores is not None:
                    raw_scores = [
                        float(x.detach().cpu().item())
                        for x in sequences_scores[raw_start:raw_end]
                    ]
                else:
                    raw_scores = [None for _ in raw_texts]

                gold_reactants = canonicalize_gold_reactants(sample)

                # 先保留 beam 原始顺序，再做 per-sample 去重。
                unique_candidates = []
                seen_reactants = set()

                for raw_rank, (raw_text, score) in enumerate(
                    zip(raw_texts, raw_scores),
                    start=1,
                ):
                    reactants_str, valid_all, num_reactants = normalize_reactants_text(raw_text)

                    if not reactants_str:
                        continue

                    if reactants_str in seen_reactants:
                        continue

                    seen_reactants.add(reactants_str)

                    unique_rank = len(unique_candidates) + 1

                    unique_candidates.append({
                        "unique_rank": unique_rank,
                        "raw_rank": raw_rank,
                        "raw_text": raw_text,
                        "reactants_str": reactants_str,
                        "reactants_smiles": reactants_str.split("."),
                        "valid_smiles": bool(valid_all),
                        "num_reactants": int(num_reactants),
                        "is_gold": reactants_str == gold_reactants,
                        "sequence_score": score,
                    })

                if len(unique_candidates) == 0:
                    total_empty_candidate_sets += 1

                num_unique = len(unique_candidates)
                num_valid = sum(1 for c in unique_candidates if c["valid_smiles"])

                total_unique_candidates += num_unique
                total_valid_unique_candidates += num_valid

                matched_rank = None
                for c in unique_candidates:
                    if c["is_gold"]:
                        matched_rank = c["unique_rank"]
                        break

                if matched_rank is not None:
                    total_gold_in_unique_topk += 1
                    total_gold_rank_sum += matched_rank
                    total_gold_rank_count += 1

                out_obj = {
                    "id": sample.get("id"),
                    "split": split,

                    "target_smiles": canonicalize_smiles(sample.get("target_smiles", "")),
                    "current_smiles": canonicalize_smiles(
                        sample.get("current_smiles") or sample.get("input_current") or ""
                    ),
                    "gold_reactants": gold_reactants,

                    "route_id": sample.get("route_id"),
                    "step_order": sample.get("step_order"),
                    "depth": sample.get("depth"),

                    "max_depth": max_depth,
                    "prompt_mode": "route_context",
                    "num_candidates": num_unique,
                    "num_valid_candidates": num_valid,
                    "gold_matched_rank": matched_rank,

                    "candidates": unique_candidates,
                }

                if save_prompt:
                    out_obj["prompt"] = prompts[i]

                write_jsonl_line(fout, out_obj)

    avg_unique_candidates = total_unique_candidates / total if total > 0 else 0.0
    avg_valid_unique_candidates = total_valid_unique_candidates / total if total > 0 else 0.0
    gold_recall_unique_topk = total_gold_in_unique_topk / total if total > 0 else 0.0
    avg_gold_rank = (
        total_gold_rank_sum / total_gold_rank_count
        if total_gold_rank_count > 0
        else None
    )

    stats = {
        "split": split,
        "data_path": str(data_path),
        "output_jsonl": str(output_jsonl),
        "num_samples": total,
        "topk": topk,
        "num_beams": num_beams,
        "max_depth": max_depth,
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
        "rdkit_available": HAS_RDKIT,

        "avg_unique_candidates_per_sample": avg_unique_candidates,
        "avg_valid_unique_candidates_per_sample": avg_valid_unique_candidates,
        "empty_candidate_sets": total_empty_candidate_sets,
        "empty_candidate_set_rate": total_empty_candidate_sets / total if total > 0 else 0.0,

        "gold_recall_unique_topk": gold_recall_unique_topk,
        "num_gold_in_unique_topk": total_gold_in_unique_topk,
        "avg_gold_rank_when_found": avg_gold_rank,
    }

    return stats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./single_step_no_overlap",
        help="去重后的 single-step 数据目录。"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="./molt5_route_context_sft_maxdepth14/checkpoint-best",
        help="route-context SFT 模型路径。"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dpo_candidates_route_context_sft_top20",
        help="候选输出目录。"
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        help="要生成候选的 split，默认 train valid。"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="每个样本返回多少个候选。DPO 建议 10 或 20。"
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="beam 数。默认等于 topk。"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="CUDA 上使用 fp16 autocast。"
    )

    parser.add_argument(
        "--save_prompt",
        action="store_true",
        help="是否在 JSONL 中保存完整 prompt。默认不保存，避免文件过大。"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="调试用：每个 split 最多处理多少样本。默认处理全部。"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"找不到模型目录：{model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===== Generate Route-context Candidates =====")
    print(f"model_dir        : {model_dir}")
    print(f"data_dir         : {data_dir}")
    print(f"output_dir       : {output_dir}")
    print(f"splits           : {args.splits}")
    print(f"topk             : {args.topk}")
    print(f"num_beams        : {args.num_beams if args.num_beams is not None else args.topk}")
    print(f"batch_size       : {args.batch_size}")
    print(f"max_depth        : {args.max_depth}")
    print(f"device           : {device}")
    print(f"fp16             : {args.fp16}")
    print(f"save_prompt      : {args.save_prompt}")
    print(f"max_samples      : {args.max_samples}")

    tokenizer = T5Tokenizer.from_pretrained(
        str(model_dir),
        model_max_length=args.max_source_length,
    )

    model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    all_stats = {}

    for split in args.splits:
        data_path = get_split_file(data_dir, split)

        if not data_path.exists():
            print(f"[跳过] 找不到 {split} 数据：{data_path}")
            continue

        output_jsonl = output_dir / f"{split}_candidates_top{args.topk}.jsonl"

        stats = generate_candidates_for_split(
            split=split,
            data_path=data_path,
            output_jsonl=output_jsonl,
            model=model,
            tokenizer=tokenizer,
            device=device,
            topk=args.topk,
            num_beams=args.num_beams,
            batch_size=args.batch_size,
            max_depth=args.max_depth,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            fp16=args.fp16,
            save_prompt=args.save_prompt,
            max_samples=args.max_samples,
        )

        all_stats[split] = stats

        stats_path = output_dir / f"{split}_candidate_stats.json"
        write_json(stats, stats_path)

        print(f"\n[{split}] stats:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    write_json(all_stats, output_dir / "all_candidate_stats.json")

    print(f"\n[完成] 所有候选已保存到：{output_dir}")


if __name__ == "__main__":
    main()