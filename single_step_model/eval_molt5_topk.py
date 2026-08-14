# eval_molt5_topk.py

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


ATOM_MAP_RE = re.compile(r":\d+(?=\])")

ORIGINAL_TASK_PREFIX = "Please predict the reactant of the product:\n"

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


def canonicalize_smiles(smi: str) -> str:
    """
    删除原子映射并 canonicalize SMILES。
    如果 RDKit 解析失败，则退回到去 atom mapping 后的原字符串。
    """
    smi = strip_atom_mapping(smi)

    if not smi:
        return ""

    if not HAS_RDKIT:
        return smi

    mol = None

    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
    except Exception:
        mol = None

    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is not None:
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
        except Exception:
            mol = None

    if mol is None:
        return smi

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return smi


def normalize_reactants_text(text: str) -> str:
    """
    将模型输出或 gold reactants 统一处理成：
    canonical_reactant_1.canonical_reactant_2...

    兼容：
    1. 直接输出 reactants；
    2. 输出 product>>reactants；
    3. 输出带空格或换行。
    """
    if text is None:
        return ""

    text = str(text).strip()

    if ">>" in text:
        text = text.split(">>", 1)[1]

    text = re.sub(r"\s+", "", text)

    if not text:
        return ""

    parts = [p for p in text.split(".") if p.strip()]
    parts = [canonicalize_smiles(p) for p in parts]
    parts = [p for p in parts if p]

    if not parts:
        return ""

    return ".".join(sorted(parts))


def canonicalize_gold_reactants(sample: Dict[str, Any]) -> str:
    """
    优先使用 reactants_str。
    如果没有 reactants_str，则从 reactants_smiles 生成。
    """
    if sample.get("reactants_str"):
        return normalize_reactants_text(sample["reactants_str"])

    reactants = sample.get("reactants_smiles", [])

    if isinstance(reactants, list):
        clean = [canonicalize_smiles(x) for x in reactants]
        clean = [x for x in clean if x]
        return ".".join(sorted(clean))

    return normalize_reactants_text(str(reactants))


def build_original_prompt(sample: Dict[str, Any]) -> str:
    """
    原始 RetroInText / MolT5 prompt。
    用于评估原始 MolT5 baseline。
    """
    current = sample.get("current_smiles") or sample.get("input_current") or ""
    current = canonicalize_smiles(current)

    return ORIGINAL_TASK_PREFIX + current


def build_route_context_prompt(sample: Dict[str, Any], max_depth: int) -> str:
    """
    新 route-context prompt。
    用于评估 route-context SFT 后的 MolT5。

    注意：
    current_depth = step_order - 1，是搜索时可获得的信息；
    max_depth 是固定搜索预算，不使用 gold route depth；
    不使用 materials / intermediates / depth_remaining / route_id。
    """
    target = canonicalize_smiles(sample.get("target_smiles", ""))
    current = canonicalize_smiles(
        sample.get("current_smiles") or sample.get("input_current") or ""
    )

    step_order = int(sample.get("step_order", 1))
    current_depth = max(step_order - 1, 0)

    return (
        f"{ORIGINAL_TASK_PREFIX}"
        f"<target> {target}\n"
        f"<current> {current}\n"
        f"<current_depth> {current_depth}\n"
        f"<max_depth> {max_depth}\n"
        f"<goal> purchasable_starting_materials"
    )


def build_model_input(
    sample: Dict[str, Any],
    prompt_mode: str,
    max_depth: int,
) -> str:
    if prompt_mode == "original":
        return build_original_prompt(sample)

    if prompt_mode == "route_context":
        return build_route_context_prompt(sample, max_depth=max_depth)

    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_multi_refs(
    samples: List[Dict[str, Any]],
    prompt_mode: str,
    max_depth: int,
) -> Dict[str, Set[str]]:
    """
    如果同一个模型输入有多个 gold reactants，
    则 multi-reference 模式下命中任意一个都算正确。
    """
    refs: Dict[str, Set[str]] = {}

    for sample in samples:
        model_input = build_model_input(
            sample=sample,
            prompt_mode=prompt_mode,
            max_depth=max_depth,
        )
        gold = canonicalize_gold_reactants(sample)

        refs.setdefault(model_input, set())
        refs[model_input].add(gold)

    return refs


@torch.no_grad()
def evaluate_topk(
    samples: List[Dict[str, Any]],
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    device: torch.device,
    prompt_mode: str,
    max_depth: int,
    topk: int,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    multi_reference: bool,
    fp16: bool,
    save_details: bool,
) -> Dict[str, Any]:

    model.eval()

    refs_by_input = build_multi_refs(
        samples=samples,
        prompt_mode=prompt_mode,
        max_depth=max_depth,
    )

    total = len(samples)
    hit_counts = [0 for _ in range(topk)]
    valid_pred_top1 = 0

    details = [] if save_details else None

    for start in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch = samples[start:start + batch_size]

        model_inputs = [
            build_model_input(
                sample=x,
                prompt_mode=prompt_mode,
                max_depth=max_depth,
            )
            for x in batch
        ]

        enc = tokenizer(
            model_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        gen_kwargs = {
            "num_beams": topk,
            "num_return_sequences": topk,
            "early_stopping": True,
            "max_length": max_target_length,
        }

        if fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output_ids = model.generate(**enc, **gen_kwargs)
        else:
            output_ids = model.generate(**enc, **gen_kwargs)

        decoded = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for i, sample in enumerate(batch):
            raw_preds = decoded[i * topk:(i + 1) * topk]

            # top-k rank 必须按 beam 原始顺序计算，不先去重。
            normalized_preds_in_order = [
                normalize_reactants_text(p)
                for p in raw_preds
            ]

            if normalized_preds_in_order and normalized_preds_in_order[0]:
                valid_pred_top1 += 1

            model_input = model_inputs[i]

            if multi_reference:
                gold_refs = refs_by_input[model_input]
            else:
                gold_refs = {canonicalize_gold_reactants(sample)}

            matched_rank = None

            for rank, pred in enumerate(normalized_preds_in_order[:topk], start=1):
                if pred in gold_refs:
                    matched_rank = rank
                    break

            if matched_rank is not None:
                for k in range(matched_rank, topk + 1):
                    hit_counts[k - 1] += 1

            if save_details:
                unique_preds = []
                seen = set()
                for p in normalized_preds_in_order:
                    if p not in seen:
                        seen.add(p)
                        unique_preds.append(p)

                details.append({
                    "id": sample.get("id"),
                    "model_input": model_input,
                    "prompt_mode": prompt_mode,

                    "target_smiles": sample.get("target_smiles"),
                    "current_smiles": sample.get("current_smiles"),
                    "route_id": sample.get("route_id"),
                    "step_order": sample.get("step_order"),

                    "gold_reactants": canonicalize_gold_reactants(sample),
                    "gold_refs": sorted(list(gold_refs)),
                    "matched_rank": matched_rank,

                    "raw_predictions": raw_preds,
                    "normalized_predictions_in_order": normalized_preds_in_order,
                    "unique_normalized_predictions": unique_preds,
                })

    metrics = {
        "num_samples": total,
        "prompt_mode": prompt_mode,
        "topk": topk,
        "multi_reference": multi_reference,
        "valid_pred_top1_rate": valid_pred_top1 / total if total > 0 else 0.0,
        "rdkit_available": HAS_RDKIT,
    }

    for k in range(1, topk + 1):
        metrics[f"top_{k}"] = hit_counts[k - 1] / total if total > 0 else 0.0

    return {
        "metrics": metrics,
        "details": details,
    }


def print_metrics(metrics: Dict[str, Any]):
    print("\n===== Top-k Metrics =====")
    print(f"num_samples           : {metrics['num_samples']}")
    print(f"prompt_mode           : {metrics['prompt_mode']}")
    print(f"multi_reference       : {metrics['multi_reference']}")
    print(f"valid_pred_top1_rate  : {metrics['valid_pred_top1_rate']:.6f}")

    for k in range(1, metrics["topk"] + 1):
        print(f"top-{k:<2}: {metrics[f'top_{k}']:.6f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help=(
            "模型路径。"
            "原始 MolT5 用 ../molt5；"
            "SFT 后模型用 ./molt5_route_context_sft_maxdepth14/checkpoint-best"
        )
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="./single_step_no_overlap/test_single_step_no_train_valid_overlap.json",
        help="评估数据文件，默认使用去重后的 test。"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="评估结果输出目录。"
    )

    parser.add_argument(
        "--prompt_mode",
        type=str,
        required=True,
        choices=["original", "route_context"],
        help=(
            "original: 原始 RetroInText prompt，评估原始 MolT5；"
            "route_context: 新 prompt，评估 route-context SFT 后模型。"
        )
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=14,
        help="route_context prompt 中使用的固定 max_depth。"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
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
        "--multi_reference",
        action="store_true",
        help="如果同一 prompt 有多个 gold reactants，命中任意一个都算对。"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="CUDA 上使用 fp16 autocast。"
    )

    parser.add_argument(
        "--save_details",
        action="store_true",
        help="是否保存每条样本的 top-k 预测详情。默认不保存，避免 JSON 太大。"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_file = Path(args.data_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"找不到模型目录：{model_dir}")

    if not data_file.exists():
        raise FileNotFoundError(f"找不到数据文件：{data_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===== MolT5 Top-k Evaluation =====")
    print(f"model_dir      : {model_dir}")
    print(f"data_file      : {data_file}")
    print(f"output_dir     : {output_dir}")
    print(f"prompt_mode    : {args.prompt_mode}")
    print(f"max_depth      : {args.max_depth}")
    print(f"topk           : {args.topk}")
    print(f"batch_size     : {args.batch_size}")
    print(f"device         : {device}")
    print(f"multi_reference: {args.multi_reference}")
    print(f"save_details   : {args.save_details}")

    samples = load_json(data_file)

    if not isinstance(samples, list):
        raise ValueError(f"{data_file} 顶层必须是 list。")

    tokenizer = T5Tokenizer.from_pretrained(
        str(model_dir),
        model_max_length=args.max_source_length,
    )

    model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    result = evaluate_topk(
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_mode=args.prompt_mode,
        max_depth=args.max_depth,
        topk=args.topk,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        multi_reference=args.multi_reference,
        fp16=args.fp16,
        save_details=args.save_details,
    )

    metrics = result["metrics"]

    # 补充记录
    metrics.update({
        "model_dir": str(model_dir),
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "max_depth": args.max_depth,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "task_prefix": ORIGINAL_TASK_PREFIX,
    })

    write_json(metrics, output_dir / "topk_metrics.json")

    if args.save_details:
        write_json(result["details"], output_dir / "topk_details.json")

    print_metrics(metrics)

    print(f"\n[完成] metrics 保存到：{output_dir / 'topk_metrics.json'}")
    if args.save_details:
        print(f"[完成] details 保存到：{output_dir / 'topk_details.json'}")


if __name__ == "__main__":
    main()