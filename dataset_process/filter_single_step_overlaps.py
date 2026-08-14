# filter_single_step_overlaps.py

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


ATOM_MAP_RE = re.compile(r":\d+(?=\])")

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


def normalize_reactants_from_text(text: str) -> str:
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

    return ".".join(sorted(parts))


def normalize_reactants_from_sample(sample: Dict[str, Any]) -> str:
    if sample.get("reactants_str"):
        return normalize_reactants_from_text(sample["reactants_str"])

    reactants = sample.get("reactants_smiles", [])

    if isinstance(reactants, list):
        clean = [canonicalize_smiles(x) for x in reactants]
        clean = [x for x in clean if x]
        return ".".join(sorted(clean))

    return normalize_reactants_from_text(str(reactants))


def get_reaction_key(sample: Dict[str, Any]) -> str:
    current = canonicalize_smiles(
        sample.get("current_smiles", sample.get("input_current", ""))
    )
    reactants = normalize_reactants_from_sample(sample)
    return f"{current}>>{reactants}"


def get_context_reaction_key(sample: Dict[str, Any]) -> str:
    target = canonicalize_smiles(sample.get("target_smiles", ""))
    current = canonicalize_smiles(
        sample.get("current_smiles", sample.get("input_current", ""))
    )
    reactants = normalize_reactants_from_sample(sample)
    return f"{target}||{current}>>{reactants}"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def key_set(samples: List[Dict[str, Any]], key_type: str) -> Set[str]:
    if key_type == "reaction":
        return set(get_reaction_key(x) for x in samples)

    if key_type == "context_reaction":
        return set(get_context_reaction_key(x) for x in samples)

    raise ValueError(f"Unknown key_type: {key_type}")


def filter_by_seen_keys(
    samples: List[Dict[str, Any]],
    seen_keys: Set[str],
    key_type: str,
):
    kept = []
    removed = []

    for sample in samples:
        if key_type == "reaction":
            key = get_reaction_key(sample)
        elif key_type == "context_reaction":
            key = get_context_reaction_key(sample)
        else:
            raise ValueError(f"Unknown key_type: {key_type}")

        if key in seen_keys:
            removed.append({
                "id": sample.get("id"),
                "key": key,
                "target_smiles": sample.get("target_smiles"),
                "current_smiles": sample.get("current_smiles"),
                "reactants_str": normalize_reactants_from_sample(sample),
                "route_id": sample.get("route_id"),
                "step_order": sample.get("step_order"),
            })
        else:
            kept.append(sample)

    return kept, removed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./single_step",
        help="原始 single-step 数据目录。"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./single_step_no_overlap",
        help="去重后输出目录。"
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default="train_single_step_dedup.json",
    )

    parser.add_argument(
        "--valid_file",
        type=str,
        default="valid_single_step_dedup.json",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default="test_single_step_dedup.json",
    )

    parser.add_argument(
        "--key_type",
        type=str,
        default="reaction",
        choices=["reaction", "context_reaction"],
        help=(
            "reaction: 按 current>>reactants 去重，推荐用于 single-step 泄露检查；"
            "context_reaction: 按 target||current>>reactants 去重。"
        )
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / args.train_file
    valid_path = data_dir / args.valid_file
    test_path = data_dir / args.test_file

    train_samples = load_json(train_path)
    valid_samples = load_json(valid_path)
    test_samples = load_json(test_path)

    if not isinstance(train_samples, list):
        raise ValueError("train file 顶层必须是 list。")
    if not isinstance(valid_samples, list):
        raise ValueError("valid file 顶层必须是 list。")
    if not isinstance(test_samples, list):
        raise ValueError("test file 顶层必须是 list。")

    # 1. train 保持不动
    train_keys = key_set(train_samples, args.key_type)

    # 2. 删除 valid 中与 train 重复的样本
    filtered_valid, removed_valid = filter_by_seen_keys(
        samples=valid_samples,
        seen_keys=train_keys,
        key_type=args.key_type,
    )

    filtered_valid_keys = key_set(filtered_valid, args.key_type)

    # 3. 删除 test 中与 train + filtered_valid 重复的样本
    train_valid_keys = set(train_keys)
    train_valid_keys.update(filtered_valid_keys)

    filtered_test, removed_test = filter_by_seen_keys(
        samples=test_samples,
        seen_keys=train_valid_keys,
        key_type=args.key_type,
    )

    # 4. 输出文件
    write_json(
        train_samples,
        output_dir / "train_single_step_dedup.json",
    )

    write_json(
        filtered_valid,
        output_dir / "valid_single_step_no_train_overlap.json",
    )

    write_json(
        filtered_test,
        output_dir / "test_single_step_no_train_valid_overlap.json",
    )

    write_json(
        removed_valid,
        output_dir / "removed_valid_overlap_with_train.json",
    )

    write_json(
        removed_test,
        output_dir / "removed_test_overlap_with_train_valid.json",
    )

    report = {
        "rdkit_available": HAS_RDKIT,
        "key_type": args.key_type,
        "input_files": {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
        },
        "output_files": {
            "train": str(output_dir / "train_single_step_dedup.json"),
            "valid": str(output_dir / "valid_single_step_no_train_overlap.json"),
            "test": str(output_dir / "test_single_step_no_train_valid_overlap.json"),
            "removed_valid": str(output_dir / "removed_valid_overlap_with_train.json"),
            "removed_test": str(output_dir / "removed_test_overlap_with_train_valid.json"),
        },
        "counts": {
            "train_original": len(train_samples),
            "valid_original": len(valid_samples),
            "test_original": len(test_samples),
            "valid_removed_overlap_with_train": len(removed_valid),
            "test_removed_overlap_with_train_valid": len(removed_test),
            "valid_after_filter": len(filtered_valid),
            "test_after_filter": len(filtered_test),
        },
        "rates": {
            "valid_removed_rate": len(removed_valid) / len(valid_samples) if len(valid_samples) > 0 else 0.0,
            "test_removed_rate": len(removed_test) / len(test_samples) if len(test_samples) > 0 else 0.0,
        },
        "examples": {
            "removed_valid_first_20": removed_valid[:20],
            "removed_test_first_20": removed_test[:20],
        }
    }

    write_json(report, output_dir / "overlap_filter_report.json")

    print("\n===== Single-step Overlap Filtering =====")
    print(f"Key type: {args.key_type}")
    print(f"Train original: {len(train_samples)}")
    print(f"Valid original: {len(valid_samples)}")
    print(f"Test original : {len(test_samples)}")

    print("\n[Valid filtering]")
    print(f"Removed valid overlap with train: {len(removed_valid)}")
    print(f"Valid after filter: {len(filtered_valid)}")

    print("\n[Test filtering]")
    print(f"Removed test overlap with train + filtered_valid: {len(removed_test)}")
    print(f"Test after filter: {len(filtered_test)}")

    print("\nSaved files:")
    print(output_dir / "train_single_step_dedup.json")
    print(output_dir / "valid_single_step_no_train_overlap.json")
    print(output_dir / "test_single_step_no_train_valid_overlap.json")
    print(output_dir / "overlap_filter_report.json")


if __name__ == "__main__":
    main()