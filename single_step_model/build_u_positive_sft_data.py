#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build U-guided positive-only SFT data.

Input:
  1. original single-step train/valid json
  2. dpo_pairs_u/train_scored_candidates.jsonl

Output:
  train_u_positive_sft.json
  valid_gold_sft.json
  report.json

This does NOT use rejected candidates.
It only adds high-U generated candidates as extra positive targets.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from rdkit import Chem
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False
    Chem = None


ATOM_MAP_RE = re.compile(r":\d+(?=\])")


def strip_atom_mapping(smi: str) -> str:
    if smi is None:
        return ""
    return ATOM_MAP_RE.sub("", str(smi).strip())


def canonicalize_mol(smi: str) -> Optional[str]:
    if smi is None:
        return None

    smi = strip_atom_mapping(str(smi).strip().replace(" ", ""))

    if not smi:
        return None

    if not HAS_RDKIT:
        return smi

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None


def canonicalize_side(side: str) -> Optional[str]:
    if side is None:
        return None

    side = str(side).strip().replace(" ", "").rstrip(".")

    if not side:
        return None

    parts = [p for p in side.split(".") if p.strip()]
    if not parts:
        return None

    out = []

    for p in parts:
        cp = canonicalize_mol(p)
        if cp is None:
            return None
        out.append(cp)

    return ".".join(sorted(out))


def split_reactants(side: str) -> List[str]:
    side = str(side).strip().replace(" ", "").rstrip(".")
    if not side:
        return []
    return [x for x in side.split(".") if x]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def get_gold_reactants(sample: Dict[str, Any]) -> Optional[str]:
    if sample.get("reactants_str"):
        return canonicalize_side(sample["reactants_str"])

    xs = sample.get("reactants_smiles", [])
    if isinstance(xs, list) and xs:
        return canonicalize_side(".".join(xs))

    return None


def normalize_gold_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    gold = get_gold_reactants(sample)
    if not gold:
        return None

    out = dict(sample)
    out["reactants_str"] = gold
    out["reactants_smiles"] = split_reactants(gold)
    out["target_source"] = "gold"
    out["original_id"] = str(sample.get("id", ""))

    return out


def candidate_passes_filters(
    cand: Dict[str, Any],
    gold_reactants: str,
    args,
) -> Tuple[bool, str]:

    reactants = canonicalize_side(
        cand.get("reactants_str")
        or cand.get("raw_text")
        or ""
    )

    if not reactants:
        return False, "invalid_reactants"

    if reactants == gold_reactants:
        return False, "same_as_gold"

    if not bool(cand.get("valid_smiles", False)):
        return False, "valid_smiles_false"

    features = cand.get("features", {})

    valid_score = safe_float(features.get("valid_score"), 0.0)
    bad_penalty = safe_float(features.get("bad_action_penalty"), 999.0)
    forward_plausibility = safe_float(features.get("forward_plausibility"), 0.0)
    route_future_proxy = safe_float(features.get("route_future_proxy"), 0.0)
    infomax_similarity = safe_float(features.get("infomax_similarity"), 0.0)
    bb_ratio = safe_float(features.get("bb_ratio"), 0.0)
    utility = safe_float(cand.get("utility"), -999.0)

    if valid_score < args.min_valid_score:
        return False, "low_valid_score"

    if bad_penalty > args.max_bad_action_penalty:
        return False, "bad_action_penalty"

    if forward_plausibility < args.min_forward_plausibility:
        return False, "low_forward_plausibility"

    if route_future_proxy < args.min_route_future_proxy:
        return False, "low_route_future_proxy"

    if infomax_similarity < args.min_infomax_similarity:
        return False, "low_infomax_similarity"

    if bb_ratio < args.min_bb_ratio:
        return False, "low_bb_ratio"

    if utility < args.min_utility:
        return False, "low_utility"

    return True, "pass"


def build_positive_data(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_path = Path(args.train_json)
    valid_path = Path(args.valid_json)
    scored_train_path = Path(args.train_scored_candidates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_json(train_path)
    valid_samples = load_json(valid_path)

    sample_index = {}
    gold_records = []

    for sample in tqdm(train_samples, desc="load gold train samples"):
        sid = str(sample.get("id", ""))
        norm = normalize_gold_sample(sample)
        if norm is None:
            continue
        sample_index[sid] = norm
        gold_records.append(norm)

    valid_gold_records = []

    for sample in tqdm(valid_samples, desc="load gold valid samples"):
        norm = normalize_gold_sample(sample)
        if norm is not None:
            valid_gold_records.append(norm)

    pseudo_records = []
    reject_reasons = {}
    selected_per_sample = {}

    for rec in tqdm(iter_jsonl(scored_train_path), desc="select U-positive candidates"):
        sid = str(rec.get("id", ""))

        if sid not in sample_index:
            reject_reasons["sample_not_found"] = reject_reasons.get("sample_not_found", 0) + 1
            continue

        base_sample = sample_index[sid]
        gold_reactants = base_sample["reactants_str"]

        candidates = rec.get("candidates", [])
        passed = []

        for cand in candidates:
            ok, reason = candidate_passes_filters(cand, gold_reactants, args)

            if not ok:
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue

            reactants = canonicalize_side(
                cand.get("reactants_str")
                or cand.get("raw_text")
                or ""
            )

            if not reactants:
                reject_reasons["invalid_after_canonical"] = reject_reasons.get("invalid_after_canonical", 0) + 1
                continue

            passed.append((safe_float(cand.get("utility"), -999.0), reactants, cand))

        if not passed:
            continue

        passed.sort(key=lambda x: x[0], reverse=True)

        n_added = 0
        seen_reactants = set()

        for utility, reactants, cand in passed:
            if reactants in seen_reactants:
                continue
            seen_reactants.add(reactants)

            pseudo = dict(base_sample)
            pseudo["id"] = f"{sid}__u_pos_{n_added + 1}"
            pseudo["original_id"] = sid
            pseudo["reactants_str"] = reactants
            pseudo["reactants_smiles"] = split_reactants(reactants)
            pseudo["target_source"] = "u_positive"
            pseudo["pseudo_utility"] = utility
            pseudo["pseudo_features"] = cand.get("features", {})
            pseudo["pseudo_unique_rank"] = cand.get("unique_rank")
            pseudo["pseudo_raw_rank"] = cand.get("raw_rank")

            pseudo_records.append(pseudo)
            n_added += 1

            if n_added >= args.max_pseudo_per_sample:
                break

        if n_added > 0:
            selected_per_sample[sid] = n_added

    # Limit pseudo/gold ratio
    max_pseudo_total = int(len(gold_records) * args.max_pseudo_to_gold_ratio)

    if len(pseudo_records) > max_pseudo_total:
        random.shuffle(pseudo_records)
        pseudo_records = pseudo_records[:max_pseudo_total]

    train_expanded = gold_records + pseudo_records
    random.shuffle(train_expanded)

    train_out = output_dir / "train_u_positive_sft.json"
    valid_out = output_dir / "valid_gold_sft.json"

    write_json(train_expanded, train_out)
    write_json(valid_gold_records, valid_out)

    report = {
        "train_json": str(train_path),
        "valid_json": str(valid_path),
        "train_scored_candidates": str(scored_train_path),
        "output_dir": str(output_dir),
        "rdkit_available": HAS_RDKIT,
        "num_gold_train": len(gold_records),
        "num_valid_gold": len(valid_gold_records),
        "num_pseudo_positive": len(pseudo_records),
        "num_train_expanded": len(train_expanded),
        "pseudo_to_gold_ratio": len(pseudo_records) / len(gold_records) if gold_records else 0.0,
        "num_samples_with_pseudo": len(selected_per_sample),
        "filters": {
            "max_pseudo_per_sample": args.max_pseudo_per_sample,
            "max_pseudo_to_gold_ratio": args.max_pseudo_to_gold_ratio,
            "min_valid_score": args.min_valid_score,
            "max_bad_action_penalty": args.max_bad_action_penalty,
            "min_forward_plausibility": args.min_forward_plausibility,
            "min_route_future_proxy": args.min_route_future_proxy,
            "min_infomax_similarity": args.min_infomax_similarity,
            "min_bb_ratio": args.min_bb_ratio,
            "min_utility": args.min_utility,
        },
        "reject_reasons": reject_reasons,
        "train_out": str(train_out),
        "valid_out": str(valid_out),
    }

    write_json(report, output_dir / "u_positive_sft_data_report.json")

    print("\n===== U-positive SFT data report =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_json",
        type=str,
        default="./single_step_no_overlap/train_single_step_dedup.json",
    )
    parser.add_argument(
        "--valid_json",
        type=str,
        default="./single_step_no_overlap/valid_single_step_no_train_overlap.json",
    )
    parser.add_argument(
        "--train_scored_candidates",
        type=str,
        default="./dpo_pairs_u/train_scored_candidates.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_u_positive_v1",
    )

    parser.add_argument("--max_pseudo_per_sample", type=int, default=1)
    parser.add_argument("--max_pseudo_to_gold_ratio", type=float, default=0.5)

    parser.add_argument("--min_valid_score", type=float, default=1.0)
    parser.add_argument("--max_bad_action_penalty", type=float, default=0.0)
    parser.add_argument("--min_forward_plausibility", type=float, default=0.5)
    parser.add_argument("--min_route_future_proxy", type=float, default=0.0)
    parser.add_argument("--min_infomax_similarity", type=float, default=0.0)
    parser.add_argument("--min_bb_ratio", type=float, default=0.0)
    parser.add_argument("--min_utility", type=float, default=1.5)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    build_positive_data(args)


if __name__ == "__main__":
    main()