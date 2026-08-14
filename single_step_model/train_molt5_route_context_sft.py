# train_molt5_route_context_sft.py

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)


SPECIAL_TOKENS = [
    "<target>",
    "<current>",
    "<current_depth>",
    "<max_depth>",
    "<goal>",
]


TASK_PREFIX = "Please predict the reactant of the product:\n"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_route_context_prompt(sample: Dict[str, Any], max_depth: int) -> str:
    """
    注意：
    current_depth 使用 step_order - 1，这是推理时可获得的信息；
    max_depth 使用固定搜索预算，不使用 gold route 的真实 depth；
    不输入 materials / intermediates / depth_remaining，避免泄露。
    """
    target = sample["target_smiles"]
    current = sample["current_smiles"]

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


class RouteContextSFTDataset(Dataset):
    def __init__(
        self,
        json_path: Path,
        tokenizer: T5Tokenizer,
        max_depth: int = 14,
        max_source_length: int = 512,
        max_target_length: int = 256,
    ):
        self.samples = load_json(json_path)
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        if not isinstance(self.samples, list):
            raise ValueError(f"{json_path} 顶层必须是 list。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        source_text = build_route_context_prompt(
            sample=sample,
            max_depth=self.max_depth,
        )

        target_text = sample["reactants_str"]

        source_enc = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = source_enc["input_ids"].squeeze(0)
        attention_mask = source_enc["attention_mask"].squeeze(0)
        labels = target_enc["input_ids"].squeeze(0)

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def add_special_tokens_and_resize(
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
):
    special_tokens_dict = {
        "additional_special_tokens": SPECIAL_TOKENS
    }

    num_added = tokenizer.add_special_tokens(special_tokens_dict)

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    return num_added


def save_checkpoint(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    output_dir: Path,
    name: str,
    extra: Dict[str, Any],
):
    ckpt_dir = output_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    write_json(extra, ckpt_dir / "training_state.json")

    print(f"[保存] checkpoint -> {ckpt_dir}")


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    fp16: bool = False,
    grad_clip: float = 1.0,
):
    model.train()

    total_loss = 0.0
    total_steps = 0

    scaler = None
    if fp16 and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    pbar = tqdm(dataloader, desc="train", leave=False)

    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        scheduler.step()

        loss_value = float(loss.detach().cpu().item())

        total_loss += loss_value
        total_steps += 1

        pbar.set_postfix({
            "loss": f"{loss_value:.4f}"
        })

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate_loss(
    model,
    dataloader,
    device,
    fp16: bool = False,
):
    model.eval()

    total_loss = 0.0
    total_steps = 0

    pbar = tqdm(dataloader, desc="valid", leave=False)

    for batch in pbar:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        if fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
        else:
            outputs = model(**batch)
            loss = outputs.loss

        loss_value = float(loss.detach().cpu().item())

        total_loss += loss_value
        total_steps += 1

        pbar.set_postfix({
            "loss": f"{loss_value:.4f}"
        })

    return total_loss / max(total_steps, 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./single_step",
        help="single-step 数据目录。"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="../molt5",
        help="原始 MolT5 路径。当前 datasets 目录下默认 ../molt5。"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./molt5_route_context_sft_maxdepth14",
        help="SFT 后模型输出目录。"
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
        "--max_depth",
        type=int,
        default=14,
        help="固定搜索最大深度，不使用 gold route depth。"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
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
        "--warmup_ratio",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--save_every_epochs",
        type=int,
        default=10,
        help="每隔多少个 epoch 保存一次 checkpoint。默认 10。"
    )

    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="valid loss 连续多少个 epoch 没有改善则停止。"
    )

    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="valid loss 至少下降多少才算改善。"
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="CUDA 上使用混合精度训练。"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / args.train_file
    valid_path = data_dir / args.valid_file

    if not train_path.exists():
        raise FileNotFoundError(f"找不到训练集：{train_path}")

    if not valid_path.exists():
        raise FileNotFoundError(f"找不到验证集：{valid_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===== Route-context MolT5 SFT =====")
    print(f"model_dir   : {model_dir}")
    print(f"train_path  : {train_path}")
    print(f"valid_path  : {valid_path}")
    print(f"output_dir  : {output_dir}")
    print(f"device      : {device}")
    print(f"max_depth   : {args.max_depth}")
    print(f"lr          : {args.learning_rate}")
    print(f"batch_size  : {args.batch_size}")
    print(f"epochs      : {args.epochs}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"fp16        : {args.fp16}")

    tokenizer = T5Tokenizer.from_pretrained(
        str(model_dir),
        model_max_length=args.max_source_length,
    )

    model = T5ForConditionalGeneration.from_pretrained(str(model_dir))

    num_added = add_special_tokens_and_resize(
        tokenizer=tokenizer,
        model=model,
    )

    print(f"Added special tokens: {num_added}")
    print(f"Tokenizer size      : {len(tokenizer)}")

    model.to(device)

    train_dataset = RouteContextSFTDataset(
        json_path=train_path,
        tokenizer=tokenizer,
        max_depth=args.max_depth,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    valid_dataset = RouteContextSFTDataset(
        json_path=valid_path,
        tokenizer=tokenizer,
        max_depth=args.max_depth,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    config_to_save = vars(args).copy()
    config_to_save.update({
        "special_tokens": SPECIAL_TOKENS,
        "task_prefix": TASK_PREFIX,
        "prompt_format": (
            "Please predict the reactant of the product:\\n"
            "<target> {target_smiles}\\n"
            "<current> {current_smiles}\\n"
            "<current_depth> {step_order_minus_1}\\n"
            "<max_depth> {fixed_max_depth}\\n"
            "<goal> purchasable_starting_materials"
        ),
        "num_train_samples": len(train_dataset),
        "num_valid_samples": len(valid_dataset),
        "total_training_steps": total_training_steps,
        "warmup_steps": warmup_steps,
    })

    write_json(config_to_save, output_dir / "sft_config.json")

    best_valid_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    training_log = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            fp16=args.fp16,
            grad_clip=args.grad_clip,
        )

        valid_loss = evaluate_loss(
            model=model,
            dataloader=valid_loader,
            device=device,
            fp16=args.fp16,
        )

        current_lr = scheduler.get_last_lr()[0]

        improved = valid_loss < (best_valid_loss - args.early_stop_min_delta)

        if improved:
            best_valid_loss = valid_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                name="checkpoint-best",
                extra={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "best_valid_loss": best_valid_loss,
                    "best_epoch": best_epoch,
                    "learning_rate": current_lr,
                },
            )

        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "best_valid_loss": best_valid_loss,
            "best_epoch": best_epoch,
            "learning_rate": current_lr,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
        }

        training_log.append(epoch_record)
        write_json(training_log, output_dir / "training_log.json")

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.6f} "
            f"valid_loss={valid_loss:.6f} "
            f"best_valid_loss={best_valid_loss:.6f} "
            f"best_epoch={best_epoch} "
            f"lr={current_lr:.8f} "
            f"improved={improved} "
            f"no_improve={epochs_without_improvement}"
        )

        if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                name=f"checkpoint-epoch-{epoch}",
                extra=epoch_record,
            )

        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"[Early Stop] valid loss 连续 "
                f"{args.early_stop_patience} 个 epoch 没有改善，停止训练。"
            )
            break

    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        name="checkpoint-last",
        extra={
            "last_epoch": training_log[-1]["epoch"] if training_log else 0,
            "best_valid_loss": best_valid_loss,
            "best_epoch": best_epoch,
        },
    )

    print("\n===== 训练完成 =====")
    print(f"Best epoch      : {best_epoch}")
    print(f"Best valid loss : {best_valid_loss:.6f}")
    print(f"Best checkpoint : {output_dir / 'checkpoint-best'}")
    print(f"Last checkpoint : {output_dir / 'checkpoint-last'}")


if __name__ == "__main__":
    main()