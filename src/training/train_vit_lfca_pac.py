"""DiffWSS phase-2 training and validation loops.

Model contract:
    outputs = model(images, image_labels)
    outputs is a dict with scalar "total_loss" and optional scalar loss terms.
    During validation, "val_loss" may be returned; otherwise "total_loss" is used.

The model forward is responsible for the DiG-specific ViT, frozen DDPM,
LFCA, CAM and PAC computations. This module owns orchestration, optimization,
validation, logging, checkpointing and resume.
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_to_device(batch, device):
    if isinstance(batch, dict):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        images = batch.get("image", batch.get("images"))
        labels = batch.get("image_labels", batch.get("labels"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, labels = batch[0].to(device), batch[1].to(device)
    else:
        raise TypeError("Batch must be a dict or (images, image_labels).")
    if images is None or labels is None:
        raise KeyError("Batch needs image/images and image_labels/labels.")
    return images, labels


def _loss_tensor(outputs, validation=False):
    if not isinstance(outputs, dict):
        raise TypeError("model(images, labels) must return a dict.")
    loss = outputs.get("val_loss") if validation else outputs.get("total_loss")
    if loss is None and validation:
        loss = outputs.get("total_loss")
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise ValueError("Model output must contain a scalar loss tensor.")
    return loss


def _record_losses(totals, outputs, n):
    for name, value in outputs.items():
        if "loss" in name.lower() and torch.is_tensor(value) and value.numel() == 1:
            totals[name] = totals.get(name, 0.0) + value.detach().float().item() * n


def _averages(totals, count):
    return {k: v / count for k, v in totals.items()} if count else {}


def train_one_epoch(model, loader, optimizer, device, epoch, *,
                    scaler=None, amp=True, grad_clip_norm=1.0):
    model.train()
    totals, count = {}, 0
    use_amp = bool(amp and device.type == "cuda")
    bar = tqdm(loader, desc=f"Train {epoch:03d}", leave=False)

    for batch in bar:
        images, labels = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            outputs = model(images, labels)
            loss = _loss_tensor(outputs)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        n = images.shape[0]
        count += n
        _record_losses(totals, outputs, n)
        bar.set_postfix(loss=f"{loss.detach().item():.4f}")

    return _averages(totals, count)


@torch.inference_mode()
def validate_one_epoch(model, loader, device, epoch, *, amp=True):
    model.eval()
    totals, count = {}, 0
    use_amp = bool(amp and device.type == "cuda")
    bar = tqdm(loader, desc=f"Valid {epoch:03d}", leave=False)

    for batch in bar:
        images, labels = _batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            outputs = model(images, labels)
            loss = _loss_tensor(outputs, validation=True)
        n = images.shape[0]
        count += n
        _record_losses(totals, outputs, n)
        bar.set_postfix(val_loss=f"{loss.detach().item():.4f}")

    return _averages(totals, count)


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_val_loss, history, config=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_val_loss": best_val_loss,
        "history": history,
        "config": config or {},
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict"):
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict"):
        scaler.load_state_dict(state["scaler_state_dict"])
    return state


def fit(model, train_loader: DataLoader, val_loader: DataLoader,
        optimizer, device, *, epochs: int, output_dir,
        scheduler=None, scaler=None, amp=True, grad_clip_norm=1.0,
        patience: Optional[int] = None, start_epoch=1,
        best_val_loss=math.inf, config=None):
    device = torch.device(device)
    model.to(device)
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = {"train": [], "validation": []}
    no_improve = 0

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            scaler=scaler, amp=amp, grad_clip_norm=grad_clip_norm)
        val_metrics = validate_one_epoch(
            model, val_loader, device, epoch, amp=amp)

        val_loss = val_metrics.get("val_loss", val_metrics.get("total_loss"))
        if val_loss is None:
            raise KeyError("Validation must report val_loss or total_loss.")

        history["train"].append({"epoch": epoch, **train_metrics})
        history["validation"].append({"epoch": epoch, **val_metrics})

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(ckpt_dir / "last.pt", epoch, model, optimizer,
                        scheduler, scaler, best_val_loss, history, config)

        if val_loss < best_val_loss:
            best_val_loss, no_improve = val_loss, 0
            save_checkpoint(ckpt_dir / "best.pt", epoch, model, optimizer,
                            scheduler, scaler, best_val_loss, history, config)
        else:
            no_improve += 1

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        csv_path = output_dir / "history.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(f"Epoch {epoch:03d}/{epochs} | "
              f"train={train_metrics.get('total_loss', float('nan')):.5f} | "
              f"val={val_loss:.5f} | best={best_val_loss:.5f}")

        if patience is not None and no_improve >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    return history
