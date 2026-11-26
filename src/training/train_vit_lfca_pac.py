from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_device(batch, device):

    if isinstance(batch, dict):
        images = batch.get("image", batch.get("images"))
        images_diff = batch.get(
            "image_diff",
            batch.get("images_diff", images),
        )
        labels = batch.get("image_labels", batch.get("labels"))

    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images = batch[0]
        images_diff = batch[1] if len(batch) >= 3 else batch[0]
        labels = batch[1] if len(batch) == 2 else batch[2]

    else:
        raise TypeError(
            "Expected a dictionary or tuple containing images and labels."
        )

    if images is None or labels is None:
        raise KeyError("Images or image-level labels are missing.")

    images = images.to(device, non_blocking=True)
    images_diff = images_diff.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    return images, images_diff, labels


def get_loss(outputs, validation=False):

    if not isinstance(outputs, dict):
        raise TypeError("Model must return a dictionary.")

    loss = outputs.get("total_loss")

    if validation:
        loss = outputs.get("val_loss", loss)

    if loss is None:
        raise ValueError("Model output does not contain a loss.")

    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise ValueError("Loss must be a scalar tensor.")

    return loss


def record_losses(totals, outputs, batch_size):

    for name, value in outputs.items():

        if (
            "loss" in name.lower()
            and torch.is_tensor(value)
            and value.numel() == 1
        ):

            totals[name] = (
                totals.get(name, 0.0)
                + value.detach().float().item() * batch_size
            )


def average_losses(totals, count):

    if count == 0:
        return {}

    return {
        name: value / count
        for name, value in totals.items()
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    scaler=None,
    amp=True,
    grad_clip_norm=1.0
):

    model.train()

    totals = {}
    count = 0

    use_amp = amp and device.type == "cuda"

    progress = tqdm(
        loader,
        desc=f"Train {epoch:03d}",
        leave=True
    )

    for batch in progress:

        images, images_diff, labels = batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(
                images,
                labels,
                images_diff=images_diff,
                epoch=epoch,
            )
            loss = get_loss(outputs)

        if use_amp and scaler is not None:

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm
                )

            scaler.step(optimizer)
            scaler.update()

        else:

            loss.backward()

            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm
                )

            optimizer.step()

        batch_size = images.shape[0]

        count += batch_size

        record_losses(
            totals,
            outputs,
            batch_size
        )

        progress.set_postfix(
            loss=f"{loss.detach().item():.5f}"
        )

    return average_losses(totals, count)


@torch.inference_mode()
def validate_one_epoch(
    model,
    loader,
    device,
    epoch,
    amp=True
):

    model.eval()

    totals = {}
    count = 0

    use_amp = amp and device.type == "cuda"

    progress = tqdm(
        loader,
        desc=f"Validation {epoch:03d}",
        leave=True
    )

    for batch in progress:

        images, images_diff, labels = batch_to_device(batch, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(
                images,
                labels,
                images_diff=images_diff,
                epoch=epoch,
            )

            loss = get_loss(
                outputs,
                validation=True
            )

        batch_size = images.shape[0]

        count += batch_size

        record_losses(
            totals,
            outputs,
            batch_size
        )

        progress.set_postfix(
            val_loss=f"{loss.item():.5f}"
        )

    return average_losses(totals, count)


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_loss,
    history,
    config=None,
    best_miou=0.0
):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None else None
        ),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None else None
        ),
        "best_val_loss": best_val_loss,
        "best_miou": best_miou,
        "history": history,
        "config": config or {}
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu"
):

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        state = checkpoint.get("optimizer_state_dict")

        if state is not None:
            optimizer.load_state_dict(state)

    if scheduler is not None:
        state = checkpoint.get("scheduler_state_dict")

        if state is not None:
            scheduler.load_state_dict(state)

    if scaler is not None:
        state = checkpoint.get("scaler_state_dict")

        if state is not None:
            scaler.load_state_dict(state)

    return checkpoint


def save_history(history, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "history.json"

    json_path.write_text(
        json.dumps(history, indent=4)
    )


def append_csv(metrics, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "history.csv"

    rows = []
    fieldnames = []

    if csv_path.exists():
        with csv_path.open("r", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

    for key in metrics.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    with csv_path.open(
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(metrics)


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device,
    epochs,
    output_dir,
    scheduler=None,
    scaler=None,
    amp=True,
    grad_clip_norm=1.0,
    patience=20,
    start_epoch=1,
    best_val_loss=math.inf,
    history=None,
    config=None,
    save_every=5,
    miou_evaluator=None,
    miou_start_epoch=10,
    best_miou=0.0
):

    device = torch.device(device)

    model.to(device)

    output_dir = Path(output_dir)

    checkpoint_dir = output_dir / "checkpoints"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if history is None:
        history = {
            "train": [],
            "validation": [],
            "miou": []
        }

    if "miou" not in history:
        history["miou"] = []

    no_improvement = 0

    for epoch in range(start_epoch, epochs + 1):

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            amp=amp,
            grad_clip_norm=grad_clip_norm
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            amp=amp
        )

        val_loss = val_metrics.get(
            "val_loss",
            val_metrics.get("total_loss")
        )

        if val_loss is None:
            raise RuntimeError(
                "Validation did not return a valid loss."
            )

        miou_metrics = None
        use_miou = miou_evaluator is not None and epoch >= miou_start_epoch

        if use_miou:
            miou_metrics = miou_evaluator(model, epoch)
            current_score = miou_metrics["miou"]
            history["miou"].append({
                "epoch": epoch,
                **miou_metrics
            })
            score_improved = current_score > best_miou

            if score_improved:
                best_miou = current_score

        else:
            current_score = val_loss
            score_improved = val_loss < best_val_loss

            if score_improved:
                best_val_loss = val_loss

        history["train"].append({
            "epoch": epoch,
            **train_metrics
        })

        history["validation"].append({
            "epoch": epoch,
            **val_metrics
        })

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(
            checkpoint_dir / "last.pt",
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_val_loss,
            history,
            config,
            best_miou
        )

        if epoch % save_every == 0:

            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val_loss,
                history,
                config
            )

        if score_improved:

            no_improvement = 0

            save_checkpoint(
                checkpoint_dir / "best.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val_loss,
                history,
                config
            )

            if use_miou:
                print("New best checkpoint saved by CAM mIoU.")
            else:
                print("New best checkpoint saved by validation loss.")

        else:
            no_improvement += 1

        metrics = {
            "epoch": epoch,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            },
            **(
                {
                    "miou": miou_metrics["miou"],
                    "threshold": miou_metrics["threshold"]
                }
                if miou_metrics is not None
                else {}
            )
        }

        append_csv(metrics, output_dir)

        save_history(
            history,
            output_dir
        )

        print(
            f"\nEpoch {epoch}/{epochs}"
            f"\nTrain loss: {train_metrics.get('total_loss', float('nan')):.5f}"
            f"\nValidation loss: {val_loss:.5f}"
            f"\nBest validation loss: {best_val_loss:.5f}"
            + (
                f"\nCAM mIoU: {miou_metrics['miou']:.3f}"
                f"\nBest CAM mIoU: {best_miou:.3f}"
                f"\nThreshold: {miou_metrics['threshold']}"
                if miou_metrics is not None
                else ""
            )
            + f"\nNo improvement: {no_improvement}/{patience}"
        )

        if patience is not None and no_improvement >= patience:

            print(
                f"Early stopping at epoch {epoch}."
            )

            break

    return history