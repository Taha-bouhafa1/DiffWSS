from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.cam.fusion import extract_cams


def collect_cam_dicts(model, loader, device, size=(512, 512)):
    model.eval()

    items = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["image_labels"].to(device, non_blocking=True)

            cams = extract_cams(model, images, labels, size=size)
            cams = cams.float().cpu().numpy()
            labels_np = labels.cpu().numpy()

            for name, cam, label in zip(batch["name"], cams, labels_np):
                active = np.flatnonzero(label > 0.5)
                items.append(
                    (str(name), {int(index): cam[index] for index in active})
                )

    return items


def evaluate_cam_dicts(items, mask_dir, num_classes=6, thresholds=None):
    mask_dir = Path(mask_dir)

    if thresholds is None:
        thresholds = [index / 100.0 for index in range(35, 70)]

    ground_truth = []
    for name, _ in items:
        with Image.open(mask_dir / f"{name}.png") as loaded:
            ground_truth.append(np.asarray(loaded).astype(np.int16))

    best_miou = -1.0
    best_threshold = None
    best_precision = None
    best_recall = None
    best_iou = None

    for threshold in thresholds:
        true_positive = np.zeros(num_classes, dtype=np.float64)
        predicted = np.zeros(num_classes, dtype=np.float64)
        target = np.zeros(num_classes, dtype=np.float64)

        for (_, cam_dict), gt in zip(items, ground_truth):
            height, width = gt.shape
            scores = np.zeros(
                (num_classes + 1, height, width),
                dtype=np.float32,
            )
            scores[0, :, :] = threshold

            for class_id, cam in cam_dict.items():
                if class_id < num_classes:
                    scores[class_id + 1, :, :] = cam

            prediction = scores.argmax(axis=0).astype(np.int16)
            valid = gt < 255
            shifted = gt + 1

            for class_id in range(num_classes):
                class_shifted = class_id + 1
                predicted_mask = prediction == class_shifted
                true_positive[class_id] += np.sum(
                    (shifted == class_shifted) & predicted_mask & valid
                )
                predicted[class_id] += np.sum(predicted_mask & valid)
                target[class_id] += np.sum((shifted == class_shifted) & valid)

        iou = true_positive / (
            target + predicted - true_positive + 1e-10
        )
        precision = true_positive / (predicted + 1e-10)
        recall = true_positive / (target + 1e-10)
        miou = float(np.mean(iou) * 100.0)

        if miou > best_miou:
            best_miou = miou
            best_threshold = threshold
            best_precision = precision.tolist()
            best_recall = recall.tolist()
            best_iou = iou.tolist()
        else:
            break

    return {
        "miou": best_miou,
        "threshold": best_threshold,
        "iou_per_class": best_iou,
        "precision_per_class": best_precision,
        "recall_per_class": best_recall,
    }
