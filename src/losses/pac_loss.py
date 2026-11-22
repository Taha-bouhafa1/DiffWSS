import torch
import torch.nn as nn
import torch.nn.functional as F


class PACLoss(nn.Module):

    def __init__(self, weight=1.0, eps=1e-5):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def max_norm(self, cams):
        batch_size, num_classes, height, width = cams.shape

        cams = F.relu(cams)

        cams_flat = cams.reshape(batch_size, num_classes, -1)

        max_values = cams_flat.max(dim=-1)[0].view(
            batch_size, num_classes, 1, 1
        )

        min_values = cams_flat.min(dim=-1)[0].view(
            batch_size, num_classes, 1, 1
        )

        cams = F.relu(cams - min_values - self.eps) / (
            max_values - min_values + self.eps
        )

        return cams

    def forward(self, original_rcams, denoised_rcams, image_labels):

        if original_rcams.shape != denoised_rcams.shape:
            raise ValueError(
                "Original and denoised refined CAMs must have identical shapes"
            )

        batch_size, num_classes, _, _ = original_rcams.shape

        if image_labels.shape != (batch_size, num_classes):
            raise ValueError(
                f"Expected image labels shape {(batch_size, num_classes)}, "
                f"got {tuple(image_labels.shape)}"
            )

        original_normalized = self.max_norm(original_rcams).detach()

        denoised_normalized = self.max_norm(denoised_rcams)

        class_mask = image_labels.to(
            device=original_rcams.device,
            dtype=original_rcams.dtype
        ).view(batch_size, num_classes, 1, 1)

        loss = (
            (original_normalized - denoised_normalized) * class_mask
        ).abs().mean()

        return self.weight * loss
