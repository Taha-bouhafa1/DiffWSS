import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lfca import LFCA, DiffusionCAMHead
from src.losses.classification_loss import DiGClassificationLoss
from src.losses.pac_loss import PACLoss


class DiffWSSModel(nn.Module):

    def __init__(
        self,
        vit,
        diffusion_feature_extractor,
        num_classes=6,
        vit_dim=768,
        diffusion_dim=512,
        num_heads=8,
        pac_weight=1.0,
        lfca_weight=1.0,
        classification_alpha=0.2,
        temperature=5.0,
    ):
        super().__init__()

        self.vit = vit
        self.diffusion_feature_extractor = diffusion_feature_extractor

        self.num_classes = num_classes

        self.lfca = LFCA(
            vit_dim=vit_dim,
            diffusion_dim=diffusion_dim,
            num_classes=num_classes,
            num_heads=num_heads,
        )

        self.diffusion_cam_head = DiffusionCAMHead(
            diffusion_dim=diffusion_dim,
            num_classes=num_classes,
        )

        self.classification_loss = DiGClassificationLoss(
            alpha=classification_alpha,
            temperature=temperature,
        )

        self.pac_loss = PACLoss(weight=pac_weight)

        self.lfca_weight = lfca_weight

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _imagenet_to_ddpm(self, images):
        images_01 = images * self.imagenet_std + self.imagenet_mean
        images_01 = images_01.clamp(0.0, 1.0)
        return images_01 * 2.0 - 1.0

    def _ddpm_to_imagenet(self, images):
        images_01 = (images + 1.0) * 0.5
        return (images_01 - self.imagenet_mean) / self.imagenet_std

    def _get_query_tokens(self, vit_outputs):
        q = vit_outputs["q"][-1]

        batch_size, num_heads, num_tokens, head_dim = q.shape

        query_tokens = (
            q.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, num_tokens, num_heads * head_dim)
        )

        return query_tokens

    def _get_patch_logits(self, cam_logits):
        flat_cams = cam_logits.flatten(2)

        top_k = min(3, flat_cams.shape[-1])

        positive_scores = torch.topk(
            F.relu(flat_cams),
            k=top_k,
            dim=-1,
        )[0].mean(dim=-1)

        negative_scores = torch.topk(
            F.relu(-flat_cams),
            k=top_k,
            dim=-1,
        )[0].mean(dim=-1)

        return positive_scores - negative_scores

    def _max_norm(self, cams, eps=1e-5):
        batch_size, num_classes, _, _ = cams.shape

        cams = F.relu(cams)

        cams_flat = cams.flatten(2)

        max_values = cams_flat.max(dim=-1)[0].view(
            batch_size, num_classes, 1, 1
        )

        min_values = cams_flat.min(dim=-1)[0].view(
            batch_size, num_classes, 1, 1
        )

        return F.relu(cams - min_values - eps) / (
            max_values - min_values + eps
        )

    def forward(self, images, image_labels):

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [B, 3, H, W].")

        if image_labels.shape != (
            images.shape[0],
            self.num_classes,
        ):
            raise ValueError("Image-level labels have an incorrect shape.")

        image_labels = image_labels.float()

        ddpm_images = self._imagenet_to_ddpm(images)

        diffusion_outputs = self.diffusion_feature_extractor(
            ddpm_images,
            extract_pac_image=True,
        )

        diffusion_features = diffusion_outputs["features"]

        denoised_image = diffusion_outputs["denoised_image"]

        if len(diffusion_features) == 0:
            raise RuntimeError("No diffusion features were extracted.")

        if isinstance(diffusion_features[0], (list, tuple)):
            diffusion_features = diffusion_features[0]

        if len(diffusion_features) != 5:
            raise RuntimeError("Expected five diffusion feature tensors.")

        vit_outputs = self.vit(images)

        query_tokens = self._get_query_tokens(vit_outputs)

        diffusion_cls_tokens, diffusion_spatial_features = self.lfca(
            query_tokens,
            diffusion_features,
        )

        diffusion_cam_logits = self.diffusion_cam_head(
            diffusion_spatial_features
        )

        diffusion_cls_logits = self.vit.class_head(
            diffusion_cls_tokens
        ).squeeze(-1)

        diffusion_patch_logits = self._get_patch_logits(
            diffusion_cam_logits
        )

        denoised_imagenet = self._ddpm_to_imagenet(
            denoised_image
        )

        denoised_outputs = self.vit(denoised_imagenet)

        classification_losses = self.classification_loss(
            vit_cls_logits=vit_outputs["cls"],
            vit_patch_logits=vit_outputs["pcls"],
            labels=image_labels,
            diffusion_cls_logits=diffusion_cls_logits,
            diffusion_patch_logits=diffusion_patch_logits,
            denoised_cls_logits=denoised_outputs["cls"],
            denoised_patch_logits=denoised_outputs["pcls"],
        )

        original_cams = vit_outputs["cams"]

        if original_cams.shape[-2:] != diffusion_cam_logits.shape[-2:]:
            original_cams = F.interpolate(
                original_cams,
                size=diffusion_cam_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        class_mask = image_labels.view(
            images.shape[0],
            self.num_classes,
            1,
            1,
        )

        loss_lfca = (
            (
                self._max_norm(diffusion_cam_logits)
                - self._max_norm(original_cams).detach()
            )
            * class_mask
        ).abs().mean()

        loss_lfca = self.lfca_weight * loss_lfca

        loss_pac = self.pac_loss(
            vit_outputs["rcams"],
            denoised_outputs["rcams"],
            image_labels,
        )

        total_loss = (
            classification_losses["total"]
            + loss_lfca
            + loss_pac
        )

        return {
            "total_loss": total_loss,
            "loss_vit": classification_losses["loss_vit"],
            "loss_diffusion": classification_losses["loss_diffusion"],
            "loss_denoised": classification_losses["loss_denoised"],
            "loss_lfca": loss_lfca,
            "loss_pac": loss_pac,
        }
