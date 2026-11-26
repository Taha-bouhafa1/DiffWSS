import torch
import torch.nn as nn
import torch.nn.functional as F


class DiGClassificationLoss(nn.Module):

    def __init__(
        self,
        alpha=0.2,
        temperature=5.0,
        weight_vit=1.0,
        weight_diffusion=1.0,
        weight_denoised=1.0
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError("Temperature must be positive.")

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1.")

        self.alpha = alpha
        self.temperature = temperature

        self.weight_vit = weight_vit
        self.weight_diffusion = weight_diffusion
        self.weight_denoised = weight_denoised

        self.kl_loss = nn.KLDivLoss(
            reduction="batchmean",
            log_target=True
        )

    def _check_shapes(self, logits, labels, name):

        if logits.shape != labels.shape:
            raise ValueError(
                f"{name}: expected logits and labels with identical "
                f"shapes, got {logits.shape} and {labels.shape}"
            )

    def multilabel_loss(self, logits, labels):

        self._check_shapes(
            logits,
            labels,
            "Multilabel classification"
        )

        return F.multilabel_soft_margin_loss(
            logits,
            labels
        )

    def knowledge_distillation(
        self,
        student_logits,
        teacher_logits,
        labels
    ):

        self._check_shapes(
            student_logits,
            labels,
            "Student predictions"
        )

        self._check_shapes(
            teacher_logits,
            labels,
            "Teacher predictions"
        )

        temperature = self.temperature

        # Student predictions
        student_probs = F.softmax(
            student_logits / temperature,
            dim=1
        )

        # Teacher predictions, detached
        teacher_probs = F.softmax(
            teacher_logits.detach() / temperature,
            dim=1
        )

        # Reproduce the original DiG probability construction
        student_probs = torch.cat(
            (
                student_probs,
                1.0 - student_probs
            ),
            dim=1
        )

        teacher_probs = torch.cat(
            (
                teacher_probs,
                1.0 - teacher_probs
            ),
            dim=1
        )

        student_log_probs = torch.log(student_probs)

        teacher_log_probs = torch.log(teacher_probs)

        kl = self.kl_loss(
            student_log_probs,
            teacher_log_probs
        )

        kl = kl * (temperature ** 2) * 0.5

        classification = self.multilabel_loss(
            student_logits,
            labels
        )

        return (
            self.alpha * classification
            + (1.0 - self.alpha) * kl
        )

    def forward(
        self,
        vit_cls_logits,
        vit_patch_logits,
        labels,
        diffusion_cls_logits=None,
        diffusion_patch_logits=None,
        denoised_cls_logits=None,
        denoised_patch_logits=None
    ):

        # 1. Original ViT classification
        loss_vit = (
            self.multilabel_loss(vit_cls_logits, labels)
            + self.multilabel_loss(vit_patch_logits, labels)
        )

        total_loss = self.weight_vit * loss_vit

        losses = {
            "loss_vit": loss_vit
        }

        # 2. Diffusion classification and knowledge distillation
        if (
            diffusion_cls_logits is not None
            and diffusion_patch_logits is not None
        ):

            loss_diffusion = (
                self.knowledge_distillation(
                    diffusion_cls_logits,
                    vit_cls_logits,
                    labels
                )
                +
                self.knowledge_distillation(
                    diffusion_patch_logits,
                    vit_patch_logits,
                    labels
                )
            )

            total_loss = (
                total_loss
                + self.weight_diffusion * loss_diffusion
            )

            losses["loss_diffusion"] = loss_diffusion

        # 3. Denoised-image classification
        if (
            denoised_cls_logits is not None
            and denoised_patch_logits is not None
        ):

            loss_denoised = (
                self.multilabel_loss(
                    denoised_cls_logits,
                    labels
                )
                +
                self.multilabel_loss(
                    denoised_patch_logits,
                    labels
                )
            )

            total_loss = (
                total_loss
                + self.weight_denoised * loss_denoised
            )

            losses["loss_denoised"] = loss_denoised

        losses["total"] = total_loss

        return losses
