import torch

from src.losses.classification_loss import DiGClassificationLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B, C = 2, 6

criterion = DiGClassificationLoss().to(device)

# Image-level labels
labels = torch.randint(0, 2, (B, C)).float().to(device)

# Original ViT predictions
vit_cls = torch.randn(B, C, device=device, requires_grad=True)
vit_patch = torch.randn(B, C, device=device, requires_grad=True)

# Diffusion/LFCA predictions
diff_cls = torch.randn(B, C, device=device, requires_grad=True)
diff_patch = torch.randn(B, C, device=device, requires_grad=True)

# Denoised-image predictions
den_cls = torch.randn(B, C, device=device, requires_grad=True)
den_patch = torch.randn(B, C, device=device, requires_grad=True)

losses = criterion(
    vit_cls_logits=vit_cls,
    vit_patch_logits=vit_patch,
    labels=labels,
    diffusion_cls_logits=diff_cls,
    diffusion_patch_logits=diff_patch,
    denoised_cls_logits=den_cls,
    denoised_patch_logits=den_patch
)

losses["total"].backward()

for name, loss in losses.items():
    print(f"{name}: {loss.item():.6f}")

print("ViT gradients:", vit_cls.grad is not None)
print("Diffusion gradients:", diff_cls.grad is not None)
print("Denoised gradients:", den_cls.grad is not None)
