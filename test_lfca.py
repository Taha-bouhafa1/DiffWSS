import torch

from src.models.lfca import LFCA, DiffusionCAMHead

device = torch.device("cpu")

lfca = LFCA(
    vit_dim=768,
    diffusion_dim=512,
    num_classes=6,
    num_heads=8
).to(device)

cam_head = DiffusionCAMHead(
    diffusion_dim=512,
    num_classes=6
).to(device)

query_tokens = torch.randn(1, 1030, 768, device=device)

diffusion_features = [
    torch.randn(1, 256, 32, 32, device=device),
    torch.randn(1, 128, 32, 32, device=device),
    torch.randn(1, 64, 32, 32, device=device),
    torch.randn(1, 32, 32, 32, device=device),
    torch.randn(1, 32, 32, 32, device=device)
]

class_features, spatial_features = lfca(
    query_tokens,
    diffusion_features
)

diffusion_cams = cam_head(spatial_features)

print("Class features:", class_features.shape)
print("Spatial features:", spatial_features.shape)
print("Diffusion CAMs:", diffusion_cams.shape)