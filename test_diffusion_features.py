import torch

from src.models.ddpm_unet import Unet
from src.models.diffusion import GaussianDiffusion
from src.models.diffusion_features import DiffusionFeatureExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"

unet = Unet(
    dim=32,
    dim_mults=(1, 2, 4, 8),
    channels=3,
    resnet_block_groups=8,
    self_condition=False
).to(device)

state_dict = torch.load(
    "checkpoints/ddpm/ddpm_unet_ema_latest.pth",
    map_location=device,
    weights_only=True
)

unet.load_state_dict(state_dict, strict=True)

diffusion = GaussianDiffusion(
    model=unet,
    image_size=512,
    timesteps=1000,
    objective="pred_v",
    beta_schedule="sigmoid",
    auto_normalize=False
).to(device)

extractor = DiffusionFeatureExtractor(diffusion).to(device)

images = torch.randn(2, 3, 512, 512, device=device)

features = extractor(images)

print("Number of timesteps:", len(features))

for timestep, timestep_features in zip(extractor.timesteps, features):
    print(f"\nTimestep {timestep}:")
    for i, feature in enumerate(timestep_features):
        print(f"Feature {i}: {feature.shape}")

print("\nDDPM parameters trainable:",
      any(p.requires_grad for p in extractor.parameters()))
