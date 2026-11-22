import torch

from src.models.ddpm_unet import Unet
from src.models.diffusion import GaussianDiffusion
from src.models.diffusion_features import DiffusionFeatureExtractor


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 256

unet = Unet(
    dim=32,
    channels=3,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model=unet,
    image_size=IMAGE_SIZE,
    timesteps=1000,
    objective="pred_v",
    beta_schedule="sigmoid",
    auto_normalize=False
)

checkpoint = torch.load(
    "checkpoints/ddpm/ddpm_unet_ema_latest.pth",
    map_location="cpu",
    weights_only=True
)

diffusion.model.load_state_dict(checkpoint, strict=True)

diffusion = diffusion.to(device)
diffusion.eval()

extractor = DiffusionFeatureExtractor(
    diffusion=diffusion
).to(device)

images = torch.randn(
    1, 3, IMAGE_SIZE, IMAGE_SIZE,
    device=device
)

outputs = extractor(images)

features = outputs["features"]
denoised_image = outputs["denoised_image"]

print("Number of timesteps:", len(features))

for timestep_index, timestep_features in enumerate(features):

    print(f"\nTimestep {timestep_index}:")

    for feature in timestep_features:
        print(feature.shape)

print("\nDenoised image:", tuple(denoised_image.shape))

assert len(features) == len(extractor.timesteps)
assert all(len(item) == 5 for item in features)
assert denoised_image.shape == images.shape
assert bool(torch.isfinite(denoised_image).all())

print("\nDiffusion feature extraction: PASSED")
