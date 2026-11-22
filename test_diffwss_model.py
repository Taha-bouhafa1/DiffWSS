
import torch
import torch.nn as nn

from src.models.vit import ViTBase16
from src.models.diffwss_model import DiffWSSModel


class MockDiffusionFeatureExtractor(nn.Module):

    def forward(self, images, extract_pac_image=True):
        batch_size = images.shape[0]
        device = images.device

        features = [
            torch.randn(batch_size, 256, 4, 4, device=device),
            torch.randn(batch_size, 128, 4, 4, device=device),
            torch.randn(batch_size, 64, 4, 4, device=device),
            torch.randn(batch_size, 32, 4, 4, device=device),
            torch.randn(batch_size, 32, 4, 4, device=device),
        ]

        return {
            "features": features,
            "denoised_image": torch.randn_like(images)
            if extract_pac_image else None,
        }


def main():

    torch.manual_seed(42)

    device = torch.device("cpu")

    print("Device:", device)

    vit = ViTBase16(
        num_classes=6,
        img_size=64,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        pretrained=False,
    )

    diffusion_extractor = MockDiffusionFeatureExtractor()

    model = DiffWSSModel(
        vit=vit,
        diffusion_feature_extractor=diffusion_extractor,
        num_classes=6,
        vit_dim=768,
        diffusion_dim=512,
        num_heads=8,
    ).to(device)

    model.train()

    images = torch.randn(1, 3, 64, 64, device=device)

    image_labels = torch.tensor(
        [[1, 1, 0, 1, 0, 0]],
        dtype=torch.float32,
        device=device,
    )

    print("\nRunning forward pass...")

    outputs = model(images, image_labels)

    print("\nLosses:")

    for name, value in outputs.items():
        print(f"{name}: {value.item():.6f}")
        assert torch.isfinite(value), f"{name} contains NaN or Inf"

    assert outputs["total_loss"].ndim == 0

    print("\nRunning backward pass...")

    outputs["total_loss"].backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    assert len(gradients) > 0
    assert all(torch.isfinite(grad).all() for grad in gradients)

    print("Gradient check: PASSED")

    print("\nChecking frozen DDPM extractor interface...")

    assert all(
        not parameter.requires_grad
        for parameter in diffusion_extractor.parameters()
    )

    print("Frozen extractor check: PASSED")

    print("\n================================")
    print("DIFFWSS INTEGRATION TEST PASSED")
    print("================================")


if __name__ == "__main__":
    main()
