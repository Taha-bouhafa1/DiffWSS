import torch
import torch.nn as nn


class DiffusionFeatureExtractor(nn.Module):

    def __init__(self, diffusion, timesteps=(0, 1, 2, 3, 4)):
        super().__init__()

        self.diffusion = diffusion
        self.timesteps = tuple(timesteps)

        self.diffusion.eval()

        for parameter in self.diffusion.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, images, noise=None):

        batch_size = images.shape[0]
        device = images.device

        if noise is None:
            noise = torch.randn_like(images)

        all_features = []

        for timestep in self.timesteps:
            t = torch.full(
                (batch_size,),
                timestep,
                device=device,
                dtype=torch.long
            )

            noisy_images = self.diffusion.q_sample(
                images,
                t,
                noise
            )

            _, features = self.diffusion.model(
                noisy_images,
                t
            )

            all_features.append(features)

        return all_features
