import torch
import torch.nn as nn


class DiffusionFeatureExtractor(nn.Module):

    def __init__(
        self,
        diffusion,
        timesteps=(0, 1, 2, 3, 4),
        expected_channels=(256, 128, 64, 32, 32),
        pac_timestep=100
    ):
        super().__init__()

        self.diffusion = diffusion
        self.timesteps = tuple(timesteps)
        self.expected_channels = tuple(expected_channels)
        self.pac_timestep = pac_timestep

        self.diffusion.eval()

        for parameter in self.diffusion.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def _unet_output(self, noisy_images, timesteps):

        output = self.diffusion.model(
            noisy_images,
            timesteps,
            None
        )

        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "Expected U-Net output: "
                "(predicted_output, feature_maps)"
            )

        return output

    @torch.no_grad()
    def _validate_features(self, features, batch_size, index):

        if len(features) != len(self.expected_channels):
            raise RuntimeError(
                f"Timestep {index}: expected "
                f"{len(self.expected_channels)} feature maps, "
                f"received {len(features)}"
            )

        for feature_index, (feature, expected_channels) in enumerate(
            zip(features, self.expected_channels)
        ):

            if feature.ndim != 4:
                raise RuntimeError(
                    f"Timestep {index}, feature {feature_index}: "
                    "must be 4-dimensional"
                )

            if feature.shape[0] != batch_size:
                raise RuntimeError(
                    f"Timestep {index}, feature {feature_index}: "
                    "batch size mismatch"
                )

            if feature.shape[1] != expected_channels:
                raise RuntimeError(
                    f"Timestep {index}, feature {feature_index}: "
                    f"expected {expected_channels} channels, "
                    f"received {feature.shape[1]}"
                )

    @torch.no_grad()
    def forward(self, images, noise=None, extract_pac_image=True):

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected images [B, 3, H, W], received {images.shape}"
            )

        batch_size = images.shape[0]
        device = images.device

        if noise is not None and noise.shape != images.shape:
            raise ValueError(
                "Noise and images must have identical shapes"
            )

        self.diffusion.eval()
        self.diffusion.model.eval()

        all_features = []

        for timestep_index, timestep in enumerate(self.timesteps):

            t = torch.full(
                (batch_size,),
                timestep,
                device=device,
                dtype=torch.long
            )

            step_noise = torch.randn_like(images)

            noisy_images = self.diffusion.q_sample(
                images,
                t,
                step_noise
            )

            _, features = self._unet_output(noisy_images, t)

            self._validate_features(
                features,
                batch_size,
                timestep_index
            )

            spatial_size = features[0].shape[-2:]

            for feature in features[1:]:
                if feature.shape[-2:] != spatial_size:
                    raise RuntimeError(
                        f"Timestep {timestep_index}: feature spatial "
                        "size mismatch"
                    )

            all_features.append(
                torch.cat(features, dim=1)
            )

        denoised_image = None

        if extract_pac_image:
            t = torch.full(
                (batch_size,),
                self.pac_timestep,
                device=device,
                dtype=torch.long
            )

            pac_noise = (
                noise if noise is not None else torch.randn_like(images)
            )

            noisy_images = self.diffusion.q_sample(
                images,
                t,
                pac_noise
            )

            predicted_velocity, _ = self._unet_output(
                noisy_images,
                t
            )

            reconstructed = self.diffusion.predict_start_from_v(
                noisy_images,
                t,
                predicted_velocity
            )

            denoised_image = self.diffusion.unnormalize(reconstructed)

        return {
            "features": all_features,
            "denoised_image": denoised_image
        }
