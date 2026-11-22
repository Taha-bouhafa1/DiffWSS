import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.GroupNorm(groups, out_channels)

        self.branch = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.branch(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity

        return self.act2(out)


class LFCA(nn.Module):

    def __init__(
        self,
        vit_dim=768,
        diffusion_dim=512,
        num_classes=6,
        num_heads=8,
        drop_path=0.1
    ):
        super().__init__()

        if vit_dim % num_heads != 0:
            raise ValueError(
                "vit_dim must be divisible by num_heads"
            )

        if diffusion_dim % num_heads != 0:
            raise ValueError(
                "diffusion_dim must be divisible by num_heads"
            )

        self.vit_dim = vit_dim
        self.diffusion_dim = diffusion_dim
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.head_dim = vit_dim // num_heads
        self.value_head_dim = diffusion_dim // num_heads

        self.expected_channels = [256, 128, 64, 32, 32]

        self.feature_projections = nn.ModuleList([
            nn.Conv2d(256, diffusion_dim, kernel_size=1),
            nn.Conv2d(128, diffusion_dim, kernel_size=1),
            nn.Conv2d(64, diffusion_dim, kernel_size=1),
            nn.Conv2d(32, diffusion_dim, kernel_size=1),
            nn.Conv2d(32, diffusion_dim, kernel_size=1)
        ])

        self.diff_init = ResidualBlock(
            diffusion_dim * 5,
            diffusion_dim,
            groups=1
        )

        self.norm_q = nn.LayerNorm(
            vit_dim,
            eps=1e-6
        )

        self.norm_k = nn.LayerNorm(
            vit_dim,
            eps=1e-6
        )

        self.norm_v = nn.LayerNorm(
            diffusion_dim,
            eps=1e-6
        )

        self.query = nn.Linear(
            vit_dim,
            vit_dim
        )

        self.key = nn.Conv2d(
            diffusion_dim,
            vit_dim,
            kernel_size=1
        )

        self.value = nn.Conv2d(
            diffusion_dim,
            diffusion_dim,
            kernel_size=3,
            padding=1
        )

        self.mlp_c0 = nn.Linear(
            diffusion_dim,
            vit_dim
        )

        self.mlp_c1 = nn.Sequential(
            nn.Linear(vit_dim, vit_dim * 4),
            nn.GELU(),
            nn.Linear(vit_dim * 4, vit_dim)
        )

        self.norm_c1 = nn.LayerNorm(
            vit_dim,
            eps=1e-6
        )

        self.mlp_p0 = nn.Conv2d(
            diffusion_dim,
            diffusion_dim,
            kernel_size=3,
            padding=1
        )

        self.norm_p1 = nn.GroupNorm(
            1,
            diffusion_dim,
            eps=1e-6
        )

        self.mlp_p1 = nn.Sequential(
            nn.Conv2d(
                diffusion_dim,
                diffusion_dim * 4,
                kernel_size=1
            ),
            nn.GELU(),
            nn.Conv2d(
                diffusion_dim * 4,
                diffusion_dim,
                kernel_size=1
            )
        )

        self.drop_path = nn.Dropout(drop_path)

    def forward(self, query_tokens, diffusion_features):

        if len(diffusion_features) != 5:
            raise ValueError(
                f"LFCA expects 5 feature maps, "
                f"received {len(diffusion_features)}"
            )

        if query_tokens.ndim != 3:
            raise ValueError(
                "query_tokens must have shape [B, N, vit_dim]"
            )

        batch_size = query_tokens.shape[0]

        if query_tokens.shape[-1] != self.vit_dim:
            raise ValueError(
                f"Expected ViT dimension {self.vit_dim}, "
                f"received {query_tokens.shape[-1]}"
            )

        if query_tokens.shape[1] <= self.num_classes:
            raise ValueError(
                "Query tokens must contain class and patch tokens"
            )

        projected_features = []

        for index, (feature, projection, expected_channels) in enumerate(
            zip(
                diffusion_features,
                self.feature_projections,
                self.expected_channels
            )
        ):

            if feature.ndim != 4:
                raise ValueError(
                    f"Feature {index} must have shape [B,C,H,W]"
                )

            if feature.shape[0] != batch_size:
                raise ValueError(
                    f"Feature {index}: batch size mismatch"
                )

            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"Feature {index}: expected "
                    f"{expected_channels} channels, "
                    f"received {feature.shape[1]}"
                )

            projected_features.append(
                projection(feature.detach())
            )

        spatial_size = projected_features[0].shape[-2:]

        for index, feature in enumerate(projected_features):

            if feature.shape[-2:] != spatial_size:
                raise ValueError(
                    f"Feature {index} has spatial size "
                    f"{feature.shape[-2:]}, expected {spatial_size}"
                )

        diffusion_features = torch.cat(
            projected_features,
            dim=1
        )

        kv = self.diff_init(diffusion_features)

        num_patches = (
            query_tokens.shape[1] - self.num_classes
        )

        patch_height = int(num_patches ** 0.5)
        patch_width = patch_height

        if patch_height * patch_width != num_patches:
            raise ValueError(
                "The number of ViT patch tokens must form a square grid"
            )

        query_input = query_tokens

        query = self.query(
            self.norm_q(query_tokens)
        )

        key = self.key(kv)
        value = self.value(kv)

        query = query.reshape(
            batch_size,
            query.shape[1],
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        key = key.flatten(2).transpose(1, 2)
        key = self.norm_k(key)

        key = key.reshape(
            batch_size,
            key.shape[1],
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        value = value.flatten(2).transpose(1, 2)
        value = self.norm_v(value)

        value = value.reshape(
            batch_size,
            value.shape[1],
            self.num_heads,
            self.value_head_dim
        ).transpose(1, 2)

        attention = torch.matmul(
            query,
            key.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        attention = F.softmax(
            attention,
            dim=-1
        )

        output = torch.matmul(
            attention,
            value
        )

        output = output.transpose(
            1, 2
        ).reshape(
            batch_size,
            query_tokens.shape[1],
            self.diffusion_dim
        )

        output_cls = output[:, :self.num_classes]
        output_patch = output[:, self.num_classes:]

        output_cls = self.mlp_c0(output_cls)

        output_cls = (
            output_cls +
            query_input[:, :self.num_classes]
        )

        output_cls = (
            self.mlp_c1(
                self.norm_c1(output_cls)
            ) + output_cls
        )

        output_patch = output_patch.transpose(
            1, 2
        ).reshape(
            batch_size,
            self.diffusion_dim,
            patch_height,
            patch_width
        )

        output_patch = self.mlp_p0(output_patch)

        output_patch = F.interpolate(
            output_patch,
            size=spatial_size,
            mode="bilinear",
            align_corners=False
        )

        output_patch = (
            kv + self.drop_path(output_patch)
        )

        output_patch = (
            self.mlp_p1(
                self.norm_p1(output_patch)
            ) + output_patch
        )

        return output_cls, output_patch


class DiffusionCAMHead(nn.Module):

    def __init__(
        self,
        diffusion_dim=512,
        num_classes=6
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                diffusion_dim,
                diffusion_dim,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                diffusion_dim,
                num_classes,
                kernel_size=1
            )
        )

    def forward(self, spatial_features):
        return self.head(spatial_features)