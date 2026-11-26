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


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):

        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class LFCA(nn.Module):

    def __init__(
        self,
        vit_dim=768,
        diffusion_dim=512,
        num_classes=6,
        num_heads=8,
        drop_path=0.1,
        diffusion_feature_channels=None
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

        self.feature_channels = (
            diffusion_feature_channels
            if diffusion_feature_channels is not None
            else diffusion_dim * 5
        )

        self.diff_init = ResidualBlock(
            self.feature_channels,
            diffusion_dim,
            groups=1
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

        self.drop_path = DropPath(drop_path)

        self.init_weight()
        self._reset_parameters()

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(
                module,
                (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm),
            ):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.constant_(self.key.bias, 0)
        nn.init.constant_(self.value.bias, 0)
        nn.init.trunc_normal_(self.mlp_p1[0].weight, std=0.02)
        nn.init.trunc_normal_(self.mlp_p1[2].weight, std=0.02)

    def forward(self, query_tokens, diffusion_features):

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

        if diffusion_features.ndim != 4:
            raise ValueError(
                "diffusion_features must have shape [B, C, H, W]"
            )

        if diffusion_features.shape[0] != batch_size:
            raise ValueError(
                "diffusion_features batch size mismatch"
            )

        if diffusion_features.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} diffusion channels, "
                f"received {diffusion_features.shape[1]}"
            )

        spatial_size = diffusion_features.shape[-2:]

        kv = self.diff_init(diffusion_features.detach())

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

        query = self.norm_k(
            self.query(query_tokens)
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
            self.drop_path(
                self.mlp_p1(
                    self.norm_p1(output_patch)
                )
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