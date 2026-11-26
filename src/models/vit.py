import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape

        qkv = self.qkv(x).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)

        output = (attention @ v).transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            channels
        )

        output = self.proj(output)

        return output, attention, q, k


class MLP(nn.Module):
    def __init__(self, dim=768, hidden_dim=3072):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.mlp = MLP(
            dim,
            int(dim * mlp_ratio)
        )

    def forward(self, x):
        attention_output, attention, q, k = self.attn(
            self.norm1(x)
        )

        x = x + attention_output
        x = x + self.mlp(self.norm2(x))

        return x, attention, q, k


class ViTBase16(nn.Module):
    def __init__(
        self,
        num_classes=6,
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        pretrained=False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.base_grid_size = img_size // patch_size

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, num_classes, embed_dim)
        )

        num_patches = self.base_grid_size ** 2

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_classes + num_patches,
                embed_dim
            )
        )

        self.pos_drop = nn.Dropout(0.0)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.class_head = nn.Linear(embed_dim, 1)

        self.cam_head = nn.Conv2d(
            embed_dim,
            num_classes,
            kernel_size=3,
            padding=1
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if pretrained:
            self.load_imagenet_pretrained()

    def load_imagenet_pretrained(self):
        from torchvision.models import (
            vit_b_16,
            ViT_B_16_Weights
        )

        pretrained_model = vit_b_16(
            weights=ViT_B_16_Weights.IMAGENET1K_V1
        )

        state = pretrained_model.state_dict()

        with torch.no_grad():
            self.patch_embed.proj.weight.copy_(
                state["conv_proj.weight"]
            )

            self.patch_embed.proj.bias.copy_(
                state["conv_proj.bias"]
            )

            pretrained_cls = state["class_token"]

            self.cls_token.copy_(
                pretrained_cls.repeat(
                    1,
                    self.num_classes,
                    1
                )
            )

            source_pos = state["encoder.pos_embedding"]

            source_cls_pos = source_pos[:, :1]
            source_patch_pos = source_pos[:, 1:]

            source_grid = int(
                math.sqrt(source_patch_pos.shape[1])
            )

            source_patch_pos = source_patch_pos.reshape(
                1,
                source_grid,
                source_grid,
                self.embed_dim
            ).permute(0, 3, 1, 2)

            target_patch_pos = F.interpolate(
                source_patch_pos,
                size=(
                    self.base_grid_size,
                    self.base_grid_size
                ),
                mode="bicubic",
                align_corners=False
            )

            target_patch_pos = target_patch_pos.flatten(
                2
            ).transpose(1, 2)

            target_cls_pos = source_cls_pos.repeat(
                1,
                self.num_classes,
                1
            )

            target_pos = torch.cat(
                [
                    target_cls_pos,
                    target_patch_pos
                ],
                dim=1
            )

            self.pos_embed.copy_(target_pos)

            for i, block in enumerate(self.blocks):
                prefix = f"encoder.layers.encoder_layer_{i}."

                block.norm1.weight.copy_(
                    state[prefix + "ln_1.weight"]
                )

                block.norm1.bias.copy_(
                    state[prefix + "ln_1.bias"]
                )

                block.attn.qkv.weight.copy_(
                    state[prefix + "self_attention.in_proj_weight"]
                )

                block.attn.qkv.bias.copy_(
                    state[prefix + "self_attention.in_proj_bias"]
                )

                block.attn.proj.weight.copy_(
                    state[prefix + "self_attention.out_proj.weight"]
                )

                block.attn.proj.bias.copy_(
                    state[prefix + "self_attention.out_proj.bias"]
                )

                block.norm2.weight.copy_(
                    state[prefix + "ln_2.weight"]
                )

                block.norm2.bias.copy_(
                    state[prefix + "ln_2.bias"]
                )

                block.mlp.fc1.weight.copy_(
                    state[prefix + "mlp.0.weight"]
                )

                block.mlp.fc1.bias.copy_(
                    state[prefix + "mlp.0.bias"]
                )

                block.mlp.fc2.weight.copy_(
                    state[prefix + "mlp.3.weight"]
                )

                block.mlp.fc2.bias.copy_(
                    state[prefix + "mlp.3.bias"]
                )

            self.norm.weight.copy_(
                state["encoder.ln.weight"]
            )

            self.norm.bias.copy_(
                state["encoder.ln.bias"]
            )

        del pretrained_model
        del state

    def interpolate_pos_encoding(self, height, width):
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        cls_pos = self.pos_embed[:, :self.num_classes]

        patch_pos = self.pos_embed[:, self.num_classes:]

        patch_pos = patch_pos.reshape(
            1,
            self.base_grid_size,
            self.base_grid_size,
            self.embed_dim
        ).permute(0, 3, 1, 2)

        if (
            grid_h != self.base_grid_size
            or grid_w != self.base_grid_size
        ):
            patch_pos = F.interpolate(
                patch_pos,
                size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False
            )

        patch_pos = patch_pos.flatten(2).transpose(1, 2)

        return torch.cat(
            [cls_pos, patch_pos],
            dim=1
        )

    def forward_features(self, images):
        batch_size, _, height, width = images.shape

        x = self.patch_embed(images)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        x = x + self.interpolate_pos_encoding(
            height,
            width
        )

        x = self.pos_drop(x)

        attention_weights = []
        q_list = []
        k_list = []

        for block in self.blocks:
            x, attention, q, k = block(x)

            attention_weights.append(attention)
            q_list.append(q)
            k_list.append(k)

        x_cls = x[:, :self.num_classes]
        x_patch = x[:, self.num_classes:]

        return (
            x_cls,
            x_patch,
            attention_weights,
            q_list,
            k_list
        )

    def forward(self, images, return_att=False):
        batch_size, _, height, width = images.shape

        if height % self.patch_size != 0:
            raise ValueError("Image height must be divisible by patch_size")

        if width % self.patch_size != 0:
            raise ValueError("Image width must be divisible by patch_size")

        (
            x_cls,
            x_patch,
            attention_weights,
            q_list,
            k_list
        ) = self.forward_features(images)

        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        patch_features = x_patch.transpose(
            1,
            2
        ).reshape(
            batch_size,
            self.embed_dim,
            grid_h,
            grid_w
        )

        cls_logits = x_cls.mean(-1)

        cam_logits = self.cam_head(patch_features)

        flat_cams = cam_logits.flatten(2)

        top_k = min(3, flat_cams.shape[-1])

        positive_scores = torch.topk(
            F.relu(flat_cams),
            k=top_k,
            dim=-1
        )[0].mean(dim=-1)

        negative_scores = torch.topk(
            F.relu(-flat_cams),
            k=top_k,
            dim=-1
        )[0].mean(dim=-1)

        patch_logits = positive_scores - negative_scores

        attention_weights = torch.stack(
            attention_weights,
            dim=0
        ).mean(dim=2)

        class_attention = attention_weights[
            -len(self.blocks):
        ].sum(dim=0)[
            :,
            :self.num_classes,
            self.num_classes:
        ]

        mtatt = class_attention.reshape(
            batch_size,
            self.num_classes,
            grid_h,
            grid_w
        )

        patch_attention = torch.sum(
            attention_weights[
                :,
                :,
                self.num_classes:,
                self.num_classes:
            ],
            dim=0
        )

        cams = F.relu(cam_logits) * mtatt

        refined_cams = torch.bmm(
            patch_attention,
            cams.flatten(2).transpose(1, 2)
        ).transpose(1, 2).reshape(
            batch_size,
            self.num_classes,
            grid_h,
            grid_w
        )

        outputs = {
            "cls": cls_logits,
            "pcls": patch_logits,
            "cams": F.relu(cam_logits),
            "fcams": cams,
            "attn": patch_attention,
            "mtatt": mtatt,
            "rcams": refined_cams,
            "feat": patch_features,
            "q": q_list,
            "k": k_list
        }

        if return_att:
            return refined_cams

        return outputs