import torch
import torch.nn.functional as F
from torch import nn


class AerialSwinEncoder(nn.Module):
    def __init__(
        self,
        num_classes=6,
        embed_dim=768,
        num_heads=12,
        img_size=512,
    ):
        super().__init__()

        import satlaspretrain_models

        self.backbone = satlaspretrain_models.Weights().get_pretrained_model(
            "Aerial_SwinB_SI"
        )

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.backbone_dim = 512
        self.grid_size = img_size // 16
        self.num_tokens = 6 + self.grid_size * self.grid_size

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        self.cls_queries = nn.Parameter(
            torch.zeros(1, 6, self.backbone_dim)
        )
        nn.init.trunc_normal_(self.cls_queries, std=0.02)

        self.patch_norm = nn.LayerNorm(self.backbone_dim)
        self.cls_norm = nn.LayerNorm(self.backbone_dim)
        self.cls_attn = nn.MultiheadAttention(
            self.backbone_dim,
            num_heads=8,
            batch_first=True,
        )
        self.cls_head = nn.Linear(self.backbone_dim, num_classes)
        self.cam_head = nn.Conv2d(
            self.backbone_dim,
            num_classes,
            kernel_size=3,
            padding=1,
        )
        self.q_proj = nn.Linear(self.backbone_dim, embed_dim)
        self.q_proj_prev = nn.Linear(self.backbone_dim, embed_dim)

    def _project_queries(self, tokens, projection):
        batch_size = tokens.shape[0]
        q = projection(tokens)
        q = q.reshape(
            batch_size,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        return q

    def forward(self, images):
        images = (images * self.std + self.mean).clamp(0.0, 1.0)

        features = self.backbone(images)
        feat = features[2]

        batch_size, channels, height, width = feat.shape
        patch_tokens = feat.flatten(2).transpose(1, 2)
        patch_tokens = self.patch_norm(patch_tokens)

        pooled = patch_tokens.mean(dim=1)
        queries = self.cls_queries.expand(batch_size, -1, -1) + pooled.unsqueeze(1)
        queries = self.cls_norm(queries)
        attended, _ = self.cls_attn(queries, patch_tokens, patch_tokens)
        cls_tokens = self.cls_norm(queries + attended)

        cls = self.cls_head(cls_tokens.mean(dim=1))

        cam_logits = self.cam_head(feat)
        flat_cams = cam_logits.reshape(batch_size, self.num_classes, -1)
        positive = F.relu(flat_cams).topk(3, dim=-1).values.mean(dim=-1)
        negative = F.relu(-flat_cams).topk(3, dim=-1).values.mean(dim=-1)
        pcls = positive - negative

        cams = F.relu(cam_logits)

        scale = channels ** -0.5
        attention_scores = torch.bmm(
            patch_tokens,
            patch_tokens.transpose(1, 2),
        ) * scale
        patch_attention = F.softmax(attention_scores, dim=-1)
        rcams = torch.bmm(
            patch_attention,
            cams.reshape(batch_size, self.num_classes, -1).transpose(1, 2),
        ).reshape(batch_size, self.num_classes, height, width)

        final_tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        previous_tokens = torch.cat([queries, patch_tokens], dim=1)

        q_last = self._project_queries(final_tokens, self.q_proj)
        q_prev = self._project_queries(previous_tokens, self.q_proj_prev)

        return {
            "q": [q_prev, q_last],
            "cls": cls,
            "pcls": pcls,
            "cams": cams,
            "rcams": rcams,
        }
