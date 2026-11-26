import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path

import modal

app = modal.App("diffwss-visualizations")

VOLUME_NAME = "diffwss-data"
DATA_MOUNT = "/mnt/data"
DATASET_ROOT = Path(DATA_MOUNT) / "upload_phase2"
OUTPUT_DIR = Path(DATA_MOUNT) / "diffwss_run_60_satlas"

image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "numpy",
    "Pillow",
    "tqdm",
    "einops",
    "matplotlib",
    "timm",
    "huggingface_hub",
    "satlaspretrain-models",
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    volumes={DATA_MOUNT: volume},
    timeout=24 * 60 * 60,
)
def generate(split="val", checkpoint_name="best.pt", run_dir="diffwss_run_60_satlas", num_samples=10, sample_offset=0):
    import random
    import sys

    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms
    from tqdm import tqdm

    sys.path.insert(0, str(DATASET_ROOT))

    from src.cam.fusion import extract_cams, make_pseudo_labels
    from src.cam.visualization import (
        CLASS_NAMES,
        cam_overlay,
        colorize_mask,
    )
    from src.models.ddpm_unet import Unet
    from src.models.diffusion import GaussianDiffusion
    from src.models.diffusion_features import DiffusionFeatureExtractor
    from src.models.diffwss_model import DiffWSSModel
    from src.models.satlas_swin import AerialSwinEncoder

    class PotsdamData(Dataset):
        def __init__(self, split, transform=None):
            self.image_paths = sorted(
                (DATASET_ROOT / f"data/{split}/images").glob("*.png")
            )
            labels = np.load(
                DATASET_ROOT / f"data/{split}/image_labels.npy",
                allow_pickle=True,
            ).astype(np.float32)
            names = np.load(
                DATASET_ROOT / f"data/{split}/tile_names.npy",
                allow_pickle=True,
            ).astype(str)
            self.labels_by_name = {
                str(n): labels[i] for i, n in enumerate(names)
            }
            self.names = [p.stem for p in self.image_paths]
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            image = Image.open(self.image_paths[index]).convert("RGB")
            tensor = image if self.transform is None else self.transform(image)
            label = torch.from_numpy(
                self.labels_by_name[self.names[index]].copy()
            )
            return {
                "image": tensor,
                "image_labels": label,
                "name": self.names[index],
            }

    def build_model():
        unet = Unet(dim=32, channels=3, dim_mults=(1, 2, 4, 8))
        ddpm_state = torch.load(
            DATASET_ROOT / "checkpoints/ddpm/ddpm_unet_ema_latest.pth",
            map_location="cpu",
            weights_only=True,
        )
        unet.load_state_dict(ddpm_state, strict=True)

        diffusion = GaussianDiffusion(
            model=unet,
            image_size=512,
            timesteps=1000,
            objective="pred_v",
            beta_schedule="sigmoid",
            auto_normalize=False,
        ).to(device)
        diffusion.eval()

        diffusion_timesteps = (0, 1, 2, 3, 4)
        diffusion_expected_channels = (256, 128, 64, 32, 32)

        feature_extractor = DiffusionFeatureExtractor(
            diffusion=diffusion,
            timesteps=diffusion_timesteps,
            expected_channels=diffusion_expected_channels,
            pac_timestep=100,
        ).to(device)
        feature_extractor.eval()

        diffusion_feature_channels = (
            sum(diffusion_expected_channels) * len(diffusion_timesteps)
        )

        vit = AerialSwinEncoder(
            num_classes=6,
            img_size=512,
            embed_dim=768,
            num_heads=12,
        )

        model = DiffWSSModel(
            vit=vit,
            diffusion_feature_extractor=feature_extractor,
            num_classes=6,
            vit_dim=768,
            diffusion_dim=512,
            diffusion_feature_channels=diffusion_feature_channels,
            num_heads=8,
            pac_weight=10.0,
            lfca_weight=1.0,
            lfca_start_epoch=10,
        ).to(device)
        return model

    def load_weights(model):
        checkpoint_path = output_dir / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            checkpoint_path = output_dir / "checkpoints/last.pt"
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        state = checkpoint["model_state_dict"]
        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        model.load_state_dict(state, strict=True)
        return checkpoint_path, checkpoint.get("epoch")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda")
    output_dir = Path(DATA_MOUNT) / run_dir
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    dataset = PotsdamData(split, transform)

    labels = np.load(
        DATASET_ROOT / f"data/{split}/image_labels.npy",
        allow_pickle=True,
    ).astype(np.float32)
    names = np.load(
        DATASET_ROOT / f"data/{split}/tile_names.npy",
        allow_pickle=True,
    ).astype(str)
    labels_by_name = {
        str(name): labels[index] for index, name in enumerate(names)
    }

    class_indices = {}
    for class_id in range(len(CLASS_NAMES)):
        selected = []
        for index, name in enumerate(dataset.names):
            label = labels_by_name.get(name)
            if label is not None and label[class_id] > 0.5:
                selected.append(index)
            if len(selected) >= num_samples + sample_offset:
                break
        class_indices[class_id] = selected

    model = build_model()
    checkpoint_path, checkpoint_epoch = load_weights(model)
    model.eval()

    threshold = 0.35
    history_path = output_dir / "history.json"

    if history_path.exists():
        history = json.loads(history_path.read_text())
        miou_entries = history.get("miou", [])
        if miou_entries:
            threshold = float(miou_entries[-1]["threshold"])

    out_dir = output_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    for old_path in out_dir.glob("*.png"):
        old_path.unlink()

    selected_indices = []
    for indices in class_indices.values():
        for index in indices[sample_offset:]:
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= num_samples:
                break
        if len(selected_indices) >= num_samples:
            break

    titles = [
        "Image",
        "ViT CAM",
        "Diffusion CAM",
        "Denoised image",
        "Pseudo mask",
        "GT mask",
    ]

    print("Checkpoint:", checkpoint_path.name, "epoch:", checkpoint_epoch)
    print("Split:", split)
    print("Samples:", len(selected_indices))

    with torch.inference_mode():
        for sample_index in tqdm(selected_indices, desc="Visualizing"):
            sample = dataset[sample_index]
            name = sample["name"]

            image = Image.open(
                DATASET_ROOT / f"data/{split}/images/{name}.png"
            ).convert("RGB").resize((512, 512))

            gt = Image.open(
                DATASET_ROOT / f"data/{split}/masks/{name}.png"
            ).convert("L").resize((512, 512), Image.Resampling.NEAREST)

            images = sample["image"].unsqueeze(0).to(device)
            image_labels = sample["image_labels"].unsqueeze(0).to(device)

            cams = extract_cams(model, images, image_labels)

            diffusion_cams = model.get_diffusion_cams(images)
            diffusion_flipped = model.get_diffusion_cams(
                torch.flip(images, dims=(3,))
            )
            diffusion_cams = (
                diffusion_cams
                + torch.flip(diffusion_flipped, dims=(3,))
            )
            diffusion_cams = F.interpolate(
                diffusion_cams,
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
            diffusion_cams = diffusion_cams * image_labels.view(
                1,
                -1,
                1,
                1,
            )
            diffusion_cams = model._max_norm(diffusion_cams)

            ddpm_images = model._imagenet_to_ddpm(images)
            denoised_outputs = model.diffusion_feature_extractor(
                ddpm_images,
                extract_pac_image=True,
            )
            denoised_01 = torch.clamp(
                (denoised_outputs["denoised_image"] + 1.0) * 0.5,
                0.0,
                1.0,
            )
            denoised = Image.fromarray(
                (
                    denoised_01[0]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    * 255.0
                ).astype(np.uint8)
            )

            pseudo = make_pseudo_labels(
                cams,
                image_labels,
                threshold=threshold,
            )[0]

            vit_np = cams[0].cpu().numpy()
            diffusion_np = diffusion_cams[0].cpu().numpy()

            cells = [
                image,
                cam_overlay(image, vit_np, pseudo),
                cam_overlay(image, diffusion_np, pseudo),
                denoised,
                colorize_mask(pseudo),
                colorize_mask(np.asarray(gt, dtype=np.uint8)),
            ]

            figure, axes = plt.subplots(2, 3, figsize=(12, 8))

            for axis, title, cell in zip(axes.flat, titles, cells):
                axis.imshow(cell)
                axis.set_title(title)
                axis.axis("off")

            figure.suptitle(name)
            figure.tight_layout()

            plot_path = out_dir / f"{name}.png"
            figure.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(figure)

    volume.commit()


@app.local_entrypoint()
def main(
    split: str = "val",
    checkpoint_name: str = "best.pt",
    run_dir: str = "diffwss_run_60_satlas",
    num_samples: int = 10,
    sample_offset: int = 0,
):
    generate.remote(
        split=split,
        checkpoint_name=checkpoint_name,
        run_dir=run_dir,
        num_samples=num_samples,
        sample_offset=sample_offset,
    )
