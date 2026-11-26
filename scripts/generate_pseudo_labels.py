import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path

import modal

app = modal.App("diffwss-pseudo-labels")

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
def generate(split="train", batch_size=4, checkpoint_name="best.pt"):
    import sys

    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from tqdm import tqdm

    sys.path.insert(0, str(DATASET_ROOT))

    from src.cam.fusion import extract_cams
    from src.models.ddpm_unet import Unet
    from src.models.diffusion import GaussianDiffusion
    from src.models.diffusion_features import DiffusionFeatureExtractor
    from src.models.diffwss_model import DiffWSSModel
    from src.models.satlas_swin import AerialSwinEncoder

    class PotsdamWSSLDataset(Dataset):
        def __init__(self, image_dir, labels_path, names_path, transform=None):
            self.image_paths = sorted(Path(image_dir).glob("*.png"))
            labels = np.load(labels_path, allow_pickle=True).astype(np.float32)
            names = np.load(names_path, allow_pickle=True).astype(str)
            self.labels_by_name = {
                str(n): labels[i] for i, n in enumerate(names)
            }
            self.names = [p.stem for p in self.image_paths]
            self.transform = transform

            missing = [
                n for n in self.names if n not in self.labels_by_name
            ]
            if missing:
                raise RuntimeError(
                    f"Missing labels for {len(missing)} tiles"
                )

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            image = Image.open(self.image_paths[index]).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            label = torch.from_numpy(
                self.labels_by_name[self.names[index]].copy()
            )
            return {
                "image": image,
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
        checkpoint_path = OUTPUT_DIR / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            checkpoint_path = OUTPUT_DIR / "checkpoints/last.pt"
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

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    dataset = PotsdamWSSLDataset(
        DATASET_ROOT / f"data/{split}/images",
        DATASET_ROOT / f"data/{split}/image_labels.npy",
        DATASET_ROOT / f"data/{split}/tile_names.npy",
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = build_model()
    checkpoint_path, checkpoint_epoch = load_weights(model)
    model.eval()

    out_dir = OUTPUT_DIR / "pseudo_labels" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Checkpoint:", checkpoint_path.name, "epoch:", checkpoint_epoch)
    print("Split:", split, "images:", len(dataset))
    print("Output:", out_dir)

    count = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Pseudo {split}"):
            images = batch["image"].to(device, non_blocking=True)
            image_labels = batch["image_labels"].to(device, non_blocking=True)

            cams = extract_cams(
                model,
                images,
                image_labels,
                size=(images.shape[-2], images.shape[-1]),
            )
            cams = cams.detach().cpu().numpy()

            for name, labels, cam in zip(
                batch["name"],
                image_labels.detach().cpu().numpy(),
                cams,
            ):
                cam_dict = {
                    index: cam[index]
                    for index in range(cam.shape[0])
                    if labels[index] > 1e-5
                }
                np.save(out_dir / f"{name}.npy", cam_dict)
                count += 1

    volume.commit()
    print("Saved pseudo-labels:", count)


@app.local_entrypoint()
def main(
    split: str = "train",
    batch_size: int = 4,
    checkpoint_name: str = "best.pt",
):
    generate.spawn(
        split=split,
        batch_size=batch_size,
        checkpoint_name=checkpoint_name,
    )
    print(f"Pseudo-label generation spawned with {checkpoint_name}.")
