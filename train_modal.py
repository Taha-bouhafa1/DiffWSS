import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path

import modal

app = modal.App("diffwss-train")

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
def train(resume=False):
    import sys
    import math

    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    sys.path.insert(0, str(DATASET_ROOT))

    import src.training.train_vit_lfca_pac as trainer
    from src.cam.miou import collect_cam_dicts, evaluate_cam_dicts
    from src.models.ddpm_unet import Unet
    from src.models.diffusion import GaussianDiffusion
    from src.models.diffusion_features import DiffusionFeatureExtractor
    from src.models.diffwss_model import DiffWSSModel
    from src.models.satlas_swin import AerialSwinEncoder
    from src.training.train_vit_lfca_pac import set_seed

    if not DATASET_ROOT.exists():
        raise RuntimeError(f"Missing dataset root: {DATASET_ROOT}")

    class PotsdamWSSLDataset(Dataset):
        def __init__(
            self,
            image_dir,
            labels_path,
            names_path,
            transform=None,
            transform2=None,
        ):
            self.image_paths = sorted(Path(image_dir).glob("*.png"))
            labels = np.load(labels_path, allow_pickle=True).astype(np.float32)
            names = np.load(names_path, allow_pickle=True).astype(str)
            self.labels_by_name = {
                str(n): labels[i] for i, n in enumerate(names)
            }
            self.names = [p.stem for p in self.image_paths]
            self.transform = transform
            self.transform2 = transform2

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
            image_diff = Image.open(self.image_paths[index]).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            if self.transform2 is not None:
                image_diff = self.transform2(image_diff)

            label = torch.from_numpy(
                self.labels_by_name[self.names[index]].copy()
            )
            return {
                "image": image,
                "image_diff": image_diff,
                "image_labels": label,
                "name": self.names[index],
            }

    original_save_checkpoint = trainer.save_checkpoint
    original_save_history = trainer.save_history

    def save_checkpoint_and_commit(*args, **kwargs):
        original_save_checkpoint(*args, **kwargs)
        volume.commit()

    def save_history_and_commit(*args, **kwargs):
        original_save_history(*args, **kwargs)
        volume.commit()

    trainer.save_checkpoint = save_checkpoint_and_commit
    trainer.save_history = save_history_and_commit

    set_seed(0)

    device = torch.device("cuda")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            (512, 512),
            scale=(0.3, 1.2),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_transform2 = transforms.Compose([
        transforms.RandomResizedCrop(
            (512, 512),
            scale=(0.3, 1.2),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    unet = Unet(dim=32, channels=3, dim_mults=(1, 2, 4, 8))

    checkpoint_state = torch.load(
        DATASET_ROOT / "checkpoints/ddpm/ddpm_unet_ema_latest.pth",
        map_location="cpu",
        weights_only=True,
    )
    unet.load_state_dict(checkpoint_state, strict=True)

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

    train_dataset = PotsdamWSSLDataset(
        DATASET_ROOT / "data/train/images",
        DATASET_ROOT / "data/train/image_labels.npy",
        DATASET_ROOT / "data/train/tile_names.npy",
        train_transform,
        train_transform2,
    )

    val_dataset = PotsdamWSSLDataset(
        DATASET_ROOT / "data/val/images",
        DATASET_ROOT / "data/val/image_labels.npy",
        DATASET_ROOT / "data/val/tile_names.npy",
        val_transform,
        val_transform,
    )

    batch_size = 4
    workers = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    val_mask_dir = DATASET_ROOT / "data/val/masks"

    def miou_evaluator(model, epoch):
        cam_items = collect_cam_dicts(model, val_loader, device)
        return evaluate_cam_dicts(cam_items, val_mask_dir, num_classes=6)

    epochs = 60
    save_every = 5
    base_lr = (5e-4 * 32) / 512.0
    warmup_lr = 1e-6
    min_lr = 1e-5
    warmup_epochs = 5

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=0.05,
    )

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            warmup_value = (
                warmup_lr
                + (base_lr - warmup_lr) * epoch / warmup_epochs
            )
            return warmup_value / base_lr

        progress = (
            (epoch - warmup_epochs)
            / max(1, epochs - warmup_epochs)
        )
        cosine_value = (
            min_lr
            + 0.5
            * (base_lr - min_lr)
            * (1.0 + math.cos(math.pi * progress))
        )
        return cosine_value / base_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    scaler = torch.amp.GradScaler("cuda")

    torch.backends.cudnn.benchmark = True

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": base_lr,
        "warmup_lr": warmup_lr,
        "min_lr": min_lr,
        "warmup_epochs": warmup_epochs,
        "weight_decay": 0.05,
        "timesteps": list(feature_extractor.timesteps),
        "pac_timestep": feature_extractor.pac_timestep,
        "lfca_start_epoch": model.lfca_start_epoch,
        "W": [1.0, 1.0, 1.0, 1.0, 10.0],
        "T": model.classification_loss.temperature,
        "A": model.classification_loss.alpha,
        "save_every": save_every,
        "gpu": "L40S",
    }

    start_epoch = 1
    best_val_loss = float("inf")
    best_miou = 0.0
    history = None

    checkpoint_path = OUTPUT_DIR / "checkpoints/last.pt"

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state = checkpoint["model_state_dict"]
        state = {
            key.replace("module.", "", 1): value
            for key, value in state.items()
        }

        model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if (
            scaler is not None
            and checkpoint.get("scaler_state_dict") is not None
        ):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        best_miou = float(checkpoint.get("best_miou", 0.0))
        history = checkpoint["history"]

    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset batches:", len(train_loader), len(val_loader))
    print("Resuming from epoch:", start_epoch)

    trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        output_dir=OUTPUT_DIR,
        scheduler=scheduler,
        scaler=None,
        amp=False,
        grad_clip_norm=None,
        patience=None,
        save_every=save_every,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        history=history,
        config=config,
        miou_evaluator=miou_evaluator,
        miou_start_epoch=10,
        best_miou=best_miou,
    )

    volume.commit()

    print("Training finished.")
    print("Output:", OUTPUT_DIR)
    print(
        "Checkpoints:",
        sorted(p.name for p in (OUTPUT_DIR / "checkpoints").glob("*.pt")),
    )


@app.local_entrypoint()
def main(resume: bool = False):
    print("Starting training on Modal...")
    train.spawn(resume=resume)
    print("Training spawned. You can shut down this PC; monitor the app in Modal.")
