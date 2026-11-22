"""Tests for the DiffWSS training engine. Runnable directly or with pytest."""
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.training.train_vit_lfca_pac import (
    train_one_epoch, validate_one_epoch, save_checkpoint, load_checkpoint
)


class _ToyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 16, 16),
            "image_labels": torch.randint(0, 2, (6,)).float()
        }


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(3, 6)

    def forward(self, images, labels):
        logits = self.head(self.pool(images).flatten(1))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return {"total_loss": loss, "loss_vit": loss, "loss_pac": loss * 0.0}


def test_train_and_validation_epoch():
    device = torch.device("cpu")
    model = _ToyModel()
    loader = DataLoader(_ToyDataset(), batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train = train_one_epoch(model, loader, optimizer, device, 1, amp=False)
    valid = validate_one_epoch(model, loader, device, 1, amp=False)

    assert "total_loss" in train
    assert "total_loss" in valid
    assert torch.isfinite(torch.tensor(train["total_loss"]))
    assert torch.isfinite(torch.tensor(valid["total_loss"]))


def test_checkpoint_save_and_load():
    model = _ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "checkpoint.pt"
        save_checkpoint(path, 1, model, optimizer, None, None, 0.5,
                        {"train": [], "validation": []})

        restored = _ToyModel()
        state = load_checkpoint(path, restored)
        assert state["epoch"] == 1
        for key, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[key])


if __name__ == "__main__":
    test_train_and_validation_epoch()
    test_checkpoint_save_and_load()
    print("Training pipeline: PASSED")
