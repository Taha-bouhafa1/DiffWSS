import json
import sys
from pathlib import Path

LOSS_PLOTS = (
    ("total_loss", "total_loss.png", "Total loss"),
    ("loss_vit", "loss_vit.png", "ViT loss"),
    ("loss_diffusion", "loss_diffusion.png", "Diffusion loss"),
    ("loss_denoised", "loss_denoised.png", "Denoised loss"),
    ("loss_lfca", "loss_lfca.png", "LFCA loss"),
    ("loss_pac", "loss_pac.png", "PAC loss"),
)


def plot_loss_curves(history_path, output_dir):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    history = json.loads(Path(history_path).read_text())
    train = history.get("train", [])
    validation = history.get("validation", [])

    if not train and not validation:
        raise ValueError("History is empty.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    for key, filename, title in LOSS_PLOTS:
        train_records = [record for record in train if key in record]
        val_records = [record for record in validation if key in record]

        if not train_records and not val_records:
            continue

        figure, ax = plt.subplots(figsize=(8, 5))

        if train_records:
            ax.plot(
                [record["epoch"] for record in train_records],
                [record[key] for record in train_records],
                marker="o",
                label="Train",
            )

        if val_records:
            ax.plot(
                [record["epoch"] for record in val_records],
                [record[key] for record in val_records],
                marker="s",
                label="Validation",
            )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        figure.tight_layout()

        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

    return paths


def plot_miou(history_path, output_dir):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    history = json.loads(Path(history_path).read_text())
    entries = history.get("miou", [])

    if not entries:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [record["epoch"] for record in entries],
        [record["miou"] for record in entries],
        marker="o",
        color="tab:green",
        label="mIoU",
    )
    ax.set_title("CAM mIoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figure.tight_layout()

    path = output_dir / "miou.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


if __name__ == "__main__":
    history_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "history.json"
    )
    output_dir = Path(
        sys.argv[2] if len(sys.argv) > 2 else history_path.parent / "loss_plots"
    )

    paths = plot_loss_curves(history_path, output_dir)
    miou_path = plot_miou(history_path, output_dir)
    if miou_path is not None:
        paths.append(miou_path)

    for path in paths:
        print(path)
