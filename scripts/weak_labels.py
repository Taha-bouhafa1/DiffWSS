from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

METADATA = ROOT / "data/processed/tile_metadata.csv"

CLASSES = [
    "impervious",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter"
]

def main():
    df = pd.read_csv(METADATA)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].copy()

        labels = split_df[CLASSES].to_numpy(dtype=np.uint8)
        names = split_df["tile"].to_numpy(dtype=str)

        output_dir = ROOT / "data/processed" / split

        np.save(output_dir / "image_labels.npy", labels)
        np.save(output_dir / "tile_names.npy", names)

        print(f"\n{split.upper()}")
        print("Labels shape:", labels.shape)
        print("Names shape:", names.shape)

    print("\nWeak labels successfully extracted.")

if __name__ == "__main__":
    main()
