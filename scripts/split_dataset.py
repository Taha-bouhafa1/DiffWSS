from pathlib import Path
import csv
import random

import numpy as np
import rasterio


# Paths
MASK_DIR = Path("data/raw/5_Labels_all")
OUTPUT_FILE = Path("data/processed/scene_split.csv")


# Official Potsdam RGB colour palette
CLASSES = {
    "impervious": (255, 255, 255),
    "building": (0, 0, 255),
    "low_vegetation": (0, 255, 255),
    "tree": (0, 255, 0),
    "car": (255, 255, 0),
    "clutter": (255, 0, 0),
}


# Split configuration
SEED = 42

SPLIT_SIZES = {
    "train": 26,
    "val": 6,
    "test": 6,
}

MAX_COLOR_DISTANCE = 100


def get_class_presence(mask_path):
    """
    Detect the presence of each semantic class
    using nearest-colour RGB matching.
    """

    colors = np.array(
        list(CLASSES.values()),
        dtype=np.float32
    )

    class_names = list(CLASSES.keys())

    presence = {
        name: 0 for name in class_names
    }

    with rasterio.open(mask_path) as src:

        if src.count < 3:
            raise ValueError(
                f"{mask_path.name} does not have 3 RGB bands."
            )

        for row in range(0, src.height, 256):

            height = min(256, src.height - row)

            block = src.read(
                [1, 2, 3],
                window=(
                    (row, row + height),
                    (0, src.width)
                )
            )

            pixels = block.transpose(
                1, 2, 0
            ).reshape(-1, 3).astype(np.float32)

            distances = np.linalg.norm(
                pixels[:, None, :] - colors[None, :, :],
                axis=2
            )

            nearest = distances.argmin(axis=1)
            minimum = distances.min(axis=1)

            if minimum.max() > MAX_COLOR_DISTANCE:
                raise ValueError(
                    f"Unexpected colours in {mask_path.name}. "
                    f"Maximum distance: {minimum.max():.2f}"
                )

            for index, name in enumerate(class_names):

                if np.any(nearest == index):
                    presence[name] = 1

    return presence


def main():

    random.seed(SEED)

    masks = sorted(
        MASK_DIR.glob("*_label.tif")
    )

    expected_total = sum(SPLIT_SIZES.values())

    if len(masks) != expected_total:
        raise ValueError(
            f"Expected {expected_total} masks, "
            f"but found {len(masks)}."
        )

    scenes = []

    print("Analysing class presence in each scene...\n")

    for mask_path in masks:

        scene_name = mask_path.name.replace(
            "_label.tif", ""
        )

        presence = get_class_presence(mask_path)

        scenes.append({
            "scene": scene_name,
            "presence": presence
        })

        print(f"Processed: {scene_name}")

    # Calculate how frequently each class appears
    class_frequency = {
        class_name: sum(
            scene["presence"][class_name]
            for scene in scenes
        )
        for class_name in CLASSES
    }

    # Prioritise scenes containing rare classes
    def rarity_score(scene):

        return sum(
            scene["presence"][name]
            / max(class_frequency[name], 1)
            for name in CLASSES
        )

    random.shuffle(scenes)

    scenes.sort(
        key=rarity_score,
        reverse=True
    )

    split_names = list(SPLIT_SIZES.keys())

    assigned = {
        name: [] for name in split_names
    }

    total_sizes = sum(SPLIT_SIZES.values())

    targets = {
        split: {
            class_name: (
                class_frequency[class_name]
                * SPLIT_SIZES[split]
                / total_sizes
            )
            for class_name in CLASSES
        }
        for split in split_names
    }

    counts = {
        split: {
            class_name: 0
            for class_name in CLASSES
        }
        for split in split_names
    }

    # Greedy multilabel scene assignment
    for scene in scenes:

        available = [
            split
            for split in split_names
            if len(assigned[split]) < SPLIT_SIZES[split]
        ]

        best_split = max(
            available,
            key=lambda split: (
                sum(
                    scene["presence"][name]
                    * (
                        targets[split][name]
                        - counts[split][name]
                    )
                    / max(class_frequency[name], 1)
                    for name in CLASSES
                )
                +
                (
                    SPLIT_SIZES[split]
                    - len(assigned[split])
                )
                / SPLIT_SIZES[split] * 0.01
            )
        )

        assigned[best_split].append(scene)

        for name in CLASSES:

            counts[best_split][name] += (
                scene["presence"][name]
            )

    # Save CSV
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scene",
                "split",
                *CLASSES.keys()
            ]
        )

        writer.writeheader()

        for split, split_scenes in assigned.items():

            for scene in split_scenes:

                writer.writerow({
                    "scene": scene["scene"],
                    "split": split,
                    **scene["presence"]
                })

    # Display results
    print("\n" + "=" * 45)
    print("DATASET SPLIT COMPLETED")
    print("=" * 45)

    for split, split_scenes in assigned.items():

        print(
            f"\n{split.upper()}: "
            f"{len(split_scenes)} scenes"
        )

        for class_name in CLASSES:

            print(
                f"  {class_name}: "
                f"{counts[split][class_name]} scenes"
            )

    print("\nClass frequency across all scenes:")

    for name, frequency in class_frequency.items():
        print(f"  {name}: {frequency}/38")

    print(f"\nCSV saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()