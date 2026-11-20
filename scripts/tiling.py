from pathlib import Path
import csv
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

RGB_DIR = ROOT / "data/raw/2_Ortho_RGB"
MASK_DIR = ROOT / "data/raw/5_Labels_all"
SPLIT_FILE = ROOT / "data/processed/scene_split.csv"
OUTPUT_DIR = ROOT / "data/processed"

TILE_SIZE = 512
IGNORE_INDEX = 255
MAX_COLOR_DISTANCE = 100
CHUNK_SIZE = 256

CLASS_NAMES = [
    "impervious",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter",
]

PALETTE = np.array([
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0],
], dtype=np.int32)


def find_rgb(scene):
    candidates = [
        RGB_DIR / f"{scene}_RGB.tif",
        RGB_DIR / f"{scene}_RGB.tiff",
        RGB_DIR / f"{scene}.tif",
        RGB_DIR / f"{scene}.tiff",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"RGB image not found: {scene}")


def find_mask(scene):
    candidates = [
        MASK_DIR / f"{scene}_label.tif",
        MASK_DIR / f"{scene}_label.tiff",
        MASK_DIR / f"{scene}.tif",
        MASK_DIR / f"{scene}.tiff",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"Label mask not found: {scene}")


def convert_mask(rgb_mask):
    height, width, _ = rgb_mask.shape

    result = np.full(
        (height, width),
        IGNORE_INDEX,
        dtype=np.uint8
    )

    threshold = MAX_COLOR_DISTANCE ** 2

    for start in range(0, height, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, height)

        pixels = rgb_mask[start:end].astype(np.int32)

        differences = (
            pixels[:, :, None, :]
            - PALETTE[None, None, :, :]
        )

        distances = np.sum(
            differences * differences,
            axis=3
        )

        nearest_classes = np.argmin(distances, axis=2)
        nearest_distances = np.min(distances, axis=2)

        valid = nearest_distances <= threshold

        chunk_result = result[start:end]
        chunk_result[valid] = nearest_classes[valid]

    return result


def pad_image(image):
    height, width = image.shape[:2]

    new_height = (
        (height + TILE_SIZE - 1) // TILE_SIZE
    ) * TILE_SIZE

    new_width = (
        (width + TILE_SIZE - 1) // TILE_SIZE
    ) * TILE_SIZE

    padded = np.pad(
        image,
        (
            (0, new_height - height),
            (0, new_width - width),
            (0, 0)
        ),
        mode="edge"
    )

    return padded, height, width


def pad_mask(mask, new_height, new_width):
    height, width = mask.shape

    padded = np.full(
        (new_height, new_width),
        IGNORE_INDEX,
        dtype=np.uint8
    )

    padded[:height, :width] = mask

    return padded


def get_image_labels(mask_tile, valid_height, valid_width):
    valid_pixels = mask_tile[:valid_height, :valid_width]

    return [
        int(np.any(valid_pixels == class_id))
        for class_id in range(len(CLASS_NAMES))
    ]


def main():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Scene split file not found: {SPLIT_FILE}"
        )

    with open(SPLIT_FILE, "r", newline="") as file:
        scenes = list(csv.DictReader(file))

    if not scenes:
        raise ValueError("No scenes found in scene_split.csv")

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    for split in split_counts:
        (OUTPUT_DIR / split / "images").mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_DIR / split / "masks").mkdir(
            parents=True,
            exist_ok=True
        )

    metadata_path = OUTPUT_DIR / "tile_metadata.csv"
    total_tiles = 0

    with open(metadata_path, "w", newline="") as metadata_file:
        writer = csv.writer(metadata_file)

        writer.writerow([
            "tile",
            "scene",
            "split",
            "row",
            "col",
            "valid_height",
            "valid_width",
            *CLASS_NAMES
        ])

        for scene_info in tqdm(scenes, desc="Processing scenes"):
            scene = scene_info["scene"]
            split = scene_info["split"]

            if split not in split_counts:
                raise ValueError(f"Unknown split: {split}")

            rgb_path = find_rgb(scene)
            mask_path = find_mask(scene)

            print(f"\nProcessing {scene} [{split}]")

            with rasterio.open(rgb_path) as src:
                if src.count < 3:
                    raise ValueError(
                        f"RGB image has fewer than 3 bands: {rgb_path}"
                    )

                rgb = np.moveaxis(
                    src.read([1, 2, 3]),
                    0,
                    -1
                )

            with rasterio.open(mask_path) as src:
                if src.count < 3:
                    raise ValueError(
                        f"Mask has fewer than 3 bands: {mask_path}"
                    )

                mask_rgb = np.moveaxis(
                    src.read([1, 2, 3]),
                    0,
                    -1
                )

            if rgb.shape[:2] != mask_rgb.shape[:2]:
                raise ValueError(
                    f"Image/mask dimensions differ for {scene}: "
                    f"{rgb.shape[:2]} vs {mask_rgb.shape[:2]}"
                )

            mask = convert_mask(mask_rgb)

            rgb_padded, original_h, original_w = pad_image(rgb)

            padded_h, padded_w = rgb_padded.shape[:2]

            mask_padded = pad_mask(
                mask,
                padded_h,
                padded_w
            )

            rows = padded_h // TILE_SIZE
            cols = padded_w // TILE_SIZE

            for row in range(rows):
                for col in range(cols):
                    y1 = row * TILE_SIZE
                    x1 = col * TILE_SIZE

                    y2 = y1 + TILE_SIZE
                    x2 = x1 + TILE_SIZE

                    image_tile = rgb_padded[y1:y2, x1:x2]
                    mask_tile = mask_padded[y1:y2, x1:x2]

                    valid_height = max(
                        0,
                        min(TILE_SIZE, original_h - y1)
                    )

                    valid_width = max(
                        0,
                        min(TILE_SIZE, original_w - x1)
                    )

                    tile_name = (
                        f"{scene}_r{row:02d}_c{col:02d}"
                    )

                    image_output = (
                        OUTPUT_DIR
                        / split
                        / "images"
                        / f"{tile_name}.png"
                    )

                    mask_output = (
                        OUTPUT_DIR
                        / split
                        / "masks"
                        / f"{tile_name}.png"
                    )

                    Image.fromarray(
                        image_tile.astype(np.uint8)
                    ).save(image_output)

                    Image.fromarray(
                        mask_tile
                    ).save(mask_output)

                    labels = get_image_labels(
                        mask_tile,
                        valid_height,
                        valid_width
                    )

                    writer.writerow([
                        tile_name,
                        scene,
                        split,
                        row,
                        col,
                        valid_height,
                        valid_width,
                        *labels
                    ])

                    total_tiles += 1
                    split_counts[split] += 1

    print("\nTILING COMPLETED")
    print(f"Scenes: {len(scenes)}")
    print(f"Tile size: {TILE_SIZE}x{TILE_SIZE}")
    print(f"Train tiles: {split_counts['train']}")
    print(f"Validation tiles: {split_counts['val']}")
    print(f"Test tiles: {split_counts['test']}")
    print(f"Total tiles: {total_tiles}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()