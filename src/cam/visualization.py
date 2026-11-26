import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = [
    "impervious",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter",
]

PALETTE = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 128, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ],
    dtype=np.uint8,
)

TITLES = [
    "Image",
    "ViT CAM",
    "Diffusion CAM",
    "Pseudo mask",
    "GT mask",
]

CELL_SIZE = 256
HEADER_HEIGHT = 36
TITLE_HEIGHT = 24
LEGEND_HEIGHT = 32


def colorize_mask(mask):
    mask = np.asarray(mask)
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask < len(PALETTE)
    colored[valid] = PALETTE[mask[valid]]
    return Image.fromarray(colored, mode="RGB")


def jet(heat):
    heat = np.clip(heat, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * heat - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * heat - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * heat - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def cam_overlay(image, cams, pseudo):
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    selected = np.take_along_axis(
        cams,
        pseudo[None, :, :].astype(np.int64),
        axis=0,
    )[0]
    selected = selected / (float(selected.max()) + 1e-6)
    heat = jet(selected)
    blended = 0.55 * image_array + 0.45 * heat
    return Image.fromarray(
        np.clip(blended * 255.0, 0, 255).astype(np.uint8),
        mode="RGB",
    )


def make_image_row(cells, cell_size=CELL_SIZE):
    row = Image.new(
        "RGB",
        (len(cells) * cell_size, cell_size),
        "white",
    )

    for index, cell in enumerate(cells):
        resized = cell.resize((cell_size, cell_size))
        row.paste(resized, (index * cell_size, 0))

    return row


def make_class_grid(class_id, num_samples):
    columns = len(TITLES)

    grid = Image.new(
        "RGB",
        (
            columns * CELL_SIZE,
            HEADER_HEIGHT
            + TITLE_HEIGHT
            + num_samples * (CELL_SIZE + TITLE_HEIGHT)
            + LEGEND_HEIGHT,
        ),
        "white",
    )
    draw = ImageDraw.Draw(grid)

    color = tuple(int(channel) for channel in PALETTE[class_id])
    draw.rectangle([4, 8, 18, 22], fill=color, outline="black")
    draw.text(
        (24, 6),
        f"Class {class_id}: {CLASS_NAMES[class_id]}",
        fill="black",
    )

    for column, title in enumerate(TITLES):
        draw.text(
            (column * CELL_SIZE + 4, HEADER_HEIGHT + 4),
            title,
            fill="black",
        )

    legend_y = (
        HEADER_HEIGHT
        + TITLE_HEIGHT
        + num_samples * (CELL_SIZE + TITLE_HEIGHT)
    )
    x = 4

    for name, palette_color in zip(CLASS_NAMES, PALETTE):
        draw.rectangle(
            [x, legend_y + 6, x + 14, legend_y + 20],
            fill=tuple(int(channel) for channel in palette_color),
            outline="black",
        )
        draw.text((x + 20, legend_y + 4), name, fill="black")
        x += 24 + len(name) * 7 + 12

    return grid
