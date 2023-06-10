import os

from PIL import Image, ImageDraw, ImageFont


def combine_images(columns: int, space: int, images: list) -> Image:
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    width_max = max([image.width for image in images])
    height_max = max([image.height for image in images])
    background_width = width_max * columns + (space * columns) - space
    background_height = height_max * rows + (space * rows) - space
    background = Image.new(
        "RGBA", (background_width, background_height), (255, 255, 255, 255)
    )
    x = 0
    y = 0
    for i, img in enumerate(images):
        x_offset = int((width_max - img.width) / 2)
        y_offset = int((height_max - img.height) / 2)
        background.paste(img, (x + x_offset, y + y_offset))
        x += width_max + space
        if not (i + 1) % columns:
            y += height_max + space
            x = 0
    return background


def to_thumbnail(
    image: Image,
    label: str,
    font: ImageFont,
    basewidth: int = 400,
) -> Image:
    wpercent = basewidth / float(image.width)
    hsize = int(float(image.height) * float(wpercent))

    thumbnail = image.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(thumbnail)
    draw.multiline_text(
        (10, 10),
        f"label:{label}",
        fill="white",
        font=font,
    )

    return thumbnail


def combine_prediction_visualisation(
    list_of_images: list,
    list_of_labels: list,
    font: ImageFont,
    basewidth: int,
) -> list:
    images = [
        to_thumbnail(image, label, font, basewidth)
        for (image, label) in zip(list_of_images, list_of_labels)
    ]

    return images


def generate_plots(images: list, labels: list, image_path: str, file_name: str) -> None:
    basewidth = 400

    font = ImageFont.truetype("Courier New Bold.ttf", 30)
    thumbnails = combine_prediction_visualisation(images, labels, font, basewidth)

    collage_image = combine_images(4, 10, thumbnails)
    collage_image.save(os.path.join(image_path, file_name))
    collage_image.show()
