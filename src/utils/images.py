import imageio
import numpy as np
from matplotlib import pyplot as plt


def read_image(image_name: str) -> np.ndarray:
    image_path = f"assets/images/{image_name}"
    image = imageio.imread(image_path)
    image = image.astype(np.float_)
    return image

def plot_image(image: np.ndarray):
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def plot_images(images: list[np.ndarray], titles: list[str], figsize=(16, 16)):
    fig, axs = plt.subplots(1, len(images), figsize=figsize)
    for i, (image, title) in enumerate(zip(images, titles)):
        axs[i].imshow(image, cmap="gray")
        axs[i].set_title(title)
        axs[i].axis("off")
    plt.show()
