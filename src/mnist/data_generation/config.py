import os
import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class VideoConfig:
    """Configuration for the Moving MNIST dataset"""

    num_frames: int = 20
    num_images: int = 100
    original_size: int = 28
    nums_per_image: int = 2
    shape: tuple[int, int] = (64, 64)


# helper functions copied from moving_mnist.py
def arr_from_img(image, mean: float = 0, std: float = 1) -> np.ndarray:
    """
    Get normalized np.float32 arrays of shape (width, height, channel) from image
    """
    width, height = image.size
    arr = image.getdata()
    channel = int(np.product(arr.size) / (width * height))

    return (
        np.asarray(arr, dtype=np.float32)
        .reshape((height, width, channel))
        .transpose(2, 1, 0)
        / 255.0
        - mean
    ) / std


def get_image_from_array(
    x: np.ndarray, index: int, mean: float = 0, std: float = 1
) -> np.ndarray:
    """
    Args:
        x: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    """
    channel, width, height = x.shape[1], x.shape[2], x.shape[3]
    ret = (
        (((x[index] + mean) * 255.0) * std)
        .reshape(channel, width, height)
        .transpose(2, 1, 0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    if channel == 1:
        ret = ret.reshape(height, width)
    return ret


def load_dataset(training: bool = True) -> np.ndarray:
    """loads mnist from web on demand"""

    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(file_name: str, source: str = "http://yann.lecun.com/exdb/mnist/"):
        print(f"Downloading  {file_name}")
        urlretrieve(source + file_name, file_name)

    import gzip

    def load_mnist_images(file_name: str) -> np.ndarray:
        if not os.path.exists(file_name):
            download(file_name)
        with gzip.open(file_name, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    if training:
        return load_mnist_images("train-images-idx3-ubyte.gz")
    return load_mnist_images("t10k-images-idx3-ubyte.gz")
