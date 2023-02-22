###########################################################################################
#
# Module to generate custom moving mnist video datasets
# refactored version of https://gist.github.com/tencia/afb129122a64bde3bd0c
#
# usage:
# `python data_generator.py`
#
# Notes:
# Under development. Currently only makes standard and position glitch datasets.
#
###########################################################################################

import math
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image


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


@dataclass
class VideoConfig:
    """Configuration for the Moving MNIST dataset"""

    num_frames: int = 20
    num_images: int = 100
    original_size: int = 28
    nums_per_image: int = 2
    shape: tuple[int, int] = (64, 64)


class TrajectoryGenerator(ABC):
    """Abstract base class for Moving MNIST trajectory generation"""

    def __init__(self, config: VideoConfig = VideoConfig()) -> None:
        self.config = config
        self.limits = self.get_movement_limits(config)

        direction = self.get_random_direction(config.nums_per_image)
        speed = self.get_random_speed(config.nums_per_image)

        self.velocity = self.get_initial_velocity(direction, speed)
        self.position = self.get_initial_position(self.config, *self.limits)

    @staticmethod
    def get_random_direction(nums_per_image: int = 2) -> np.ndarray:
        """Get a random direction for the Moving MNIST trajectory"""
        return np.pi * (np.random.rand(nums_per_image) * 2 - 1)

    @staticmethod
    def get_random_speed(nums_per_image: int = 2) -> np.ndarray:
        """Get a random speed for the Moving MNIST trajectory"""
        return np.random.randint(5, size=nums_per_image) + 2

    @staticmethod
    def get_initial_velocity(direction: np.ndarray, speed: np.ndarray) -> np.ndarray:
        """Get the initial velocity for the Moving MNIST trajectory"""
        return np.asarray(
            [
                (speed * math.cos(direc), speed * math.sin(direc))
                for direc, speed in zip(direction, speed)
            ]
        )

    @staticmethod
    def get_movement_limits(config: VideoConfig) -> tuple[float, float]:
        """Get the limits for the Moving MNIST trajectory"""
        return (
            config.shape[0] - config.original_size,
            config.shape[1] - config.original_size,
        )

    @staticmethod
    def get_initial_position(
        config: VideoConfig, x_lim: float, y_lim: float
    ) -> np.ndarray:
        """Get the initial position for the Moving MNIST trajectory"""
        return np.asarray(
            [
                (np.random.rand() * x_lim, np.random.rand() * y_lim)
                for _ in range(config.nums_per_image)
            ]
        )

    def bounce_on_wall(self, pos: np.ndarray, next_pos: np.ndarray) -> None:
        """Check if the MNIST digit hit the wall and bounce it off."""
        for i, pos in enumerate(next_pos):
            for j, coord in enumerate(pos):
                if coord < -2 or coord > self.limits[j] + 2:
                    self.velocity[i] = list(
                        list(self.velocity[i][:j])
                        + [-1 * self.velocity[i][j]]
                        + list(self.velocity[i][j + 1 :])
                    )

    @abstractmethod
    def _generate_single(self, glitch_frame: int = 0) -> np.ndarray:
        """Generate a single trajectory for the moving MNIST dataset"""

    def generate(self, glitch_frame: int = 0) -> np.ndarray:
        """Generate the whole dataset"""
        dataset = [
            self._generate_single(glitch_frame) for _ in range(self.config.num_images)
        ]
        return np.asarray(dataset)


class StandardTrajectoryGenerator(TrajectoryGenerator):
    """Standard Moving MNIST trajectory generator"""

    def _generate_single(self, glitch_frame: int = 0) -> np.ndarray:
        list_of_positions = [self.position]

        for _ in range(self.config.num_frames - 1):
            self.bounce_on_wall(self.position, self.position + self.velocity)
            self.position = self.position + self.velocity
            list_of_positions.append(self.position)

        return np.asarray(list_of_positions)


@dataclass
class PositionGlitchTrajectoryGenerator(TrajectoryGenerator):
    """Moving MNIST trajectory generator with position glitches"""

    def _generate_single(self, glitch_frame: int = 10) -> np.ndarray:
        list_of_positions = [self.position]

        for frame_idx in range(self.config.num_frames - 1):
            self.bounce_on_wall(self.position, self.position + self.velocity)
            self.position = self.position + self.velocity

            if frame_idx == glitch_frame:
                self.position = self.get_initial_position(self.config, *self.limits)

            list_of_positions.append(self.position)

        return np.asarray(list_of_positions)


@dataclass
class MNISTSampler(ABC):
    """Abstract base class for sampling MNIST digits"""

    config: VideoConfig = VideoConfig()
    idx_max: int = 60_000

    def _sample_ids_for_frame(self) -> np.ndarray:
        """Sample MNIST digits' indices for one frame"""
        return np.random.randint(0, self.idx_max, self.config.nums_per_image)

    @abstractmethod
    def _sample_ids_for_trajectory(self) -> np.ndarray:
        """Map the MNIST digits' indices across the length of the trajectory"""

    @abstractmethod
    def sample_ids_for_dataset(self) -> np.ndarray:
        """Sample MNIST digits' indices for the whole dataset"""


class StandardMNISTSampler(MNISTSampler):
    """Standard Moving MNIST sampler"""

    def _sample_ids_for_trajectory(self) -> np.ndarray:
        return np.broadcast_to(
            self._sample_ids_for_frame(),
            shape=(self.config.num_frames, self.config.nums_per_image),
        )

    def sample_ids_for_dataset(self) -> np.ndarray:
        data = [
            self._sample_ids_for_trajectory() for _ in range(self.config.num_images)
        ]
        return np.asarray(data)


@dataclass
class MovingMNISTFactory:
    """Moving MNIST dataset generator factory class."""

    trajectory_generator: TrajectoryGenerator
    mnist_sampler: MNISTSampler
    mnist_data: np.ndarray  # load with load_dataset(True)
    config: VideoConfig = VideoConfig()

    def _map_digits_to_positions_single_image(
        self, mnist_ids: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Map the MNIST digits to the positions in each trajectory"""
        canvas_combined = np.zeros((1, *self.config.shape), dtype=np.float32)

        for mnist_idx, position in zip(mnist_ids, positions):
            mnist_image = Image.fromarray(
                get_image_from_array(self.mnist_data, mnist_idx)
            ).resize(
                (self.config.original_size, self.config.original_size), Image.ANTIALIAS
            )
            canvas = Image.new("L", self.config.shape)
            canvas.paste(mnist_image, tuple(position.astype(int)))
            canvas_combined += arr_from_img(canvas)

        final_image = (
            (canvas_combined * 255)
            .clip(0, 255)
            .astype(np.uint8)
            .reshape(self.config.shape)
        )

        return final_image.T

    def _map_digits_to_positions(self, mnist_ids: np.ndarray, positions: np.ndarray):
        """Map the MNIST digits to the positions in the whole dataset"""
        # do it with np.apply_over_axes(); currently dumb but works

        dataset = np.empty((positions.shape[0], positions.shape[1], *self.config.shape))

        for n in range(positions.shape[0]):
            for frame_idx in range(positions.shape[1]):
                dataset[
                    n, frame_idx, :, :
                ] = self._map_digits_to_positions_single_image(
                    mnist_ids[n, frame_idx, :], positions[n, frame_idx, :, :]
                )
        return dataset

    def make(self) -> np.ndarray:
        """Make the Moving MNIST dataset"""
        mnist_ids = self.mnist_sampler.sample_ids_for_dataset()
        positions = self.trajectory_generator.generate()

        return self._map_digits_to_positions(mnist_ids, positions)

    def save(self, path: str, dataset: np.ndarray) -> None:
        """Save the Moving MNIST dataset to a file"""
        np.save(path, dataset)

    def load(self, path: str) -> np.ndarray:
        """Load the Moving MNIST dataset from a file"""
        return np.load(path)


def main():
    """Main function"""

    print("Generating standard Moving MNIST dataset...\n")

    config = VideoConfig()
    trajectory_generator = StandardTrajectoryGenerator(config)
    mnist_sampler = StandardMNISTSampler(config=config)
    mnist_data = load_dataset(training=True)

    factory = MovingMNISTFactory(
        trajectory_generator=trajectory_generator,
        mnist_sampler=mnist_sampler,
        mnist_data=mnist_data,
        config=config,
    )
    dataset = factory.make()
    factory.save("test_factory_data.npy", dataset)

    print("\nDone! Saved to data/test_factory_data.npy")


if __name__ == "__main__":
    main()
