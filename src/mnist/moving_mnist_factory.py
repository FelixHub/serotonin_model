from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import VideoConfig, arr_from_img, get_image_from_array
from .digit_sampler import MNISTSampler
from .trajectory_generator import TrajectoryGenerator


@dataclass
class MovingMNISTFactory:
    """Moving MNIST dataset generator factory class."""

    trajectory_generator: TrajectoryGenerator
    mnist_sampler: MNISTSampler
    mnist_data: np.ndarray  # load with load_dataset(True)
    config: VideoConfig

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


if __name__ == "__main__":
    print("This module defines the factory classes.")
    print("Please use the data_generator.py script to generate data.")
