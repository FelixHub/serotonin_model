from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from config import VideoConfig


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

    def sample_ids_for_dataset(self) -> np.ndarray:
        """Sample MNIST digits' indices for the whole dataset"""
        data = [
            self._sample_ids_for_trajectory() for _ in range(self.config.num_images)
        ]
        return np.asarray(data)


class Standard(MNISTSampler):
    """Standard Moving MNIST sampler"""

    def _sample_ids_for_trajectory(self) -> np.ndarray:
        return np.broadcast_to(
            self._sample_ids_for_frame(),
            shape=(self.config.num_frames, self.config.nums_per_image),
        )


@dataclass
class DigitGlitch(MNISTSampler):
    """Digit Glitch Moving MNIST sampler"""

    glitch_frame: int = 10

    def _sample_ids_for_trajectory(self) -> np.ndarray:
        data_1 = np.broadcast_to(
            self._sample_ids_for_frame(),
            shape=(self.glitch_frame, self.config.nums_per_image),
        )
        data_2 = np.broadcast_to(
            self._sample_ids_for_frame(),
            shape=(
                self.config.num_frames - self.glitch_frame,
                self.config.nums_per_image,
            ),
        )
        return np.vstack((data_1, data_2))


if __name__ == "__main__":
    print("This module defines the MNIST digit sampler classes.")
    print("Please use the data_generator.py script to generate data.")
