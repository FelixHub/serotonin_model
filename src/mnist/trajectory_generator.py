import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .config import VideoConfig


@dataclass(kw_only=True)
class TrajectoryGenerator(ABC):
    """Abstract base class for Moving MNIST trajectory generation"""

    config: VideoConfig = VideoConfig()
    glitch_frame: int = 0

    def __post_init__(self) -> None:
        self.position, self.velocity = self.reset_position_and_velocity()

    def reset_position_and_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        position = self.get_random_position()

        direction = self.get_random_direction()
        speed = self.get_random_speed()

        velocity = self.compute_velocity(direction, speed)

        return position, velocity

    def get_random_direction(self) -> np.ndarray:
        """Get a random direction for the Moving MNIST trajectory"""
        return np.pi * (np.random.rand(self.config.nums_per_image) * 2 - 1)

    def get_random_speed(self) -> np.ndarray:
        """Get a random speed for the Moving MNIST trajectory"""
        return np.random.randint(5, size=self.config.nums_per_image) + 2

    @staticmethod
    def compute_velocity(direction: np.ndarray, speed: np.ndarray) -> np.ndarray:
        """Get the velocity for the Moving MNIST trajectory from dir, speed"""
        return np.asarray(
            [
                (spd * math.cos(dir), spd * math.sin(dir))
                for dir, spd in zip(direction, speed)
            ]
        )

    @property
    def movement_limits(self) -> tuple[float, float]:
        """Get the limits for the Moving MNIST trajectory"""
        return (
            self.config.shape[0] - self.config.original_size,
            self.config.shape[1] - self.config.original_size,
        )

    def get_random_position(self) -> np.ndarray:
        """Get the initial position for the Moving MNIST trajectory"""
        x_lim, y_lim = self.movement_limits
        return np.asarray(
            [
                (np.random.rand() * x_lim, np.random.rand() * y_lim)
                for _ in range(self.config.nums_per_image)
            ]
        )

    def bounce_if_wall_hit(self, next_pos: np.ndarray) -> None:
        """Check if the MNIST digit hit the wall and bounce it off."""
        for i, pos in enumerate(next_pos):
            for j, coord in enumerate(pos):
                if coord < -2 or coord > self.movement_limits[j] + 2:
                    self.velocity[i] = list(
                        list(self.velocity[i][:j])
                        + [-1 * self.velocity[i][j]]
                        + list(self.velocity[i][j + 1 :])
                    )

    def _generate_single(self) -> np.ndarray:
        """Generate a single trajectory for the moving MNIST dataset"""
        self.position, self.velocity = self.reset_position_and_velocity()
        list_of_positions = [self.position]

        for frame_idx in range(self.config.num_frames - 1):
            self.bounce_if_wall_hit(next_pos=self.position + self.velocity)
            self.position = self.position + self.velocity

            if frame_idx == self.glitch_frame - 1:
                self._execute_glitch()
            list_of_positions.append(self.position)

        return np.asarray(list_of_positions)

    def generate(self) -> np.ndarray:
        """Generate the whole dataset"""
        dataset = [self._generate_single() for _ in range(self.config.num_images)]
        return np.asarray(dataset)

    @abstractmethod
    def _execute_glitch(self) -> None:
        """Execute the glitch for the Moving MNIST trajectory"""


class Standard(TrajectoryGenerator):
    """Standard Moving MNIST trajectory generator"""

    def _execute_glitch(self) -> None:
        return


@dataclass(kw_only=True)
class PositionGlitch(TrajectoryGenerator):
    """Moving MNIST trajectory generator with random position glitches"""

    glitch_frame: int = 10

    def _execute_glitch(self) -> None:
        self.position = self.get_random_position()


@dataclass(kw_only=True)
class OffsetPositionGlitch(TrajectoryGenerator):
    """Moving MNIST trajectory generator with offset position glitches"""

    glitch_frame: int = 10

    def _offset_position(self) -> np.ndarray:
        """Offset the position of the Moving MNIST digits"""
        position = self.position.copy()
        for i, pos in enumerate(self.position):
            for j, coord in enumerate(pos):
                offset = self.movement_limits[j] / 2
                if coord > offset:
                    position[i][j] -= offset
                elif coord < offset:
                    position[i][j] += offset
        return position

    def _execute_glitch(self) -> None:
        self.position = self._offset_position()


@dataclass(kw_only=True)
class SpeedGlitch(TrajectoryGenerator):
    """Moving MNIST trajectory generator with speed glitches"""

    glitch_frame: int = 10
    velocity_gain: float = 2

    def _execute_glitch(self) -> None:
        self.velocity *= self.velocity_gain


@dataclass(kw_only=True)
class DirectionGlitch(TrajectoryGenerator):
    """Moving MNIST trajectory generator with direction glitches"""

    glitch_frame: int = 10

    def _execute_glitch(self) -> None:
        direction = self.get_random_direction()
        speed = np.sqrt(self.velocity[:, 0] ** 2 + self.velocity[:, 1] ** 2)
        self.velocity = self.compute_velocity(direction, speed)


@dataclass(kw_only=True)
class TimedBounce(TrajectoryGenerator):
    """Moving MNIST trajectory generator with timed bounces"""

    glitch_frame: int = 5

    def reset_position_and_velocity(self) -> None:
        self.position, self.velocity = super().reset_position_and_velocity()
        position = self.calculate_initial_position_for_bounce()
        return position, self.velocity

    def calculate_initial_position_for_bounce(self) -> np.ndarray:
        """Calculate the initial position to time bounce at glitch_frame"""
        pos = 0 * self.position

        for i, v in enumerate(self.velocity):
            # get the axis with the higher speed
            j = np.argmax(np.abs(v))

            # if the speed is positive, offset initial pos from right edge
            # else offset from left edge
            if v[j] > 0:
                pos[i, j] = self.movement_limits[j] - v[j] * self.glitch_frame
            elif v[j] < 0:
                pos[i, j] = -v[j] * self.glitch_frame

            if v[1 - j] > 0:
                tmp_pos = self.movement_limits[1 - j] - (v[1 - j] * self.glitch_frame)
                pos[i, 1 - j] = np.random.rand() * tmp_pos
            elif v[1 - j] < 0:
                tmp_pos = -v[1 - j] * self.glitch_frame
                pos[i, 1 - j] = tmp_pos + np.random.rand() * (
                    self.movement_limits[1 - j] - tmp_pos
                )
        return pos

    def _execute_glitch(self) -> None:
        return


if __name__ == "__main__":
    print("This module defines the trajectory generator classes.")
    print("Please use the data_generator.py script to generate data.")
